"""KG 순회 -> evidence path -> (persona, query_type) 시나리오 -> LLM 으로 질문/정답 생성.

RAGAS testset 생성 철학(path -> scenario -> question/reference)을 networkx + pydantic 로 재구현.
LLM 은 '질문 문장화'와 'reference 정리'에만 쓰고, 사실은 KG path 에서만 가져온다.
"""
from __future__ import annotations

import json
import pathlib
from typing import Literal, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from .personas import DEFAULT_PERSONAS, PersonaSpec

QueryType = Literal["single_hop", "multi_hop_2", "multi_hop_3"]


class PathEvidence(BaseModel):
    nodes: list[str]
    relations: list[str]   # len == len(nodes) - 1

    @property
    def hops(self) -> int:
        return len(self.relations)

    @property
    def text(self) -> str:
        parts = [self.nodes[0]]
        for rel, nxt in zip(self.relations, self.nodes[1:]):
            parts.append(f"-[{rel}]->")
            parts.append(nxt)
        return " ".join(parts)


class TestItem(BaseModel):
    id: str
    split: Literal["general_rag", "graph_reasoning"]
    query_type: str
    persona: str
    question: str
    reference: str
    evidence_path: list[str]
    evidence_relations: list[str]
    expected_entities: list[str] = Field(default_factory=list)
    review_status: Literal["pending", "keep", "edit", "drop"] = "pending"
    reviewer_note: str = ""


def enumerate_paths(graph, min_hops: int = 1, max_hops: int = 3) -> list[PathEvidence]:
    """방향 단순 경로(cycle 없음) 후보를 hop 길이별로 모은다."""
    import networkx as nx

    seen = set()
    out: list[PathEvidence] = []
    nodes = list(graph.nodes)
    for src in nodes:
        for tgt in nodes:
            if src == tgt:
                continue
            for p in nx.all_simple_paths(graph, src, tgt, cutoff=max_hops):
                hops = len(p) - 1
                if hops < min_hops:
                    continue
                key = tuple(p)
                if key in seen:
                    continue
                seen.add(key)
                rels = [graph[p[i]][p[i + 1]]["relation"] for i in range(hops)]
                out.append(PathEvidence(nodes=p, relations=rels))
    return out


_QT = {1: "single_hop", 2: "multi_hop_2", 3: "multi_hop_3"}


class _GeneratedQA(BaseModel):
    question: str = Field(description="한국어 사용자 질문 (의문문)")
    reference: str = Field(description="evidence path 만 근거로 한 한국어 정답")
    expected_entities: list[str] = Field(default_factory=list, description="정답에 등장해야 하는 핵심 엔티티")


_SYSTEM = """너는 RAG 평가셋 생성기다. 주어진 지식그래프 evidence path 에 근거해 한국어 질문과 정답을 만든다.
- path 에 있는 사실만 쓰고, 그래프에 없는 사실을 지어내지 마라.
- query_type 이 multi_hop 이면 질문이 path 의 여러 관계를 모두 거쳐야 답할 수 있게 만든다(중간 노드가 답에 필요).
- persona 의 말투/관심사를 반영하되, 정답은 간결하게 핵심 근거만 담는다."""


def generate_testset_from_kg(
    graph,
    llm,
    personas: Optional[list[PersonaSpec]] = None,
    min_hops: int = 1,
    max_hops: int = 3,
    max_per_type: int = 6,
) -> list[TestItem]:
    """KG 를 순회해 평가셋 후보를 생성한다(review_status='pending').

    같은 근거가 과도하게 반복되지 않도록 hop 길이별 max_per_type 개만 취한다(결정적: 앞에서부터).
    최종 gold 는 사람이 keep/edit/drop 으로 검수해 고정하는 것을 전제로 한다.
    """
    personas = personas or DEFAULT_PERSONAS
    paths = enumerate_paths(graph, min_hops, max_hops)

    # hop 길이별로 그룹화 후 앞에서 max_per_type 개
    by_hops: dict[int, list[PathEvidence]] = {}
    for p in paths:
        by_hops.setdefault(p.hops, []).append(p)

    qa_llm = llm.with_structured_output(_GeneratedQA)
    items: list[TestItem] = []
    counter = 0
    for hops in sorted(by_hops):
        for p in by_hops[hops][:max_per_type]:
            persona = personas[counter % len(personas)]
            qt = _QT.get(hops, "multi_hop_3")
            split = "general_rag" if hops == 1 else "graph_reasoning"
            gen = qa_llm.invoke([
                SystemMessage(content=_SYSTEM),
                HumanMessage(content=(
                    f"persona: {persona.name} ({persona.style})\n"
                    f"query_type: {qt}\n"
                    f"evidence_path: {p.text}\n"
                )),
            ])
            counter += 1
            items.append(TestItem(
                id=f"kg-{counter:03d}",
                split=split,
                query_type=qt,
                persona=persona.name,
                question=gen.question,
                reference=gen.reference,
                evidence_path=p.nodes,
                evidence_relations=p.relations,
                expected_entities=gen.expected_entities or [p.nodes[0], p.nodes[-1]],
            ))
    return items


def validate_item(item: TestItem) -> list[str]:
    """자동 품질 체크. 문제 목록을 반환(비면 통과).

    multi-hop 가짜(첫/끝 엣지만으로 답해지는 경우)를 거르려고, 경로의 **중간 노드**가
    질문이나 정답에 실제로 쓰였는지 본다(중간 노드가 답에 필요한지의 휴리스틱).
    """
    issues = []
    if "?" not in item.question:
        issues.append("question_not_interrogative")
    evidence_text = " ".join(item.evidence_path)
    for ent in item.expected_entities:
        if ent not in evidence_text and ent not in item.reference:
            issues.append(f"entity_not_in_evidence:{ent}")
    if item.query_type.startswith("multi_hop"):
        if len(item.evidence_relations) < 2:
            issues.append("multi_hop_requires_two_edges")
        # 중간 노드가 질문/정답에 등장해야 진짜 multi-hop (single-edge 로 풀리지 않게)
        qr = f"{item.question} {item.reference}"
        for node in item.evidence_path[1:-1]:
            if node not in qr:
                issues.append(f"multi_hop_middle_node_unused:{node}")
    return issues


def to_eval_dataset(items: list[TestItem], rag_fn):
    """검수된 TestItem 들에 RAG 를 실행해 평가용 EvaluationDataset 으로.

    rag_fn(question) -> (answer: str, contexts: list[str] | list[Document]).
    """
    from ..schema import EvaluationDataset

    rows = []
    for it in items:
        answer, contexts = rag_fn(it.question)
        rows.append({
            "user_input": it.question,
            "retrieved_contexts": contexts,
            "response": answer,
            "reference": it.reference,
        })
    return EvaluationDataset.from_list(rows)


def save_jsonl(items: list[TestItem], path) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(it.model_dump_json() + "\n")


def load_jsonl(path) -> list[TestItem]:
    lines = pathlib.Path(path).read_text(encoding="utf-8").splitlines()
    return [TestItem.model_validate_json(line) for line in lines if line.strip()]
