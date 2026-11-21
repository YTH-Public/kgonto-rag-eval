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
    answerable: bool = True   # False 면 그래프/문서에 답이 없어 "모른다"가 정답 (abstention)
    review_status: Literal["pending", "keep", "edit", "drop"] = "pending"
    reviewer_note: str = ""


def enumerate_paths(graph, min_hops: int = 1, max_hops: int = 3) -> list[PathEvidence]:
    """방향 단순 경로(같은 노드를 두 번 지나지 않음) 후보를 hop 길이별로 모은다.

    같은 두 노드 사이에 관계가 여럿이면(MultiDiGraph) 관계마다 다른 경로로 셉니다.
    """
    import networkx as nx

    seen = set()
    out: list[PathEvidence] = []
    nodes = list(graph.nodes)
    multi = graph.is_multigraph()
    for src in nodes:
        for tgt in nodes:
            if src == tgt:
                continue
            for edges in nx.all_simple_edge_paths(graph, src, tgt, cutoff=max_hops):
                hops = len(edges)
                if hops < min_hops:
                    continue
                path_nodes = [edges[0][0]] + [e[1] for e in edges]
                if multi:
                    rels = [graph.edges[e]["relation"] for e in edges]
                else:
                    rels = [graph[u][v]["relation"] for u, v in edges]
                key = (tuple(path_nodes), tuple(rels))
                if key in seen:
                    continue
                seen.add(key)
                out.append(PathEvidence(nodes=path_nodes, relations=rels))
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
    최종 정답 데이터는 사람이 keep/edit/drop 으로 검수해 고정하는 것을 전제로 한다.
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
    한계: 이름이 문장에 있다고 진짜 multi-hop 이라는 보장은 아닙니다. 핵심 엣지 하나를 가려도
    답할 수 있는지는 사람이 확인하세요. 통과해도 review_status 는 pending 그대로입니다.
    """
    issues = []
    if "?" not in item.question:
        issues.append("question_not_interrogative")
    # abstention(모름) 문항은 근거가 없는 게 정상 -> 근거 관련 검사는 건너뛴다
    if not item.answerable:
        if item.evidence_path:
            issues.append("abstention_should_have_no_evidence")
        return issues
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


def to_eval_dataset(items: list[TestItem], rag_fn, allow_status=("keep", "edit")):
    """사람이 검수해 승인한 TestItem 에만 RAG 를 실행해 평가용 EvaluationDataset 으로.

    rag_fn(question) -> (answer: str, contexts: list[str] | list[Document]).
        contexts 는 **그때 실제로 검색된** 근거여야 합니다(정답 쪽을 나중에 붙이지 마세요).
    allow_status: 실행할 review_status. 기본은 keep · edit 만(pending · drop 은 건너뜀).
        edit 은 고친 뒤 다시 확인했다는 뜻으로 씁니다.
    """
    from ..schema import EvaluationDataset

    allowed = set(allow_status)
    rows = []
    for it in items:
        if it.review_status not in allowed:
            continue
        answer, contexts = rag_fn(it.question)
        rows.append({
            "sample_id": it.id,
            "user_input": it.question,
            "retrieved_contexts": list(contexts),
            "response": answer,
            "reference": it.reference,
            "metadata": {"split": it.split, "query_type": it.query_type, "answerable": it.answerable},
        })
    return EvaluationDataset.from_list(rows)


def leakage_report(items: list[TestItem], split_of=lambda it: it.split) -> dict:
    """평가셋 누수 점검 두 가지(결정적).

    answer_in_question: 정답 엔티티(경로의 출발 노드 제외)가 질문에 그대로 들어 있는 문항.
        질문만 보고 답을 알 수 있으면 검색 평가가 부풀려집니다.
    split_overlap: 같은 근거 경로가 서로 다른 split 에 들어간 경우.
    세 번째 누수(평가셋을 만든 KG 와 채점 정답이 같은 KG 라 순환하는 문제)는 코드로 잡을 수 없어 사람이 확인합니다.
    """
    in_question = []
    for it in items:
        if not it.answerable:
            continue
        start = it.evidence_path[0] if it.evidence_path else None
        leaked = [e for e in it.expected_entities if e != start and e and e in it.question]
        if leaked:
            in_question.append({"id": it.id, "leaked": leaked})
    by_path: dict = {}
    for it in items:
        if it.evidence_path:
            by_path.setdefault(tuple(it.evidence_path), set()).add(split_of(it))
    overlap = [list(p) for p, splits in by_path.items() if len(splits) > 1]
    return {"answer_in_question": in_question, "split_overlap": overlap}


def save_jsonl(items: list[TestItem], path) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(it.model_dump_json() + "\n")


def load_jsonl(path) -> list[TestItem]:
    lines = pathlib.Path(path).read_text(encoding="utf-8").splitlines()
    return [TestItem.model_validate_json(line) for line in lines if line.strip()]
