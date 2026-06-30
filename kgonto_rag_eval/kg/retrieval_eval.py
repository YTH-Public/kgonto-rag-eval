"""검색/근거 품질 평가 (결정적, LLM 불필요).

지식 그래프 RAG 에서 "질문에 필요한 근거(경로/트리플)를 검색이 실제로 가져왔는가" 를 봅니다.
답변 점수(RAGAS 계열) 와 분리해, 검색 단계의 성공/실패를 먼저 드러내기 위한 지표입니다.

지표와 근거:
- path_recall: 정보검색 Recall@k 표준을 KG 근거 경로(엔티티)에 적용한 graph-aware recall.
- triple_coverage: RAGAS context recall(Es et al. 2024, arXiv:2309.15217) 과 KG 트리플 recall 을
  결합한 project-defined metric (표준 단일 지표가 아니라 수업용 조합 지표).
"""
from __future__ import annotations

import re
from typing import Any, Iterable


_KOREAN_PARTICLES = frozenset("은는이가을를와과의에로도만")


def _entity_in_text(entity: str, text: str) -> bool:
    """Return True for a real mention, avoiding prefix substring hits."""
    if not entity:
        return False
    pattern = re.compile(re.escape(entity))
    for m in pattern.finditer(text):
        before = text[m.start() - 1] if m.start() > 0 else ""
        after = text[m.end()] if m.end() < len(text) else ""
        if before and (before.isalnum() or before == "_"):
            continue
        if after and (after.isalnum() or after == "_") and after not in _KOREAN_PARTICLES:
            continue
        return True
    return False


def _as_text(retrieved: Any) -> str:
    """검색 결과를 하나의 텍스트로 합칩니다. 문자열/문자열리스트/Document리스트/트리플 모두 수용."""
    if retrieved is None:
        return ""
    if isinstance(retrieved, str):
        return retrieved
    if hasattr(retrieved, "page_content"):
        return str(retrieved.page_content)
    if isinstance(retrieved, dict):
        return " ".join(str(retrieved.get(k, "")) for k in ("source", "relation", "target")) or str(retrieved)
    if isinstance(retrieved, (tuple, list)) and len(retrieved) == 3 and not any(isinstance(x, (tuple, list, dict)) for x in retrieved):
        return f"{retrieved[0]} {retrieved[1]} {retrieved[2]}"
    parts: list[str] = []
    for r in retrieved:
        if isinstance(r, str):
            parts.append(r)
        elif hasattr(r, "page_content"):          # langchain Document
            parts.append(r.page_content)
        elif isinstance(r, (tuple, list)) and len(r) == 3:
            parts.append(f"{r[0]} {r[1]} {r[2]}")
        elif isinstance(r, dict):
            parts.append(" ".join(str(r.get(k, "")) for k in ("source", "relation", "target")) or str(r))
        else:
            parts.append(str(r))
    return "\n".join(parts)


def _triples(x: Any) -> set[tuple]:
    out: set[tuple] = set()
    for r in x or []:
        if isinstance(r, dict):
            out.add((r["source"], r["relation"], r["target"]))
        elif isinstance(r, (tuple, list)) and len(r) == 3:
            out.add(tuple(r))
    return out


def path_triples(evidence_path: list[str], evidence_relations: list[str]) -> list[tuple]:
    """TestItem 의 evidence_path/relations 를 (s, rel, o) 트리플 리스트로 펼칩니다."""
    return [(evidence_path[i], evidence_relations[i], evidence_path[i + 1])
            for i in range(len(evidence_relations))]


def path_recall(gold_entities: Iterable[str], retrieved: Any) -> dict:
    """gold 근거 경로의 엔티티 중 검색 결과에 나타난 비율 (graph-aware recall).

    gold_entities: 정답 근거 경로의 노드 이름들 (예: TestItem.evidence_path).
    retrieved: 검색 결과. 텍스트 / 문자열 리스트 / langchain Document 리스트 / (s,rel,o) 트리플 모두 가능.
        벡터 RAG(텍스트)와 그래프 RAG(트리플) 에 같은 기준으로 적용해 baseline 비교가 됩니다.

    근거: 정보검색의 Recall@k 표준을 KG 근거 경로에 적용한 graph-aware recall 입니다
    (Recall@k 자체는 IR 표준 지표, 경로 적용은 GraphRAG 평가 관행).
    """
    gold = [g for g in gold_entities if g]
    text = _as_text(retrieved)
    found = [g for g in gold if _entity_in_text(g, text)]
    recall = len(found) / len(gold) if gold else 0.0
    return {
        "path_recall": round(recall, 3),
        "found": found,
        "missing": [g for g in gold if g not in found],
        "gold_total": len(gold),
    }


def triple_coverage(gold_triples: Iterable, retrieved_triples: Iterable) -> dict:
    """gold 근거 트리플 중 검색 결과 트리플이 덮은 비율 (트리플 단위 recall).

    트리플을 돌려주는 검색기(그래프/하이브리드)용입니다. 검색이 텍스트만 주면 path_recall 을 쓰세요.

    근거: RAGAS context recall(Es et al. 2024, arXiv:2309.15217) 과 KG 트리플 recall 을
    결합한 project-defined metric 입니다(표준 단일 지표가 아니라 수업용 조합 지표).
    """
    gold, ret = _triples(gold_triples), _triples(retrieved_triples)
    covered = gold & ret
    cov = len(covered) / len(gold) if gold else 0.0
    return {
        "triple_coverage": round(cov, 3),
        "covered": sorted(covered),
        "missing": sorted(gold - ret),
        "gold_total": len(gold),
    }
