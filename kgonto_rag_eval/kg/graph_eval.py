"""KG/온톨로지 평가 (결정적, LLM 불필요).

- compare_kg: 추출 KG vs 정답 KG 의 트리플 precision/recall/F1.
  근거: 관계추출/KG 구축의 트리플 단위 P/R/F1 (head·relation·tail 정확 일치;
        예: Mondal et al. 2021, "End-to-End NLP Knowledge Graph Construction", arXiv:2106.01167).
- check_ontology / schema_violation_rate: 허용 타입·관계 위반, dangling 관계 검사.
  근거: RDF 그래프 제약 검증(W3C SHACL, https://www.w3.org/TR/shacl/) 의 동기.
- non_isolated_node_ratio: 엣지를 가진 노드 비율(구조 보조 지표).
  근거: GraphRAG-Bench(arXiv:2506.02404) 의 구축 평가에서 쓰는 non-isolated node 관점.
"""
from __future__ import annotations

from typing import Any, Iterable, Optional


def _as_triples(x: Any) -> set[tuple]:
    """(s,rel,o) 집합으로. 입력: 트리플 iterable 또는 15 relations dict 리스트."""
    out: set[tuple] = set()
    for r in x:
        if isinstance(r, dict):
            out.add((r["source"], r["relation"], r["target"]))
        else:
            s, rel, o = r
            out.add((s, rel, o))
    return out


def compare_kg(pred: Iterable, reference: Iterable) -> dict:
    """추출 KG(pred)를 정답(reference)과 트리플 단위로 비교합니다. precision/recall/F1 + 누락/추가."""
    P, G = _as_triples(pred), _as_triples(reference)
    tp = len(P & G)
    precision = tp / len(P) if P else 0.0
    recall = tp / len(G) if G else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "precision_raw": precision,   # 회귀 테스트·세밀 비교용 원점수
        "recall_raw": recall,
        "f1_raw": f1,
        "matched": tp,
        "pred_total": len(P),
        "reference_total": len(G),
        "missing": sorted(G - P),   # 정답에 있는데 못 뽑음
        "extra": sorted(P - G),     # pred 에만 있음(노이즈 또는 정답 누락)
    }


def check_ontology(
    entities: Iterable[dict],
    relations: Iterable,
    allowed_types: Optional[Iterable[str]] = None,
    allowed_relations: Optional[Iterable[str]] = None,
    subclass_rules: Optional[dict[str, str]] = None,
) -> dict:
    """온톨로지 적합성 검사 (결정적).

    - allowed_types: 허용 엔티티 타입 화이트리스트.
    - allowed_relations: 허용 관계 타입 화이트리스트.
    - subclass_rules: {하위클래스: 상위클래스}. 클래스 계층 존재 확인용(관계 domain/range 검사는 아님).
    반환: {ok, issues:[...], stats}.
    """
    ents = list(entities)
    names = {e["name"] for e in ents}
    types_in = {e.get("type") for e in ents}
    triples = _as_triples(relations)
    rels_in = {rel for _, rel, _ in triples}

    issues: list[str] = []
    if allowed_types is not None:
        allowed_types = set(allowed_types)
        for e in ents:
            if e.get("type") not in allowed_types:
                issues.append(f"bad_entity_type:{e['name']}={e.get('type')}")
    if allowed_relations is not None:
        allowed_relations = set(allowed_relations)
        for s, rel, o in triples:
            if rel not in allowed_relations:
                issues.append(f"bad_relation_type:{s}-{rel}->{o}")
    # dangling: 관계 양 끝이 엔티티 목록에 있는지
    for s, rel, o in triples:
        if s not in names:
            issues.append(f"dangling_source:{s}")
        if o not in names:
            issues.append(f"dangling_target:{o}")
    # subClassOf 일관성: 규칙의 상/하위 클래스가 실제 타입에 존재하는지
    if subclass_rules:
        for sub, sup in subclass_rules.items():
            if sub not in types_in:
                issues.append(f"subclass_missing:{sub}")
            if sup not in types_in and (allowed_types is None or sup not in allowed_types):
                issues.append(f"superclass_unknown:{sup}")

    return {
        "ok": not issues,
        "issues": issues,
        "stats": {
            "entities": len(ents),
            "relations": len(triples),
            "entity_types": sorted(t for t in types_in if t),
            "relation_types": sorted(rels_in),
        },
    }


def schema_violation_rate(
    entities: Iterable[dict],
    relations: Iterable,
    allowed_types: Optional[Iterable[str]] = None,
    allowed_relations: Optional[Iterable[str]] = None,
) -> dict:
    """온톨로지/스키마 제약을 위반한 항목 비율 (결정적).

    check_ontology 의 위반 건수(bad type/관계, dangling)를 (엔티티+관계) 전체 대비 비율로 냅니다.
    "그래프가 약속한 스키마를 얼마나 지켰나" 를 한 숫자로 봅니다.
    근거: RDF 그래프 제약 검증(W3C SHACL, https://www.w3.org/TR/shacl/) 의 동기.
    """
    ents = list(entities)
    triples = _as_triples(relations)
    report = check_ontology(ents, triples, allowed_types, allowed_relations)
    names = {e["name"] for e in ents}
    violated_items: set[tuple] = set()
    if allowed_types is not None:
        allowed_types = set(allowed_types)
        for e in ents:
            if e.get("type") not in allowed_types:
                violated_items.add(("entity", e["name"]))
    if allowed_relations is not None:
        allowed_relations = set(allowed_relations)
    for s, rel, o in triples:
        if (allowed_relations is not None and rel not in allowed_relations) or s not in names or o not in names:
            violated_items.add(("relation", s, rel, o))
    total = len(ents) + len(triples)
    violations = len(violated_items)
    return {
        "schema_violation_rate": round(violations / total, 3) if total else 0.0,
        "violations": violations,
        "checked": total,
        "issues": report["issues"],
    }


def non_isolated_node_ratio(entities: Iterable, relations: Iterable) -> dict:
    """엣지를 하나 이상 가진 노드의 비율 (구조 보조 지표, 정답성 지표가 아님).

    검색·추론에 참여 가능한 노드가 충분히 연결됐는지 봅니다. 고립 노드가 많으면
    그래프는 "예쁘게" 보여도 실제 검색에 안 잡힐 수 있습니다.
    근거: GraphRAG-Bench(arXiv:2506.02404) 의 구축 평가에서 쓰는 non-isolated node 관점.
    """
    ents = list(entities)
    names = {e["name"] if isinstance(e, dict) else e for e in ents}
    triples = _as_triples(relations)
    connected: set = set()
    for s, _, o in triples:
        connected.add(s)
        connected.add(o)
    non_isolated = names & connected
    ratio = len(non_isolated) / len(names) if names else 0.0
    return {
        "non_isolated_node_ratio": round(ratio, 3),
        "non_isolated": len(non_isolated),
        "total_nodes": len(names),
        "isolated": sorted(names - connected),
    }
