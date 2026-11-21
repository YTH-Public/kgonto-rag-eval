"""KG/온톨로지 평가 (결정적, LLM 불필요).

- compare_kg: 추출 KG vs 정답 KG 의 트리플 precision/recall/F1.
  근거: 관계추출/KG 구축의 트리플 단위 P/R/F1 (head·relation·tail 정확 일치;
        예: Mondal et al. 2021, "End-to-End NLP Knowledge Graph Construction", arXiv:2106.01167).
- check_ontology / schema_violation_rate: 허용 타입 · 관계 · (주어 타입, 관계, 목적어 타입) 패턴 위반, dangling 관계.
  근거: RDF 그래프 제약 검증(W3C SHACL, https://www.w3.org/TR/shacl/) 의 동기. 추론 엔진은 아닙니다.
- non_isolated_node_ratio: 엣지를 가진 노드 비율(구조 보조 지표).
  근거: GraphRAG-Bench(arXiv:2506.02404) 의 구축 평가 관점을 이 프로젝트에서 단순화한 지표입니다.
"""
from __future__ import annotations

from typing import Any, Callable, Iterable, Optional, Sequence

from ._triples import as_triples


def compare_kg(
    pred: Iterable,
    reference: Iterable,
    normalize: Optional[Callable[[str], str]] = None,
    scope: Sequence[str] = (),
) -> dict:
    """추출 KG(pred)를 정답(reference)과 트리플 단위로 비교합니다. precision/recall/F1 + 누락/추가.

    normalize: 주어 · 목적어에 같은 정리 함수를 적용합니다(예: "(주)대창" -> "대창").
        모든 괄호 · 공백을 지우는 식의 넓은 정리는 서로 다른 대상을 합칠 수 있으니 피하세요.
    scope: dict 입력일 때 문서 범위 칸(예: ("rcept_no",)). 다른 보고서의 같은 이름 트리플을 섞지 않습니다.
    """
    P = as_triples(pred, normalize, scope)
    G = as_triples(reference, normalize, scope)
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
        "missing": sorted(G - P, key=str),   # 정답에 있는데 못 뽑음
        "extra": sorted(P - G, key=str),     # pred 에만 있음(노이즈 또는 정답 누락)
    }


def _as_set(x: Optional[Iterable]) -> Optional[set]:
    """허용 목록을 한 번만 집합으로 (generator 를 두 번 읽어 비는 문제 방지)."""
    return None if x is None else set(x)


def _violations(ents, triples, allowed_types, allowed_relations, allowed_patterns):
    """위반 항목과 이유. 반환: (issues 문자열 목록, 위반 항목 키 집합)."""
    names = {e["name"] for e in ents}
    type_of = {e["name"]: e.get("type") for e in ents}
    issues: list[str] = []
    bad: set[tuple] = set()
    if allowed_types is not None:
        for e in ents:
            if e.get("type") not in allowed_types:
                issues.append(f"bad_entity_type:{e['name']}={e.get('type')}")
                bad.add(("entity", e["name"]))
    for s, rel, o in sorted(triples, key=str):
        key = ("relation", s, rel, o)
        if allowed_relations is not None and rel not in allowed_relations:
            issues.append(f"bad_relation_type:{s}-{rel}->{o}")
            bad.add(key)
        if s not in names:
            issues.append(f"dangling_source:{s}")
            bad.add(key)
        if o not in names:
            issues.append(f"dangling_target:{o}")
            bad.add(key)
        if allowed_patterns is not None and s in names and o in names:
            pattern = (type_of[s], rel, type_of[o])
            if pattern not in allowed_patterns:
                issues.append(f"bad_pattern:{pattern[0]}-{rel}->{pattern[2]} ({s}->{o})")
                bad.add(key)
    return issues, bad


def check_ontology(
    entities: Iterable[dict],
    relations: Iterable,
    allowed_types: Optional[Iterable[str]] = None,
    allowed_relations: Optional[Iterable[str]] = None,
    subclass_rules: Optional[dict[str, str]] = None,
    allowed_patterns: Optional[Iterable[tuple]] = None,
) -> dict:
    """온톨로지 적합성 검사 (결정적).

    - allowed_types / allowed_relations: 허용 목록.
    - allowed_patterns: 허용 (주어 타입, 관계, 목적어 타입) 조합. 예: {("회사", "공급", "회사")}
      관계 이름은 맞아도 타입 조합이 틀린 경우(품목이 회사에 공급)를 잡습니다.
    - subclass_rules: {하위클래스: 상위클래스}. 클래스 이름이 있는지만 확인합니다(상속 추론은 하지 않음).
    반환: {ok, issues:[...], stats}.
    """
    ents = list(entities)
    triples = as_triples(relations)
    types_in = {e.get("type") for e in ents}
    allowed_types = _as_set(allowed_types)
    issues, _ = _violations(ents, triples, allowed_types, _as_set(allowed_relations), _as_set(allowed_patterns))
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
            "relation_types": sorted({rel for _, rel, _ in triples}),
        },
    }


def schema_violation_rate(
    entities: Iterable[dict],
    relations: Iterable,
    allowed_types: Optional[Iterable[str]] = None,
    allowed_relations: Optional[Iterable[str]] = None,
    allowed_patterns: Optional[Iterable[tuple]] = None,
) -> dict:
    """스키마 제약을 위반한 항목 비율 (결정적).

    위반 항목(엔티티 + 관계) 수 / 전체 항목 수. 항목 하나에 이유가 여럿이어도 한 번만 셉니다.
    """
    ents = list(entities)
    triples = as_triples(relations)
    issues, bad = _violations(ents, triples, _as_set(allowed_types), _as_set(allowed_relations),
                              _as_set(allowed_patterns))
    total = len(ents) + len(triples)
    return {
        "schema_violation_rate": round(len(bad) / total, 3) if total else 0.0,
        "violations": len(bad),
        "checked": total,
        "issues": issues,
    }


def non_isolated_node_ratio(entities: Iterable, relations: Iterable) -> dict:
    """엣지를 하나 이상 가진 노드의 비율 (구조 보조 지표, 정답성 지표가 아님).

    검색·추론에 참여 가능한 노드가 충분히 연결됐는지 봅니다. 고립 노드가 많으면
    그래프는 "예쁘게" 보여도 실제 검색에 안 잡힐 수 있습니다.
    """
    ents = list(entities)
    names = {e["name"] if isinstance(e, dict) else e for e in ents}
    connected: set = set()
    for s, _, o in as_triples(relations):
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
