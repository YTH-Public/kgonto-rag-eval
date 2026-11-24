"""GraphRAG 의 중간 단계를 채점하는 함수 (결정적, LLM · DB 불필요).

Neo4j 같은 DB 에 붙지 않습니다. 미리 실행해 저장한 결과(행 · ID)를 비교합니다.

- compare_result_rows      : 질의(Cypher 등)의 **실행 결과**를 정답 결과와 비교 (질의 글자를 비교하지 않음)
- entity_link_recall       : 질문의 출발점(노드)을 맞게 찾았는가
- evidence_bundle_coverage : 답에 필요한 근거 묶음을 다 가져왔는가 (묶음 안은 모두 필요, 묶음끼리는 하나만)

근거: 실행 결과 비교는 CypherBench(ACL 2025)의 execution accuracy 아이디어를 좁혀 쓴 것입니다.
한 스냅샷에서 결과가 같다고 질의의 뜻까지 같다는 증명은 아닙니다(우연히 같은 결과가 나올 수 있음).
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any, Iterable, Mapping, Optional, Sequence

_MODES = ("scalar", "set", "multiset", "ordered")
_TYPES = ("any", "string", "integer", "number", "boolean")


class _RowError(ValueError):
    """행을 계약에 맞게 바꿀 수 없음"""


def _freeze(v: Any) -> Any:
    """비교할 수 있게(해시 가능하게) 바꿈. 1 과 True 를 같은 값으로 보지 않도록 bool 은 따로 표시."""
    if isinstance(v, bool):
        return ("bool", v)
    if isinstance(v, Mapping):
        return ("map", tuple(sorted((str(k), _freeze(x)) for k, x in v.items())))
    if isinstance(v, (list, tuple)):
        return ("list", tuple(_freeze(x) for x in v))
    if isinstance(v, (set, frozenset)):
        return ("set", tuple(sorted((_freeze(x) for x in v), key=repr)))
    if isinstance(v, float):
        if not math.isfinite(v):
            raise _RowError(f"NaN · 무한대는 비교할 수 없습니다: {v!r}")
        return int(v) if v.is_integer() else v
    if v is None or isinstance(v, (str, int)):
        return v
    return ("other", str(v))   # 날짜 등


def _thaw(v: Any) -> Any:
    """_freeze 를 되돌림 - 차이를 보여 줄 때 씀 (목록 · 집합은 목록, 날짜 등은 글자)"""
    if not isinstance(v, tuple):
        return v
    kind, value = v
    if kind == "map":
        return {k: _thaw(x) for k, x in value}
    return [_thaw(x) for x in value] if kind in ("list", "set") else value


def _convert(value: Any, kind: str, column: str) -> Any:
    if value is None or kind == "any":
        return _freeze(value)
    if kind == "string":
        if not isinstance(value, (str, int)) or isinstance(value, bool):
            raise _RowError(f"{column}: 글자가 아닙니다({value!r}).")
        return str(value)   # ID 앞자리 0 을 지키려면 DB 에서도 글자로 받아야 합니다
    if kind == "boolean":
        if not isinstance(value, bool):
            raise _RowError(f"{column}: 참/거짓이 아닙니다({value!r}).")
        return _freeze(value)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _RowError(f"{column}: 숫자가 아닙니다({value!r}).")
    frozen = _freeze(value)
    if kind == "integer" and not isinstance(frozen, int):
        raise _RowError(f"{column}: 정수가 아닙니다({value!r}).")
    return frozen


def _normalize(rows: Iterable[dict], columns: Mapping[str, str], aliases: Mapping[str, str]) -> list[tuple]:
    out = []
    for row in rows:
        renamed: dict[str, Any] = {}
        for key, value in row.items():
            name = aliases.get(key, key)
            if name in renamed:
                raise _RowError(f"두 열이 같은 이름이 됩니다: {name}")
            renamed[name] = value
        lost = [c for c in columns if c not in renamed]
        if lost:
            raise _RowError(f"필요한 열이 없습니다: {lost} (있는 열: {sorted(renamed)})")
        out.append(tuple(_convert(renamed[c], kind, c) for c, kind in columns.items()))
    return out


def _show(counter: Counter, columns: Mapping[str, str]) -> list[dict]:
    return [dict(zip(columns, map(_thaw, key))) for key in counter.elements()]


def compare_result_rows(
    predicted: Optional[Iterable[Mapping]],
    reference: Iterable[Mapping],
    mode: str = "multiset",
    columns: Optional[Mapping[str, str]] = None,
    aliases: Optional[Mapping[str, str]] = None,
    predicted_status: str = "ok",
    predicted_snapshot: Any = None,
    reference_snapshot: Any = None,
) -> dict:
    """질의 실행 결과(행 목록)를 정답 결과와 비교합니다.

    mode: scalar(값 하나) | set(중복 · 순서 무시) | multiset(순서 무시, 중복은 셈) | ordered(순서까지).
    columns: {열 이름: 종류}. 종류는 any · string · integer · number · boolean. 비우면 정답 행의 열을 any 로 씁니다.
        여기 적은 열만 비교하고, 예측에 더 있는 열은 보지 않습니다(extra_columns 에 적음).
        정답이 빈 결과면 columns 를 꼭 적으세요.
    aliases: 예측 열 이름 -> 계약 열 이름. 예: {"n": "count"}. 값을 보고 나중에 정하지 말고 미리 정해 두세요.
    predicted_status: 실행이 안 됐으면 "query_error" · "timeout" · "blocked" · "truncated" 같은 값. ex 는 0 이 됩니다.
        실행은 됐는데 결과가 없으면 "ok" 와 빈 목록입니다. 오류를 빈 목록으로 바꾸지 마세요.
    *_snapshot: 둘 다 주고 서로 다르면 다른 시점의 데이터이므로 비교하지 않습니다(status="invalid").

    반환: {"ex": 1.0 | 0.0 | None, "status", "reason", "matched", "precision", "recall", "f1",
          "missing", "extra", "empty_match", "extra_columns"}
    ex 가 None 이면 채점 대상이 아닙니다. 실행 오류는 0 으로 남기고 분모에서 빼지 않습니다.
    정답이 계약에 맞지 않으면 ValueError 입니다(정답을 고쳐야 하므로).
    """
    if mode not in _MODES:
        raise ValueError(f"mode 는 {_MODES} 가운데 하나입니다: {mode!r}")
    reference = [dict(r) for r in reference]
    if columns is None:
        columns = {name: "any" for r in reference for name in r}
    columns = dict(columns)
    if not columns:
        raise ValueError("비교할 열이 없습니다. 정답이 빈 결과면 columns 를 적으세요.")
    unknown = [k for k in columns.values() if k not in _TYPES]
    if unknown:
        raise ValueError(f"열 종류는 {_TYPES} 가운데 하나입니다: {unknown}")
    gold = _normalize(reference, columns, {})
    if mode == "scalar" and (len(gold) != 1 or len(columns) != 1):
        raise ValueError("scalar 의 정답은 한 행 · 한 열이어야 합니다.")
    base = {"ex": None, "status": "ok", "reason": "", "matched": 0, "precision": None, "recall": None, "f1": None,
            "missing": [], "extra": [], "empty_match": False, "extra_columns": []}
    if predicted_snapshot is not None and reference_snapshot is not None and predicted_snapshot != reference_snapshot:
        return {**base, "status": "invalid", "reason": f"데이터 시점이 다릅니다: {predicted_snapshot} / {reference_snapshot}"}
    failed = {**base, "ex": 0.0, "missing": _show(Counter(gold), columns), "recall": 0.0 if gold else None}
    if predicted_status != "ok":
        return {**failed, "status": predicted_status, "reason": "질의가 정상으로 끝나지 않았습니다."}
    if predicted is None:
        raise ValueError("predicted_status 가 ok 이면 predicted 는 목록이어야 합니다(결과가 없으면 빈 목록).")
    predicted = [dict(r) for r in predicted]
    aliases = dict(aliases or {})
    extra_columns = sorted({aliases.get(k, k) for r in predicted for k in r} - set(columns))
    try:
        pred = _normalize(predicted, columns, aliases)
    except _RowError as e:
        return {**failed, "status": "normalization_error", "reason": str(e), "extra_columns": extra_columns}
    if mode == "set":
        pred, gold = list(dict.fromkeys(pred)), list(dict.fromkeys(gold))
    p, g = Counter(pred), Counter(gold)
    matched = sum((p & g).values())
    same = pred == gold if mode in ("ordered", "scalar") else p == g
    both_empty = not pred and not gold
    precision = matched / len(pred) if pred else (1.0 if both_empty else None)
    recall = matched / len(gold) if gold else (1.0 if both_empty else None)
    f1 = 1.0 if both_empty else 2 * matched / (len(pred) + len(gold))
    reason = "" if same else ("행은 같은데 순서가 다릅니다." if p == g else "행이 다릅니다.")
    return {**base, "ex": 1.0 if same else 0.0, "reason": reason, "matched": matched, "precision": precision, "recall": recall,
            "f1": f1, "missing": _show(g - p, columns), "extra": _show(p - g, columns), "empty_match": both_empty,
            "extra_columns": extra_columns}


def entity_link_recall(reference_ids: Iterable, retrieved_ids: Sequence, k: Optional[int] = None) -> dict:
    """질문의 출발점(정답 노드 ID)이 찾은 후보 k 개 안에 있는 비율.

    이름이 아니라 안정된 ID(조문 번호 · 사업자번호 · 노드 ID)를 넣으세요. 동명이인을 가르기 위해서입니다.
    ID 는 그대로 비교합니다("001" 과 1 은 다름). 정답이 없으면 recall 은 None 입니다.
    반환: {"recall", "all_found", "found", "missing", "ranks": {ID: 순위(1부터)}, "k"}
    """
    if k is not None and (isinstance(k, bool) or not isinstance(k, int) or k < 1):
        raise ValueError("k 는 1 이상의 정수여야 합니다.")
    if isinstance(reference_ids, str) or isinstance(retrieved_ids, str):
        raise ValueError("ID 는 목록으로 넣으세요. 하나면 [ID] 입니다.")
    gold = list(dict.fromkeys(reference_ids))
    candidates = list(dict.fromkeys(retrieved_ids))[:k]
    ranks = {c: i for i, c in enumerate(candidates, 1)}
    found = [g for g in gold if g in ranks]
    return {"recall": len(found) / len(gold) if gold else None, "all_found": len(found) == len(gold) if gold else None,
            "found": found, "missing": [g for g in gold if g not in ranks], "ranks": {g: ranks[g] for g in found}, "k": k}


def evidence_bundle_coverage(reference_bundles: Iterable[Iterable], retrieved_facts: Iterable) -> dict:
    """답에 필요한 근거 묶음을 얼마나 가져왔는지.

    reference_bundles: 근거 묶음의 목록. **묶음 안의 사실은 모두 필요**하고, 묶음이 여럿이면 **하나만 다 있으면** 됩니다.
        예: [[("법 제29조", "위임", "영 제30조"), ("영 제30조", "위임", "고시")],     # 길 하나
             [("법 제29조", "인용", "고시")]]                                          # 다른 길
        '모든 경로'를 묻는 질문이면 모든 경로를 한 묶음에 넣으세요.
    retrieved_facts: 답 모델에 **실제로 보낸** 사실. 찾았지만 잘려서 못 보낸 것은 넣지 않습니다.
    사실은 ID(글자 · 숫자)나 (주어, 관계, 목적어) 튜플입니다. 방향 · 시점이 중요하면 튜플에 함께 넣으세요.
    반환: {"coverage": 가장 많이 채운 묶음의 비율, "complete", "best_bundle": 그 묶음의 순번(0부터), "missing"}
    """
    if isinstance(retrieved_facts, (str, bytes)):
        raise ValueError("retrieved_facts 는 사실의 목록이어야 합니다. 사실이 하나면 [사실] 입니다.")
    bundles = []
    for bundle in reference_bundles:
        if isinstance(bundle, str):
            raise ValueError("근거 묶음은 사실의 목록이어야 합니다. 사실이 하나면 [사실] 입니다.")
        facts = list(dict.fromkeys(tuple(f) if isinstance(f, list) else f for f in bundle))
        if not facts:
            raise ValueError("빈 근거 묶음이 있습니다.")
        bundles.append(facts)
    if not bundles:
        return {"coverage": None, "complete": None, "best_bundle": None, "missing": []}
    got = {tuple(f) if isinstance(f, list) else f for f in retrieved_facts}
    scores = [sum(f in got for f in b) / len(b) for b in bundles]   # 서로 다른 묶음의 조각을 섞어 세지 않음
    best = scores.index(max(scores))
    return {"coverage": scores[best], "complete": all(f in got for f in bundles[best]), "best_bundle": best,
            "missing": [f for f in bundles[best] if f not in got]}
