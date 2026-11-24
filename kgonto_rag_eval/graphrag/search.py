"""GraphRAG 검색 단계 평가 (모델 없음, DB 에 붙지 않음).

- compare_result_rows      : 질의(Cypher 등)의 **실행 결과**를 정답 결과와 비교 (질의 글자를 비교하지 않음)
- entity_link_recall       : 질문의 출발점(노드)을 맞게 찾았는가
- evaluate_cypher          : 질문 세트의 Cypher 를 실행해(실행은 넘겨받은 함수가) 결과를 한 번에 비교
- evidence_bundle_coverage : 답에 필요한 근거 묶음을 다 가져왔는가 (묶음 안은 모두 필요, 묶음끼리는 하나만)

근거: 실행 결과 비교는 CypherBench(ACL 2025)의 execution accuracy 아이디어를 좁혀 쓴 것입니다.
한 스냅샷에서 결과가 같다고 질의의 뜻까지 같다는 증명은 아닙니다(우연히 같은 결과가 나올 수 있음).
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Iterable, Mapping, Optional, Sequence

from ..evaluate import evaluate
from ..inputs import _missing, as_id, as_records
from ..metrics.base import Metric
from ..schema import EvalReport, EvalSample, EvaluationDataset, MetricResult

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
    ranks: dict = {}
    for i, c in enumerate(list(retrieved_ids)[:k], 1):      # 겹친 후보가 있어도 원래 순위 그대로
        ranks.setdefault(c, i)
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


_WRITE = re.compile(r"(?<![.:$\w])(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|INSERT|FOREACH|LOAD\s+CSV|CALL)\b", re.I)

_ADMIN = re.compile(r"(?:^|;)\s*(?:CYPHER\s+\d+\s+)?(?:(?:EXPLAIN|PROFILE)\s+)?"
                    r"(ALTER|RENAME|GRANT|DENY|REVOKE|START|STOP|TERMINATE|ENABLE|DEALLOCATE|REALLOCATE)\b", re.I)   # 문장 첫머리만 (변수 이름 start 와 구별)


def write_word(cypher: str) -> Optional[str]:
    """Cypher 글에 DB 를 바꾸거나 프로시저를 부르는 말이 있으면 그 말을, 없으면 None.

    글자 · 주석 · 백틱 이름 안은 보지 않고, 하위 질의 CALL { } 와 CALL (변수) { } 는 허용합니다.
    글자로만 거르는 것이라 안전을 보장하지 않습니다. 읽기 전용 계정 · 읽기 트랜잭션을 함께 쓰세요.
    """
    text = re.sub(r"'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\"|`[^`]*`|//[^\n]*|/\*.*?\*/", " ", cypher, flags=re.S)
    admin = _ADMIN.search(text)
    if admin:
        return admin.group(1).upper()
    for m in _WRITE.finditer(text):
        word = re.sub(r"\s+", " ", m.group(1)).upper()
        if word == "CALL" and re.match(r"\s*(\([^()]*\)\s*)?\{", text[m.end():]):
            continue
        return word
    return None


def _as_rows(result) -> tuple[list[dict], Optional[list]]:
    """실행 결과(DataFrame · 사전 목록) -> (사전 목록, 열 이름). 빈 목록이면 열을 알 수 없어 None"""
    columns = None
    if hasattr(result, "columns") and hasattr(result, "to_dict"):
        columns, result = list(result.columns), result.to_dict(orient="records")
    if result is None or isinstance(result, (str, bytes, dict)):
        raise TypeError(f"실행 결과가 행 목록이 아닙니다: {type(result).__name__}")
    rows = [{k: None if _missing(v) else v for k, v in dict(r).items()} for r in result]
    if columns is None and rows:
        columns = list(rows[0])
    if any(set(r) != set(columns or ()) for r in rows):
        raise ValueError("행마다 열이 다릅니다. 실행 결과의 모든 행은 같은 열이어야 합니다.")
    return rows, columns


def _by_position(rows: list[dict]) -> list[dict]:
    """열 이름을 자리로 바꿈 (count(c) 와 건수처럼 이름만 다른 열을 맞춰 봄)"""
    order = list(rows[0]) if rows else []
    return [{f"{i + 1}열": r[c] for i, c in enumerate(order)} for r in rows]


class _Ex(Metric):
    """저장해 둔 비교 결과를 점수로 (실행은 evaluate_cypher 가 먼저 함)"""
    needs_llm = False

    def __init__(self, name: str, parts: list, by: str, read_only: bool, modes: list):
        self.name, self.parts, self.by, self.read_only, self.modes = name, parts, by, read_only, modes   # 채점 조건 지문에 들어감

    def score(self, sample: EvalSample, llm=None) -> MetricResult:
        results = [sample.metadata["results"][p] for p in self.parts]
        broken = [r["reason"] for r in results if r["status"] == "reference_error"]
        if broken:
            return MetricResult(metric=self.name, reason=broken[0], error=broken[0], failed_component="reference")
        score = min(r["ex"] for r in results)
        return MetricResult(metric=self.name, score=score, verdict="pass" if score else "fail",
                            reason=" / ".join(dict.fromkeys(r["reason"] for r in results if r["reason"])),
                            details=results[0] if len(results) == 1 else {"ex": dict(zip(self.parts, (r["ex"] for r in results)))})


def evaluate_cypher(run, questions, *, question: str = "question", gold: str = "gold_cypher", cypher: str = "cypher",
                    id: Optional[str] = None, mode: Optional[str] = None, by: str = "position", read_only: bool = True,
                    run_meta: Optional[dict] = None) -> EvalReport:
    """만든 Cypher 와 정답 Cypher 를 같은 데이터에서 실행해 **결과**를 비교합니다(글자를 비교하지 않음).

    run   : Cypher 글 -> 행 목록(사전 목록이나 DataFrame) 함수. 저장한 결과를 돌려주는 함수도 됩니다.
            {"원본": run1, "반례": run2} 처럼 여러 데이터를 주면 ex@원본 · ex@반례 와 모두 맞았는지(ex@all)를 냅니다.
    questions 의 열 : question · gold_cypher(정답 Cypher) · cypher(만든 Cypher, 빈 칸이면 못 만든 것).
            열 이름이 다르면 question= · gold= · cypher= · id= 에 내 열 이름을 적습니다.
            [{"question": "계약은 몇 건?", "gold_cypher": "MATCH (c:Contract) RETURN count(c)", "cypher": "..."}]
    mode  : 비교 방식이 적힌 열 이름 - multiset(기본, 순서 무시) · set(중복도 무시) · ordered(순서까지) · scalar
    by    : "position"(열을 자리로 맞춤, 열 수가 같아야 함) · "name"(열 이름으로 맞춤, 열 이름이 같아야 함)
    read_only : True 면 만든 Cypher 에 쓰는 말이 있을 때 실행하지 않고 0점(blocked)

    1 = 결과가 같음, 0 = 다름 · 실행 실패 · 막힘 · 못 만듦. 정답 Cypher 가 실패하면 점수 없이 오류로 남깁니다.
    결과가 우연히 같을 수 있으므로 반례 데이터(빠뜨리기 쉬운 조건이 갈리는 데이터)로도 실행하세요.
    """
    if by not in ("position", "name"):
        raise ValueError(f"by 는 position · name 가운데 하나입니다: {by!r}")
    runs = dict(run) if isinstance(run, Mapping) else {"": run}
    if not runs or not all(callable(r) for r in runs.values()):
        raise TypeError("run 에는 Cypher 글을 받아 행 목록을 돌려주는 함수를 넣으세요.")
    if "all" in runs or any("__" in str(n) for n in runs):
        raise ValueError(f"데이터 이름으로 'all' 과 '__' 는 쓰지 않습니다: {list(runs)}")
    rows, seen = [], set()
    for i, record in enumerate(as_records(questions)):                          # 실행하기 전에 질문 표부터 검사
        lost = [c for c in (question, gold, cypher, id, mode) if c is not None and c not in record]
        if lost:
            raise KeyError(f"{i}번째 질문에 {lost} 열이 없습니다. 있는 열: {list(record)}")
        for c in (question, gold):
            if not isinstance(record[c], str) or not record[c].strip():
                raise ValueError(f"{i}번째 질문의 '{c}' 칸이 비었거나 글자가 아닙니다.")
        how = record[mode] if mode is not None and not _missing(record[mode]) else "multiset"
        if how not in _MODES:
            raise ValueError(f"{i}번째 질문의 '{mode}' 는 {_MODES} 가운데 하나입니다: {how!r}")
        sample_id = as_id(record[id], f"{i}번째 질문의 '{id}'") if id is not None else None
        if sample_id is not None and sample_id in seen:
            raise ValueError(f"{i}번째 질문의 '{id}' 가 앞과 겹칩니다: {sample_id!r}")
        seen.add(sample_id)
        made = record[cypher]
        rows.append((sample_id, record[question], record[gold], made if isinstance(made, str) and made.strip() else "", how))

    def one(call, want: str, made: str, how: str) -> dict:
        try:
            reference, want_columns = _as_rows(call(want))
        except Exception as e:
            return {"ex": None, "status": "reference_error", "reason": f"정답 Cypher 실행 실패: {type(e).__name__}: {str(e)[:200]}"}
        zero = {"ex": 0.0, "reference": reference[:5]}
        if not made:
            return {**zero, "status": "no_query", "reason": "만든 Cypher 가 없음"}
        blocked = write_word(made) if read_only else None
        if blocked:
            return {**zero, "status": "blocked", "reason": f"읽기 전용인데 {blocked} 가 있음"}
        try:
            predicted, made_columns = _as_rows(call(made))
        except Exception as e:
            return {**zero, "status": "query_error", "reason": f"{type(e).__name__}: {str(e)[:200]}"}
        shown = {"reference": reference[:5], "predicted": predicted[:5], "n_reference": len(reference), "n_predicted": len(predicted)}
        a, b = (_by_position(predicted), _by_position(reference)) if by == "position" else (predicted, reference)
        columns = {c: "any" for r in (b or a) for c in r} or {"1열": "any"}      # 둘 다 비었으면 열을 알 수 없음
        if how == "scalar" and (len(b) != 1 or len(columns) != 1):
            return {"ex": None, "status": "reference_error", "reason": "scalar 의 정답은 한 행 · 한 열이어야 합니다."}
        if made_columns is not None and want_columns is not None:
            if by == "position" and len(made_columns) != len(want_columns):
                return {"ex": 0.0, "status": "ok", "reason": f"열 수가 다름(만든 것 {len(made_columns)}, 정답 {len(want_columns)})", **shown}
            if by == "name" and set(made_columns) != set(want_columns):
                return {"ex": 0.0, "status": "ok", "reason": f"열 이름이 다름(만든 것 {made_columns}, 정답 {want_columns})", **shown}
        try:
            result = compare_result_rows(a, b, mode=how, columns=columns)
        except ValueError as e:                             # 정답이 비교 방식과 맞지 않음(scalar 인데 여러 행 등)
            return {"ex": None, "status": "reference_error", "reason": f"정답 결과를 비교할 수 없음: {e}"}
        unknown = result["empty_match"] and (made_columns is None or want_columns is None)
        reason = "둘 다 빈 결과(열은 확인하지 못함)" if unknown else result["reason"]
        return {"ex": result["ex"], "status": result["status"], "reason": reason, "empty_match": result["empty_match"],
                "missing": result["missing"][:5], "extra": result["extra"][:5], **shown}

    samples = [EvalSample(sample_id=sample_id, user_input=text, response=made, reference=want,
                          metadata={"mode": how, "results": {name: one(call, want, made, how) for name, call in runs.items()}})
               for sample_id, text, want, made, how in rows]
    parts = {f"ex@{n}" if n else "ex": [n] for n in runs}
    if len(runs) > 1:
        parts["ex@all"] = list(runs)
    report = evaluate(EvaluationDataset(samples=samples), [_Ex(n, p, by, read_only, [r[4] for r in rows]) for n, p in parts.items()], run_meta=run_meta)
    for row in report.rows:                                 # 세부는 지표 칸에 있으므로 표에서는 뺌
        row.pop("meta__results")
    return report
