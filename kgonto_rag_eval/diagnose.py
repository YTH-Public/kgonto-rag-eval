"""점수를 조합해 '의심할 곳'을 찾습니다(모델을 다시 부르지 않음).

원인을 확정하지 않습니다. 후보를 좁혀 주는 것이고, 확정은 근거와 답을 읽어서 합니다.
"""
from __future__ import annotations

_ANSWER = ("fact_coverage", "answer_correctness")   # 답이 맞는지 보는 지표 (앞의 것부터 씀)


def diagnose(report, low: float = 0.5, high: float = 0.8) -> list[dict]:
    """문항마다 의심할 곳을 [{"sample_id", "suspect", "because", "check"}, ...] 로 돌려줍니다.

    low 미만을 낮음, high 이상을 높음으로 봅니다(그 사이는 판단하지 않음). 없는 지표가 필요한 규칙은 건너뜁니다.
    한 문항에 여러 후보가 나올 수 있고, 아무 규칙에도 걸리지 않으면 나오지 않습니다.
    """
    if not 0 <= low <= high <= 1:
        raise ValueError("0 <= low <= high <= 1 이어야 합니다.")
    out: list[dict] = []
    for row in report.rows:
        out += _diagnose_row(row, list(report.summary), low, high)
    return out


def _diagnose_row(row: dict, names: list[str], low: float, high: float) -> list[dict]:
    found: list[dict] = []

    def is_low(name: str) -> bool:
        return row.get(name) is not None and row[name] < low

    def is_high(name: str) -> bool:
        return row.get(name) is not None and row[name] >= high

    def add(suspect: str, used: list[str], check: str) -> None:
        because = ", ".join(f"{n}={row[n]:.2f}" for n in used)
        found.append({"sample_id": row["sample_id"], "suspect": suspect, "because": because, "check": check})

    failed = [n for n in names if row.get(f"{n}__error") or row.get(f"{n}__status") == "error"]
    if failed:
        found.append({"sample_id": row["sample_id"], "suspect": "채점 실패", "because": ", ".join(failed),
                      "check": "점수가 낮은 것이 아니라 채점이 안 된 것입니다. 오류 이유부터 봅니다."})
    refused = (row.get("answered_relevancy__details") or {}).get("answered") is False
    wrong = next((n for n in _ANSWER if is_low(n)), None)
    if refused and is_high("context_recall"):
        add("지나친 거절", ["answered_relevancy", "context_recall"], "근거에 답이 있는데 답하지 않았는지 봅니다.")
    if refused and is_low("context_recall"):
        add("근거가 없어 거절", ["answered_relevancy", "context_recall"], "옳은 거절일 수 있습니다. 검색을 먼저 봅니다.")
    if not refused and is_low("context_recall") and wrong:
        add("검색이 근거를 못 찾음", ["context_recall", wrong], "검색된 근거에 정답의 내용이 있는지 봅니다.")
    if not refused and is_high("context_recall") and wrong:
        add("근거는 있는데 답이 빠뜨림", ["context_recall", wrong], "어느 사실이 빠졌는지 봅니다.")
    if not refused and is_high("faithfulness") and is_low("answer_correctness"):
        add("다른 자료에 충실한 답", ["faithfulness", "answer_correctness"], "질문과 다른 대상 · 조건의 근거인지 봅니다.")
    if is_high("fact_coverage") and is_low("answer_correctness"):
        add("덧붙인 내용 또는 필수사실", ["fact_coverage", "answer_correctness"],
            "필수사실 밖의 틀린 내용인지, 필수사실에 대상 · 조건이 빠졌는지 봅니다.")
    if row.get("cited_retrieved") is not None and row["cited_retrieved"] < 1:
        add("검색되지 않은 출처", ["cited_retrieved"], "지어낸 출처인지 봅니다.")
    return found
