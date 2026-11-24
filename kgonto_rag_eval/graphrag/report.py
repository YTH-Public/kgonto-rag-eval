"""단계별 기록을 한 표로 - 문항마다 처음 끊긴 단계(의심)."""
from __future__ import annotations

from numbers import Real
from typing import Mapping, Sequence, Union

from ..inputs import _missing, as_records

_WORDS = {"해당 없음", "오류"}


def stage_table(records, stages: Union[Sequence[str], Mapping[str, str]], *, id: str = "id", cut: float = 1.0,
                keep: Sequence[str] = ()):
    """문항마다 단계의 통과 · 끊김을 한 줄로 놓고, **처음 끊긴 단계**를 적습니다.

    records : 표나 사전 목록. 한 줄이 문항 하나(여러 조건을 섞었으면 id 를 조건까지 붙여 다르게)
    stages  : 단계 열 이름을 앞에서부터. {열 이름: 보여 줄 이름} 도 됩니다.
              예 ["graph", "entry", "query", "sent", "answer"]
    칸의 값 : True/False, 0 ~ 1 점수(cut 이상이면 통과), "해당 없음"(그 길에는 없는 단계), "오류"(채점 실패), 빈 칸(안 봄)
    keep    : 그대로 남길 열 (시간 · 토큰 · 근거 글자 수 같은 비용도 여기에. 점수와 섞지 않습니다)
    반환    : id · 단계 열들(통과 · 끊김 · 해당 없음 · 오류 · 안 봄) · first_break · note · keep 열

    first_break 는 원인이 아니라 **먼저 볼 곳**입니다. 여러 단계가 함께 틀릴 수 있습니다. 종합 점수는 내지 않습니다.
    """
    import pandas as pd

    names = dict(stages) if isinstance(stages, Mapping) else {s: s for s in stages}
    if isinstance(stages, str) or not names:
        raise ValueError("stages 는 단계 열 이름의 목록입니다.")
    taken = [id, *names.values(), "first_break", "note", *keep]
    if len(set(taken)) != len(taken):
        raise ValueError(f"표의 열 이름이 겹칩니다: {sorted({str(c) for c in taken if taken.count(c) > 1})}")
    rows, seen = [], set()
    for i, record in enumerate(as_records(records)):
        lost = [c for c in (id, *names, *keep) if c not in record]
        if lost:
            raise KeyError(f"{i}번째 기록에 {lost} 열이 없습니다. 있는 열: {list(record)}")
        if _missing(record[id]) or record[id] in seen:
            raise ValueError(f"{i}번째 기록의 '{id}' 가 비었거나 겹칩니다: {record[id]!r}")
        seen.add(record[id])
        marks = {}
        for column, shown in names.items():
            value = record[column]
            if type(value).__module__ == "numpy" and hasattr(value, "item"):
                value = value.item()
            if _missing(value):
                marks[shown] = "안 봄"
            elif isinstance(value, str) and value in _WORDS:
                marks[shown] = value
            elif isinstance(value, Real) and 0 <= value <= 1:
                marks[shown] = "통과" if value >= cut else "끊김"
            else:
                raise ValueError(f"{i}번째 기록의 '{column}' 는 참/거짓 · 0 ~ 1 · '해당 없음' · '오류' · 빈 칸이어야 합니다: {value!r}")
        order = list(marks)
        broken = next((s for s in order if marks[s] == "끊김"), None)
        before = order[:order.index(broken)] if broken else []
        unsure = [s for s in before if marks[s] in ("안 봄", "오류")]
        notes = []
        if unsure:
            notes.append("앞 단계를 확인하지 못함: " + ", ".join(unsure))
        if broken and broken != order[-1] and marks[order[-1]] == "통과":
            notes.append(f"{broken}에서 끊겼는데 {order[-1]}는 통과 - 다른 근거나 모델의 기억으로 맞혔을 수 있음")
        rows.append({id: record[id], **marks, "first_break": broken, "note": " / ".join(notes), **{k: record[k] for k in keep}})
    return pd.DataFrame(rows)
