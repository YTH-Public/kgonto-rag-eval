"""평가 입력 만들기: 수업에서 쓰던 표 · 기록을 그대로 EvaluationDataset 으로."""
from __future__ import annotations

from numbers import Integral
from typing import Any, Optional, Sequence

from .schema import EvalSample, EvaluationDataset


def _missing(value: Any) -> bool:
    """None 이나 pandas 의 빈 칸(NaN · NaT · NA)인지"""
    if value is None or type(value).__name__ in ("NAType", "NaTType"):
        return True
    return isinstance(value, float) and value != value


def _text(value: Any, where: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{where} 값이 글자가 아닙니다: {value!r}"[:200])
    return value


def _chunk(item: Any, source: Optional[str], where: str):
    """근거 한 조각 -> (글, 출처). 글자 · Document · {"text" 또는 "page_content"} 사전만 받습니다."""
    if isinstance(item, dict):
        keys = [k for k in ("text", "page_content") if k in item]
        if len(keys) != 1:
            raise ValueError(f"{where} 사전에는 'text' 와 'page_content' 가운데 하나만 있어야 합니다: {list(item)}")
        text, info = item[keys[0]], {**(item.get("metadata") or {}), **item}
    elif hasattr(item, "page_content"):
        text, info = item.page_content, getattr(item, "metadata", None) or {}
    else:
        text, info = item, {}
    text = _text(text, where)
    if source is None:
        return text, None
    if _missing(info.get(source)):
        raise ValueError(f"{where} 에 출처 '{source}' 칸이 없거나 비었습니다: {list(info)}")
    return text, info[source]


def to_dataset(records, *, question: str = "question", answer: str = "answer", contexts: Optional[str] = None,
               reference: Optional[str] = None, id: Optional[str] = None, keep: Sequence[str] = (),
               source: Optional[str] = None) -> EvaluationDataset:
    """표(DataFrame)나 기록(사전 목록)을 평가 입력으로 바꿉니다. 인자에는 내 자료의 **열 이름**을 적습니다.

    question · answer : 질문 · 답 열 (꼭 있어야 함)
    contexts  : 근거 열. 칸마다 목록 - 글자, Document, {"text": ...} 또는 {"page_content": ...}
    reference : 기준 답 열 (빈 칸은 기준 답 없음)
    id        : 문항 번호 열. 겹치거나 비면 오류. 안 적으면 순번
    keep      : metadata 에 그대로 남길 열 이름 목록
    source    : 근거 조각의 출처 이름(예 "page"). metadata["retrieved_ids"] 에 순서대로 남김

    없는 열 · 모르는 모양은 오류로 알립니다(짐작하지 않음). 원본은 고치지 않습니다.
    """
    if hasattr(records, "columns") and hasattr(records, "to_dict"):
        if not records.columns.is_unique:
            raise ValueError(f"열 이름이 겹칩니다: {list(records.columns)}")
        records = records.to_dict(orient="records")
    if isinstance(records, (str, bytes, dict)):
        raise TypeError("records 는 표(DataFrame)나 사전 목록이어야 합니다.")
    if isinstance(keep, str):
        raise TypeError(f"keep 은 목록으로 적습니다: keep=[{keep!r}]")
    if source is not None and contexts is None:
        raise ValueError("source 는 contexts 와 함께 씁니다.")
    if source is not None and "retrieved_ids" in keep:
        raise ValueError("source 를 쓰면 retrieved_ids 가 만들어지므로 keep 에는 적지 않습니다.")
    records = list(records)
    if not records:
        raise ValueError("기록이 하나도 없습니다.")

    samples, seen = [], set()
    for i, record in enumerate(records):
        if not isinstance(record, dict):
            raise TypeError(f"{i}번째 기록이 사전이 아닙니다: {type(record).__name__}")
        for name in [question, answer, contexts, reference, id, *keep]:
            if name is not None and name not in record:
                raise KeyError(f"{i}번째 기록에 '{name}' 열이 없습니다. 있는 열: {list(record)}")
        where = f"{i}번째 기록의"
        row: dict = {"user_input": _text(record[question], f"{where} '{question}'"),
                     "response": _text(record[answer], f"{where} '{answer}'")}
        if not row["user_input"].strip():
            raise ValueError(f"{where} '{question}' 칸이 비어 있습니다.")
        row["metadata"] = {name: None if _missing(record[name]) else record[name] for name in keep}
        if reference is not None and not _missing(record[reference]):
            row["reference"] = _text(record[reference], f"{where} '{reference}'")
        if contexts is not None:
            items = record[contexts]
            if not isinstance(items, (list, tuple)):
                raise ValueError(f"{where} '{contexts}' 칸은 목록이어야 합니다(근거가 없으면 빈 목록): {type(items).__name__}")
            pairs = [_chunk(item, source, f"{where} '{contexts}'[{j}]") for j, item in enumerate(items)]
            row["retrieved_contexts"] = [text for text, _ in pairs]
            if source is not None:
                row["metadata"]["retrieved_ids"] = [value for _, value in pairs]
        if id is not None:
            value = record[id]
            if isinstance(value, bool) or not isinstance(value, (str, Integral)) or not str(value).strip():
                raise ValueError(f"{where} '{id}' 값은 글자나 정수여야 합니다: {value!r}")
            row["sample_id"] = str(value)
            if row["sample_id"] in seen:
                raise ValueError(f"{where} '{id}' 값이 앞의 문항과 겹칩니다: {value!r}")
            seen.add(row["sample_id"])
        samples.append(EvalSample(**row))
    return EvaluationDataset(samples=samples)
