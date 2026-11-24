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
        text, top, info = item[keys[0]], item, item.get("metadata")
    elif hasattr(item, "page_content"):
        text, top, info = item.page_content, {}, item.metadata
    else:
        text, top, info = item, {}, None
    text = _text(text, where)
    if source is None:
        return text, None
    if info is not None and not isinstance(info, dict):
        raise ValueError(f"{where} 의 metadata 는 사전이어야 합니다(예 {{'{source}': 7}}): {info!r}"[:200])
    info = info or {}
    if source in top and source in info and top[source] != info[source]:
        raise ValueError(f"{where} 의 '{source}' 와 metadata 의 '{source}' 가 다릅니다: {top[source]!r}, {info[source]!r}")
    value = top[source] if source in top else info.get(source)
    if isinstance(value, bool) or not isinstance(value, (str, Integral)) or not str(value).strip():
        raise ValueError(f"{where} 의 출처 '{source}' 는 빈 칸이 아닌 글자나 정수여야 합니다: {value!r}"[:200])
    return text, value


def _add(metadata: dict, name: str, value: Any, where: str) -> None:
    """metadata 에 값을 더합니다. 같은 이름에 다른 값이 있으면 덮어쓰지 않고 오류"""
    if name in metadata and metadata[name] != value:
        raise ValueError(f"{where} metadata 에 '{name}' 이름의 다른 값이 이미 있습니다: {metadata[name]!r} / {value!r}"[:300])
    metadata[name] = value


def to_dataset(records, *, question: str = "question", answer: str = "answer", contexts: Optional[str] = None,
               reference: Optional[str] = None, id: Optional[str] = None, keep: Sequence[str] = (),
               source: Optional[str] = None) -> EvaluationDataset:
    """표(DataFrame)나 기록(사전 목록)을 평가 입력으로 바꿉니다. 인자에는 내 자료의 **열 이름**을 적습니다.

    question · answer : 질문 · 답 열 (꼭 있어야 함)
    contexts  : 근거 열. 칸마다 목록 - 글자, Document, {"text": ...} 또는 {"page_content": ...}
    reference : 기준 답 열 (빈 칸은 기준 답 없음)
    id        : 문항 번호 열. 겹치거나 비면 오류. 안 적으면 순번
    keep      : metadata 에 그대로 남길 열 이름 목록 (이미 있는 metadata 열은 적지 않아도 남습니다)
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
        metadata = record.get("metadata")
        if not _missing(metadata) and not isinstance(metadata, dict):
            raise ValueError(f"{where} 'metadata' 칸은 사전이어야 합니다: {metadata!r}"[:200])
        row["metadata"] = {} if _missing(metadata) else dict(metadata)
        for name in keep:
            if name != "metadata":
                _add(row["metadata"], name, None if _missing(record[name]) else record[name], where)
        if reference is not None and not _missing(record[reference]):
            row["reference"] = _text(record[reference], f"{where} '{reference}'")
        if contexts is not None:
            items = record[contexts]
            if not isinstance(items, (list, tuple)):
                raise ValueError(f"{where} '{contexts}' 칸은 목록이어야 합니다(근거가 없으면 빈 목록): {type(items).__name__}")
            pairs = [_chunk(item, source, f"{where} '{contexts}'[{j}]") for j, item in enumerate(items)]
            row["retrieved_contexts"] = [text for text, _ in pairs]
            if source is not None:
                _add(row["metadata"], "retrieved_ids", [value for _, value in pairs], where)
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
