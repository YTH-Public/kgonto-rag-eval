"""평가 입력 만들기: 수업에서 쓰던 표 · 기록을 그대로 EvaluationDataset 으로."""
from __future__ import annotations

from numbers import Integral
from uuid import uuid4
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
    as_id(value, f"{where} 의 출처 '{source}'")
    return text, value


def as_id(value: Any, where: str) -> str:
    """ID -> 글자. 글자와 정수만 받음(12.0 처럼 빈 칸 때문에 소수가 된 열은 알림)"""
    if isinstance(value, bool) or not isinstance(value, (str, Integral)) or not str(value).strip():
        raise ValueError(f"{where} 값은 글자나 정수여야 합니다: {value!r}"[:300])
    return str(value)


def as_records(records) -> list[dict]:
    """표(DataFrame)나 사전 목록 -> 사전 목록"""
    if hasattr(records, "columns") and hasattr(records, "to_dict"):
        if not records.columns.is_unique:
            raise ValueError(f"열 이름이 겹칩니다: {list(records.columns)}")
        records = records.to_dict(orient="records")
    if isinstance(records, (str, bytes, dict)):
        raise TypeError("표(DataFrame)나 사전 목록을 넣으세요.")
    records = list(records)
    if not records:
        raise ValueError("기록이 하나도 없습니다.")
    for i, record in enumerate(records):
        if not isinstance(record, dict):
            raise TypeError(f"{i}번째 기록이 사전이 아닙니다: {type(record).__name__}")
    return records


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
    if isinstance(keep, str):
        raise TypeError(f"keep 은 목록으로 적습니다: keep=[{keep!r}]")
    if source is not None and contexts is None:
        raise ValueError("source 는 contexts 와 함께 씁니다.")
    samples, seen = [], set()
    for i, record in enumerate(as_records(records)):
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
            row["sample_id"] = as_id(record[id], f"{where} '{id}'")
            if row["sample_id"] in seen:
                raise ValueError(f"{where} '{id}' 값이 앞의 문항과 겹칩니다: {record[id]!r}")
            seen.add(row["sample_id"])
        samples.append(EvalSample(**row))
    return EvaluationDataset(samples=samples)


def _message_text(message) -> str:
    """메시지의 글. 블록 목록이면 글 블록만 이음"""
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(b if isinstance(b, str) else b.get("text", "") for b in content if isinstance(b, str) or b.get("type") == "text")


def _from_messages(messages, contexts: Optional[str], tools, skip) -> tuple[str, list, list]:
    """그래프의 메시지 -> (답, 도구 결과 글, 부른 도구 이름)"""
    last = messages[-1] if messages else None
    if getattr(last, "type", None) != "ai" or last.tool_calls:
        raise ValueError("마지막 메시지가 모델의 답이 아닙니다(중간에 멈췄을 수 있음).")
    found, used = [], []
    for m in messages:
        if m.type != "tool":
            continue
        used.append(m.name)
        text = _message_text(m).strip()
        if contexts == "tool" and m.status != "error" and (tools is None or m.name in tools) and text and text not in skip:
            found.append(text)
    return _message_text(last), found, used


def run_rag(rag, questions, *, contexts: Optional[str], question: str = "question", answer: str = "answer",
            reference: Optional[str] = None, id: Optional[str] = None, keep: Sequence[str] = (),
            tools: Optional[Sequence[str]] = None, skip: Sequence[str] = (), config: Optional[dict] = None) -> EvaluationDataset:
    """만든 RAG 를 질문마다 한 번 실행해 평가 입력을 만듭니다(채점은 evaluate 로 따로).

    rag      : LangGraph 그래프({"messages": ...}) 또는 함수(질문 -> 답 글, 또는 답과 근거가 든 사전).
               체인은 chain.invoke, 상태 칸이 다른 그래프는 lambda q: graph.invoke({"question": q}) 를 넣습니다.
    contexts : 근거를 꺼낼 곳 - "tool"(도구가 돌려준 글), 결과 사전의 칸 이름, None(근거 없음). 꼭 적습니다.
    question · reference · id · keep : questions 의 열 이름 (to_dataset 과 같음). 기준 답은 RAG 에 넘기지 않습니다.
    answer   : 결과 사전에서 답이 든 칸 이름
    tools    : contexts="tool" 일 때 근거로 볼 도구 이름 목록 (None 이면 모두)
    skip     : 근거가 아닌 도구 결과 글 (예 ["관련 근거 없음"]) - 글 전체가 같을 때만 뺌
    config   : graph.invoke 에 넘길 설정. 체크포인터가 있는 그래프는 문항마다 새 thread_id 를 넣습니다.

    contexts="tool" 이면 부른 도구 이름을 metadata["tools_used"] 에 남깁니다(안 부른 것과 못 찾은 것을 구별).
    한 문항이라도 실패하면 어느 문항인지 알리고 멈춥니다. 뒤에서 다시 검색해 근거를 끼워 넣지 않습니다.
    """
    for name, value in (("tools", tools), ("skip", skip)):
        if isinstance(value, str):
            raise TypeError(f"{name} 은 목록으로 적습니다: {name}=[{value!r}]")
    skip = [s.strip() for s in skip]
    is_graph = hasattr(rag, "invoke")
    if not is_graph and not callable(rag):
        raise TypeError("rag 에는 그래프나 함수를 넣으세요.")
    if config is not None and not is_graph:
        raise ValueError("config 는 그래프를 넣었을 때만 씁니다.")
    if "tools_used" in keep and contexts == "tool":
        raise ValueError("tools_used 는 run_rag 가 남기므로 keep 에 적지 않습니다.")
    blank = "_answer"
    while any(blank in r for r in as_records(questions)):
        blank += "_"
    base = to_dataset([{**r, blank: ""} for r in as_records(questions)], question=question, answer=blank,
                      reference=reference, id=id, keep=keep)          # 실행하기 전에 질문 표부터 검사
    samples = []
    for i, sample in enumerate(base):
        where = f"{i}번째 질문(id={sample.sample_id})"
        try:
            if is_graph:
                setting = dict(config or {})
                if getattr(rag, "checkpointer", None):
                    setting["configurable"] = {**setting.get("configurable", {}), "thread_id": f"run-rag-{uuid4()}"}
                out = rag.invoke({"messages": [("user", sample.user_input)]}, setting)
            else:
                out = rag(sample.user_input)
        except Exception as e:
            raise RuntimeError(f"{where}에서 RAG 실행이 실패했습니다: {type(e).__name__}: {e}") from e
        metadata = dict(sample.metadata)
        try:
            if isinstance(out, dict) and "__interrupt__" in out:
                raise ValueError("그래프가 사람의 확인을 기다리며 멈췄습니다.")
            if isinstance(out, dict) and "messages" in out and (contexts == "tool" or answer not in out):
                text, found, used = _from_messages(out["messages"], contexts, tools, skip)
                if contexts == "tool":
                    _add(metadata, "tools_used", used, where)
            elif contexts == "tool":
                raise ValueError("contexts='tool' 인데 결과에 messages 가 없습니다.")
            elif isinstance(out, dict):
                if answer not in out:
                    raise ValueError(f"결과에 '{answer}' 칸이 없습니다. 있는 칸: {list(out)}")
                text, found = out[answer], []
            elif contexts is None:
                text, found = out, []
            else:
                raise ValueError(f"contexts='{contexts}' 인데 결과가 사전이 아닙니다: {type(out).__name__}")
            text = _text(text, f"{where} 답")
            if contexts not in (None, "tool"):
                if contexts not in out:
                    raise ValueError(f"결과에 '{contexts}' 칸이 없습니다. 있는 칸: {list(out)}")
                if not isinstance(out[contexts], (list, tuple)):
                    raise ValueError(f"'{contexts}' 칸은 근거 목록이어야 합니다: {type(out[contexts]).__name__}")
                found = [_chunk(c, None, f"'{contexts}'[{j}]")[0] for j, c in enumerate(out[contexts])]
        except ValueError as e:
            raise ValueError(f"{where}의 결과를 읽지 못했습니다: {e}") from e
        samples.append(EvalSample(sample_id=sample.sample_id, user_input=sample.user_input, response=text,
                                  reference=sample.reference, retrieved_contexts=found, metadata=metadata))
    return EvaluationDataset(samples=samples)
