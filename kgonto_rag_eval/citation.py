"""인용(출처 표시) 검사.

세 가지를 서로 다른 함수로 봅니다. 하나가 통과해도 나머지가 통과한다는 뜻이 아닙니다.

1. resolve_citations: 답에 적힌 출처 ID 가 실제 자료로 풀리는가 (결정적)
2. quote_in_source:   따옴표로 인용한 문구가 그 자료에 그대로 있는가 (결정적, 공백 무시)
3. citation_support:  문장의 주장을 인용 자료가 뒷받침하는가 (LLM 판정)

3 은 ALCE(Gao et al. 2023, arXiv:2305.14627)의 citation recall / precision 아이디어를
수업용으로 줄인 지표입니다. 원 논문은 NLI 모델(AutoAIS)로 판정하지만, 여기서는 LLM 판정을 씁니다.
- citation_recall: 출처를 단 문장 중, 단 출처들을 합쳐 그 문장을 뒷받침하는 비율
- citation_precision: 단 출처 중, 혼자서 뒷받침하거나 빼면 뒷받침이 깨지는 출처의 비율
- 출처가 없는 문장은 recall 분모에서 빼고 uncited 로 따로 셉니다(필요하면 직접 분모에 넣으세요).
"""
from __future__ import annotations

import re
from typing import Iterable, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

DEFAULT_PATTERN = r"\[p\.(\d+)\]"   # 예: "월 20만 원입니다 [p.35]"


def _squash(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def split_cited_sentences(answer: str, pattern: str = DEFAULT_PATTERN) -> list[dict]:
    """답을 문장으로 나누고 문장마다 출처 ID 를 뽑습니다.

    반환: [{"sentence": 출처 표시를 뺀 문장, "citations": [ID, ...]}]
    출처 표시가 문장 끝 마침표 뒤에 붙어도 앞 문장의 것으로 봅니다.
    """
    pieces = re.split(r"(?<=[.!?。])\s+|\n+", answer.strip())
    out: list[dict] = []
    for piece in pieces:
        ids = re.findall(pattern, piece)
        text = re.sub(pattern, "", piece).strip()
        if not text and ids and out:   # 출처 표시만 따로 떨어진 조각
            out[-1]["citations"] += ids
            continue
        if text:
            out.append({"sentence": text, "citations": ids})
    return out


def resolve_citations(citations: Iterable[str], corpus: dict) -> dict:
    """출처 ID 가 corpus(ID -> 원문)에 있는지. 정수 · 문자열 ID 를 같은 것으로 봅니다."""
    keys = {str(k) for k in corpus}
    ids = [str(c) for c in citations]
    missing = [c for c in ids if c not in keys]
    return {"resolved_rate": round(1 - len(missing) / len(ids), 3) if ids else None,
            "resolved": len(ids) - len(missing), "total": len(ids), "missing": missing}


def quote_in_source(quote: str, source_text: str) -> bool:
    """인용 문구가 원문에 그대로 있는지(공백 · 줄바꿈 차이는 무시). 바꿔 쓴 문장은 False 가 정상입니다."""
    return bool(quote) and _squash(quote) in _squash(source_text)


class _Support(BaseModel):
    supported: bool = Field(description="근거 글이 문장의 모든 주장을 뒷받침하면 true")
    reason: str = Field(description="한국어 한 문장")


_SYSTEM = """당신은 인용 검증자입니다. '근거 글'이 '문장'의 주장을 모두 뒷받침하는지 판단합니다.
- 숫자 · 기간 · 대상 · 조건이 근거 글의 같은 대상에 대한 것이어야 합니다. 같은 글에 숫자가 있어도 다른 대상이면 false.
- 근거 글에 없는 내용이 하나라도 있으면 false. 세상 지식으로 맞아도 근거 글에 없으면 false."""


def _supports(llm, sentence: str, texts: list[str], question: Optional[str]) -> bool:
    joined = "\n\n".join(f"[근거 {i + 1}] {t}" for i, t in enumerate(texts))
    q = f"질문: {question}\n\n" if question else ""
    v = llm.with_structured_output(_Support).invoke([
        SystemMessage(content=_SYSTEM),
        HumanMessage(content=f"{q}문장: {sentence}\n\n근거 글:\n{joined}"),
    ])
    return bool(v.supported)


def citation_support(answer: str, corpus: dict, llm, question: Optional[str] = None,
                     pattern: str = DEFAULT_PATTERN) -> dict:
    """ALCE 아이디어를 적용한 수업용 인용 지표(LLM 판정).

    corpus: 출처 ID -> 원문. 풀리지 않는 ID 는 뒷받침하지 않는 출처로 셉니다.
    반환: citation_recall, citation_precision, 문장별 판정(sentences), 출처 없는 문장 수(uncited).
    호출 수: 출처 단 문장마다 1번 + (출처가 2개 이상이면) 출처마다 최대 2번.
    """
    keys = {str(k): v for k, v in corpus.items()}
    rows = []
    n_cited = n_supported = n_cites = n_good = 0
    for item in split_cited_sentences(answer, pattern):
        ids = [str(c) for c in item["citations"]]
        if not ids:
            rows.append({**item, "supported": None, "good_citations": []})
            continue
        texts = [keys[c] for c in ids if c in keys]
        joint = bool(texts) and _supports(llm, item["sentence"], texts, question)
        good: list[str] = []
        for c in ids:
            if c not in keys:
                continue
            if len(ids) == 1:
                ok = joint
            else:
                alone = _supports(llm, item["sentence"], [keys[c]], question)
                rest = [keys[o] for o in ids if o != c and o in keys]
                # 혼자 뒷받침하면 좋은 출처. 아니면 빼 봤을 때 뒷받침이 깨지는지(필요한 출처인지) 봅니다.
                ok = alone or (joint and not (rest and _supports(llm, item["sentence"], rest, question)))
            if ok:
                good.append(c)
        n_cited += 1
        n_supported += int(joint)
        n_cites += len(ids)
        n_good += len(good)
        rows.append({**item, "supported": joint, "good_citations": good})
    return {
        "citation_recall": round(n_supported / n_cited, 3) if n_cited else None,
        "citation_precision": round(n_good / n_cites, 3) if n_cites else None,
        "cited_sentences": n_cited,
        "uncited": sum(1 for r in rows if r["supported"] is None),
        "sentences": rows,
    }
