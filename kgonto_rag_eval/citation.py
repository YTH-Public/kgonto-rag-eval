"""인용(출처 표시) 검사.

세 가지를 서로 다른 함수로 봅니다. 하나가 통과해도 나머지가 통과한다는 뜻이 아닙니다.

1. resolve_citations: 답에 적힌 출처 ID 가 실제 자료로 풀리는가 (결정적)
2. quote_in_source:   따옴표로 인용한 문구가 그 자료에 그대로 있는가 (결정적, 공백 무시)
3. citation_support:  문장의 주장을 인용 자료가 뒷받침하는가 (LLM 판정)

3 은 ALCE(Gao et al. 2023, arXiv:2305.14627)의 citation recall / precision 아이디어를
수업용으로 줄인 지표입니다. 원 논문은 NLI 모델(AutoAIS)로 판정하지만, 여기서는 LLM 판정을 씁니다.
- citation_recall: 출처를 단 문장 중, 단 출처들을 합쳐 그 문장을 뒷받침하는 비율
- citation_precision: 단 출처 중 쓸모 있는 출처의 비율. 원 논문처럼 **출처들을 합쳐 문장을 뒷받침할 때만**
  (결합 판정 True) 그 출처를 쓸모 있다고 봅니다 - 혼자서 뒷받침하거나, 빼면 뒷받침이 깨지는 출처.
- 원 논문과 다른 점: 출처가 없는 문장은 recall 분모에서 빼고 uncited 로 따로 셉니다(원 논문은 0 으로 넣음).
  원 논문 방식이 필요하면 citation_recall_all(출처 없는 문장을 0 으로 넣은 값)을 쓰세요.
- 문장 하나의 판정 가운데 하나라도 실패하면 그 문장은 오류(errors)로 남기고 점수에서 뺍니다.
"""
from __future__ import annotations

import re
from typing import Iterable, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

DEFAULT_PATTERN = r"\[p\.(\d+)\]"   # 예: "월 20만 원입니다 [p.35]"


_RANGE = re.compile(r"\[p\.(\d+)\s*[-~\u2013\u2014]\s*(?:p\.)?(\d+)\]")   # [p.1-3] · [p.1~3]
MAX_RANGE = 50                        # 이보다 넓은 범위는 잘못 적은 것으로 보고 펼치지 않음


def expand_ranges(text: str) -> str:
    """출처 범위를 하나씩으로: "가 [p.1-3]" -> "가 [p.1][p.2][p.3]". 거꾸로 된 범위 · 너무 넓은 범위는 그대로 둡니다."""
    def one(m: re.Match) -> str:
        first, last = int(m.group(1)), int(m.group(2))
        if last < first or last - first >= MAX_RANGE:
            return m.group(0)
        return "".join(f"[p.{n}]" for n in range(first, last + 1))
    return _RANGE.sub(one, text or "")


def extract_citations(answer: str, pattern: str = DEFAULT_PATTERN, ranges: bool = True) -> list[str]:
    """답에 적힌 출처 ID 를 적힌 순서대로(중복 포함) 뽑습니다. ranges=True 면 [p.1-3] 을 펼칩니다."""
    text = expand_ranges(answer) if ranges else (answer or "")
    return re.findall(pattern, text)


def _squash(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def split_cited_sentences(answer: str, pattern: str = DEFAULT_PATTERN, ranges: bool = False) -> list[dict]:
    """답을 문장으로 나누고 문장마다 출처 ID 를 뽑습니다.

    반환: [{"sentence": 출처 표시를 뺀 문장, "citations": [ID, ...]}]
    출처 표시가 문장 끝 마침표 뒤에 붙어도 앞 문장의 것으로 봅니다.
    ranges=True 면 [p.1-3] 같은 범위를 펼칩니다(기본 False - 0.2 와 같은 결과).
    """
    if ranges:
        answer = expand_ranges(answer)
    cite = re.sub(r"\((?!\?)", "(?:", pattern)             # 묶음 없는 판 (옮기기용)
    # 마침표 뒤에 붙은 출처를 마침표 앞으로 옮긴다: "A. [p.1] B." -> "A [p.1]. B."
    moved = re.sub(rf"([.!?。])((?:\s*{cite})+)", lambda m: m.group(2) + m.group(1), answer.strip())
    pieces = re.split(r"(?<=[.!?。])\s+|\n+", moved)
    out: list[dict] = []
    for piece in pieces:
        ids = re.findall(pattern, piece)
        text = re.sub(r"\s+([.!?。])", lambda m: m.group(1), re.sub(pattern, "", piece)).strip()   # 출처를 뺀 자리의 공백 정리
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
    squashed = _squash(quote)
    return bool(squashed) and squashed in _squash(source_text)   # 공백뿐인 인용은 False


class _Support(BaseModel):
    supported: bool = Field(description="근거 글이 문장의 모든 주장을 뒷받침하면 true")
    reason: str = Field(description="한국어 한 문장")


_SYSTEM = """당신은 인용 검증자입니다. '근거 글'이 '문장'의 주장을 모두 뒷받침하는지 판단합니다.
- 숫자 · 기간 · 대상 · 조건이 근거 글의 같은 대상에 대한 것이어야 합니다. 같은 글에 숫자가 있어도 다른 대상이면 false.
- 근거 글에 없는 내용이 하나라도 있으면 false. 세상 지식으로 맞아도 근거 글에 없으면 false."""


def _supports(llm, sentence: str, texts: list[str], question: Optional[str]) -> _Support:
    joined = "\n\n".join(f"[근거 {i + 1}] {t}" for i, t in enumerate(texts))
    q = f"질문: {question}\n\n" if question else ""
    v = llm.with_structured_output(_Support).invoke([
        SystemMessage(content=_SYSTEM),
        HumanMessage(content=f"{q}문장: {sentence}\n\n근거 글:\n{joined}"),
    ])
    return v


def citation_support(answer: str, corpus: dict, llm, question: Optional[str] = None,
                     pattern: str = DEFAULT_PATTERN, ranges: bool = False) -> dict:
    """ALCE 아이디어를 적용한 수업용 인용 지표(LLM 판정).

    corpus: 출처 ID -> 원문. 풀리지 않는 ID 는 뒷받침하지 않는 출처로 셉니다.
    반환: citation_recall, citation_precision, 문장별 판정(sentences), 출처 없는 문장 수(uncited),
        attempted(출처가 있어 판정하려 한 문장 수 = cited_sentences + errors).
        sentences 의 각 문장에는 reason(결합 판정의 이유)과 judgements(판정마다 kind · citation · supported · reason)가 있습니다.
        kind: joint(단 출처를 모두 합침) | alone(그 출처만) | rest(그 출처를 뺀 나머지).
    호출 수: 출처 단 문장마다 1번 + (출처가 2개 이상이면) 출처마다 최대 2번.
    """
    keys = {str(k): v for k, v in corpus.items()}
    rows = []
    n_cited = n_supported = n_cites = n_good = 0
    for item in split_cited_sentences(answer, pattern, ranges):
        ids = [str(c) for c in item["citations"]]
        if not ids:
            rows.append({**item, "supported": None, "good_citations": [], "reason": "", "judgements": []})
            continue
        judgements: list[dict] = []

        def ask(kind: str, texts: list[str], citation: Optional[str] = None) -> bool:
            """판정 한 번 - 이유까지 judgements 에 남김"""
            v = _supports(llm, item["sentence"], texts, question)
            judgements.append({"kind": kind, "citation": citation, "supported": bool(v.supported), "reason": v.reason})
            return bool(v.supported)

        try:   # 문장 하나의 판정(결합 · 혼자 · 나머지)을 모두 감쌈 - 실패는 0점이 아니라 오류
            texts = [keys[c] for c in ids if c in keys]
            joint = bool(texts) and ask("joint", texts)
            good: list[str] = []
            if joint:   # 원 논문: 결합 판정이 True 일 때만 출처의 쓸모를 셈
                for c in ids:
                    if c not in keys:
                        continue
                    if len(ids) == 1:
                        good.append(c)
                        continue
                    alone = ask("alone", [keys[c]], c)
                    rest = [keys[o] for o in ids if o != c and o in keys]
                    if alone or not (rest and ask("rest", rest, c)):
                        good.append(c)
        except Exception as e:
            rows.append({**item, "supported": None, "good_citations": [], "error": f"{type(e).__name__}: {e}",
                         "reason": "", "judgements": judgements})
            continue
        n_cited += 1
        n_supported += int(joint)
        n_cites += len(ids)
        n_good += len(good)
        reason = judgements[0]["reason"] if judgements else "출처 ID 가 자료에 없습니다."
        rows.append({**item, "supported": joint, "good_citations": good, "reason": reason, "judgements": judgements})
    uncited = sum(1 for r in rows if r["supported"] is None and not r.get("error"))
    return {
        "citation_recall": round(n_supported / n_cited, 3) if n_cited else None,
        "citation_recall_all": round(n_supported / (n_cited + uncited), 3) if (n_cited + uncited) else None,
        "citation_precision": round(n_good / n_cites, 3) if n_cites else None,
        "cited_sentences": n_cited,
        "attempted": n_cited + sum(1 for r in rows if r.get("error")),
        "uncited": uncited,
        "errors": sum(1 for r in rows if r.get("error")),
        "sentences": rows,
    }
