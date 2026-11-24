"""두 답 비교(pairwise) 판정. 위치 편향을 드러내려고 순서를 바꿔 두 번 묻습니다.

근거: Zheng et al. 2023, "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena"
(arXiv:2306.05685)의 위치 편향 논의. 두 순서의 결과가 다르면 승자를 정하지 않고
"inconsistent" 로 남깁니다.
"""
from __future__ import annotations

from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field


class _Pick(BaseModel):
    reason: str = Field(description="한국어 1~2문장, 판단 이유를 먼저")
    winner: Literal["1", "2", "tie"] = Field(description="더 나은 답의 번호, 차이가 없으면 tie")


_SYSTEM = """당신은 두 답을 비교하는 평가자입니다. 기준에 따라 더 나은 답을 고릅니다.
- 답의 길이나 순서로 판단하지 마세요. 기준에 맞는 내용만 봅니다.
- 차이가 없으면 tie 를 고르세요."""


def _ask(llm, question: str, first: str, second: str, criteria: str) -> _Pick:
    v = llm.with_structured_output(_Pick).invoke([
        SystemMessage(content=_SYSTEM),
        HumanMessage(content=f"기준: {criteria}\n\n질문: {question}\n\n[답 1]\n{first}\n\n[답 2]\n{second}"),
    ])
    return v


def pairwise_judge(question: str, answer_a: str, answer_b: str, llm,
                   criteria: str = "질문에 정확하고 빠짐없이 답했는가") -> dict:
    """A · B 를 AB 순서와 BA 순서로 한 번씩 묻고 답 ID(A/B)로 되돌립니다.

    반환: {"ab", "ba": 각 순서의 결과(A/B/tie 로 되돌린 값), "ab_reason", "ba_reason",
          "final": A | B | tie | inconsistent | error, "consistent": bool, "error": 실패 이유}
    한쪽 호출이 실패해도 다른 쪽 결과는 남깁니다.
    """
    out = {"ab": None, "ba": None, "ab_reason": "", "ba_reason": "", "error": None}
    errors = []
    for name, first, second, back in (("ab", answer_a, answer_b, {"1": "A", "2": "B", "tie": "tie"}),
                                      ("ba", answer_b, answer_a, {"1": "B", "2": "A", "tie": "tie"})):
        try:   # 한 순서가 실패해도 다른 순서는 물어봄
            v = _ask(llm, question, first, second, criteria)
            out[name], out[f"{name}_reason"] = back[v.winner], v.reason
        except Exception as e:
            errors.append(f"{name}: {type(e).__name__}: {e}")
    if errors:
        out["error"] = " | ".join(errors)
        return {**out, "consistent": False, "final": "error"}
    consistent = out["ab"] == out["ba"]
    return {**out, "consistent": consistent, "final": out["ab"] if consistent else "inconsistent"}
