"""AnswerCorrectness: 답변이 기준 정답과 사실·의미상 맞는가. factual F1 + semantic adequacy."""
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from ..schema import EvalSample, MetricResult
from .base import Metric


class _CorrectnessVerdict(BaseModel):
    factual_precision: float = Field(ge=0, le=1, description="답변의 사실 주장 중 기준 정답이 뒷받침하는 비율")
    factual_recall: float = Field(ge=0, le=1, description="기준 정답의 핵심 주장 중 답변이 포함한 비율")
    semantic_adequacy: float = Field(ge=0, le=1, description="표현이 달라도 질문 요구를 충족한 정도")
    reason: str = Field(description="한국어 1~2문장")


_SYSTEM = """당신은 RAG answer correctness 평가자입니다. 답변(response)을 기준 정답(reference)과 비교합니다.
- factual_precision: 답변이 한 사실 주장 중 기준 정답이 뒷받침하는 비율. 기준에 없는 과한 부연은 깎습니다.
- factual_recall: 기준 정답의 핵심 주장 중 답변이 담은 비율.
- semantic_adequacy: 표현이 달라도 질문 요구를 충족했는지 0~1.
- 숫자, 날짜, 기관명, 법령/표준명, 관계 방향은 엄격하게 봅니다. 한국어 표현 차이는 관대하게.
- 길이나 자신감 있는 어조를 보상하지 마세요. 애매하면 높은 점수를 주지 마세요."""


class AnswerCorrectness(Metric):
    name = "answer_correctness"
    requires_reference = True

    def score(self, sample: EvalSample, llm) -> MetricResult:
        if not sample.reference:
            return MetricResult(metric=self.name, score=None, reason="기준 정답(reference)이 없습니다.",
                                failed_component="reference")
        v = llm.with_structured_output(_CorrectnessVerdict).invoke([
            SystemMessage(content=_SYSTEM),
            HumanMessage(content=(
                f"질문: {sample.user_input}\n\n기준 정답:\n{sample.reference}\n\n답변:\n{sample.response}"
            )),
        ])
        p, r = max(0.0, min(1.0, v.factual_precision)), max(0.0, min(1.0, v.factual_recall))
        f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
        semantic = max(0.0, min(1.0, v.semantic_adequacy))
        score = round(0.6 * f1 + 0.4 * semantic, 3)
        verdict = MetricResult.verdict_from_score(score)
        return MetricResult(
            metric=self.name, score=score, verdict=verdict, reason=v.reason,
            failed_component="generator" if verdict == "fail" else None,
            details={"factual_precision": round(p, 3), "factual_recall": round(r, 3),
                     "factual_f1": round(f1, 3), "semantic_adequacy": round(semantic, 3)},
        )
