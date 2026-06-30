"""AnswerRelevancy: 답변이 질문의 의도·범위·형식에 직접 답했는가. (사실성 아님)"""
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from ..schema import EvalSample, MetricResult
from .base import Metric


class _RelevancyVerdict(BaseModel):
    total_intents: int = Field(description="질문이 요구하는 의도/요구사항 수 (1~3)")
    addressed_intents: int = Field(description="답변이 직접 다룬 의도 수")
    verbosity_penalty: float = Field(ge=0, le=0.3, description="불필요한 장황함·범위 이탈 감점 0~0.3")
    reason: str = Field(description="한국어 1~2문장")


_SYSTEM = """당신은 RAG answer relevancy 평가자입니다. 사실 정확성이 아니라 '질문에 제대로 답했는가'를 봅니다.
- 질문의 요구를 1~3개 의도로 나눕니다(total_intents).
- 답변이 각 의도를 직접 다뤘는지 셉니다(addressed_intents).
- 불필요한 장황함, 질문 범위 밖 확장, 형식 미준수는 verbosity_penalty(0~0.3)로 감점.
- 길거나 자신감 있는 어조를 보상하지 마세요."""


class AnswerRelevancy(Metric):
    name = "answer_relevancy"
    requires_reference = False

    def score(self, sample: EvalSample, llm) -> MetricResult:
        if not sample.response.strip():
            return MetricResult(metric=self.name, score=0.0, verdict="fail", reason="답변이 비어 있습니다.",
                                failed_component="generator")
        v = llm.with_structured_output(_RelevancyVerdict).invoke([
            SystemMessage(content=_SYSTEM),
            HumanMessage(content=f"질문: {sample.user_input}\n\n답변:\n{sample.response}"),
        ])
        addressed = min(v.addressed_intents, v.total_intents)   # addressed 는 total 을 넘을 수 없음
        base = (addressed / v.total_intents) if v.total_intents else 0.0
        score = round(max(0.0, min(1.0, base - max(0.0, v.verbosity_penalty))), 3)
        verdict = MetricResult.verdict_from_score(score)
        return MetricResult(
            metric=self.name, score=score, verdict=verdict, reason=v.reason,
            failed_component="generator" if verdict == "fail" else None,
            details={"addressed": addressed, "total": v.total_intents,
                     "verbosity_penalty": round(v.verbosity_penalty, 3)},
        )
