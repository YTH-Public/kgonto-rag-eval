"""AnsweredRelevancy: 질문에 답했는지 먼저 보고, 답했으면 AnswerRelevancy 점수. (거절이 옳은지는 보지 않음)"""
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from ..schema import EvalSample, MetricResult
from .answer_relevancy import AnswerRelevancy


class _Answered(BaseModel):
    reason: str = Field(description="한국어 한 문장, 판단 이유를 먼저")
    answered: bool = Field(description="질문이 물은 것 가운데 하나라도 실제 내용으로 답했으면 true")


_SYSTEM = """당신은 답이 질문에 답했는지 가르는 평가자입니다. 답이 맞는지는 보지 않습니다.
- 질문이 물은 것 가운데 하나라도 실제 내용(값 · 이름 · 조건 · 방법)으로 답했으면 answered=true.
- "자료에 없습니다", "알 수 없습니다"처럼 답하지 않겠다고만 했으면 false.
- 단위나 표현이 질문과 달라도(며칠을 물었는데 개월로 답함), 조건을 들어 간접으로 답해도 true.
- 질문이 묻지 않은 다른 것을 덧붙이고 정작 물은 것은 자료에 없다고 했으면 false."""


class AnsweredRelevancy(AnswerRelevancy):
    name = "answered_relevancy"

    def score(self, sample: EvalSample, llm) -> MetricResult:
        if not sample.response.strip():
            return MetricResult(metric=self.name, score=0.0, verdict="fail", reason="답변이 비어 있습니다.",
                                failed_component="generator", details={"answered": False})
        v = llm.with_structured_output(_Answered).invoke([
            SystemMessage(content=_SYSTEM),
            HumanMessage(content=f"질문: {sample.user_input}\n\n답변:\n{sample.response}"),
        ])
        if not v.answered:   # 답하지 않은 답은 0 (옳은 거절일 수도 있음)
            return MetricResult(metric=self.name, score=0.0, verdict="fail", reason=v.reason,
                                failed_component="generator", details={"answered": False})
        result = super().score(sample, llm)   # 답했으면 원래 계산 그대로
        result.details = {**result.details, "answered": True}
        return result
