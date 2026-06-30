"""Faithfulness: 답변의 주장이 검색 문맥에 근거하는가 (환각 탐지)."""
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from ..schema import EvalSample, MetricResult
from .base import Metric


class _FaithVerdict(BaseModel):
    total_claims: int = Field(description="답변에 담긴 사실 주장의 총 개수")
    supported_claims: int = Field(description="검색 문맥이 뒷받침하는 주장 개수")
    unsupported: list[str] = Field(default_factory=list, description="문맥에 근거가 없는 주장 목록")
    reason: str = Field(description="한국어 2~3문장 근거")


_SYSTEM = """당신은 RAG faithfulness 평가자입니다.
답변을 사실 주장(claim) 단위로 나누고, 각 주장이 '참고 문맥'에 의해 뒷받침되는지 판단합니다.
- 세상 지식으로 맞더라도 문맥에 없으면 근거 없음으로 봅니다.
- total_claims: 답변의 주장 총수. supported_claims: 문맥이 뒷받침하는 수.
- 근거 없는 주장은 unsupported 에 적습니다. supported_claims 는 total_claims 를 넘지 않습니다."""


class Faithfulness(Metric):
    name = "faithfulness"
    requires_reference = False

    def score(self, sample: EvalSample, llm) -> MetricResult:
        if not sample.retrieved_contexts:
            return MetricResult(metric=self.name, score=0.0, reason="검색 문맥이 없습니다.")
        if not sample.response.strip():
            return MetricResult(metric=self.name, score=0.0, reason="답변이 비어 있습니다.")
        v = llm.with_structured_output(_FaithVerdict).invoke([
            SystemMessage(content=_SYSTEM),
            HumanMessage(content=f"참고 문맥:\n{sample.context_text}\n\n답변:\n{sample.response}"),
        ])
        supported = min(v.supported_claims, v.total_claims)   # supported 는 total 을 넘을 수 없음
        score = (supported / v.total_claims) if v.total_claims else 0.0
        score = round(max(0.0, min(1.0, score)), 3)
        verdict = MetricResult.verdict_from_score(score)
        return MetricResult(
            metric=self.name, score=score, verdict=verdict, reason=v.reason,
            failed_component="generator" if verdict == "fail" else None,
            details={"supported": supported, "total": v.total_claims, "unsupported": v.unsupported},
        )
