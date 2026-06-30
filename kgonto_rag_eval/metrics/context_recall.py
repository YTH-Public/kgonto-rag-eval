"""ContextRecall: 기준 정답(reference)의 핵심 근거가 검색 문맥에 들어왔는가. 검색기 평가."""
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from ..schema import EvalSample, MetricResult
from .base import Metric


class _RecallVerdict(BaseModel):
    total_claims: int = Field(description="기준 정답을 답하는 데 필수인 핵심 주장 수")
    supported_claims: int = Field(description="검색 문맥에서 근거를 찾을 수 있는 주장 수")
    missing: list[str] = Field(default_factory=list, description="문맥에서 근거를 못 찾은 핵심 주장")
    reason: str = Field(description="한국어 1~2문장")


_SYSTEM = """당신은 RAG context recall 평가자입니다. 생성된 답변은 보지 않습니다.
기준 정답(reference)을 핵심 주장 단위로 나누고, 각 주장이 '검색 문맥'에서 근거를 찾을 수 있는지 판단합니다.
- 질문에 답하는 데 필수인 주장만 셉니다.
- total_claims: 핵심 주장 총수. supported_claims: 문맥이 뒷받침하는 수. supported 는 total 을 넘지 않습니다.
- 애매하면 근거 있다고 보지 마세요."""


class ContextRecall(Metric):
    name = "context_recall"
    requires_reference = True

    def score(self, sample: EvalSample, llm) -> MetricResult:
        if not sample.reference:
            return MetricResult(metric=self.name, score=None, reason="기준 정답(reference)이 없습니다.",
                                failed_component="reference")
        if not sample.retrieved_contexts:
            return MetricResult(metric=self.name, score=0.0, verdict="fail", reason="검색 문맥이 없습니다.",
                                failed_component="retriever")
        v = llm.with_structured_output(_RecallVerdict).invoke([
            SystemMessage(content=_SYSTEM),
            HumanMessage(content=f"기준 정답:\n{sample.reference}\n\n검색 문맥:\n{sample.context_text}"),
        ])
        supported = min(v.supported_claims, v.total_claims)
        score = (supported / v.total_claims) if v.total_claims else 0.0
        score = round(max(0.0, min(1.0, score)), 3)
        verdict = MetricResult.verdict_from_score(score)
        return MetricResult(
            metric=self.name, score=score, verdict=verdict, reason=v.reason,
            failed_component="retriever" if verdict == "fail" else None,
            details={"supported": supported, "total": v.total_claims, "missing": v.missing},
        )
