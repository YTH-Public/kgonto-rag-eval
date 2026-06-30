"""ContextPrecision: 관련 문맥이 상위에 랭크됐는가 (순위 인식). 리랭커/top-k 품질."""
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from ..schema import EvalSample, MetricResult
from .base import Metric


class _PrecisionVerdict(BaseModel):
    relevant_flags: list[bool] = Field(
        description="검색 순서대로 각 문맥이 질문에 관련 있으면 true, 아니면 false. 문맥 개수와 같은 길이."
    )
    reason: str = Field(description="한국어 1~2문장")


_SYSTEM = """당신은 RAG context precision 평가자입니다.
검색된 문맥을 순서대로 보고, 각 문맥이 질문에 관련 있는지 true/false 로 표시합니다(relevant_flags).
- 관련 문맥이 상위(앞쪽)에 올수록 좋은 검색입니다.
- 주제만 비슷한 것은 관련 없음(false)으로 봅니다. 문맥 개수만큼 정확히 표시하세요."""


def _average_precision(flags: list[bool]) -> float:
    """관련 문맥이 상위에 있을수록 높은 점수 (RAGAS context precision 방식).
    score = sum_k(P@k * rel_k) / (관련 문맥 수)."""
    num_relevant = sum(1 for f in flags if f)
    if num_relevant == 0:
        return 0.0
    hits = 0
    weighted = 0.0
    for k, rel in enumerate(flags, start=1):
        if rel:
            hits += 1
            weighted += hits / k   # 그 위치까지의 precision@k
    return weighted / num_relevant


class ContextPrecision(Metric):
    name = "context_precision"
    requires_reference = False

    def score(self, sample: EvalSample, llm) -> MetricResult:
        n = len(sample.retrieved_contexts)
        if n == 0:
            return MetricResult(metric=self.name, score=0.0, verdict="fail", reason="검색 문맥이 없습니다.",
                                failed_component="retriever")
        v = llm.with_structured_output(_PrecisionVerdict).invoke([
            SystemMessage(content=_SYSTEM),
            HumanMessage(content=f"질문: {sample.user_input}\n\n검색 문맥(순서대로):\n{sample.context_text}"),
        ])
        flags = list(v.relevant_flags)[:n]
        flags += [False] * (n - len(flags))   # 부족하면 무관으로 채움
        score = round(_average_precision(flags), 3)
        verdict = MetricResult.verdict_from_score(score)
        return MetricResult(
            metric=self.name, score=score, verdict=verdict, reason=v.reason,
            failed_component="retriever" if verdict == "fail" else None,
            details={"relevant_flags": flags},
        )
