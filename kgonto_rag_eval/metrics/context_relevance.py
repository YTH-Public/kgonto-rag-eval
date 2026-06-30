"""ContextRelevance: 검색된 각 문맥이 질문에 유용한가. per-context 판정 후 평균."""
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from ..schema import EvalSample, MetricResult
from .base import Metric

_LABEL_SCORE = {"relevant": 1.0, "partial": 0.5, "irrelevant": 0.0}


class _ContextVerdict(BaseModel):
    index: int = Field(description="문맥 번호(1부터)")
    label: str = Field(description="relevant | partial | irrelevant")


class _RelevanceVerdict(BaseModel):
    items: list[_ContextVerdict] = Field(description="각 문맥 조각의 관련성 판정")
    reason: str = Field(description="한국어 1~2문장")


_SYSTEM = """당신은 RAG context relevance 평가자입니다.
검색된 각 문맥 조각이 질문에 답하는 데 유용한지 독립적으로 판정합니다.
- relevant: 답변에 필요한 정보를 담음. partial: 보조적으로만 도움. irrelevant: 무관하거나 distractor.
- 주제만 비슷한 것은 relevant 가 아닙니다. 질문의 답 경로에 기여해야 합니다.
- 모든 문맥 번호(1..N)에 대해 하나씩 판정하세요."""


class ContextRelevance(Metric):
    name = "context_relevance"
    requires_reference = False

    def score(self, sample: EvalSample, llm) -> MetricResult:
        n = len(sample.retrieved_contexts)
        if n == 0:
            return MetricResult(metric=self.name, score=0.0, verdict="fail", reason="검색 문맥이 없습니다.",
                                failed_component="retriever")
        v = llm.with_structured_output(_RelevanceVerdict).invoke([
            SystemMessage(content=_SYSTEM),
            HumanMessage(content=f"질문: {sample.user_input}\n\n검색 문맥:\n{sample.context_text}"),
        ])
        labels = [it.label for it in sorted(v.items, key=lambda x: x.index)][:n]
        # judge 가 일부 문맥만 판정하면 누락분이 분모에서 빠져 과대평가됨 -> 부족분은 irrelevant 로 채움
        if len(labels) < n:
            labels = labels + ["irrelevant"] * (n - len(labels))
        score = round(sum(_LABEL_SCORE.get(l, 0.0) for l in labels) / n, 3)
        verdict = MetricResult.verdict_from_score(score)
        return MetricResult(
            metric=self.name, score=score, verdict=verdict, reason=v.reason,
            failed_component="retriever" if verdict == "fail" else None,
            details={"labels": labels},
        )
