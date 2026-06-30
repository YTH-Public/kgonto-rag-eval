"""지표 베이스. 모든 지표는 Metric 을 상속하고 score(sample, llm) 를 구현합니다."""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..schema import EvalSample, MetricResult


class Metric(ABC):
    name: str = "metric"
    requires_reference: bool = False   # reference 가 있어야 평가 가능한 지표면 True

    @abstractmethod
    def score(self, sample: EvalSample, llm) -> MetricResult:
        """샘플 한 건을 0~1 로 채점. 평가 불가면 score=None."""
        ...

    def _no_reference(self, sample: EvalSample) -> bool:
        return self.requires_reference and not sample.reference
