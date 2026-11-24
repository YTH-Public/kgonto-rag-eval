"""지표 베이스. 모든 지표는 Metric 을 상속하고 score(sample, llm) 를 구현합니다."""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..schema import EvalSample, MetricResult


class Metric(ABC):
    name: str = "metric"
    requires_reference: bool = False   # reference 가 있어야 평가 가능한 지표면 True
    needs_llm: bool = True             # 모델을 부르지 않는 코드 지표면 False (기본 판정 모델을 만들지 않음)

    @abstractmethod
    def score(self, sample: EvalSample, llm) -> MetricResult:
        """샘플 한 건을 0~1 로 채점. 평가 불가면 score=None."""
        ...

    def _no_reference(self, sample: EvalSample) -> bool:
        return self.requires_reference and not (sample.reference or "").strip()

    def judge_error(self, message: str, **details) -> MetricResult:
        """judge 출력이 규칙을 어겼을 때. 점수를 만들지 않고 오류로 남깁니다."""
        return MetricResult(metric=self.name, score=None, verdict=None, reason=message,
                            failed_component="judge", error=message, details=details)


def check_counts(total: int, part: int, part_name: str = "supported") -> str | None:
    """judge 가 낸 개수의 앞뒤가 맞는지. 문제가 있으면 이유 문자열, 없으면 None."""
    if total < 0 or part < 0:
        return f"개수가 음수입니다(total={total}, {part_name}={part})."
    if part > total:
        return f"{part_name}={part} 가 total={total} 보다 큽니다."
    return None
