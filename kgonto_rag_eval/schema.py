"""평가 데이터 스키마. RAGAS 의 EvaluationDataset 관례를 따르되 가볍게."""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


def _to_text(ctx: Any) -> str:
    """문맥 조각을 텍스트로. langchain Document(page_content) 와 문자열 모두 수용."""
    if isinstance(ctx, str):
        return ctx
    return getattr(ctx, "page_content", str(ctx))


class EvalSample(BaseModel):
    """평가 한 건. retrieved_contexts 는 문자열/Document 모두 받아 문자열로 정규화."""
    user_input: str
    retrieved_contexts: list[str] = Field(default_factory=list)
    response: str = ""
    reference: Optional[str] = None

    @field_validator("retrieved_contexts", mode="before")
    @classmethod
    def _normalize_contexts(cls, v):
        if v is None:
            return []
        return [_to_text(c) for c in v]

    @property
    def context_text(self) -> str:
        """문맥 조각들을 번호 매겨 한 덩어리로."""
        return "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(self.retrieved_contexts))


class MetricResult(BaseModel):
    """지표 한 개의 채점 결과. (공통 스키마: score/verdict/reason/failed_component/details)"""
    metric: str
    score: Optional[float] = None   # 0~1, 평가 불가(예: reference 없음)면 None
    verdict: Optional[str] = None   # pass | partial | fail
    reason: str = ""
    failed_component: Optional[str] = None   # retriever | generator | reference | judge_uncertain
    details: dict = Field(default_factory=dict)

    @staticmethod
    def verdict_from_score(score: Optional[float]) -> Optional[str]:
        if score is None:
            return None
        if score >= 0.8:
            return "pass"
        if score >= 0.5:
            return "partial"
        return "fail"


class EvaluationDataset(BaseModel):
    """평가 샘플 묶음."""
    samples: list[EvalSample]

    @classmethod
    def from_list(cls, rows: list[dict]) -> "EvaluationDataset":
        return cls(samples=[EvalSample(**r) for r in rows])

    @classmethod
    def from_pandas(cls, df) -> "EvaluationDataset":
        return cls.from_list(df.to_dict(orient="records"))

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)


class EvalReport(BaseModel):
    """평가 결과: 샘플별 점수 행 + 지표별 평균."""
    rows: list[dict]
    summary: dict

    def to_dict(self) -> dict:
        return {"rows": self.rows, "summary": self.summary}

    def to_pandas(self):
        import pandas as pd  # pandas 는 선택 의존성

        return pd.DataFrame(self.rows)

    def __repr__(self) -> str:
        return f"EvalReport(summary={self.summary})"
