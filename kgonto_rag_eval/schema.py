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
    """평가 한 건.

    sample_id: 기준선 비교 · 원인 추적용 안정 ID (없으면 evaluate 가 순번을 씁니다).
    retrieved_contexts: **그때 실제로 검색된** 근거 목록. 문자열 / Document 리스트.
        문자열 하나를 그대로 넣으면 글자 단위로 쪼개지므로 오류로 막습니다.
    metadata: split · system_version 같은 부가 정보(점수에는 쓰지 않음).
    """
    sample_id: Optional[str] = None
    user_input: str
    retrieved_contexts: list[str] = Field(default_factory=list)
    response: str = ""
    reference: Optional[str] = None
    metadata: dict = Field(default_factory=dict)

    @field_validator("retrieved_contexts", mode="before")
    @classmethod
    def _normalize_contexts(cls, v):
        if v is None:
            return []
        if isinstance(v, (str, bytes)) or hasattr(v, "page_content"):
            raise ValueError("retrieved_contexts 는 목록이어야 합니다. 조각 하나면 [텍스트] 로 넣으세요.")
        return [_to_text(c) for c in v]

    @property
    def context_text(self) -> str:
        """문맥 조각들을 번호 매겨 한 덩어리로."""
        return "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(self.retrieved_contexts))


class MetricResult(BaseModel):
    """지표 한 개의 채점 결과.

    score: 0~1. 평가할 수 없으면(reference 없음 · judge 오류) None.
    error: judge 출력이 규칙을 어겼거나 호출이 실패하면 그 이유. 이때 score 는 None 입니다
        (오류를 낮은 품질 점수로 숨기지 않기 위해).
    """
    metric: str
    score: Optional[float] = None
    verdict: Optional[str] = None   # pass | partial | fail
    reason: str = ""
    failed_component: Optional[str] = None   # retriever | generator | reference | judge
    error: Optional[str] = None
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
    """평가 결과.

    rows: 샘플별 점수 행(입력 · 판정 세부 · 오류 포함).
    summary: 지표별 평균(유효 점수만, 반올림).
    stats: 지표별 {mean(반올림 전), valid, missing, errors} - 평균이 몇 건에서 나왔는지 함께 봅니다.
    """
    rows: list[dict]
    summary: dict
    stats: dict = Field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"rows": self.rows, "summary": self.summary, "stats": self.stats}

    def to_pandas(self):
        import pandas as pd  # pandas 는 선택 의존성

        return pd.DataFrame(self.rows)

    def explain(self, sample_id, metric: Optional[str] = None) -> dict:
        """샘플 하나의 점수와 저장해 둔 이유 · 세부를 꺼냅니다(모델을 다시 부르지 않음).

        report.explain("P064", "answer_relevancy") -> {"sample_id", "user_input", "response", "reference",
            "metric", "score", "verdict", "reason", "failed_component", "details", "error"}
        metric 을 빼면 {지표 이름: 위와 같은 dict} 를 "metrics" 에 담아 돌려줍니다.
        """
        found = [row for row in self.rows if str(row.get("sample_id")) == str(sample_id)]
        if not found:
            raise KeyError(f"sample_id 가 없습니다: {sample_id}")
        if len(found) > 1:
            raise ValueError(f"sample_id 가 {len(found)}번 나옵니다: {sample_id}")
        row = found[0]
        names = list(self.summary)
        if metric is not None and metric not in names:
            raise KeyError(f"지표가 없습니다: {metric} (있는 지표: {names})")

        def one(name: str) -> dict:
            parts = ("verdict", "reason", "failed_component", "details", "error")
            return {"metric": name, "score": row.get(name), **{k: row.get(f"{name}__{k}") for k in parts}}

        base = {k: row.get(k) for k in ("sample_id", "user_input", "response", "reference")}
        if metric is not None:
            return {**base, **one(metric)}
        return {**base, "metrics": {name: one(name) for name in names}}

    def __repr__(self) -> str:
        return f"EvalReport(summary={self.summary})"
