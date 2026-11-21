"""evaluate(): 데이터셋 × 지표를 채점해 EvalReport 로."""
from __future__ import annotations

from typing import Sequence

from .llm import default_judge
from .metrics.base import Metric
from .schema import EvaluationDataset, EvalReport, MetricResult


def evaluate(
    dataset,
    metrics: Sequence[Metric],
    llm=None,
    keep_reason: bool = True,
    keep_details: bool = True,
) -> EvalReport:
    """RAG 평가 데이터셋을 지표들로 채점.

    dataset: EvaluationDataset 또는 dict 리스트.
    metrics: Metric 인스턴스 리스트.
    llm: 평가자 LLM. 노트북에서 직접 넣는 것을 권장합니다(미지정 시 default_judge()).
    keep_reason: 지표별 reason · failed_component 를 행에 남깁니다.
    keep_details: 지표별 판정 세부(details)와 오류(error)를 행에 남깁니다.

    한 지표의 judge 호출이 실패해도 전체를 멈추지 않고 그 칸을 오류로 남깁니다.
    summary 는 유효 점수만의 평균이고, stats 에 유효 · 누락 · 오류 개수가 함께 있습니다.
    """
    if not isinstance(dataset, EvaluationDataset):
        dataset = EvaluationDataset.from_list(list(dataset))
    judge = llm or default_judge()

    rows: list[dict] = []
    for i, sample in enumerate(dataset):
        row: dict = {
            "sample_id": sample.sample_id if sample.sample_id is not None else str(i),
            "user_input": sample.user_input,
            "response": sample.response,
            "reference": sample.reference,
            "n_contexts": len(sample.retrieved_contexts),
            **{f"meta__{k}": v for k, v in sample.metadata.items()},
        }
        for m in metrics:
            try:
                res = m.score(sample, judge)
            except Exception as e:   # judge 호출 · 구조화 출력 검증 실패
                res = MetricResult(metric=m.name, failed_component="judge",
                                   error=f"{type(e).__name__}: {str(e)[:300]}")
            row[m.name] = res.score
            row[f"{m.name}__verdict"] = res.verdict
            if keep_reason:
                row[f"{m.name}__reason"] = res.reason
                row[f"{m.name}__failed_component"] = res.failed_component
            if keep_details:
                row[f"{m.name}__details"] = res.details
                row[f"{m.name}__error"] = res.error
        rows.append(row)

    summary: dict = {}
    stats: dict = {}
    for m in metrics:
        vals = [r[m.name] for r in rows if r.get(m.name) is not None]
        errors = sum(1 for r in rows if r.get(f"{m.name}__error"))
        mean = sum(vals) / len(vals) if vals else None
        summary[m.name] = round(mean, 3) if mean is not None else None
        stats[m.name] = {"mean": mean, "valid": len(vals), "missing": len(rows) - len(vals),
                         "errors": errors}

    return EvalReport(rows=rows, summary=summary, stats=stats)
