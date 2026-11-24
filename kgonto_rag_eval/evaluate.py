"""evaluate(): 데이터셋 × 지표를 채점해 EvalReport 로."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
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
    keep_contexts: bool = True,
    *,
    max_workers: int = 1,
) -> EvalReport:
    """RAG 평가 데이터셋을 지표들로 채점.

    dataset: EvaluationDataset 또는 dict 리스트.
    metrics: Metric 인스턴스 리스트.
    llm: 평가자 LLM. 노트북에서 직접 넣는 것을 권장합니다(미지정 시 default_judge()).
        모든 지표가 코드 지표(needs_llm=False)면 넣지 않아도 되고, 기본 모델도 만들지 않습니다.
    keep_reason: 지표별 reason · failed_component 를 행에 남깁니다.
    keep_details: 지표별 판정 세부(details)와 오류(error)를 행에 남깁니다.
    keep_contexts: 채점에 쓴 근거 원문(retrieved_contexts)을 행에 남깁니다(판정 오류 추적용).
    max_workers: 동시에 채점할 샘플 수. 1 이면 차례로(기본), 4 면 샘플 네 개를 함께 채점합니다.
        샘플 하나의 지표들은 차례로 채점하고, 행 순서는 입력 순서 그대로입니다.
        채점하면서 자기 상태를 바꾸는 지표를 직접 만들었다면 1 로 두세요.

    한 지표의 judge 호출이 실패해도 전체를 멈추지 않고 그 칸을 오류로 남깁니다.
    summary 는 유효 점수만의 평균이고, stats 에 유효 · 누락 · 오류 개수가 함께 있습니다.
    """
    if not isinstance(dataset, EvaluationDataset):
        dataset = EvaluationDataset.from_list(list(dataset))
    if isinstance(max_workers, bool) or not isinstance(max_workers, int) or max_workers < 1:
        raise ValueError(f"max_workers 는 1 이상의 정수여야 합니다: {max_workers!r}")
    metrics = list(metrics)
    judge = llm
    if judge is None and len(dataset) and any(getattr(m, "needs_llm", True) for m in metrics):
        judge = default_judge()   # 모델이 필요한 지표가 있을 때만 만듦

    def score_sample(i: int, sample) -> tuple[dict, list[str]]:
        """샘플 하나를 모든 지표로 채점 -> (행, 오류 난 지표 이름)"""
        failed: list[str] = []
        row: dict = {
            "idx": i,
            "sample_id": sample.sample_id if sample.sample_id is not None else str(i),
            "user_input": sample.user_input,
            "response": sample.response,
            "reference": sample.reference,
            "n_contexts": len(sample.retrieved_contexts),
            **{f"meta__{k}": v for k, v in sample.metadata.items()},
        }
        if keep_contexts:
            row["retrieved_contexts"] = list(sample.retrieved_contexts)
        for m in metrics:
            try:
                res = m.score(sample, judge)
            except Exception as e:   # judge 호출 · 구조화 출력 검증 실패
                res = MetricResult(metric=m.name, failed_component="judge",
                                   error=f"{type(e).__name__}: {str(e)[:300]}")
            row[m.name] = res.score
            if res.error:
                failed.append(m.name)
            row[f"{m.name}__verdict"] = res.verdict
            if keep_reason:
                row[f"{m.name}__reason"] = res.reason
                row[f"{m.name}__failed_component"] = res.failed_component
            if keep_details:
                row[f"{m.name}__details"] = res.details
                row[f"{m.name}__error"] = res.error
        return row, failed

    if max_workers == 1:
        scored = [score_sample(i, s) for i, s in enumerate(dataset)]
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            scored = list(pool.map(score_sample, range(len(dataset)), dataset))   # map 은 입력 순서대로 돌려줌
    rows = [row for row, _ in scored]
    errors = {m.name: sum(failed.count(m.name) for _, failed in scored) for m in metrics}

    summary: dict = {}
    stats: dict = {}
    for m in metrics:
        vals = [r[m.name] for r in rows if r.get(m.name) is not None]
        mean = sum(vals) / len(vals) if vals else None
        summary[m.name] = round(mean, 3) if mean is not None else None
        stats[m.name] = {"mean": mean, "valid": len(vals), "missing": len(rows) - len(vals),
                         "errors": errors[m.name]}

    return EvalReport(rows=rows, summary=summary, stats=stats)
