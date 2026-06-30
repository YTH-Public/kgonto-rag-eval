"""evaluate(): 데이터셋 × 지표를 채점해 EvalReport 로."""
from __future__ import annotations

from typing import Optional, Sequence

from .llm import default_judge
from .metrics.base import Metric
from .schema import EvaluationDataset, EvalReport


def evaluate(
    dataset,
    metrics: Sequence[Metric],
    llm=None,
    keep_reason: bool = True,
) -> EvalReport:
    """RAG 평가 데이터셋을 지표들로 채점.

    dataset: EvaluationDataset 또는 dict 리스트.
    metrics: Metric 인스턴스 리스트.
    llm: 평가자 LLM(미지정 시 gpt-5.4-mini). 더 강하거나 다른 judge 를 주입하는 것이 1급 경로.
    keep_reason: True 면 각 지표의 reason/failed_component 도 행에 보존(왜 떨어졌는지 진단).
    """
    if not isinstance(dataset, EvaluationDataset):
        dataset = EvaluationDataset.from_list(list(dataset))
    judge = llm or default_judge()

    rows: list[dict] = []
    for i, sample in enumerate(dataset):
        row: dict = {"idx": i, "user_input": sample.user_input}
        for m in metrics:
            res = m.score(sample, judge)
            row[m.name] = res.score
            row[f"{m.name}__verdict"] = res.verdict
            if keep_reason:
                row[f"{m.name}__reason"] = res.reason
                row[f"{m.name}__failed_component"] = res.failed_component
        rows.append(row)

    # 지표별 평균(None 제외)
    summary: dict = {}
    for m in metrics:
        vals = [r[m.name] for r in rows if r.get(m.name) is not None]
        summary[m.name] = round(sum(vals) / len(vals), 3) if vals else None

    return EvalReport(rows=rows, summary=summary)
