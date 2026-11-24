"""evaluate(): 데이터셋 × 지표를 채점해 EvalReport 로."""
from __future__ import annotations

import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Optional, Sequence

from .llm import default_judge
from .metrics.base import Metric
from .schema import EvaluationDataset, EvalReport, MetricResult


_CODE_ERRORS = (KeyError, IndexError, AttributeError, TypeError, ZeroDivisionError)   # 지표 코드 · 입력의 문제로 보는 예외


def _sha(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def _meta(dataset, metrics, judge, max_workers: int, run_meta: Optional[dict]) -> dict:
    """무엇으로 채점했는지 - 기준선을 비교해도 되는지 볼 때 씁니다"""
    from . import __version__

    questions = [[s.sample_id, s.user_input, s.reference, s.metadata.get("expected_facts")] for s in dataset]
    info = {}
    for m in metrics:
        module = sys.modules.get(type(m).__module__)
        prompt = getattr(module, "_SYSTEM", None)
        info[m.name] = {"class": type(m).__name__, "prompt_sha": _sha(prompt) if prompt else None}
    model = getattr(judge, "model_name", None) or getattr(judge, "model", None)
    return {"package_version": __version__, "created_at": datetime.now().isoformat(timespec="seconds"), "n_samples": len(dataset),
            "metrics": info, "judge": str(model) if model else None,
            "judge_effort": getattr(judge, "reasoning_effort", None) if isinstance(getattr(judge, "reasoning_effort", None), str) else None,
            "questions_sha": _sha(questions), "max_workers": max_workers, "run_meta": dict(run_meta or {})}


def _failed_component(metric, e: Exception) -> str:
    """예외가 난 곳 - 코드 지표나 입력 · 코드 실수면 metric, 나머지는 판정 모델 호출 실패로 봄"""
    if not getattr(metric, "needs_llm", True) or isinstance(e, _CODE_ERRORS):
        return "metric"
    return "judge"


def evaluate(
    dataset,
    metrics: Sequence[Metric],
    llm=None,
    keep_reason: bool = True,
    keep_details: bool = True,
    keep_contexts: bool = True,
    *,
    max_workers: int = 1,
    run_meta: Optional[dict] = None,
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
    run_meta: report.meta["run_meta"] 에 남길 정보(데이터 판 · 시스템 판 등).

    한 지표의 judge 호출이 실패해도 전체를 멈추지 않고 그 칸을 오류로 남깁니다.
    summary 는 유효 점수만의 평균이고, stats 에 유효 · 누락 · 오류 개수가 함께 있습니다.
    """
    if not isinstance(dataset, EvaluationDataset):
        dataset = EvaluationDataset.from_list(list(dataset))
    if isinstance(max_workers, bool) or not isinstance(max_workers, int) or max_workers < 1:
        raise ValueError(f"max_workers 는 1 이상의 정수여야 합니다: {max_workers!r}")
    metrics = list(metrics)
    names = [m.name for m in metrics]
    if len(set(names)) != len(names):
        raise ValueError(f"지표 이름이 겹칩니다: {sorted({n for n in names if names.count(n) > 1})}")
    ids = [s.sample_id for s in dataset if s.sample_id is not None]
    if len(set(ids)) != len(ids):
        raise ValueError(f"sample_id 가 겹칩니다: {sorted({i for i in ids if ids.count(i) > 1})}")
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
                res = MetricResult(metric=m.name, failed_component=_failed_component(m, e),
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
            row[f"{m.name}__status"] = res.status
            row[f"{m.name}__reason_code"] = res.reason_code
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
                         "errors": errors[m.name],
                         "not_applicable": sum(1 for r in rows if r[f"{m.name}__status"] == "not_applicable")}

    return EvalReport(rows=rows, summary=summary, stats=stats, meta=_meta(dataset, metrics, judge, max_workers, run_meta))
