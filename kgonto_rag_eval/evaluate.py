"""evaluate(): 데이터셋 × 지표를 채점해 EvalReport 로."""
from __future__ import annotations

import dis
import functools
import hashlib
import inspect
import json
import re
import sys
import sysconfig
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Optional, Sequence

from pydantic import ValidationError

from .llm import default_judge
from .metrics.base import Metric
from .schema import EvaluationDataset, EvalReport, MetricResult


_CODE_ERRORS = (KeyError, IndexError, AttributeError, TypeError, ZeroDivisionError, ValidationError)   # 지표 코드 · 입력의 문제로 보는 예외
_SECRET = re.compile(r"api[_-]?key|secret|passw(or)?d|authorization|(access|auth|bearer)[_-]?token", re.I)
_ROW_NAMES = {"idx", "sample_id", "user_input", "response", "reference", "n_contexts", "retrieved_contexts"}   # 표의 칸이라 지표 이름으로 못 씀
_JUDGE_OPTIONS = ("temperature", "top_p", "seed", "reasoning_effort", "reasoning", "use_responses_api")   # 기록해도 되는 것만


def _sha(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def _plain(value) -> bool:
    """기록해도 되는 단순한 값인지 (글자 · 숫자 · 참거짓 · 그 묶음)"""
    if isinstance(value, dict):
        return all(isinstance(k, str) and _plain(v) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return all(_plain(v) for v in value)
    return value is None or isinstance(value, (str, int, float, bool))


def _code_parts(code) -> list:
    """함수의 실행 내용 - 파일 위치 · 줄 번호는 빼고 (노트북에서 만든 지표도 비교할 수 있게)"""
    consts = [_code_parts(c) if inspect.iscode(c) else sorted(map(repr, c)) if isinstance(c, frozenset) else repr(c) for c in code.co_consts]
    n_args = code.co_argcount + code.co_kwonlyargcount
    shape = [code.co_argcount, code.co_posonlyargcount, code.co_kwonlyargcount, code.co_flags & 0x0C, list(code.co_varnames[:n_args])]
    return [code.co_code.hex(), list(code.co_names), consts, shape]


def _global_names(code) -> set:
    """함수가 바깥(전역)에서 가져다 쓰는 이름. 안쪽 함수까지 봅니다."""
    names = {i.argval for i in dis.get_instructions(code) if i.opname == "LOAD_GLOBAL"}
    return names.union(*[_global_names(c) for c in code.co_consts if inspect.iscode(c)])


def _function_parts(fn, seen: tuple = ()) -> list:
    """계산 + 기본 인자 + 바깥에서 가져다 쓴 단순한 값 · 직접 만든 함수"""
    def part(x):
        return _part(x, (*seen, fn))
    cells = []
    for x in fn.__closure__ or ():
        try:
            cells.append(part(x.cell_contents))
        except ValueError:                         # 아직 값이 없는 칸
            cells.append("empty")
    outside = [[n, part(fn.__globals__[n])] for n in sorted(_global_names(fn.__code__)) if n in fn.__globals__]
    return [_code_parts(fn.__code__), part(fn.__defaults__), part(fn.__kwdefaults__), cells, outside]


_STDLIB = sysconfig.get_path("stdlib").replace("\\", "/").lower()


def _user_function(fn) -> bool:
    """사용자가 쓴 함수인지 - 설치한 라이브러리 · 파이썬 기본 라이브러리 · 이 패키지의 함수는 이름만 봅니다(판으로 비교)"""
    path = fn.__code__.co_filename.replace("\\", "/").lower()
    return not ("site-packages/" in path or "dist-packages/" in path or path.startswith((_STDLIB + "/", "<frozen"))
                or (fn.__module__ or "").startswith("kgonto_rag_eval."))


def _part(x, seen: tuple = ()):
    """지표가 가져다 쓰는 값 하나 -> 비교할 수 있는 모양. 글자 · 숫자는 그대로, 직접 만든 함수는 다른 파일에 있어도 속까지, 그 밖의 객체는 이름만"""
    if _plain(x):
        return x
    if isinstance(x, functools.partial):
        return ["partial", _part(x.func, seen), _part(x.args, seen), _part(x.keywords, seen)]
    if isinstance(x, (list, tuple)):
        return [_part(v, seen) for v in x]
    if isinstance(x, dict):
        return [[repr(k), _part(v, seen)] for k, v in x.items()]
    if inspect.isfunction(x) and x not in seen and _user_function(x):
        return _function_parts(x, seen)
    return f"{getattr(x, '__module__', '')}.{getattr(x, '__qualname__', type(x).__name__)}"


def _code_sha(cls) -> str:
    """지표의 프롬프트 · 스키마 · 계산이 바뀌었는지 볼 지문. 물려받은 클래스까지 봅니다.

    패키지의 지표는 모듈 글 전체, 직접 만든 지표는 클래스 안 함수의 바이트코드 · 기본 인자 · 클래스에 적은 설정과,
    함수가 가져다 쓴 바깥의 글자 · 숫자 · 함수로 봅니다.
    (파이썬 판이 바뀌면 달라지고, 바깥의 객체(클래스 · 프롬프트 틀) 속은 보지 못합니다 -> run_meta["conditions"] 에 적습니다.)
    """
    parts = []
    for c in cls.__mro__:
        if c is object or c.__module__ == "kgonto_rag_eval.metrics.base":
            continue
        if c.__module__.startswith("kgonto_rag_eval."):
            parts.append(inspect.getsource(sys.modules[c.__module__]).replace("\r\n", "\n"))
        else:
            for k, v in sorted(vars(c).items()):
                if inspect.isfunction(v):
                    parts.append([k, _function_parts(v)])
                elif not k.startswith("__") and _plain(v):
                    parts.append([k, v])           # 클래스에 적은 설정
    return _sha(parts)


def _check_run_meta(value, path: str = "run_meta") -> None:
    """비밀 값으로 보이는 이름이 있으면 멈춤 - report 와 기준선 파일에 그대로 남기 때문"""
    if isinstance(value, dict):
        for k, v in value.items():
            if _SECRET.search(str(k)):
                raise ValueError(f"{path} 에 비밀 값으로 보이는 이름이 있습니다: {k} (키 · 비밀번호는 넣지 마세요)")
            _check_run_meta(v, f"{path}.{k}")
    elif isinstance(value, (list, tuple)):
        for v in value:
            _check_run_meta(v, path)


def _meta(dataset, metrics, judge, max_workers: int, run_meta: Optional[dict]) -> dict:
    """무엇으로 채점했는지 - 기준선을 비교해도 되는지 볼 때 씁니다"""
    from . import __version__

    questions = [[s.sample_id, s.user_input, s.reference, s.metadata.get("expected_facts")] for s in dataset]
    info = {}
    for m in metrics:
        config = getattr(m, "__dict__", {})
        info[m.name] = {"class": type(m).__name__, "code_sha": _code_sha(type(m)),
                        "config_sha": _sha({k: _part(v) for k, v in config.items()}) if config else None}
    model = getattr(judge, "model_name", None) or getattr(judge, "model", None)
    return {"package_version": __version__, "created_at": datetime.now().isoformat(timespec="seconds"), "n_samples": len(dataset),
            "metrics": info, "judge": str(model) if model else None,
            "judge_options": {k: v for k in _JUDGE_OPTIONS if (v := getattr(judge, k, None)) is not None and _plain(v)},
            "questions_sha": _sha(questions), "max_workers": max_workers, "run_meta": json.loads(json.dumps(run_meta or {}, ensure_ascii=False, default=str))}


def _failed_component(metric, e: Exception) -> str:
    """예외가 난 곳. 코드 지표면 metric, 모델 호출(invoke) 안에서 났으면 judge, 그 밖의 코드 · 입력 실수는 metric."""
    if not getattr(metric, "needs_llm", True):
        return "metric"
    tb = e.__traceback__
    while tb:
        frame = tb.tb_frame
        if frame.f_code.co_name in ("invoke", "ainvoke") and frame.f_locals.get("self") is not metric:   # 지표 자신의 함수 이름은 빼고
            return "judge"
        tb = tb.tb_next
    return "metric" if isinstance(e, _CODE_ERRORS) else "judge"


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
    run_meta: report.meta["run_meta"] 에 남길 정보. 채점 조건(데이터 판 · 판정 규칙 판)은 {"conditions": {...}} 에 넣으면
            compare_baseline(strict_meta=True) 가 비교합니다. 시스템 판처럼 비교 대상인 것은 그 밖에 넣습니다. 비밀 값은 넣지 마세요.

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
    bad = [n for n in names if not isinstance(n, str) or not n.strip() or n in _ROW_NAMES or "__" in n]
    if bad:
        raise ValueError(f"지표 이름으로 쓸 수 없습니다(표의 칸 이름이거나 '__' 가 있음): {bad}")
    _check_run_meta(run_meta or {})
    ids = [s.sample_id if s.sample_id is not None else str(i) for i, s in enumerate(dataset)]   # 안 준 ID 는 순번
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
