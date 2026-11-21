"""평가 기준선 저장 · 비교 (회귀 테스트용, LLM 불필요).

기준선 파일(JSON)에는 행 ID 별 원점수(반올림 전)와 버전 정보가 들어갑니다.
비교는 같은 sample_id 끼리만 하고, 한쪽에만 있는 ID 와 점수가 없는(None) 칸은 따로 알려 줍니다.
"""
from __future__ import annotations

import json
import pathlib
from typing import Optional, Sequence


def _scores(report, metrics: Sequence[str]) -> dict:
    out: dict = {}
    for row in report.rows:
        out[str(row["sample_id"])] = {m: row.get(m) for m in metrics}
    return out


def save_baseline(report, path, metrics: Optional[Sequence[str]] = None, meta: Optional[dict] = None) -> dict:
    """report(EvalReport) 를 기준선 파일로 저장합니다. meta 에 dataset · judge · prompt 버전을 적으세요."""
    metrics = list(metrics or report.summary.keys())
    data = {"meta": meta or {}, "metrics": metrics, "scores": _scores(report, metrics)}
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


def compare_baseline(report, path, tolerance: float = 0.1, metrics: Optional[Sequence[str]] = None) -> dict:
    """기준선과 비교합니다. 지표 평균이 tolerance 보다 많이 떨어지면 regressed.

    평균은 **양쪽 모두 점수가 있는 같은 ID** 만으로 냅니다(분모가 달라 생기는 착시 방지).
    반환: {metric: {"baseline", "current", "delta", "paired", "regressed"}}, "only_in_baseline", "only_in_current",
          "ok": 모든 지표가 regressed 가 아니고 짝지은 행이 하나 이상이면 True
    """
    base = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    metrics = list(metrics or base["metrics"])
    cur = _scores(report, metrics)
    shared = sorted(set(base["scores"]) & set(cur))
    result: dict = {"only_in_baseline": sorted(set(base["scores"]) - set(cur)),
                    "only_in_current": sorted(set(cur) - set(base["scores"])), "meta": base.get("meta", {})}
    ok = True
    for m in metrics:
        pairs = [(base["scores"][i].get(m), cur[i].get(m)) for i in shared]
        pairs = [(b, c) for b, c in pairs if b is not None and c is not None]
        if not pairs:
            result[m] = {"baseline": None, "current": None, "delta": None, "paired": 0, "regressed": None}
            ok = False
            continue
        b = sum(p[0] for p in pairs) / len(pairs)
        c = sum(p[1] for p in pairs) / len(pairs)
        regressed = (b - c) > tolerance
        ok = ok and not regressed
        result[m] = {"baseline": b, "current": c, "delta": c - b, "paired": len(pairs), "regressed": regressed}
    result["ok"] = ok
    return result
