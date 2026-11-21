"""평가 기준선 저장 · 비교 (회귀 테스트용, LLM 불필요).

기준선 파일(JSON)에는 행 ID 별 원점수(반올림하지 않은 값)와 버전 정보가 들어갑니다.
비교는 같은 sample_id 끼리 합니다. 기준선에서 점수가 있던 칸이 지금 없거나(None · 행 없음) 실패하면
기본으로 실패(ok=False)입니다. 일부만 비교하려면 metrics · allow_missing 으로 범위를 직접 정하세요.
"""
from __future__ import annotations

import json
import pathlib
from typing import Optional, Sequence


def _scores(report, metrics: Sequence[str]) -> dict:
    out: dict = {}
    for row in report.rows:
        key = str(row["sample_id"])
        if key in out:
            raise ValueError(f"sample_id 가 중복됩니다: {key} (기준선은 행마다 다른 ID 가 필요합니다)")
        out[key] = {m: row.get(m) for m in metrics}
    return out


def save_baseline(report, path, metrics: Optional[Sequence[str]] = None, meta: Optional[dict] = None) -> dict:
    """report(EvalReport) 를 기준선 파일로 저장합니다. meta 에 dataset · judge · prompt 버전을 적으세요."""
    metrics = list(metrics or report.summary.keys())
    if not metrics:
        raise ValueError("저장할 지표가 없습니다.")
    data = {"meta": meta or {}, "metrics": metrics, "scores": _scores(report, metrics)}
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


def compare_baseline(report, path, tolerance: float = 0.1, metrics: Optional[Sequence[str]] = None,
                     allow_missing: bool = False) -> dict:
    """기준선과 비교합니다.

    - 평균은 양쪽 모두 점수가 있는 같은 ID 만으로 냅니다(분모가 달라 생기는 착시 방지).
    - 평균이 tolerance 보다 많이 떨어지면 그 지표는 regressed.
    - 기준선에서 점수가 있던 칸이 지금 비었으면 lost 에 적고, allow_missing=False(기본)면 실패합니다.
    - 비교할 지표나 짝지은 행이 없으면 실패합니다.
    반환: {metric: {"baseline", "current", "delta", "paired", "regressed", "lost"}},
          "only_in_baseline", "only_in_current", "meta", "ok"
    """
    base = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    metrics = list(metrics or base["metrics"])
    cur = _scores(report, metrics)
    result: dict = {"only_in_baseline": sorted(set(base["scores"]) - set(cur)),
                    "only_in_current": sorted(set(cur) - set(base["scores"])), "meta": base.get("meta", {})}
    ok = bool(metrics)
    for m in metrics:
        lost, pairs = [], []
        for sid, scores in base["scores"].items():
            b = scores.get(m)
            if b is None:
                continue
            c = cur.get(sid, {}).get(m)
            if c is None:
                lost.append(sid)
            else:
                pairs.append((b, c))
        if not pairs:
            result[m] = {"baseline": None, "current": None, "delta": None, "paired": 0,
                         "regressed": None, "lost": lost}
            ok = False
            continue
        b = sum(p[0] for p in pairs) / len(pairs)
        c = sum(p[1] for p in pairs) / len(pairs)
        regressed = (b - c) > tolerance
        ok = ok and not regressed and (allow_missing or not lost)
        result[m] = {"baseline": b, "current": c, "delta": c - b, "paired": len(pairs),
                     "regressed": regressed, "lost": lost}
    result["ok"] = ok
    return result
