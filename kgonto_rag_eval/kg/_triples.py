"""트리플 입력을 한 가지 모양으로 바꾸는 공용 함수."""
from __future__ import annotations

from typing import Any, Callable, Iterable, Optional, Sequence


def _pick(r: dict, *keys: str):
    for k in keys:
        if k in r:
            return r[k]
    raise KeyError(f"{keys[0]} (또는 {', '.join(keys[1:])}) 칸이 없습니다: {sorted(r)}")


def as_triples(
    x: Any,
    normalize: Optional[Callable[[str], str]] = None,
    scope: Sequence[str] = (),
) -> set[tuple]:
    """(s, rel, o) 집합으로 바꿉니다.

    x: 트리플 iterable, 또는 dict 리스트(source/relation/target 또는 subject/relation/object 칸).
    normalize: 주어 · 목적어 이름에 적용할 함수(예: 법인 표기 정리). 관계 방향은 바꾸지 않습니다.
    scope: dict 의 칸 이름들(예: ("rcept_no",)). 주면 키 앞에 그 값이 붙어
        (문서, s, rel, o) 처럼 문서 범위 안에서만 같은 트리플로 셉니다.
    """
    norm = normalize or (lambda s: s)
    out: set[tuple] = set()
    for r in x or []:
        if isinstance(r, dict):
            s = _pick(r, "source", "subject")
            o = _pick(r, "target", "object")
            key = tuple(r.get(k) for k in scope) + (norm(s), r["relation"], norm(o))
        else:
            s, rel, o = r
            if scope:
                raise ValueError("scope 는 dict 입력에서만 쓸 수 있습니다.")
            key = (norm(s), rel, norm(o))
        out.add(key)
    return out


def iter_triples(x: Iterable) -> list[tuple]:
    """중복을 지우지 않고 (s, rel, o) 목록으로. 같은 쌍의 다른 관계를 모두 보존합니다."""
    out = []
    for r in x or []:
        if isinstance(r, dict):
            out.append((_pick(r, "source", "subject"), r["relation"], _pick(r, "target", "object")))
        else:
            s, rel, o = r
            out.append((s, rel, o))
    return out
