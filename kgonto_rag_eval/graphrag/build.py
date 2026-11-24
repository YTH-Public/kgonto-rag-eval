"""그래프 구축 평가 (모델 · DB 없음). 표(또는 사전 목록)를 넣고 표를 받습니다.

열 이름은 이렇게 맞춥니다(Neo4j 에서는 RETURN ... AS 로 이름을 붙여 꺼냅니다).
- 노드 표   : id, label, name
- 관계 표   : source, relation, target (source · target 은 노드 id). 있으면 chunk_id(출처), quote(근거 글)
- 원문 표   : id, text
"""
from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from typing import Iterable, Optional, Sequence

from ..citation import quote_in_source
from ..inputs import _missing, as_records
from .search import _freeze

_TRIPLE = ("source", "relation", "target")


def _rows(table, columns: Sequence[str], what: str, empty_ok: bool = False) -> list[dict]:
    """표 -> 사전 목록. 필요한 열이 없거나 비었으면 오류"""
    if empty_ok and table is not None and len(table) == 0:
        return []
    rows = as_records(table)
    for i, row in enumerate(rows):
        lost = [c for c in columns if c not in row]
        if lost:
            raise KeyError(f"{what} {i}번째 줄에 {lost} 열이 없습니다. 있는 열: {list(row)}")
        blank = [c for c in columns if _missing(row[c]) or not str(row[c]).strip()]
        if blank:
            raise ValueError(f"{what} {i}번째 줄의 {blank} 칸이 비었습니다: {row}"[:300])
    return rows


def _table(rows: list[dict], columns: Sequence[str]):
    import pandas as pd

    return pd.DataFrame(rows, columns=list(columns))


def check_build(nodes, relations, *, schema: Optional[Iterable[tuple]] = None, chunks=None,
                placeholders: Sequence[str] = ("당사", "회사", "동사", "자사")):
    """적재한 그래프의 모양과 출처를 검사합니다.

    schema : 허용하는 (출발 라벨, 관계, 도착 라벨) 목록
    chunks : 원문 표. 주면 출처 번호가 실제로 있는지, 근거 글(quote)이 그 원문에 있는지 봅니다.
    반환   : 검사 · 문제 수(bad) · 전체 수(total) · 상태(status: 통과 · 문제 있음 · 확인 안 함) · 예(examples, 3개까지)

    모양이 맞아도 내용이 사실이라는 뜻은 아닙니다. 근거 글이 원문에 있어도 그 글이 관계를 **뒷받침하는지**는
    사람이나 판정 모델이 따로 봐야 합니다.
    """
    nodes = _rows(nodes, ("id", "label", "name"), "노드 표")
    relations = _rows(relations, _TRIPLE, "관계 표", empty_ok=True)
    label = {n["id"]: n["label"] for n in nodes}
    out: list[dict] = []

    def add(name: str, bad: list, total: int, checked: bool = True):
        status = "확인 안 함" if not checked or not total else "문제 있음" if bad else "통과"
        out.append({"check": name, "bad": len(bad) if checked else None, "total": total, "status": status, "examples": bad[:3]})

    triples = [tuple(r[c] for c in _TRIPLE) for r in relations]
    ids = Counter(n["id"] for n in nodes)
    add("같은 id 의 노드가 둘 이상", [i for i, n in ids.items() if n > 1], len(ids))
    names = defaultdict(set)
    for n in nodes:
        names[n["label"], n["name"]].add(n["id"])
    add("라벨과 이름이 같은데 id 가 다른 노드", [k for k, v in names.items() if len(v) > 1], len(names))
    add("가리키는 말이 이름으로 남은 노드", [n["id"] for n in nodes if str(n["name"]).strip() in placeholders], len(nodes))
    linked = {t[0] for t in triples} | {t[2] for t in triples}
    add("관계가 하나도 없는 노드", [i for i in ids if i not in linked], len(ids))
    add("끝점 노드가 없는 관계", [t for t in triples if t[0] not in label or t[2] not in label], len(triples))
    allowed = None if schema is None else {tuple(p) for p in schema}
    wrong = [] if allowed is None else [t for t in triples if t[0] in label and t[2] in label
                                        and (label[t[0]], t[1], label[t[2]]) not in allowed]
    add("스키마에 없는 관계 모양", wrong, len(triples), allowed is not None)
    same = Counter((t, str(r.get("chunk_id"))) for t, r in zip(triples, relations))
    add("출처까지 같은 관계가 두 번", [k[0] for k, n in same.items() if n > 1], len(same))
    sources = [(t, r.get("chunk_id"), r.get("quote")) for t, r in zip(triples, relations)]
    has = [x for x in sources if not _missing(x[1]) and str(x[1]).strip()]
    add("출처(chunk_id)가 빈 관계", [x[0] for x in sources if x not in has], len(sources))
    text = None if chunks is None else {str(c["id"]): str(c["text"]) for c in _rows(chunks, ("id", "text"), "원문 표")}
    add("출처 번호가 원문에 없는 관계", [x[0] for x in has if text is not None and str(x[1]) not in text], len(has), text is not None)
    quoted = [x for x in has if text is not None and str(x[1]) in text and not _missing(x[2]) and str(x[2]).strip()]
    add("근거 글(quote)이 그 원문에 없는 관계", [x[0] for x in quoted if not quote_in_source(str(x[2]), text[str(x[1])])],
        len(quoted), bool(quoted))
    return _table(out, ("check", "bad", "total", "status", "examples"))


def why_missed(pred, gold, *, pred_names: Optional[Iterable] = None, scope: Sequence[str] = (), normalize=None):
    """정답에 있는데 못 뽑은 관계마다 **가능한 까닭**을 적습니다. 까닭이 둘 이상이면 모두 적습니다.

    pred · gold : 관계 표(source · relation · target, scope 열이 있으면 같은 문서 안에서만 비교)
    pred_names  : 같은 실행에서 뽑은 개체 이름 목록. scope 를 쓰면 (문서, 이름) 묶음의 목록.
                  관계에 안 쓰인 개체까지 모두 넣어야 '개체를 못 뽑음'과 '관계를 못 뽑음'을 가를 수 있습니다.
    normalize   : 이름을 맞출 때 쓸 함수 (양쪽에 똑같이 적용)
    반환        : scope 열 · source · relation · target · reasons(까닭 목록)

    까닭은 관계 표에서 보이는 것만으로 좁힌 후보입니다. 개체 목록이 없으면 가를 수 없는 줄은 '알 수 없음'입니다.
    """
    norm = normalize or (lambda s: s)
    columns = (*scope, *_TRIPLE)

    def keys(table, what):
        return [(tuple(r[c] for c in scope), norm(r["source"]), r["relation"], norm(r["target"]))
                for r in _rows(table, columns, what, empty_ok=True)]
    found = set(keys(pred, "추출 관계 표"))
    by_doc = defaultdict(list)
    for d, s, rel, o in found:
        by_doc[d].append((s, rel, o))
    names = None
    if pred_names is not None:
        names = set()
        for one in pred_names:
            if scope and (not isinstance(one, (tuple, list)) or len(one) != len(scope) + 1):
                raise ValueError(f"scope 를 쓰면 pred_names 에는 (문서, 이름) 묶음을 넣습니다: {one!r}")
            names.add((tuple(one[:-1]), norm(one[-1])) if scope else ((), norm(one)))
    out = []
    for d, s, rel, o in dict.fromkeys(keys(gold, "정답 관계 표")):
        if (d, s, rel, o) in found:
            continue
        near, reasons = by_doc[d], []
        if (o, rel, s) in near:
            reasons.append("방향이 반대")
        if any(t[0] == s and t[2] == o for t in near):
            reasons.append("관계 종류가 다름")
        if any(t[0] == o and t[2] == s and t[1] != rel for t in near):
            reasons.append("방향과 관계 종류가 다름")
        if names is not None:
            lost = [n for n in dict.fromkeys((s, o)) if (d, n) not in names]
            if lost:
                reasons.append("개체를 못 뽑음: " + ", ".join(map(str, lost)))
            elif not reasons:
                reasons.append("개체는 뽑았는데 관계를 못 뽑음")
        elif not reasons:
            reasons.append("알 수 없음(개체 목록 없음)")
        out.append({**dict(zip(scope, d)), "source": s, "relation": rel, "target": o, "reasons": reasons})
    return _table(out, (*columns, "reasons"))


def check_load(approved, loaded, *, key: Sequence[str] = _TRIPLE, values: Sequence[str] = (), again=None):
    """넣기로 한 목록(approved)과 DB 에서 꺼낸 표(loaded)를 대조합니다. 문제가 없으면 빈 표입니다.

    key    : 한 줄을 알아보는 열 (여러 문서를 넣었으면 문서 번호 열도 함께)
    values : 값까지 같아야 하는 열 (금액 · 기간 등)
    again  : 같은 자료를 한 번 더 넣은 뒤 꺼낸 표. 주면 두 번 넣어도 그대로인지도 봅니다.
    반환   : problem(빠짐 · 더 들어감 · 중복 · 줄어듦 · 값이 다름 / 다시 넣으니 ...) · key 열 · expected · found

    loaded 는 approved 와 같은 범위만 꺼내야 합니다(다른 문서가 섞이면 '더 들어감'이 됩니다).
    """
    key, values = tuple(key), tuple(values)

    def group(table, what):
        out = defaultdict(list)
        for row in _rows(table, key, what, empty_ok=True):
            lost = [c for c in values if c not in row]
            if lost:
                raise KeyError(f"{what}에 {lost} 열이 없습니다. 있는 열: {list(row)}")
            out[tuple(row[c] for c in key)].append(tuple(None if _missing(row[c]) else _freeze(row[c]) for c in values))
        return out

    def compare(expected, found, before: str):
        for k in sorted(set(expected) | set(found), key=str):
            a, b = expected.get(k, []), found.get(k, [])
            if Counter(a) == Counter(b):
                continue
            problem = ("빠짐" if not b else "더 들어감" if not a else "중복" if len(b) > len(a)
                       else "줄어듦" if len(b) < len(a) else "값이 다름")
            out.append({"problem": before + problem, **dict(zip(key, k)),
                        "expected": a if values else len(a), "found": b if values else len(b)})
    out: list[dict] = []
    first = group(loaded, "적재된 표")
    compare(group(approved, "승인 목록"), first, "")
    if again is not None:
        compare(first, group(again, "다시 넣은 표"), "다시 넣으니 ")
    return _table(out, ("problem", *key, "expected", "found"))


def break_graph(relations, kind: str, rate: float, *, seed: int = 0, nodes=None):
    """그래프에 일부러 오류를 넣습니다(실험용). 원본은 고치지 않습니다.

    kind  : "missing"(관계를 지움) · "wrong_target"(도착점을 같은 라벨의 다른 노드로, nodes 필요) · "reversed"(방향을 뒤집음)
    rate  : 함께 바꿀 수 있는 관계 가운데 바꿀 비율 (0 ~ 1, 개수는 내림). 전체 관계 수에 대한 비율은 len(바뀐 것) / len(relations)
    seed  : 같은 값이면 같은 관계가 바뀝니다. 비율을 올리면 낮은 비율에서 바꾼 것을 포함합니다.
    반환  : (바꾼 관계 표, 바뀐 것의 표(source · relation · target · new_source · new_target))

    실제로 바뀐 수는 len(바뀐 것의 표) 입니다. 자기 자신을 가리키거나 이미 있는 관계가 되는 변경은 하지 않습니다.
    노드 표는 바꾸지 않으므로, 관계가 없어진 노드도 그대로 둡니다. 정답은 망가뜨리기 전 그래프로 정해 두세요.
    """
    if kind not in ("missing", "wrong_target", "reversed"):
        raise ValueError(f"kind 는 missing · wrong_target · reversed 가운데 하나입니다: {kind!r}")
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not 0 <= rate <= 1:
        raise ValueError(f"rate 는 0 ~ 1 입니다: {rate!r}")
    rows = [dict(r) for r in _rows(relations, _TRIPLE, "관계 표")]
    triple = lambda r: tuple(r[c] for c in _TRIPLE)
    label, same_label = {}, defaultdict(list)
    if kind == "wrong_target":
        if nodes is None:
            raise ValueError("wrong_target 은 같은 라벨의 노드를 골라야 하므로 nodes 가 필요합니다.")
        for n in sorted(_rows(nodes, ("id", "label"), "노드 표"), key=lambda n: str(n["id"])):
            label[n["id"]] = n["label"]
            same_label[n["label"]].append(n["id"])
    order = sorted(range(len(rows)), key=lambda i: (tuple(map(str, triple(rows[i]))), i))
    rng = random.Random(seed)
    rng.shuffle(order)
    exist = {triple(r) for r in rows}                       # 원래 있던 것과 새로 만든 것 (겹치지 않게)

    def new_ends(i: int):
        """바꾼 뒤의 (출발, 도착) - 바꿀 수 없으면 None"""
        s, rel, o = triple(rows[i])
        if kind == "missing":
            return (None, None)
        if kind == "reversed":
            return (o, s) if s != o and (o, rel, s) not in exist else None
        options = [x for x in same_label[label.get(o)] if x not in (s, o) and (s, rel, x) not in exist]
        return (s, random.Random(f"{seed}-{i}").choice(options)) if options else None

    plan = []                                               # 앞에서 만든 관계와 겹치지 않게, 함께 바꿀 수 있는 것 전부
    for i in order:
        ends = new_ends(i)
        if ends is not None:
            plan.append((i, ends))
            exist.add((ends[0], rows[i]["relation"], ends[1]))
    changes, gone = [], set()
    for i, ends in plan[:math.floor(rate * len(plan) + 1e-9)]:
        s, rel, o = triple(rows[i])
        changes.append({"source": s, "relation": rel, "target": o, "new_source": ends[0], "new_target": ends[1]})
        if kind == "missing":
            gone.add(i)
        else:
            rows[i].update(source=ends[0], target=ends[1])
    broken = [r for i, r in enumerate(rows) if i not in gone]
    return (_table(broken, dict.fromkeys(c for r in rows for c in r)),
            _table(changes, (*_TRIPLE, "new_source", "new_target")))
