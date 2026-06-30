"""KG 입력 추상화. entities/relations JSON, (s,rel,o) 트리플 모두 수용해 networkx 로."""
from __future__ import annotations

import json
import pathlib
from typing import Any, Iterable, Optional


def _load_json(x: Any) -> list[dict]:
    if x is None:
        return []
    if isinstance(x, (str, pathlib.Path)):
        return json.loads(pathlib.Path(x).read_text(encoding="utf-8"))
    return list(x)


def load_kg(
    entities: Any = None,
    relations: Any = None,
    triples: Optional[Iterable[tuple]] = None,
):
    """지식그래프를 networkx DiGraph 로 만든다.

    사용법 1 (JSON 포맷): load_kg(entities="entities.json", relations="relations.json")
      - entities: [{"name","type", ...속성}], relations: [{"source","relation","target"}]
      - 파일경로 또는 list[dict] 모두 가능.
    사용법 2 (트리플): load_kg(triples=[("삼성전자","HAS_DIVISION","DS부문"), ...])

    노드 속성(type 등)은 그대로 보존, 엣지에는 relation 속성을 단다.
    """
    import networkx as nx  # 선택 의존성

    g = nx.DiGraph()
    if triples is not None:
        for s, rel, o in triples:
            g.add_node(s)
            g.add_node(o)
            g.add_edge(s, o, relation=rel)
        return g

    for e in _load_json(entities):
        name = e["name"]
        g.add_node(name, **{k: v for k, v in e.items() if k != "name"})
    for r in _load_json(relations):
        g.add_edge(r["source"], r["target"], relation=r["relation"])
    return g
