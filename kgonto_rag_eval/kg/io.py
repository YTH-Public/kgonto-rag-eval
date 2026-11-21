"""KG 입력 추상화. entities/relations JSON, (s,rel,o) 트리플 모두 수용해 networkx 로."""
from __future__ import annotations

import json
import pathlib
from typing import Any, Iterable, Optional

from ._triples import iter_triples


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
    """지식그래프를 networkx MultiDiGraph 로 만든다.

    사용법 1 (JSON 포맷): load_kg(entities="entities.json", relations="relations.json")
      - entities: [{"name","type", ...속성}], relations: [{"source","relation","target"}]
        (subject / object 칸 이름도 받습니다)
    사용법 2 (트리플): load_kg(triples=[("삼성전자","HAS_DIVISION","DS부문"), ...])

    같은 두 노드 사이에 관계가 여럿이어도 모두 남깁니다(엣지 key = 관계 이름).
    노드 속성(type 등)은 그대로 보존, 엣지에는 relation 속성을 단다.
    """
    import networkx as nx  # 선택 의존성

    g = nx.MultiDiGraph()
    if triples is None:
        for e in _load_json(entities):
            g.add_node(e["name"], **{k: v for k, v in e.items() if k != "name"})
        triples = iter_triples(_load_json(relations))
    for s, rel, o in iter_triples(triples):
        g.add_edge(s, o, key=rel, relation=rel)
    return g
