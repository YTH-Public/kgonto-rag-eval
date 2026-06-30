"""결정적 KG 지표 스모크 테스트 (LLM 불필요).

    python tests/test_kg_retrieval_eval.py
"""
from kgonto_rag_eval.kg import (
    compare_kg,
    non_isolated_node_ratio,
    path_recall,
    path_triples,
    schema_violation_rate,
    source_traceability,
    triple_coverage,
)


def test_path_recall_text_and_triples():
    ref = ["삼성전자", "DS부문", "밸류체인 탄소감축"]
    # 텍스트(벡터 RAG) 검색 결과: 2/3 등장
    text = "삼성전자는 DS부문을 두고 있다. 자세한 활동은 보고서 참조."
    r = path_recall(ref, text)
    assert r["path_recall"] == round(2 / 3, 3), r
    assert "밸류체인 탄소감축" in r["missing"]
    # 트리플(그래프 RAG) 검색 결과: 3/3 등장
    triples = [("삼성전자", "HAS_DIVISION", "DS부문"),
               ("DS부문", "RUNS_INITIATIVE", "밸류체인 탄소감축")]
    assert path_recall(ref, triples)["path_recall"] == 1.0


def test_path_recall_avoids_prefix_substring_false_positive():
    r = path_recall(["삼성", "삼성전자"], "삼성전자는 DS부문을 두고 있다.")
    assert r["path_recall"] == 0.5, r
    assert r["found"] == ["삼성전자"], r
    assert r["missing"] == ["삼성"], r


def test_path_recall_accepts_single_structured_item():
    assert path_recall(["A", "B"], ("A", "R", "B"))["path_recall"] == 1.0
    assert path_recall(["A", "B"], {"source": "A", "relation": "R", "target": "B"})["path_recall"] == 1.0


def test_triple_coverage():
    ref = [("삼성전자", "HAS_DIVISION", "DS부문"),
            ("DS부문", "RUNS_INITIATIVE", "밸류체인 탄소감축")]
    retrieved = [("삼성전자", "HAS_DIVISION", "DS부문")]   # 1/2 덮음
    c = triple_coverage(ref, retrieved)
    assert c["triple_coverage"] == 0.5, c
    assert ("DS부문", "RUNS_INITIATIVE", "밸류체인 탄소감축") in c["missing"]


def test_path_triples_helper():
    tri = path_triples(["A", "B", "C"], ["R1", "R2"])
    assert tri == [("A", "R1", "B"), ("B", "R2", "C")], tri


def test_source_traceability():
    class Doc:
        def __init__(self, text, meta): self.page_content = text; self.metadata = meta
    items = [Doc("문장1", {"page": 10}), Doc("문장2", {}), {"source": "rep.pdf"}]
    t = source_traceability(items)
    assert t["traced"] == 2 and t["total"] == 3, t
    assert t["traceability_rate"] == round(2 / 3, 3)


def test_schema_violation_rate():
    ents = [{"name": "삼성전자", "type": "Company"},
            {"name": "이상한것", "type": "BadType"}]   # 타입 위반 1
    rels = [("삼성전자", "HAS_DIVISION", "DS부문")]      # dangling_target(DS부문 없음) 1
    rep = schema_violation_rate(ents, rels,
                                allowed_types={"Company", "BusinessDivision"},
                                allowed_relations={"HAS_DIVISION"})
    # 위반: bad_entity_type 1 + dangling_target 1 = 2 / (2 ent + 1 rel = 3)
    assert rep["violations"] == 2 and rep["checked"] == 3, rep
    assert rep["schema_violation_rate"] == round(2 / 3, 3)


def test_schema_violation_rate_counts_bad_relation_once():
    rep = schema_violation_rate(
        [],
        [("없는출발", "BAD_REL", "없는도착")],
        allowed_types={"Company"},
        allowed_relations={"HAS_DIVISION"},
    )
    assert rep["violations"] == 1 and rep["checked"] == 1, rep
    assert rep["schema_violation_rate"] == 1.0, rep
    assert len(rep["issues"]) == 3, rep


def test_non_isolated_node_ratio():
    ents = [{"name": "A"}, {"name": "B"}, {"name": "C"}]   # C 는 고립
    rels = [("A", "R", "B")]
    rep = non_isolated_node_ratio(ents, rels)
    assert rep["non_isolated"] == 2 and rep["total_nodes"] == 3, rep
    assert rep["non_isolated_node_ratio"] == round(2 / 3, 3)
    assert rep["isolated"] == ["C"]


def test_compare_kg_still_triple_prf1():
    pred = [("A", "R", "B"), ("A", "R", "X")]
    ref = [("A", "R", "B"), ("A", "R", "C")]
    rep = compare_kg(pred, ref)
    assert rep["matched"] == 1 and rep["precision"] == 0.5 and rep["recall"] == 0.5, rep


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
    print("\n모든 결정적 KG 지표 테스트 통과")
