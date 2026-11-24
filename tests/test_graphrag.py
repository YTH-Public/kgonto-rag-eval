import json
import pathlib

import pytest

from kgonto_rag_eval import compare_baseline, save_baseline
from kgonto_rag_eval.graphrag import (
    break_graph, check_build, check_load, compare_result_rows, entity_link_recall, evaluate_cypher, evidence_bundle_coverage,
    stage_table, why_missed, write_word,
)

pd = pytest.importorskip("pandas")
REHEARSAL = pathlib.Path(__file__).parents[2] / "neo_rehearsal_2026-11-24.json"


# ---------------------------------------------------------------- 질의 실행 결과
GOLD = [{"company": "가", "n": 2}, {"company": "나", "n": 1}]


def ex(pred, gold=GOLD, **kw):
    return compare_result_rows(pred, gold, **kw)


def test_rows_order_and_duplicates():
    assert ex(GOLD[::-1])["ex"] == 1.0
    assert ex(GOLD[::-1], mode="ordered")["ex"] == 0.0 and "순서" in ex(GOLD[::-1], mode="ordered")["reason"]
    assert ex(GOLD + GOLD[:1])["ex"] == 0.0 and ex(GOLD + GOLD[:1], mode="set")["ex"] == 1.0


def test_rows_prf():
    r = ex([{"company": "가", "n": 2}, {"company": "다", "n": 9}])
    assert (r["matched"], r["precision"], r["recall"], r["f1"]) == (1, 0.5, 0.5, 0.5)
    assert r["missing"] == [{"company": "나", "n": 1}] and r["extra"] == [{"company": "다", "n": 9}]


def test_rows_alias_and_extra_columns():
    pred = [{"c": "가", "cnt": 2, "memo": 0}, {"c": "나", "cnt": 1, "memo": 0}]
    r = ex(pred, aliases={"c": "company", "cnt": "n"})
    assert r["ex"] == 1.0 and r["extra_columns"] == ["memo"]
    assert ex(pred)["status"] == "normalization_error" and ex(pred)["ex"] == 0.0
    assert ex([{"company": "가", "n": 2, "cnt": 2}], aliases={"cnt": "n"})["status"] == "normalization_error"


def test_rows_swapped_values_not_equal():
    assert ex([{"company": 2, "n": "가"}, {"company": 1, "n": "나"}])["ex"] == 0.0


def test_rows_types():
    cols = {"id": "string", "n": "integer"}
    gold = [{"id": "007", "n": 1}]
    assert ex([{"id": "007", "n": 1.0}], gold, columns=cols)["ex"] == 1.0
    assert ex([{"id": 7, "n": 1}], gold, columns=cols)["ex"] == 0.0                # 앞자리 0
    for bad in ("1", True, 1.5, float("nan")):
        assert ex([{"id": "007", "n": bad}], gold, columns=cols)["status"] == "normalization_error"
    assert ex([{"v": True}], [{"v": 1}])["ex"] == 0.0                              # 참과 1 은 다름
    assert ex([{"v": None}], [{"v": 0}])["ex"] == 0.0
    assert ex([{"v": [1, 2]}], [{"v": (1, 2)}])["ex"] == 1.0
    with pytest.raises(ValueError):
        ex([], [{"n": "x"}], columns={"n": "integer"})                             # 정답이 계약에 안 맞음


def test_rows_empty_vs_error_vs_zero():
    cols = {"n": "integer"}
    ok = ex([], [], columns=cols)
    assert ok["ex"] == 1.0 and ok["empty_match"] and ok["f1"] == 1.0
    assert ex([{"n": 0}], [], columns=cols)["ex"] == 0.0
    assert ex([], [{"n": 0}])["ex"] == 0.0 and ex([], [{"n": 0}])["recall"] == 0.0
    err = ex(None, [{"n": 0}], predicted_status="query_error")
    assert err["ex"] == 0.0 and err["status"] == "query_error" and not err["empty_match"]
    assert ex(None, [], columns=cols, predicted_status="timeout")["ex"] == 0.0     # 빈 정답이어도 오류는 맞음이 아님
    with pytest.raises(ValueError):
        ex(None, [{"n": 0}])
    with pytest.raises(ValueError):
        ex([], [])                                                                # 빈 정답인데 열 계약이 없음


def test_rows_snapshot_and_scalar():
    r = ex(GOLD, predicted_snapshot="2026-10", reference_snapshot="2026-09")
    assert r["ex"] is None and r["status"] == "invalid"
    assert ex([{"n": 53}], [{"n": 53}], mode="scalar")["ex"] == 1.0
    assert ex([{"n": 53}, {"n": 53}], [{"n": 53}], mode="scalar")["ex"] == 0.0
    with pytest.raises(ValueError):
        ex(GOLD, GOLD, mode="scalar")
    with pytest.raises(ValueError):
        ex(GOLD, GOLD, mode="아무거나")


def test_rows_input_not_changed():
    pred = [{"c": "가", "n": 2}]
    ex(pred, aliases={"c": "company"})
    assert pred == [{"c": "가", "n": 2}]


# ---------------------------------------------------------------- 출발점 · 근거 묶음
def test_entity_link():
    r = entity_link_recall(["법 제29조", "영 제30조"], ["법 제7조", "법 제29조", "법 제29조", "영 제30조"], k=2)
    assert r["recall"] == 0.5 and r["missing"] == ["영 제30조"] and r["ranks"] == {"법 제29조": 2} and r["all_found"] is False
    assert entity_link_recall(["001"], [1])["recall"] == 0.0
    assert entity_link_recall([], ["a"])["recall"] is None
    for bad in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            entity_link_recall(["a"], ["a"], k=bad)
    with pytest.raises(ValueError):
        entity_link_recall("법 제29조", ["법 제29조"])


def test_bundles_all_inside_any_between():
    a, b, c = ("법29", "위임", "영30"), ("영30", "위임", "고시"), ("법29", "인용", "고시")
    r = evidence_bundle_coverage([[a, b], [c]], [a])
    assert r["coverage"] == 0.5 and not r["complete"] and r["best_bundle"] == 0 and r["missing"] == [b]
    assert evidence_bundle_coverage([[a, b], [c]], [c])["complete"]
    assert evidence_bundle_coverage([[a, b]], [list(a), list(b)])["complete"]                 # 목록도 받음
    assert evidence_bundle_coverage([[a, b]], [a, ("고시", "위임", "영30")])["coverage"] == 0.5     # 방향이 다르면 다른 사실


def test_bundles_do_not_mix_pieces():
    assert evidence_bundle_coverage([["a", "b"], ["c", "d"]], ["a", "c"])["coverage"] == 0.5


def test_bundles_bad_input():
    assert evidence_bundle_coverage([], ["a"])["coverage"] is None
    with pytest.raises(ValueError):
        evidence_bundle_coverage([[]], ["a"])
    with pytest.raises(ValueError):
        evidence_bundle_coverage(["a"], ["a"])


def test_bundle_string_and_shown_values():
    with pytest.raises(ValueError):
        evidence_bundle_coverage([["a"]], "abc")
    r = ex([{"v": True, "w": {"a": [1]}}], [{"v": False, "w": {"a": [2]}}])
    assert r["missing"] == [{"v": False, "w": {"a": [2]}}] and r["extra"] == [{"v": True, "w": {"a": [1]}}]


# ---------------------------------------------------------------- 구축: check_build
NODES = [{"id": "a", "label": "회사", "name": "가"}, {"id": "b", "label": "회사", "name": "나"}, {"id": "p", "label": "품목", "name": "동"}]
RELS = [{"source": "a", "relation": "공급", "target": "b", "chunk_id": 1, "quote": "가는 나에 공급"},
        {"source": "a", "relation": "생산", "target": "p", "chunk_id": "2", "quote": "동을 생산"}]
CHUNKS = [{"id": 1, "text": "가는  나에\n공급한다."}, {"id": 2, "text": "가는 동을 생산한다."}]
SCHEMA = [("회사", "공급", "회사"), ("회사", "생산", "품목")]


def build(nodes=NODES, rels=RELS, **kw):
    table = check_build(nodes, rels, **{"schema": SCHEMA, "chunks": CHUNKS, **kw}).set_index("check")
    return {name: (row["bad"], row["status"]) for name, row in table.iterrows()}


def test_build_clean():
    assert {v for v in build().values()} == {(0, "통과")}


def test_build_finds_each_problem():
    assert build(NODES + [NODES[0]])["같은 id 의 노드가 둘 이상"] == (1, "문제 있음")
    assert build(NODES + [{"id": "c", "label": "회사", "name": "가"}])["라벨과 이름이 같은데 id 가 다른 노드"][0] == 1
    assert build(NODES + [{"id": "c", "label": "회사", "name": "당사"}])["가리키는 말이 이름으로 남은 노드"][0] == 1
    assert build(NODES + [{"id": "c", "label": "회사", "name": "다"}])["관계가 하나도 없는 노드"][0] == 1
    assert build(rels=RELS + [{**RELS[0], "target": "z"}])["끝점 노드가 없는 관계"][0] == 1
    assert build(rels=RELS + [{**RELS[0], "target": "p"}])["스키마에 없는 관계 모양"][0] == 1
    assert build(rels=RELS + [RELS[0]])["출처까지 같은 관계가 두 번"][0] == 1
    assert build(rels=RELS + [{**RELS[0], "chunk_id": 2}])["출처까지 같은 관계가 두 번"][0] == 0      # 다른 출처의 같은 사실은 중복이 아님
    assert build(rels=[{**RELS[0], "chunk_id": None}])["출처(chunk_id)가 빈 관계"][0] == 1
    assert build(rels=[{**RELS[0], "chunk_id": 99}])["출처 번호가 원문에 없는 관계"][0] == 1            # 출처가 있다고 실제 있는 것은 아님
    assert build(rels=[{**RELS[0], "quote": "가는 다에 공급"}])["근거 글(quote)이 그 원문에 없는 관계"][0] == 1


def test_build_unchecked_is_not_pass():
    out = build(schema=None, chunks=None)
    assert out["스키마에 없는 관계 모양"][1] == out["출처 번호가 원문에 없는 관계"][1] == "확인 안 함"
    assert build(rels=[])["끝점 노드가 없는 관계"][1] == "확인 안 함"
    with pytest.raises(KeyError, match="name"):
        check_build([{"id": "a", "label": "회사"}], RELS)
    with pytest.raises(ValueError, match="비었습니다"):
        check_build(NODES, [{**RELS[0], "target": None}])


def test_build_takes_dataframe():
    assert check_build(pd.DataFrame(NODES), pd.DataFrame(RELS)).shape == (10, 5)


# ---------------------------------------------------------------- 구축: why_missed
def T(*t, **kw):
    return {**dict(zip(("source", "relation", "target"), t)), **kw}


def why(pred, names=None):
    return {tuple(r[["source", "relation", "target"]]): r["reasons"] for _, r in
            why_missed(pred, [T("A", "R", "B")], pred_names=names).iterrows()}.get(("A", "R", "B"))


def test_why_missed_separates_entity_and_relation():
    assert why([T("A", "R", "B")]) is None
    assert why([], names=["A", "B"]) == ["개체는 뽑았는데 관계를 못 뽑음"]
    assert why([], names=["A"]) == ["개체를 못 뽑음: B"]
    assert why([]) == ["알 수 없음(개체 목록 없음)"]                        # 관계 표만으로는 가를 수 없음
    assert why([T("B", "R", "A")], names=["A", "B"]) == ["방향이 반대"]
    assert why([T("A", "S", "B")]) == ["관계 종류가 다름"]
    assert why([T("B", "S", "A")]) == ["방향과 관계 종류가 다름"]
    assert why([T("B", "R", "A"), T("A", "S", "B")]) == ["방향이 반대", "관계 종류가 다름"]


def test_why_missed_scope_and_normalize():
    gold = [T("(주)A", "R", "B", doc="1"), T("A", "R", "B", doc="2")]
    out = why_missed([T("A", "R", "B", doc="1")], gold, pred_names=[("2", "A")], scope=("doc",), normalize=lambda s: s.replace("(주)", ""))
    assert len(out) == 1 and out.iloc[0]["doc"] == "2" and out.iloc[0]["reasons"] == ["개체를 못 뽑음: B"]
    with pytest.raises(ValueError, match="묶음"):
        why_missed([], gold, pred_names=["A"], scope=("doc",))


# ---------------------------------------------------------------- 구축: check_load
def load(approved, loaded, **kw):
    return list(check_load(approved, loaded, **kw)["problem"])


def test_check_load():
    a, b = T("A", "R", "B", amount=1), T("A", "R", "C", amount=2)
    assert load([a, b], [b, a], values=["amount"]) == []
    assert load([a, b], [a]) == ["빠짐"] and load([a], [a, b]) == ["더 들어감"]
    assert load([a], [a, a]) == ["중복"] and load([a, a], [a]) == ["줄어듦"]
    changed = [{**a, "amount": 999}]
    assert load([a], changed) == [] and load([a], changed, values=["amount"]) == ["값이 다름"]
    assert load([a], [{**a, "amount": 1.0}], values=["amount"]) == []         # 표에서 정수가 소수가 된 것은 같은 값
    assert load([a], [{**a, "amount": float("nan")}], values=["amount"]) == ["값이 다름"]
    assert load([a], [a], again=[a, a]) == ["다시 넣으니 중복"]
    assert load([a], [], again=pd.DataFrame(columns=["source", "relation", "target"])) == ["빠짐"]
    with pytest.raises(KeyError):
        load([a], [a], values=["기간"])


# ---------------------------------------------------------------- 구축: break_graph
MANY = [T(f"a{i}", "R", f"b{i}") for i in range(10)]
MANY_NODES = [{"id": f"{k}{i}", "label": k} for k in "ab" for i in range(10)]


@pytest.mark.parametrize("kind", ["missing", "wrong_target", "reversed"])
def test_break_graph(kind):
    kw = {"nodes": MANY_NODES} if kind == "wrong_target" else {}
    before = [dict(r) for r in MANY]
    low, few = break_graph(MANY, kind, 0.2, seed=1, **kw)
    high, more = break_graph(MANY, kind, 0.5, seed=1, **kw)
    assert len(few) == 2 and len(more) == 5 and MANY == before
    assert set(few["source"]) <= set(more["source"])                           # 높은 비율이 낮은 비율을 포함
    assert break_graph(MANY, kind, 0.2, seed=1, **kw)[0].equals(low) and len(break_graph(MANY, kind, 0, **kw)[1]) == 0
    assert len(low) == (8 if kind == "missing" else 10)
    left = {tuple(r) for r in low[["source", "relation", "target"]].itertuples(index=False)} & {tuple(r.values()) for r in MANY}
    assert len(left) == 8
    if kind == "wrong_target":
        assert all(str(t).startswith("b") for t in high["target"]) and (high["source"] != high["target"]).all()
    if kind == "reversed":
        assert set(zip(few["new_source"], few["new_target"])) == set(zip(few["target"], few["source"]))


def test_break_graph_skips_useless_changes():
    both = [T("a", "R", "b"), T("b", "R", "a"), T("c", "R", "c")]
    assert len(break_graph(both, "reversed", 1.0)[1]) == 0                      # 뒤집어도 이미 있는 관계 · 자기 자신
    assert len(break_graph([T("a0", "R", "b0")], "wrong_target", 1.0, nodes=[{"id": "a0", "label": "a"}, {"id": "b0", "label": "b"}])[1]) == 0
    broken, changes = break_graph(MANY, "wrong_target", 1.0, nodes=MANY_NODES, seed=3)
    assert len(changes) == 10 and not broken.duplicated().any()
    for bad in ({"kind": "x", "rate": 0.1}, {"kind": "missing", "rate": 1.5}, {"kind": "missing", "rate": True}, {"kind": "wrong_target", "rate": 0.1}):
        with pytest.raises(ValueError):
            break_graph(MANY, **bad)


# ---------------------------------------------------------------- 검색: evaluate_cypher
def runner(table):
    def run(cypher):
        if cypher not in table:
            raise RuntimeError("문법 오류")
        return table[cypher]
    return run


def cy(made, gold="G", **kw):
    return {"question": "질문", "gold_cypher": gold, "cypher": made, **kw}


DB = {"G": [{"count(c)": 2}], "같음": [{"건수": 2}], "다름": [{"건수": 3}], "두열": [{"a": "x", "n": 2}], "빈": [], "G0": [],
      "순서": [{"n": 2}, {"n": 1}], "순서2": [{"m": 1}, {"m": 2}], "두번": [{"n": 1}, {"n": 1}, {"n": 2}]}


def ex_of(*questions, **kw):
    report = evaluate_cypher(runner(DB), list(questions), **kw)
    return [r["ex"] for r in report.rows], report


def test_cypher_results():
    scores, report = ex_of(cy("같음"), cy("다름"), cy("두열"), cy("빈"), cy("문법틀림"), cy(None), cy("MATCH (n) DELETE n"), cy("같음", gold="없는정답"), cy("빈", gold="G0"))
    assert scores == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, None, 1.0]
    status = [(r["ex__details"] or {}).get("status") for r in report.rows]
    assert status[4:7] == ["query_error", "no_query", "blocked"] and report.rows[8]["ex__details"]["empty_match"]
    assert report.rows[7]["ex__failed_component"] == "reference" and report.stats["ex"]["errors"] == 1 and report.stats["ex"]["valid"] == 8
    assert "열 수" in report.rows[2]["ex__reason"]


def test_cypher_modes_and_names():
    q = dict(gold="순서")
    assert ex_of(cy("순서2", **q))[0] == [1.0]
    assert ex_of(cy("순서2", how="ordered", **q), mode="how")[0] == [0.0]
    assert ex_of(cy("두번", how="set", **q), cy("두번", how=None, **q), mode="how")[0] == [1.0, 0.0]
    assert ex_of(cy("같음"), by="name")[0] == [0.0]                             # 이름으로 맞추면 열 이름이 달라 틀림
    assert ex_of(cy("같음", gold="순서", how="scalar"), mode="how")[0] == [None]  # 정답이 여러 행인데 scalar


def test_cypher_counterexample_and_frames():
    real, other = {"G": [{"n": 1}], "우연": [{"n": 1}]}, {"G": pd.DataFrame([{"n": 1}]), "우연": pd.DataFrame([{"n": 61}])}
    report = evaluate_cypher({"원본": runner(real), "반례": runner(other)}, [cy("우연")])
    assert report.summary == {"ex@원본": 1.0, "ex@반례": 0.0, "ex@all": 0.0}
    assert report.rows[0]["ex@all__details"]["ex"] == {"원본": 1.0, "반례": 0.0} and "meta__results" not in report.rows[0]


def test_cypher_blocked_is_not_run():
    ran = []
    evaluate_cypher(lambda c: ran.append(c) or [], [cy("MATCH (n) SET n.x = 1")])
    assert ran == ["G"]
    evaluate_cypher(lambda c: ran.append(c) or [], [cy("MATCH (n) SET n.x = 1")], read_only=False)
    assert len(ran) == 3


@pytest.mark.parametrize("text, word", [
    ("MATCH (n) RETURN n", None), ("MATCH (n) SET n.x=1", "SET"), ('MATCH (n {name: "CREATE x"}) RETURN n.set', None),
    ("MATCH (n {a: 'it\\'s SET'}) RETURN n", None), ("MATCH (n {a: 'x\\\\'}) SET n.b = 'y'", "SET"),
    ("MATCH (n:`SET`) RETURN n // DELETE", None), ("MATCH (n) /* DELETE */ RETURN n", None), ("load\n csv from 'x' as r RETURN r", "LOAD CSV"),
    ("MATCH (n) CALL { WITH n MATCH (n)--(m) RETURN m } RETURN m", None), ("MATCH (n) CALL (n) { SET n.a = 1 } RETURN 1", "SET"),
    ("CALL apoc.create.node([], {})", "CALL"), ("match(n)create(m)", "CREATE"), ("INSERT (n:A)", "INSERT"), ("MATCH (n:Set) RETURN $set", None),
])
def test_write_word(text, word):
    assert write_word(text) == word


@pytest.mark.parametrize("args, error", [({"run": 3}, TypeError), ({"run": {}}, TypeError), ({"by": "x"}, ValueError), ({"run": {"all": print}}, ValueError),
                                         ({"questions": [cy("같음", gold=" ")]}, ValueError), ({"mode": "없음"}, KeyError), ({"id": "question"}, ValueError)])
def test_cypher_bad_input(args, error):
    args = {"run": runner(DB), "questions": [cy("같음"), cy("같음")], **args}
    with pytest.raises(error):
        evaluate_cypher(args.pop("run"), args.pop("questions"), **args)


def test_cypher_baseline_needs_same_gold(tmp_path):
    save_baseline(evaluate_cypher(runner(DB), [cy("같음")]), tmp_path / "c.json")
    other = evaluate_cypher(runner(DB), [cy("같음", gold="다름")])
    assert compare_baseline(other, tmp_path / "c.json", tolerance=1, strict_meta=True)["incomparable"] == ["questions_sha"]


@pytest.mark.skipif(not REHEARSAL.exists(), reason="리허설 기록이 없는 곳")
def test_same_as_rehearsal():
    records = json.loads(REHEARSAL.read_text(encoding="utf-8"))
    for name, right in (("plain", 4), ("schema", 7)):
        rows = [r for r in records if r["set"] == name]
        saved = {r["gold_cypher"]: r["gold_rows"] for r in rows} | {r["cypher"]: r["rows"] for r in rows if r["cypher"]}
        report = evaluate_cypher(runner(saved), [{"question": r["q"], "gold_cypher": r["gold_cypher"], "cypher": r["cypher"]} for r in rows])
        assert [x["ex"] for x in report.rows] == [r["ex"] for r in rows] and sum(x["ex"] for x in report.rows) == right


# ---------------------------------------------------------------- 한 표
def test_stage_table():
    rows = [{"id": 1, "graph": True, "query": "해당 없음", "sent": 0.5, "answer": 0.0, "chars": 10},
            {"id": 2, "graph": True, "query": 1.0, "sent": True, "answer": True, "chars": 20},
            {"id": 3, "graph": None, "query": False, "sent": True, "answer": 1, "chars": 30},
            {"id": 4, "graph": True, "query": "오류", "sent": True, "answer": True, "chars": 40}]
    out = stage_table(rows, {"graph": "구축", "query": "질의", "sent": "보낸 근거", "answer": "답"}, keep=["chars"])
    assert list(out["first_break"].fillna("")) == ["보낸 근거", "", "질의", ""]
    assert list(out.loc[0, ["구축", "질의", "보낸 근거", "답"]]) == ["통과", "해당 없음", "끊김", "끊김"]
    assert "구축" in out.loc[2, "note"] and "기억" in out.loc[2, "note"] and out.loc[1, "note"] == ""
    assert list(out["chars"]) == [10, 20, 30, 40]
    assert stage_table(pd.DataFrame(rows[1:2]), ["graph", "sent"], cut=0.5)["first_break"][0] is None
    for bad in ([{**rows[0], "graph": 3}], [{**rows[0], "graph": "맞음"}], [rows[0], rows[0]]):
        with pytest.raises(ValueError):
            stage_table(bad, ["graph"])
    with pytest.raises(KeyError):
        stage_table(rows, ["없음"])
