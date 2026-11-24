import csv
import json
import math
import pathlib

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from kgonto_rag_eval import compare_baseline, evaluate, evaluate_search, metric, run_rag, save_baseline
from kgonto_rag_eval.metrics.similarity import AnswerSimilarity

NB = pathlib.Path(__file__).parents[3] / "notebooks"
RANKS = NB / "03_GraphRAG고도화/01_AdvancedRAG검색/day50_질문바꾸기와-다단계검색/50-2_질의별검색순위.jsonl"
QUESTIONS = NB / "data/50-1_일상어질문.csv"


# ---------------------------------------------------------------- evaluate_search
QS = [{"no": "q1", "question": "하나", "gold": 1}, {"no": "q2", "question": "둘", "gold": 2}, {"no": "q3", "question": "셋", "gold": 9}]


def test_hit_and_mrr_by_hand():
    report = evaluate_search(lambda q: {"하나": [1, 5, 6], "둘": [7, "2", 8], "셋": [4, 5, 6, 9]}[q], QS, id="no", k=3)
    assert report.summary == {"hit@3": 0.667, "mrr@3": 0.5}            # 1위 · 2위 · 못 찾음
    assert [r["mrr@3"] for r in report.rows] == [1.0, 0.5, 0.0]
    assert [r["sample_id"] for r in report.rows] == ["q1", "q2", "q3"]
    assert report.rows[2]["meta__retrieved_ids"] == ["4", "5", "6"]


def test_store_documents_and_k():
    class Store:
        def similarity_search(self, query, k):
            return [Document(page_content=f"글{n}", metadata={"page": n}) for n in range(1, k + 1)]
    report = evaluate_search(Store(), QS, key="page", k=2)
    assert [r["hit@2"] for r in report.rows] == [1.0, 1.0, 0.0] and report.rows[0]["retrieved_contexts"] == ["글1", "글2"]
    with pytest.raises(ValueError, match="key"):
        evaluate_search(Store(), QS)
    with pytest.raises(ValueError, match="출처 '쪽'"):
        evaluate_search(Store(), QS, key="쪽")
    seen = []
    evaluate_search(lambda q, k=10: seen.append(k) or [], QS, k=4)
    assert seen == [4, 4, 4]


def test_many_gold_duplicates_and_empty():
    def search(q):
        if q == "오류":
            raise ConnectionError("끊김")
        return {"여럿": [5, 5, 1], "빈결과": [], "정답없음": [1]}[q]
    rows = [{"question": "여럿", "gold": [1, 2]}, {"question": "빈결과", "gold": [1]}, {"question": "정답없음", "gold": []},
            {"question": "오류", "gold": 1}]
    report = evaluate_search(search, rows, k=3)
    assert [r["mrr@3"] for r in report.rows] == [1 / 3, 0.0, None, None]   # 같은 문서가 두 번 나와도 3위는 3위
    assert [r["recall@3"] for r in report.rows] == [0.5, 0.0, None, None]
    assert [r["hit@3__status"] for r in report.rows] == ["ok", "ok", "not_applicable", "error"]
    assert report.stats["hit@3"]["errors"] == 1 and report.stats["hit@3"]["valid"] == 2
    assert report.rows[3]["hit@3__failed_component"] == "retriever"


@pytest.mark.parametrize("change, error", [
    ({"k": 0}, ValueError), ({"k": True}, ValueError), ({"gold": "정답"}, KeyError), ({"id": "question"}, ValueError),
    ({"search": 3}, TypeError), ({"search": lambda q: "글"}, TypeError), ({"search": lambda q: None}, TypeError), ({"search": lambda q: [None]}, ValueError),
    ({"questions": [{"question": " ", "gold": 1}]}, ValueError), ({"questions": [{"question": "a", "gold": 1.0}]}, ValueError),
])
def test_search_bad_input(change, error):
    args = {"search": lambda q: [1], "questions": [QS[0], QS[0]], **change}
    with pytest.raises(error):
        evaluate_search(args.pop("search"), args.pop("questions"), **args)


def test_search_baseline(tmp_path):
    def run(gold=1, k=3):
        return evaluate_search(lambda q: [1, 2], [{"question": "a", "gold": gold}], k=k)
    save_baseline(run(), tmp_path / "s.json")
    strict = lambda r: compare_baseline(r, tmp_path / "s.json", metrics=["hit@3"], strict_meta=True)
    assert strict(run())["ok"]
    assert strict(run(gold=2))["incomparable"] == ["questions_sha"]       # 정답이 바뀌면 비교하지 않음
    assert not strict(run(k=2))["ok"]


@pytest.mark.skipif(not (RANKS.exists() and QUESTIONS.exists()), reason="수업 자료가 없는 곳")
def test_same_as_count_hits_in_50_1():
    pd = pytest.importorskip("pandas")
    ranks = {}
    for line in RANKS.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["strategy"] == "original":
            ranks[r["retriever"], r["query"]] = r["pages"]
    questions = pd.read_csv(QUESTIONS)

    def rrf(rankings, c=60):
        scores = {}
        for ranking in rankings:
            for rank, page in enumerate(ranking, start=1):
                scores[page] = scores.get(page, 0) + 1 / (c + rank)
        return sorted(scores, key=scores.get, reverse=True)
    searches = {"dense": lambda q: ranks["dense", q], "bm25": lambda q: ranks["bm25", q],
                "rrf": lambda q: rrf([ranks["dense", q], ranks["bm25", q]])}
    hits = {}
    for name, search in searches.items():
        report = evaluate_search(search, questions, question="일상어", gold="페이지", id="질문ID")
        for row, (_, q) in zip(report.rows, questions.iterrows()):       # 문항마다 옛 계산과 같은지
            top = search(q["일상어"])[:3]
            assert row["hit@3"] == float(q["페이지"] in top)
            assert row["mrr@3"] == (1 / (top.index(q["페이지"]) + 1) if q["페이지"] in top else 0.0)
        hits[name] = round(sum(r["hit@3"] for r in report.rows))
    assert hits == {"dense": 23, "bm25": 25, "rrf": 30}


# ---------------------------------------------------------------- run_rag
class Graph:
    def __init__(self, messages, checkpointer=None):
        self.messages, self.checkpointer, self.calls = messages, checkpointer, []

    def invoke(self, state, config):
        self.calls.append((state, config))
        return {"messages": [HumanMessage(state["messages"][0][1]), *self.messages]}


def call(name="search"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": {}, "id": "1"}])


def tool(text, name="search", **kw):
    return ToolMessage(content=text, name=name, tool_call_id="1", **kw)


ONE = [{"q": "질문", "정답": "기준", "no": "a", "메모": "m"}]
ARGS = dict(question="q", reference="정답", id="no", keep=["메모"])


def test_graph_tool_contexts():
    graph = Graph([call(), tool("근거 없음"), call(), tool("근거 1"), tool("오류", status="error"), tool("37도", "weather"),
                   HumanMessage("고쳐 주세요"), AIMessage(content=[{"type": "text", "text": "답"}])])
    s = run_rag(graph, ONE, contexts="tool", tools=["search"], skip=["근거 없음"], **ARGS).samples[0]
    assert (s.sample_id, s.response, s.reference, s.retrieved_contexts) == ("a", "답", "기준", ["근거 1"])
    assert s.metadata == {"메모": "m", "tools_used": ["search", "search", "search", "weather"]}
    assert len(graph.calls) == 1 and "기준" not in str(graph.calls)      # 한 번만 실행, 정답은 넘기지 않음
    plain = run_rag(Graph([AIMessage("답")]), ONE, contexts="tool", **ARGS).samples[0]
    assert plain.retrieved_contexts == [] and plain.metadata["tools_used"] == []


def test_thread_per_question_and_config():
    graph = Graph([AIMessage("답")], checkpointer=True)
    run_rag(graph, [{"q": "1"}, {"q": "2"}], question="q", contexts=None, config={"tags": ["t"], "configurable": {"user": "u"}})
    configs = [c for _, c in graph.calls]
    assert configs[0]["configurable"]["thread_id"] != configs[1]["configurable"]["thread_id"]
    assert all(c["tags"] == ["t"] and c["configurable"]["user"] == "u" for c in configs)
    no_memory = Graph([AIMessage("답")])
    run_rag(no_memory, [{"q": "1"}], question="q", contexts=None)
    assert no_memory.calls[0][1] == {}


def test_functions_and_state_key():
    assert run_rag(lambda q: q + "!", ONE, contexts=None, **ARGS).samples[0].response == "질문!"
    s = run_rag(lambda q: {"answer": "답", "evidence": [Document(page_content="글"), "글2"]}, ONE, contexts="evidence", **ARGS).samples[0]
    assert s.retrieved_contexts == ["글", "글2"] and s.metadata == {"메모": "m"}


@pytest.mark.parametrize("rag, args", [
    (Graph([call()]), {"contexts": "tool"}),                              # 답 없이 끝남
    (Graph([call(), tool("x")]), {"contexts": "tool"}),
    (lambda q: {"__interrupt__": 1}, {"contexts": None}),
    (lambda q: "답", {"contexts": "evidence"}),
    (lambda q: "답", {"contexts": "tool"}),
    (lambda q: {"answer": "답"}, {"contexts": "evidence"}),
    (lambda q: {"answer": "답", "evidence": "글"}, {"contexts": "evidence"}),
    (lambda q: {"답": "x"}, {"contexts": None}),
    (lambda q: None, {"contexts": None}),
    (lambda q: "답", {"contexts": None, "config": {}}),
])
def test_run_rag_bad_result(rag, args):
    with pytest.raises(ValueError):
        run_rag(rag, ONE, **{**ARGS, **args})


def test_run_rag_stops_with_question_id():
    def rag(q):
        raise KeyError("x")
    with pytest.raises(RuntimeError, match="id=a"):
        run_rag(rag, ONE, contexts=None, **ARGS)
    ran = []
    with pytest.raises(ValueError):                                       # 질문 표가 잘못이면 실행하기 전에 멈춤
        run_rag(lambda q: ran.append(q) or "답", [{"q": "1"}, {"q": None}], question="q", contexts=None)
    assert ran == []


def test_real_langgraph():
    pytest.importorskip("langgraph")
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import START, MessagesState, StateGraph

    class State(MessagesState):
        evidence: list

    def node(state):
        return {"messages": [AIMessage(f"{len(state['messages'])}번째 답")], "evidence": ["근거"]}
    builder = StateGraph(State)
    builder.add_node("a", node)
    builder.add_edge(START, "a")
    graph = builder.compile(checkpointer=InMemorySaver())
    out = run_rag(graph, [{"question": "1"}, {"question": "2"}], contexts="evidence").samples
    assert [s.response for s in out] == ["1번째 답", "1번째 답"]            # 대화방이 섞이지 않음
    assert [s.retrieved_contexts for s in out] == [["근거"], ["근거"]]


# ---------------------------------------------------------------- AnswerSimilarity
class Emb:
    def __init__(self, table, model="가짜"):
        self.table, self.model = table, model

    def embed_documents(self, texts):
        return [self.table[t] for t in texts]


@pytest.mark.parametrize("a, b, score", [([1, 0], [2, 0], 1.0), ([1, 0], [0, 3], 0.0), ([1, 0], [-1, 0], 0.0), ([1, 1], [1, 0], 1 / math.sqrt(2)),
                                         ([0, 0], [1, 0], None), ([1, 0], [1, 0, 0], None), ([float("nan"), 1], [1, 0], None)])
def test_similarity(a, b, score):
    pytest.importorskip("sklearn")
    row = {"user_input": "q", "response": "답", "reference": "기준"}
    out = evaluate([row], [AnswerSimilarity(embeddings=Emb({"답": a, "기준": b}))]).rows[0]
    if score is None:
        assert out["answer_similarity__status"] == "error" and out["answer_similarity"] is None
    else:
        assert abs(out["answer_similarity"] - score) < 1e-9 and out["answer_similarity__verdict"] is None


def test_similarity_empty_and_fingerprint(tmp_path):
    emb = Emb({"답": [1, 0], "기준": [1, 0]})
    rows = [{"user_input": "q", "response": "", "reference": "기준"}, {"user_input": "q", "response": "답", "reference": None}]
    out = evaluate(rows, [AnswerSimilarity(embeddings=emb)]).rows
    assert [r["answer_similarity"] for r in out] == [0.0, None] and out[1]["answer_similarity__status"] == "not_applicable"
    pytest.importorskip("sklearn")
    row = [{"user_input": "q", "response": "답", "reference": "기준"}]
    save_baseline(evaluate(row, [AnswerSimilarity(embeddings=emb)]), tmp_path / "e.json")
    other = evaluate(row, [AnswerSimilarity(embeddings=Emb(emb.table, "다른 모델"))])
    assert compare_baseline(other, tmp_path / "e.json", strict_meta=True)["incomparable"] == ["metrics.answer_similarity"]


def test_plain_import_does_not_need_sklearn():
    import subprocess
    import sys
    code = "import sys; sys.modules['sklearn']=None; import kgonto_rag_eval as k; k.evaluate_search, k.run_rag; from kgonto_rag_eval.metrics import AnswerSimilarity"
    assert subprocess.run([sys.executable, "-c", code], cwd=pathlib.Path(__file__).parents[1]).returncode == 0
