"""v0.2 에서 고친 동작의 오프라인 테스트 (API 호출 없음, 가짜 판정자 사용).

    python -m pytest -q tests/test_v02_offline.py
"""
import pytest

from kgonto_rag_eval import EvalSample, EvaluationDataset, evaluate
from kgonto_rag_eval.baseline import compare_baseline, save_baseline
from kgonto_rag_eval.citation import citation_support, quote_in_source, resolve_citations, split_cited_sentences
from kgonto_rag_eval.judge import pairwise_judge
from kgonto_rag_eval.kg import (
    TestItem, compare_kg, enumerate_paths, leakage_report, load_kg, path_entity_recall,
    schema_violation_rate, source_traceability, to_eval_dataset, triple_coverage,
)
from kgonto_rag_eval.metrics import ContextPrecision, ContextRelevance, Faithfulness


TestItem.__test__ = False   # pytest 가 테스트 클래스로 모으지 않게


class FakeLLM:
    """with_structured_output(schema).invoke(...) 가 정해 둔 값을 차례로 돌려주는 가짜 판정자."""

    def __init__(self, outputs):
        self.outputs = list(outputs)

    def with_structured_output(self, schema):
        llm = self

        class _Runner:
            def invoke(self, messages):
                value = llm.outputs.pop(0)
                if callable(value):
                    value = value(messages)
                return schema(**value) if isinstance(value, dict) else value
        return _Runner()


def _sample(**kw):
    base = dict(user_input="q", retrieved_contexts=["c1", "c2"], response="r", reference="ref")
    base.update(kw)
    return EvalSample(**base)


# B7: 문자열 하나를 문맥 목록으로 주면 글자로 쪼개지 않고 막는다
def test_contexts_string_rejected():
    with pytest.raises(ValueError):
        EvalSample(user_input="q", retrieved_contexts="ABC")


# B6: judge 개수가 앞뒤가 안 맞으면 점수가 아니라 오류
def test_faithfulness_bad_counts_is_error():
    res = Faithfulness().score(_sample(), FakeLLM([{"total_claims": 1, "supported_claims": 3, "reason": "x"}]))
    assert res.score is None and res.error and res.failed_component == "judge"


def test_faithfulness_negative_counts_rejected_by_schema():
    report = evaluate([_sample().model_dump()], [Faithfulness()],
                      llm=FakeLLM([{"total_claims": -2, "supported_claims": -1, "reason": "x"}]))
    row = report.rows[0]
    assert row["faithfulness"] is None and row["faithfulness__error"]
    assert report.stats["faithfulness"] == {"mean": None, "valid": 0, "missing": 1, "errors": 1, "not_applicable": 0}


# B5: context relevance 번호 중복 · 빠짐은 오류
def test_context_relevance_duplicate_index_is_error():
    out = {"items": [{"index": 1, "label": "relevant"}, {"index": 1, "label": "relevant"}], "reason": "x"}
    res = ContextRelevance().score(_sample(), FakeLLM([out]))
    assert res.score is None and res.error


def test_context_precision_length_mismatch_is_error():
    res = ContextPrecision().score(_sample(), FakeLLM([{"relevant_flags": [True], "reason": "x"}]))
    assert res.score is None and res.error


def test_context_precision_rank_score():
    res = ContextPrecision().score(_sample(retrieved_contexts=["a", "b", "c"]),
                                   FakeLLM([{"relevant_flags": [True, False, True], "reason": "x"}]))
    assert res.score == pytest.approx((1 / 1 + 2 / 3) / 2)   # 반올림하지 않은 원점수


def test_evaluate_keeps_ids_and_details():
    ds = EvaluationDataset.from_list([_sample(sample_id="P001", metadata={"split": "튜닝"}).model_dump()])
    report = evaluate(ds, [Faithfulness()], llm=FakeLLM([{"total_claims": 2, "supported_claims": 1, "reason": "x"}]))
    row = report.rows[0]
    assert row["sample_id"] == "P001" and row["meta__split"] == "튜닝"
    assert row["faithfulness"] == 0.5 and row["faithfulness__details"]["total"] == 2


# B2: 관계 dict 의 source 는 출처가 아니다
def test_traceability_relation_source_is_subject():
    assert source_traceability([{"source": "A", "relation": "R", "target": "B"}])["traceability_rate"] == 0.0
    assert source_traceability([{"source": "A", "relation": "R", "target": "B", "provenance": "p.3"}])[
        "traceability_rate"] == 1.0


# B3: 허용 목록을 generator 로 줘도 정상
def test_schema_violation_accepts_generator():
    r = schema_violation_rate([{"name": "S", "type": "Company"}], [], (x for x in ["Company"]), {"R"})
    assert r["schema_violation_rate"] == 0.0


def test_schema_pattern_violation():
    ents = [{"name": "대창", "type": "회사"}, {"name": "동", "type": "품목"}]
    rels = [{"subject": "동", "relation": "공급", "object": "대창"}]
    r = schema_violation_rate(ents, rels, allowed_patterns={("회사", "공급", "회사"), ("품목", "공급된다", "회사")})
    assert r["violations"] == 1 and "bad_pattern" in r["issues"][0]


# B4: 같은 두 노드 사이 관계 여러 개 보존
def test_multigraph_keeps_parallel_relations():
    g = load_kg(triples=[("@제출회사", "생산", "PROPYLENE"), ("@제출회사", "매입", "PROPYLENE")])
    assert g.number_of_edges() == 2
    paths = enumerate_paths(g, 1, 1)
    assert sorted(p.relations[0] for p in paths) == ["매입", "생산"]


# CSV 칸 이름(subject/object) · 문서 범위 · 정규화
def test_compare_kg_csv_columns_scope_normalize():
    gold = [{"rcept_no": "1", "subject": "(주)대창", "relation": "공급", "object": "LS"},
            {"rcept_no": "2", "subject": "대창", "relation": "공급", "object": "LS"}]
    pred = [{"rcept_no": "1", "subject": "대창", "relation": "공급", "object": "LS"}]
    norm = lambda s: s.replace("(주)", "")
    r = compare_kg(pred, gold, normalize=norm, scope=("rcept_no",))
    assert r["matched"] == 1 and r["reference_total"] == 2   # 다른 보고서의 같은 트리플은 따로 셈


# 이름만 보는 지표와 방향까지 보는 지표의 차이
def test_entity_recall_ignores_direction_but_triple_coverage_does_not():
    assert path_entity_recall(["A", "B"], [("B", "WRONG", "A")])["path_entity_recall"] == 1.0
    assert triple_coverage([("A", "R", "B")], [("B", "R", "A")])["triple_coverage"] == 0.0


# B1: 검수 안 된 문항은 실행하지 않음
def test_to_eval_dataset_skips_unreviewed():
    def item(i, status):
        return TestItem(id=i, split="general_rag", query_type="single_hop", persona="p", question="q?",
                        reference="r", evidence_path=["A", "B"], evidence_relations=["R"], review_status=status)
    called = []
    ds = to_eval_dataset([item("a", "keep"), item("b", "drop"), item("c", "pending")],
                         lambda q: (called.append(q) or "ans", ["ctx"]))
    assert len(ds) == 1 and len(called) == 1 and ds.samples[0].sample_id == "a"


def test_leakage_report():
    leak = TestItem(id="x", split="general_rag", query_type="single_hop", persona="p",
                    question="기후변화는 Planet 에 있나요?", reference="Planet", evidence_path=["기후변화", "Planet"],
                    evidence_relations=["MENTIONED_IN"], expected_entities=["Planet"])
    other = leak.model_copy(update={"id": "y", "split": "graph_reasoning", "question": "어디에 있나요?"})
    r = leakage_report([leak, other])
    assert r["answer_in_question"] == [{"id": "x", "leaked": ["Planet"]}]
    assert r["split_overlap"] == [["기후변화", "Planet"]]


# 인용: 세 검사를 따로
def test_citation_helpers():
    parts = split_cited_sentences("월 20만 원입니다 [p.35]. 최장 24개월입니다. [p.36]")
    assert parts == [{"sentence": "월 20만 원입니다.", "citations": ["35"]},
                     {"sentence": "최장 24개월입니다.", "citations": ["36"]}]
    assert resolve_citations(["35", 99], {35: "x"})["missing"] == ["99"]
    assert quote_in_source("월 20만\n원", "청년 월세는 월 20만 원 지원") is True


def test_citation_support_recall_precision():
    corpus = {"35": "월 최대 20만 원", "40": "다른 제도 설명"}
    # 호출 순서: [35, 40] 함께 -> 지지 / 35 혼자 -> 지지(좋은 출처) / 40 혼자 -> 아님 / 40 을 빼고 35 만 -> 지지(40 은 불필요)
    llm = FakeLLM([{"supported": True, "reason": ""}, {"supported": True, "reason": ""},
                   {"supported": False, "reason": ""}, {"supported": True, "reason": ""}])
    r = citation_support("월 20만 원입니다 [p.35][p.40].", corpus, llm)
    assert r["citation_recall"] == 1.0 and r["citation_precision"] == 0.5


# pairwise: 순서를 바꾸면 결과가 바뀌면 승자를 정하지 않음
def test_pairwise_inconsistent():
    r = pairwise_judge("q", "A답", "B답", FakeLLM([{"reason": "", "winner": "1"}, {"reason": "", "winner": "1"}]))
    assert r["ab"] == "A" and r["ba"] == "B" and r["final"] == "inconsistent"


def test_baseline_roundtrip(tmp_path):
    def report(scores):
        rows = [{"sample_id": i, "faithfulness": s} for i, s in scores.items()]
        from kgonto_rag_eval.schema import EvalReport
        return EvalReport(rows=rows, summary={"faithfulness": None})
    save_baseline(report({"a": 1.0, "b": 0.8}), tmp_path / "b.json", meta={"judge": "fake"})
    same = compare_baseline(report({"a": 1.0, "b": 0.8, "c": 0.0}), tmp_path / "b.json")
    assert same["ok"] and same["only_in_current"] == ["c"] and same["faithfulness"]["paired"] == 2
    worse = compare_baseline(report({"a": 0.5, "b": None}), tmp_path / "b.json", tolerance=0.1)
    assert not worse["ok"] and worse["faithfulness"]["paired"] == 1
