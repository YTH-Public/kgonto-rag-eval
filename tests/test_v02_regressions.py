"""v0.2 Codex 점검(2026-11-21)에서 찾은 경계 사례의 회귀 테스트 (API 호출 없음)."""
import pytest

from kgonto_rag_eval import evaluate
from kgonto_rag_eval.baseline import compare_baseline, save_baseline
from kgonto_rag_eval.citation import citation_support, quote_in_source, split_cited_sentences
from kgonto_rag_eval.judge import pairwise_judge
from kgonto_rag_eval.kg import TestItem, compare_kg, to_eval_dataset
from kgonto_rag_eval.metrics import Faithfulness
from kgonto_rag_eval.schema import EvalReport

TestItem.__test__ = False


class FakeLLM:
    def __init__(self, outputs):
        self.outputs = list(outputs)

    def with_structured_output(self, schema):
        llm = self

        class _Runner:
            def invoke(self, messages):
                value = llm.outputs.pop(0)
                if isinstance(value, Exception):
                    raise value
                return schema(**value)
        return _Runner()


def _report(scores):
    return EvalReport(rows=[{"sample_id": i, "m": s} for i, s in scores], summary={"m": None})


# 인용: 마침표 뒤 출처는 앞 문장의 것
def test_citation_after_period_stays_with_sentence():
    assert split_cited_sentences("A. [p.1] B. [p.2]") == [
        {"sentence": "A.", "citations": ["1"]}, {"sentence": "B.", "citations": ["2"]}]
    assert split_cited_sentences("가 [p.1][p.2]. 나.\n다 [p.3]")[0]["citations"] == ["1", "2"]


def test_blank_quote_is_not_found():
    assert quote_in_source("   ", "아무 글") is False


def test_citation_judge_failure_is_error_not_zero():
    r = citation_support("A [p.1].", {"1": "x"}, FakeLLM([RuntimeError("boom")]))
    assert r["errors"] == 1 and r["citation_recall"] is None


# 기준선
def test_baseline_lost_cell_fails(tmp_path):
    save_baseline(_report([("a", 1.0), ("b", 1.0)]), tmp_path / "b.json")
    r = compare_baseline(_report([("a", 1.0), ("b", None)]), tmp_path / "b.json")
    assert r["ok"] is False and r["m"]["lost"] == ["b"]
    assert compare_baseline(_report([("a", 1.0)]), tmp_path / "b.json")["ok"] is False   # 행이 통째로 없음
    assert compare_baseline(_report([("a", 1.0)]), tmp_path / "b.json", allow_missing=True)["ok"] is True


def test_baseline_duplicate_id_rejected(tmp_path):
    with pytest.raises(ValueError):
        save_baseline(_report([("a", 0.0), ("a", 1.0)]), tmp_path / "b.json")


def test_baseline_empty_metrics_rejected(tmp_path):
    with pytest.raises(ValueError):
        save_baseline(EvalReport(rows=[], summary={}), tmp_path / "b.json")


def test_raw_score_shows_small_regression(tmp_path):
    def run(supported):
        out = [{"total_claims": 10000, "supported_claims": supported, "reason": "x"}]
        return evaluate([{"sample_id": "a", "user_input": "q", "retrieved_contexts": ["c"], "response": "r"}],
                        [Faithfulness()], llm=FakeLLM(out))
    save_baseline(run(9001), tmp_path / "b.json")
    r = compare_baseline(run(9000), tmp_path / "b.json", tolerance=0.00005)
    assert r["faithfulness"]["delta"] == pytest.approx(-0.0001) and r["ok"] is False


def test_error_count_without_details():
    report = evaluate([{"user_input": "q", "retrieved_contexts": ["c"], "response": "r"}], [Faithfulness()],
                      llm=FakeLLM([{"total_claims": 1, "supported_claims": 2, "reason": "x"}]), keep_details=False)
    assert report.stats["faithfulness"]["errors"] == 1


# 범위 칸이 없으면 비교하지 않음
def test_scope_missing_raises():
    rows = [{"subject": "A", "relation": "R", "object": "B"}]
    with pytest.raises(ValueError):
        compare_kg(rows, rows, scope=("rcept_no",))


def test_subject_wins_over_source_column():
    r = compare_kg([{"subject": "A", "relation": "R", "object": "B", "source": "p.1"}], [("A", "R", "B")])
    assert r["matched"] == 1


# 평가셋 변환
def _item(question="q?", status="keep"):
    return TestItem(id="x", split="general_rag", query_type="single_hop", persona="p", question=question,
                    reference="r", evidence_path=["A", "B"], evidence_relations=["R"], review_status=status)


def test_to_eval_dataset_rejects_string_contexts():
    with pytest.raises(Exception):
        to_eval_dataset([_item()], lambda q: ("ans", "ABC"))


def test_to_eval_dataset_skips_failed_validation():
    with pytest.warns(UserWarning):
        ds = to_eval_dataset([_item(question="not a question")], lambda q: ("ans", ["c"]))
    assert len(ds) == 0


# pairwise: 두 번째 호출이 실패해도 첫 결과와 이유는 남음
def test_pairwise_keeps_first_result_on_error():
    r = pairwise_judge("q", "a", "b", FakeLLM([{"reason": "A 가 정확", "winner": "1"}, RuntimeError("boom")]))
    assert r["final"] == "error" and r["ab"] == "A" and r["ab_reason"] == "A 가 정확"


class _Seq:
    """판정 결과를 순서대로 돌려주는 가짜 모델 (예외 객체면 던짐)"""
    def __init__(self, answers):
        self.answers = list(answers)

    def with_structured_output(self, schema):
        outer = self

        class _R:
            def invoke(self, messages):
                a = outer.answers.pop(0)
                if isinstance(a, Exception):
                    raise a
                return schema(supported=a, reason="x")
        return _R()


def test_citation_precision_needs_joint_support():
    # 결합 판정 False 면 혼자 판정이 True 여도 쓸모 있는 출처로 세지 않음 (ALCE)
    llm = _Seq([False, True, True])
    r = citation_support("월 20만 원입니다 [p.1][p.2].", {1: "a", 2: "b"}, llm)
    assert r["citation_recall"] == 0.0 and r["citation_precision"] == 0.0


def test_citation_error_in_alone_check_is_recorded():
    llm = _Seq([True, RuntimeError("judge failed")])
    r = citation_support("월 20만 원입니다 [p.1][p.2].", {1: "a", 2: "b"}, llm)
    assert r["errors"] == 1 and r["citation_recall"] is None


def test_citation_recall_all_counts_uncited():
    llm = _Seq([True])
    r = citation_support("월 20만 원입니다 [p.1]. 출처 없는 문장입니다.", {1: "a"}, llm)
    assert r["citation_recall"] == 1.0 and r["citation_recall_all"] == 0.5 and r["uncited"] == 1
