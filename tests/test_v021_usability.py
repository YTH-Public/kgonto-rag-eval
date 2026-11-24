"""v0.2.1 에서 더한 것의 오프라인 테스트 (API 호출 없음)."""
import sys
import threading
import time

import pytest

from kgonto_rag_eval import evaluate
from kgonto_rag_eval.citation import citation_support
from kgonto_rag_eval.judge import pairwise_judge
from kgonto_rag_eval.kg import DEFAULT_PERSONAS, ESG_PERSONAS
from kgonto_rag_eval.metrics import AnswerRelevancy, Faithfulness, Metric
from kgonto_rag_eval.schema import MetricResult

ev = sys.modules["kgonto_rag_eval.evaluate"]   # 모듈(패키지 최상위의 evaluate 는 함수)


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


class EchoLLM:
    """답에 적힌 숫자로 판정을 돌려주는 가짜 모델 - 순서와 상관없이 같은 값(동시 실행 검사용)."""

    def __init__(self):
        self.active = self.peak = 0
        self.lock = threading.Lock()

    def with_structured_output(self, schema):
        llm = self

        class _Runner:
            def invoke(self, messages):
                with llm.lock:
                    llm.active += 1
                    llm.peak = max(llm.peak, llm.active)
                time.sleep(0.02)
                n = int(messages[1].content.rsplit("답변:\n", 1)[1])
                with llm.lock:
                    llm.active -= 1
                if n == 3:
                    raise RuntimeError("boom")
                return schema(total_claims=10, supported_claims=n, reason=f"이유 {n}")
        return _Runner()


class Length(Metric):
    name = "length"
    needs_llm = False

    def score(self, sample, llm):
        assert llm is None
        return MetricResult(metric=self.name, score=min(1.0, len(sample.response) / 10))


def rows(n=6):
    return [{"sample_id": f"s{i}", "user_input": "q", "retrieved_contexts": ["c"], "response": str(i)} for i in range(n)]


# 코드 지표만이면 기본 모델을 만들지 않음 / 하나라도 LLM 지표면 만듦
def test_code_metric_does_not_build_judge(monkeypatch):
    monkeypatch.setattr(ev, "default_judge", lambda: (_ for _ in ()).throw(AssertionError("모델을 만들었음")))
    report = evaluate(rows(2), [Length()])
    assert report.summary["length"] == 0.1


def test_llm_metric_builds_default_judge_once(monkeypatch):
    calls = []
    monkeypatch.setattr(ev, "default_judge", lambda: calls.append(1) or EchoLLM())
    evaluate(rows(2), [Faithfulness()])
    assert len(calls) == 1


def test_custom_metric_without_flag_is_llm_metric(monkeypatch):
    class Old:                       # 0.2.0 방식: Metric 을 물려받지 않은 지표도 동작
        name = "old"

        def score(self, sample, llm):
            return MetricResult(metric="old", score=1.0 if llm == "문자열도 됨" else 0.0)
    assert evaluate(rows(1), [Old()], llm="문자열도 됨").summary["old"] == 1.0


# 동시 실행 = 차례 실행 (순서 · 점수 · 오류 수)
def test_parallel_equals_serial():
    serial = evaluate(rows(), [Faithfulness()], llm=EchoLLM())
    llm = EchoLLM()
    parallel = evaluate(rows(), [Faithfulness()], llm=llm, max_workers=4)
    assert parallel.rows == serial.rows and parallel.stats == serial.stats and parallel.summary == serial.summary
    assert [r["sample_id"] for r in parallel.rows] == [f"s{i}" for i in range(6)]
    assert parallel.stats["faithfulness"]["errors"] == 1 and llm.peak >= 2


@pytest.mark.parametrize("bad", [0, -1, 1.5, float("nan"), float("inf"), "4", True, None])
def test_max_workers_must_be_positive_int(bad):
    with pytest.raises(ValueError):
        evaluate(rows(1), [Length()], max_workers=bad)


@pytest.mark.parametrize("reason, details", [(True, True), (True, False), (False, True), (False, False)])
def test_explain_respects_keep_options(reason, details):
    out = {"total_claims": 2, "supported_claims": 1, "reason": "절반만"}
    one = evaluate(rows(1), [Faithfulness()], llm=FakeLLM([out]), keep_reason=reason, keep_details=details).explain("s0", "faithfulness")
    assert one["score"] == 0.5 and one["verdict"] == "partial"
    assert one["reason"] == ("절반만" if reason else None) and (one["details"] is not None) == details


def test_mixed_metrics_build_judge_once(monkeypatch):
    class Code(Length):
        def score(self, sample, llm):
            return MetricResult(metric=self.name, score=0.0)
    calls = []
    monkeypatch.setattr(ev, "default_judge", lambda: calls.append(1) or EchoLLM())
    evaluate(rows(2), [Code(), Faithfulness()], max_workers=2)
    assert len(calls) == 1


def test_empty_dataset_needs_no_judge(monkeypatch):
    monkeypatch.setattr(ev, "default_judge", lambda: (_ for _ in ()).throw(AssertionError()))
    assert evaluate([], [Faithfulness()]).rows == []


def test_generator_metrics_kept():
    report = evaluate(rows(1), (m for m in [Length()]))
    assert report.summary == {"length": 0.1}


# 빈 답 · 빈 근거도 verdict
def test_faithfulness_empty_has_verdict():
    base = {"user_input": "q", "response": "r", "retrieved_contexts": []}
    row = evaluate([base], [Faithfulness()], llm=FakeLLM([])).rows[0]
    assert row["faithfulness"] == 0.0 and row["faithfulness__verdict"] == "fail"
    assert row["faithfulness__failed_component"] == "retriever"
    row = evaluate([{**base, "response": " ", "retrieved_contexts": ["c"]}], [Faithfulness()], llm=FakeLLM([])).rows[0]
    assert row["faithfulness__verdict"] == "fail" and row["faithfulness__failed_component"] == "generator"


# explain
def test_explain_one_and_all():
    out = {"total_claims": 2, "supported_claims": 1, "unsupported": ["x"], "reason": "절반만"}
    report = evaluate(rows(1), [Faithfulness()], llm=FakeLLM([out]))
    one = report.explain("s0", "faithfulness")
    assert one["score"] == 0.5 and one["reason"] == "절반만" and one["details"]["unsupported"] == ["x"] and one["response"] == "0"
    assert report.explain("s0")["metrics"]["faithfulness"]["score"] == 0.5


def test_explain_wrong_names():
    report = evaluate(rows(2), [Length()])
    with pytest.raises(KeyError):
        report.explain("없음")
    with pytest.raises(KeyError):
        report.explain("s0", "faithfulness")
    report.rows.append(dict(report.rows[0]))      # 같은 ID 가 두 번 (evaluate 는 0.3 부터 미리 막음)
    with pytest.raises(ValueError):
        report.explain("s0")


# 인용 이유
def test_citation_reasons_kept():
    llm = FakeLLM([{"supported": True, "reason": "합쳐서 맞음"}, {"supported": True, "reason": "35 만으로 충분"},
                   {"supported": False, "reason": "40 은 다른 제도"}, {"supported": True, "reason": "40 을 빼도 맞음"}])
    r = citation_support("월 20만 원입니다 [p.35][p.40].", {"35": "월 최대 20만 원", "40": "다른 제도"}, llm)
    s = r["sentences"][0]
    assert s["reason"] == "합쳐서 맞음" and [j["kind"] for j in s["judgements"]] == ["joint", "alone", "alone", "rest"]
    assert s["judgements"][2]["citation"] == "40" and r["attempted"] == 1
    assert r["citation_recall"] == 1.0 and r["citation_precision"] == 0.5


def test_citation_error_keeps_partial_judgements():
    llm = FakeLLM([{"supported": True, "reason": "합쳐서 맞음"}, RuntimeError("boom")])
    r = citation_support("가 [p.1][p.2]. 나 [p.9]. 다.", {"1": "x", "2": "y"}, llm)
    assert r["errors"] == 1 and r["attempted"] == 2 and r["cited_sentences"] == 1 and r["uncited"] == 1
    assert r["sentences"][0]["judgements"][0]["kind"] == "joint"
    assert r["sentences"][1]["supported"] is False and r["sentences"][1]["reason"]     # 자료에 없는 출처


# pairwise: 한 순서가 실패해도 다른 순서 결과를 남김
def test_pairwise_first_failure_still_asks_second():
    r = pairwise_judge("q", "A답", "B답", FakeLLM([RuntimeError("boom"), {"reason": "B 가 나음", "winner": "1"}]))
    assert r["final"] == "error" and r["ab"] is None and r["ba"] == "B" and "boom" in r["error"]


def test_pairwise_both_fail():
    r = pairwise_judge("q", "A", "B", FakeLLM([RuntimeError("x"), RuntimeError("y")]))
    assert r["final"] == "error" and "x" in r["error"] and "y" in r["error"]


def test_default_personas_are_general():
    assert len(DEFAULT_PERSONAS) == 3 and not any("ESG" in p.name for p in DEFAULT_PERSONAS)
    assert [p.name for p in ESG_PERSONAS] == ["투자자", "ESG담당자", "협력회사"]   # 0.2.0 의 기본값


# 0.2.0 과 같은 점수 (지표 계산은 바뀌지 않음)
def test_scores_unchanged():
    out = {"total_intents": 2, "addressed_intents": 1, "verbosity_penalty": 0.1, "reason": "x"}
    row = evaluate(rows(1), [AnswerRelevancy()], llm=FakeLLM([out])).rows[0]
    assert row["answer_relevancy"] == pytest.approx(0.4)
