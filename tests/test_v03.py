"""v0.3 에서 더한 것의 오프라인 테스트 (API 호출 없음)."""
import json

import pytest

from kgonto_rag_eval import diagnose, evaluate, extract_citations
from kgonto_rag_eval.baseline import compare_baseline, save_baseline
from kgonto_rag_eval.citation import citation_support, split_cited_sentences
from kgonto_rag_eval.metrics import (
    ALL_METRICS, AnsweredRelevancy, CitedRetrieved, FactCoverage, FactHit, Faithfulness, Metric,
)
from kgonto_rag_eval.schema import EvalReport, EvalSample, MetricResult


class FakeLLM:
    def __init__(self, outputs):
        self.outputs, self.calls = list(outputs), 0

    def with_structured_output(self, schema):
        llm = self

        class _Runner:
            def invoke(self, messages):
                llm.calls += 1
                value = llm.outputs.pop(0)
                if isinstance(value, Exception):
                    raise value
                return schema(**value)
        return _Runner()


def sample(response="월 최대 50만 원, 만기 3년입니다.", facts=("만기 3년", "월 50만 원"), **kw):
    meta = {"expected_facts": list(facts)} if facts is not None else {}
    return EvalSample(user_input="q", response=response, retrieved_contexts=["c"], reference="r", metadata={**meta, **kw})


def fact(index, label, quotes=(), reason="x"):
    return {"index": index, "label": label, "quotes": list(quotes), "reason": reason}


# ---------------------------------------------------------------- MetricResult
@pytest.mark.parametrize("bad", [1.2, -0.1, float("nan"), float("inf")])
def test_score_range(bad):
    with pytest.raises(ValueError):
        MetricResult(metric="m", score=bad)


def test_status_filled():
    assert MetricResult(metric="m", score=0.0).status == "ok"
    assert MetricResult(metric="m").status == "not_applicable"
    assert MetricResult(metric="m", error="x").status == "error"


# ---------------------------------------------------------------- AnsweredRelevancy
def test_answered_refusal_is_zero_with_one_call():
    llm = FakeLLM([{"reason": "자료에 없다고만 함", "answered": False}])
    r = AnsweredRelevancy().score(sample("자료에 없습니다."), llm)
    assert r.score == 0.0 and r.details == {"answered": False} and llm.calls == 1 and r.metric == "answered_relevancy"


def test_answered_keeps_original_score():
    llm = FakeLLM([{"reason": "답함", "answered": True},
                   {"total_intents": 2, "addressed_intents": 1, "verbosity_penalty": 0.1, "reason": "절반"}])
    r = AnsweredRelevancy().score(sample(), llm)
    assert r.score == pytest.approx(0.4) and r.details["answered"] is True and r.details["total"] == 2 and r.metric == "answered_relevancy"


def test_answered_blank_no_call():
    llm = FakeLLM([])
    assert AnsweredRelevancy().score(sample("  "), llm).score == 0.0 and llm.calls == 0


def test_answered_second_call_failure_is_error():
    llm = FakeLLM([{"reason": "", "answered": True}, RuntimeError("boom")])
    row = evaluate([sample().model_dump()], [AnsweredRelevancy()], llm=llm).rows[0]
    assert row["answered_relevancy"] is None and row["answered_relevancy__status"] == "error"


# ---------------------------------------------------------------- FactCoverage
def test_fact_coverage_labels():
    llm = FakeLLM([{"answered": True, "items": [fact(1, "present", ["만기 3년"]), fact(2, "contradicted", ["월 최대 50만 원"])]}])
    r = FactCoverage().score(sample(), llm)
    assert r.score == 0.5 and r.details["contradicted"] == ["월 50만 원"] and r.details["missing"] == []
    assert "어긋남" in r.reason and llm.calls == 1


def test_fact_coverage_refusal_is_zero_by_code():
    llm = FakeLLM([{"answered": False, "items": [fact(1, "present", ["만기 3년"]), fact(2, "missing")]}])
    r = FactCoverage().score(sample("만기 3년인지는 자료에 없습니다."), llm)
    assert r.score == 0.0 and r.details["answered"] is False and len(r.details["missing"]) == 2 and "답하지 않음" in r.reason


def test_fact_coverage_refusal_ignores_bad_quote():
    llm = FakeLLM([{"answered": False, "items": [fact(1, "present", ["답에 없는 문구"]), fact(2, "missing")]}])
    r = FactCoverage().score(sample("자료에 없습니다."), llm)
    assert r.score == 0.0 and r.status == "ok"


def test_fact_coverage_quote_ignores_space_and_marks():
    llm = FakeLLM([{"answered": True, "items": [fact(1, "present", ["만기  **3년**"])]}])
    assert FactCoverage().score(sample("적금은 만기 3년입니다.", ["만기 3년"]), llm).score == 1.0


@pytest.mark.parametrize("items", [
    [fact(1, "present", ["답에 없는 문구"]), fact(2, "missing")],      # 지어낸 문구
    [fact(1, "present"), fact(2, "missing")],                          # 문구 없음
    [fact(1, "missing")],                                              # 번호 빠짐
    [fact(1, "missing"), fact(1, "missing")],                          # 번호 중복
    [fact(0, "missing"), fact(1, "missing")],                          # 범위 밖
])
def test_fact_coverage_bad_verdict_is_error(items):
    r = FactCoverage().score(sample(), FakeLLM([{"answered": True, "items": items}]))
    assert r.score is None and r.status == "error" and r.failed_component == "judge"


@pytest.mark.parametrize("facts", [None, "만기 3년", [""], [1], ["가", " 가 "]])
def test_fact_input_errors(facts):
    s = EvalSample(user_input="q", response="r", metadata={} if facts is None else {"expected_facts": facts})
    for metric in (FactCoverage(), FactHit()):
        r = metric.score(s, FakeLLM([]))
        assert r.score is None and r.reason_code == "invalid_input" and r.failed_component == "input"


def test_fact_empty_list_not_applicable():
    r = FactCoverage().score(sample(facts=[]), FakeLLM([]))
    assert r.status == "not_applicable" and r.reason_code == "no_required_facts"


def test_fact_blank_answer_no_call():
    llm = FakeLLM([])
    r = FactCoverage().score(sample(" "), llm)
    assert r.score == 0.0 and len(r.details["missing"]) == 2 and llm.calls == 0


# ---------------------------------------------------------------- 코드 지표
def test_fact_hit():
    assert FactHit().score(sample("만기3년 이고 매월 최대 50만 원")).score == 0.5


def test_cited_retrieved():
    m = CitedRetrieved()
    assert m.score(sample("가 [p.1-3] 나 [p.9]", retrieved_ids=[1, "2", 3])).score == 0.75
    assert m.score(sample("출처 없음", retrieved_ids=[1])).reason_code == "no_citation"
    assert m.score(sample("가 [p.1]", retrieved_ids=[])).score == 0.0
    assert m.score(sample("가 [p.1]")).reason_code == "invalid_input"
    assert m.score(sample("가 [p.1]", retrieved_ids="1")).reason_code == "invalid_input"
    assert CitedRetrieved("pages").score(sample("가 [p.1]", pages=[1])).score == 1.0


def test_code_metrics_need_no_judge(monkeypatch):
    import sys
    monkeypatch.setattr(sys.modules["kgonto_rag_eval.evaluate"], "default_judge",
                        lambda: (_ for _ in ()).throw(AssertionError()))
    report = evaluate([sample(retrieved_ids=[1]).model_dump()], [FactHit(), CitedRetrieved()])
    assert report.stats["cited_retrieved"]["not_applicable"] == 1 and report.stats["cited_retrieved"]["errors"] == 0


def test_new_metrics_not_in_all():
    assert not {"answered_relevancy", "fact_coverage", "fact_hit", "cited_retrieved"} & {m.name for m in ALL_METRICS}


# ---------------------------------------------------------------- 출처
def test_extract_citations_ranges():
    assert extract_citations("가 [p.1-3][p.7] 나 [p.4~5] 다 [p.8–p.9]") == ["1", "2", "3", "7", "4", "5", "8", "9"]
    assert extract_citations("가 [p.1-3]", ranges=False) == []
    assert extract_citations("거꾸로 [p.9-3] 너무 넓음 [p.1-900]") == []
    assert extract_citations("") == []


def test_citation_ranges_option_off_by_default():
    assert split_cited_sentences("가 [p.1-2].") == [{"sentence": "가 [p.1-2].", "citations": []}]
    assert split_cited_sentences("가 [p.1-2].", ranges=True) == [{"sentence": "가.", "citations": ["1", "2"]}]
    r = citation_support("가 [p.1-2].", {1: "x", 2: "y"},
                         FakeLLM([{"supported": True, "reason": ""}] * 3), ranges=True)
    assert r["cited_sentences"] == 1 and r["citation_precision"] == 1.0


# ---------------------------------------------------------------- evaluate
def test_duplicate_names_and_ids_rejected():
    s = sample().model_dump()
    with pytest.raises(ValueError):
        evaluate([s], [FactHit(), FactHit()])
    with pytest.raises(ValueError):
        evaluate([{**s, "sample_id": "a"}, {**s, "sample_id": "a"}], [FactHit()])
    assert len(evaluate([s, s], [FactHit()]).rows) == 2     # ID 를 안 주면 순번


def test_failed_component():
    class Broken(Metric):
        name = "broken"

        def score(self, sample, llm):
            return MetricResult(metric=self.name, score=1 / 0)

    class CodeBroken(Broken):
        name, needs_llm = "code_broken", False

        def score(self, sample, llm):
            raise ValueError("x")

    class OutOfRange(Metric):
        name = "out"
        needs_llm = False

        def score(self, sample, llm):
            return MetricResult(metric=self.name, score=3)
    report = evaluate([sample().model_dump()], [Broken(), CodeBroken(), OutOfRange(), Faithfulness()], llm=FakeLLM([RuntimeError("api")]))
    row = report.rows[0]
    assert row["broken__failed_component"] == "metric" and row["code_broken__failed_component"] == "metric"
    assert row["out"] is None and row["out__status"] == "error"
    assert row["faithfulness__failed_component"] == "judge"
    assert report.explain("0", "broken")["status"] == "error"


def test_meta_recorded():
    report = evaluate([sample().model_dump()], [FactHit()], run_meta={"system": "v1"})
    assert report.meta["package_version"].startswith("0.3") and report.meta["run_meta"] == {"system": "v1"} and report.meta["judge"] is None
    json.dumps(report.to_dict(), ensure_ascii=False)          # 파일로 남길 수 있어야 함
    other = evaluate([sample("다른 답").model_dump()], [FactHit()])
    assert other.meta["questions_sha"] == report.meta["questions_sha"]          # 답이 바뀌어도 문항은 같음
    changed = evaluate([sample(facts=["다른 사실"]).model_dump()], [FactHit()])
    assert changed.meta["questions_sha"] != report.meta["questions_sha"]


# ---------------------------------------------------------------- 기준선 조건
def test_strict_meta(tmp_path):
    def run(facts=("만기 3년",), answer="만기 3년"):
        return evaluate([{**sample(answer, facts).model_dump(), "sample_id": "a"}], [FactHit()])
    path = tmp_path / "b.json"
    save_baseline(run(), path)
    assert compare_baseline(run(answer="만기 3년입니다"), path, strict_meta=True)["ok"] is True
    changed = compare_baseline(run(facts=("만기 3년입니다",), answer="만기 3년입니다"), path, strict_meta=True)
    assert changed["ok"] is False and "questions_sha" in changed["incomparable"]
    assert "incomparable" not in compare_baseline(run(), path)                   # 기본은 0.2 와 같은 칸
    old = json.loads(path.read_text(encoding="utf-8"))
    del old["report_meta"]                                                       # 0.2 기준선
    path.write_text(json.dumps(old), encoding="utf-8")
    assert compare_baseline(run(), path)["ok"] is True
    assert compare_baseline(run(), path, strict_meta=True)["ok"] is False


# ---------------------------------------------------------------- diagnose
def report_of(**scores):
    details = scores.pop("details", {})
    row = {"sample_id": "a", **scores, **details}
    return EvalReport(rows=[row], summary={k: None for k in scores})


def suspects(**scores):
    return [d["suspect"] for d in diagnose(report_of(**scores))]


def test_diagnose_rules():
    assert suspects(context_recall=0.0, fact_coverage=0.2) == ["검색이 근거를 못 찾음"]
    assert suspects(context_recall=1.0, fact_coverage=0.2) == ["근거는 있는데 답이 빠뜨림"]
    assert suspects(faithfulness=1.0, answer_correctness=0.1) == ["다른 자료에 충실한 답"]
    assert suspects(fact_coverage=1.0, answer_correctness=0.1) == ["덧붙인 내용 또는 필수사실"]
    assert suspects(cited_retrieved=0.5) == ["검색되지 않은 출처"]
    assert suspects(context_recall=0.6, fact_coverage=0.6) == []
    assert suspects(fact_coverage=0.0) == []                  # 근거 지표가 없으면 판단하지 않음


def test_diagnose_refusal_is_separated():
    refused = {"answered_relevancy__details": {"answered": False}}
    assert suspects(answered_relevancy=0.0, context_recall=1.0, fact_coverage=0.0, details=refused) == ["지나친 거절"]
    assert suspects(answered_relevancy=0.0, context_recall=0.0, fact_coverage=0.0, details=refused) == ["근거가 없어 거절"]


def test_diagnose_error_and_bad_range():
    report = report_of(fact_coverage=None, details={"fact_coverage__error": "x"})
    assert [d["suspect"] for d in diagnose(report)] == ["채점 실패"]
    with pytest.raises(ValueError):
        diagnose(report, low=0.9, high=0.1)


# ---------------------------------------------------------------- Codex 점검에서 나온 반례
@pytest.mark.parametrize("quote, answer, ok", [
    ("1.5%", "금리는 15%입니다.", False), ("5%", "금리는 15%입니다.", False), ("15", "금리는 15.5%", False), ("150", "1,150원", False), ("1", "1,150원", False), ("2025년", "2024,2025년", True), ("2024", "2024,2025년", True), ("1.5", "11.5", False), ("5", "1.5", False),
    ("1.5%", "금리는 **1.5 %**입니다 [p.3].", True), ("19~34세", "19 ~ 34세", True), ("19~34세", "1934세", False),
    ("-5", "5", False), ("만기 3년입니다.", "만기 3년입니다 [p.1].", True), ("15%", "2026 15%", True), ("", "x", False), (" . ", "x", False),
])
def test_quote_in_answer(quote, answer, ok):
    from kgonto_rag_eval.metrics.fact_coverage import quote_in_answer
    assert quote_in_answer(quote, answer) is ok


@pytest.mark.parametrize("quotes", [["금리는 99%"], []])
def test_contradicted_needs_real_quote(quotes):
    r = FactCoverage().score(sample("금리는 1.5%입니다.", ["금리는 1.5%"]), FakeLLM([{"answered": True, "items": [fact(1, "contradicted", quotes)]}]))
    assert r.status == "error"


def test_broken_range_counts_as_outside():
    for bad in ("[p.9-3]", "[p.1-900]"):
        r = CitedRetrieved().score(sample(f"가 [p.1] 나 {bad}", retrieved_ids=[1]))
        assert r.score == 0.5 and r.details["broken"]
    assert CitedRetrieved().score(sample("가 [p.9-3]", retrieved_ids=[1])).score == 0.0


def test_failure_place_by_call_not_type():
    class Invalid(Metric):
        name = "invalid"

        def score(self, sample, llm):
            return MetricResult(metric=self.name, score=3)
    row = evaluate([sample().model_dump()], [Invalid(), AnsweredRelevancy()], llm=FakeLLM([TypeError("SDK")])).rows[0]
    assert row["invalid__failed_component"] == "metric" and row["answered_relevancy__failed_component"] == "judge"
    row = evaluate([sample().model_dump()], [AnsweredRelevancy()], llm=FakeLLM([{"reason": "이유만"}])).rows[0]   # 스키마에 안 맞는 판정
    assert row["answered_relevancy__failed_component"] == "judge"


def test_run_meta_rejects_secrets_and_is_copied():
    for bad in ({"api_key": "x"}, {"judge": {"OPENAI_API_KEY": "x"}}, {"a": [{"db_password": "x"}]}):
        with pytest.raises(ValueError):
            evaluate([sample().model_dump()], [FactHit()], run_meta=bad)
    meta = {"conditions": {"max_tokens": 5, "input_tokens": [1]}}
    report = evaluate([sample().model_dump()], [FactHit()], run_meta=meta)
    meta["conditions"]["max_tokens"] = 9
    assert report.meta["run_meta"]["conditions"]["max_tokens"] == 5


def strict_pair(tmp_path, old, new):
    path = tmp_path / "b.json"
    save_baseline(old, path)
    return compare_baseline(new, path, strict_meta=True)["incomparable"]


def test_strict_meta_sees_config_conditions_and_holes(tmp_path):
    s = {**sample("a [p.1]", left=[1], right=[1]).model_dump(), "sample_id": "a"}
    a = evaluate([s], [CitedRetrieved("left")], run_meta={"system": "v1", "conditions": {"data": 1}})
    assert strict_pair(tmp_path, a, evaluate([s], [CitedRetrieved("right")], run_meta={"conditions": {"data": 1}})) == ["metrics.cited_retrieved"]
    assert strict_pair(tmp_path, a, evaluate([s], [CitedRetrieved("left")], run_meta={"conditions": {"data": 2}})) == ["run_meta.conditions"]
    assert strict_pair(tmp_path, a, evaluate([s], [CitedRetrieved("left")], run_meta={"system": "v2", "conditions": {"data": 1}})) == []
    hole = a.model_copy(update={"meta": {"package_version": "0.3.0"}})
    assert len(strict_pair(tmp_path, hole, hole)) >= 3


def test_code_fingerprint_of_custom_metric(tmp_path):
    class Custom(Metric):
        name, needs_llm = "custom", False

        def score(self, sample, llm=None):
            return MetricResult(metric=self.name, score=1.0)

    class Other(Custom):
        def score(self, sample, llm=None):
            return MetricResult(metric=self.name, score=0.5)
    Other.__name__ = "Custom"                                    # 이름이 같아도 계산이 다르면 다른 지표
    s = {**sample().model_dump(), "sample_id": "a"}
    one = evaluate([s], [Custom()])
    assert strict_pair(tmp_path, one, evaluate([s], [Custom()])) == []
    assert strict_pair(tmp_path, one, evaluate([s], [Other()])) == ["metrics.custom"]


def test_auto_id_collision():
    s = sample().model_dump()
    with pytest.raises(ValueError):
        evaluate([s, {**s, "sample_id": "0"}], [FactHit()])


@pytest.mark.parametrize("kw", [dict(score=1.0, error="x"), dict(score=1.0, status="error"), dict(status="ok"),
                                dict(error="x", status="not_applicable")])
def test_result_contradictions(kw):
    with pytest.raises(ValueError):
        MetricResult(metric="m", **kw)


def test_code_fingerprint_sees_class_settings_and_defaults(tmp_path):
    def by_class(scale):
        class Custom(Metric):
            name, needs_llm, weight = "custom", False, scale

            def score(self, sample, llm=None):
                return MetricResult(metric=self.name, score=self.weight)
        return Custom()

    def by_default(scale):
        class Custom(Metric):
            name, needs_llm = "custom", False

            def score(self, sample, llm=None, weight=scale):
                return MetricResult(metric=self.name, score=weight)
        return Custom()

    def by_closure(scale):
        class Custom(Metric):
            name, needs_llm = "custom", False

            def score(self, sample, llm=None):
                return MetricResult(metric=self.name, score=scale)
        return Custom()
    s = {**sample().model_dump(), "sample_id": "a"}
    for make in (by_class, by_default, by_closure):
        one = evaluate([s], [make(0.5)])
        assert strict_pair(tmp_path, one, evaluate([s], [make(0.5)])) == []
        assert strict_pair(tmp_path, one, evaluate([s], [make(1.0)])) == ["metrics.custom"]


def test_metric_own_invoke_is_not_judge():
    class Local(Metric):
        name = "local"

        def invoke(self):
            return MetricResult(metric=self.name, score=3)

        def score(self, sample, llm):
            return self.invoke()
    assert evaluate([sample().model_dump()], [Local()], llm=object()).rows[0]["local__failed_component"] == "metric"
