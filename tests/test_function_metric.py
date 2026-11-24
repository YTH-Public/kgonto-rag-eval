import json
import pathlib
import re

import pytest

from kgonto_rag_eval import MetricResult, compare_baseline, evaluate, extract_citations, judge_metric, metric, save_baseline, to_dataset
from kgonto_rag_eval.metrics import Metric

ROW = {"sample_id": "a", "user_input": "질문", "response": "답", "reference": "기준"}
DATA = pathlib.Path(__file__).parents[3] / "notebooks/data/59-8_정책RAG_평가기록.jsonl"


def differs(tmp_path, one, two):
    """같은 자료를 두 지표로 채점했을 때 엄격 비교가 다르다고 보는 칸"""
    save_baseline(evaluate([ROW], [one]), tmp_path / "b.json")
    return compare_baseline(evaluate([ROW], [two]), tmp_path / "b.json", tolerance=1, strict_meta=True)["incomparable"]


@pytest.mark.parametrize("value, score, status", [
    (True, 1.0, "ok"), (False, 0.0, "ok"), (0.25, 0.25, "ok"), (1, 1.0, "ok"), (None, None, "not_applicable"),
    ((0.5, "까닭"), 0.5, "ok"), ((None, "출처 없음"), None, "not_applicable"), ((1.0, "", {"hits": [1]}), 1.0, "ok"),
    ("글", None, "error"), (2, None, "error"), (-0.1, None, "error"), (float("nan"), None, "error"), ([1], None, "error"),
    ((1.0,), None, "error"), ((1.0, 3), None, "error"), ((1.0, "", "x"), None, "error"), ((1, "", {}, 4), None, "error"),
    (MetricResult(metric="other", score=1.0), None, "error"), (MetricResult(metric="one", score=0.5), 0.5, "ok"),
])
def test_return_values(value, score, status):
    @metric
    def one(sample):
        return value
    report = evaluate([ROW], [one])
    row = report.rows[0]
    assert (row["one"], row["one__status"]) == (score, status)
    assert report.stats["one"]["errors"] == (status == "error")
    if status == "error":
        assert row["one__failed_component"] == "metric"


def test_numpy_and_details():
    np = pytest.importorskip("numpy")

    @metric
    def flag(sample):
        return np.bool_(True)

    @metric(name="비율")
    def ratio(sample):
        return np.float32(0.5), "반", {"hits": ["x"]}
    row = evaluate([ROW], [flag, ratio]).rows[0]
    assert row["flag"] == 1.0 and row["flag__verdict"] == "pass" and row["비율"] == 0.5
    assert row["비율__reason"] == "반" and row["비율__details"] == {"hits": ["x"]}
    assert flag(None)                                     # 함수로도 그대로 부를 수 있음


def test_exception_stays_in_its_cell():
    @metric
    def bad(sample):
        return sample.metadata["없음"]

    @metric
    def good(sample):
        return True
    report = evaluate([ROW], [bad, good])
    assert report.rows[0]["bad__status"] == "error" and "KeyError" in report.rows[0]["bad__error"]
    assert report.rows[0]["good"] == 1.0


def test_signature_and_reference():
    with pytest.raises(TypeError, match="needs_llm=True"):
        metric(lambda sample, llm: True)
    with pytest.raises(TypeError):
        metric(needs_llm=True)(lambda sample: True)

    @metric(needs_llm=True, requires_reference=True)
    def ask(sample, llm):
        return llm == "모델"
    out = evaluate([ROW, {**ROW, "sample_id": "b", "reference": None}], [ask], llm="모델")
    assert [r["ask"] for r in out.rows] == [1.0, None] and out.rows[1]["ask__reason_code"] == "no_reference"


@pytest.mark.parametrize("name", ["response", "sample_id", "a__b", " "])
def test_reserved_names(name):
    with pytest.raises(ValueError, match="지표 이름"):
        evaluate([ROW], [metric(name=name)(lambda sample: True)])


CUT = 1


def helper(text):
    return len(text) >= CUT


def test_fingerprint(tmp_path, monkeypatch):
    def make(limit):
        @metric
        def rule(sample):
            return len(sample.response) >= limit
        return rule

    @metric(name="rule")
    def constant(sample):
        return True

    @metric(name="rule")
    def by_helper(sample):
        return all(helper(t) for t in [sample.response])

    assert differs(tmp_path, make(1), make(1)) == []
    assert differs(tmp_path, make(1), make(2)) == ["metrics.rule"]
    assert differs(tmp_path, make(1), constant) == ["metrics.rule"]
    assert differs(tmp_path, by_helper, by_helper) == []
    save_baseline(evaluate([ROW], [by_helper]), tmp_path / "c.json")
    monkeypatch.setitem(globals(), "CUT", 99)             # 함수가 부르는 함수가 쓰는 바깥 값이 바뀜
    assert compare_baseline(evaluate([ROW], [by_helper]), tmp_path / "c.json", tolerance=1, strict_meta=True)["incomparable"] == ["metrics.rule"]


def zero():
    return 0


def one_():
    return 1


def test_fingerprint_arguments_defaults_and_partial(tmp_path):
    from functools import partial
    ns = {}
    exec("def f(s, *, a=0, b=1): return a", ns)
    first = metric(name="rule")(ns["f"])
    exec("def f(s, *, b=1, a=0): return b", ns)
    assert differs(tmp_path, first, metric(name="rule")(ns["f"])) == ["metrics.rule"]

    def make(default):
        def rule(sample, fn=default):
            return fn()
        return metric(rule)
    assert differs(tmp_path, make(zero), make(one_)) == ["metrics.rule"] and differs(tmp_path, make(zero), make(zero)) == []

    def by_value(sample, n):
        return n
    p = lambda n: metric(partial(by_value, n=n), name="rule")
    assert differs(tmp_path, p(0), p(1)) == ["metrics.rule"] and differs(tmp_path, p(0), p(0)) == []


def test_fingerprint_sees_function_in_another_file(tmp_path):
    from types import ModuleType
    rules = ModuleType("my_rules")

    def make(source):
        exec(source, rules.__dict__)
        check = rules.check

        @metric
        def rule(sample):
            return check(sample)
        return rule
    assert differs(tmp_path, make("def check(s): return 0"), make("def check(s): return 1")) == ["metrics.rule"]
    assert differs(tmp_path, make("def check(s): return 0"), make("def check(s): return 0")) == []

    @metric
    def library(sample):
        return len(json.dumps(sample.response)) > 0 and pathlib.Path("x").name == "x"
    assert differs(tmp_path, library, library) == []


def test_blank_reference_is_not_sent_and_keyword_call():
    judge = FakeJudge(True)
    out = evaluate([{**ROW, "reference": "  "}], [judge_metric("맞음", "같으면 통과", fields=["reference"])], llm=judge).rows[0]
    assert out["맞음__status"] == "not_applicable" and judge.seen == []
    assert metric(lambda sample: sample, name="x")(sample=3) == 3


def test_fingerprint_recursion_is_safe():
    @metric
    def loop(sample):
        def inner(n):
            return n == 0 or inner(n - 1)
        return inner(3)
    assert evaluate([ROW], [loop]).summary["loop"] == 1.0


class FakeJudge:
    def __init__(self, passed):
        self.passed, self.seen = passed, []

    def with_structured_output(self, schema):
        self.schema = schema
        return self

    def invoke(self, messages):
        self.seen.append("\n".join(m.content for m in messages))
        return self.schema(passed=self.passed, reason="까닭")


def test_judge_metric(tmp_path):
    polite = judge_metric("친절함", "존댓말이면 통과")
    for passed in (True, False):
        judge = FakeJudge(passed)
        row = evaluate([ROW], [polite], llm=judge).rows[0]
        assert (row["친절함"], row["친절함__reason"]) == (float(passed), "까닭")
        assert "존댓말이면 통과" in judge.seen[0] and "질문" in judge.seen[0] and "기준 답" not in judge.seen[0]
    full = judge_metric("맞음", "기준 답과 같으면 통과", fields=("response", "reference", "retrieved_contexts"))
    judge = FakeJudge(True)
    out = evaluate([ROW, {**ROW, "sample_id": "b", "reference": None}], [full], llm=judge)
    assert [r["맞음"] for r in out.rows] == [1.0, None] and len(judge.seen) == 1
    assert "기준 답" in judge.seen[0] and "(비어 있음)" in judge.seen[0]

    def both(a, b):
        save_baseline(evaluate([ROW], [a], llm=FakeJudge(True)), tmp_path / "j.json")
        return compare_baseline(evaluate([ROW], [b], llm=FakeJudge(True)), tmp_path / "j.json", strict_meta=True)["incomparable"]
    assert both(polite, judge_metric("친절함", "존댓말이면 통과")) == []
    assert both(polite, judge_metric("친절함", "반말이면 통과")) == ["metrics.친절함"]
    assert both(polite, judge_metric("친절함", "존댓말이면 통과", fields=("response",))) == ["metrics.친절함"]


@pytest.mark.parametrize("args", [{"criteria": " "}, {"criteria": "x", "fields": ("answer",)}, {"criteria": "x", "fields": ()},
                                  {"criteria": "x", "fields": ("response", "response")}])
def test_judge_metric_bad_args(args):
    with pytest.raises(ValueError):
        judge_metric("이름", **args)


@pytest.mark.skipif(not DATA.exists(), reason="수업 자료가 없는 곳")
def test_same_scores_as_classes_in_61_4():
    def squash(text):
        return re.sub(r"\s+", "", text)

    class FactHit(Metric):
        name, needs_llm = "fact_hit", False

        def score(self, sample, llm):
            facts = sample.metadata["expected_facts"]
            hits = [f for f in facts if squash(f) in squash(sample.response)]
            return MetricResult(metric=self.name, score=len(hits) / len(facts), details={"hits": hits})

    class CitedRetrieved(Metric):
        name, needs_llm = "cited_retrieved", False

        def score(self, sample, llm):
            cites = extract_citations(sample.response)
            if not cites:
                return MetricResult(metric=self.name, score=None, reason="출처 없음")
            return MetricResult(metric=self.name, score=sum(c in map(str, sample.metadata["retrieved_ids"]) for c in cites) / len(cites))

    class Answered(Metric):
        name, needs_llm = "answered", False

        def score(self, sample, llm):
            return MetricResult(metric=self.name, score=0.0 if sample.response.startswith("자료에 없") else 1.0)

    @metric
    def fact_hit(sample):
        facts = sample.metadata["expected_facts"]
        hits = [f for f in facts if squash(f) in squash(sample.response)]
        return len(hits) / len(facts), "", {"hits": hits}

    @metric
    def cited_retrieved(sample):
        cites = extract_citations(sample.response)
        if not cites:
            return None, "출처 없음"
        return sum(c in map(str, sample.metadata["retrieved_ids"]) for c in cites) / len(cites)

    @metric
    def answered(sample):
        return not sample.response.startswith("자료에 없")

    records = [json.loads(line) for line in DATA.read_text(encoding="utf-8").splitlines()]
    data = to_dataset(records, contexts="retrieved", id="sample_id", keep=["expected_facts"], source="page")
    old, new = evaluate(data, [FactHit(), CitedRetrieved(), Answered()]), evaluate(data, [fact_hit, cited_retrieved, answered])
    assert old.summary == new.summary and old.stats == new.stats
    for a, b in zip(old.rows, new.rows):
        a.pop("answered__verdict"), b.pop("answered__verdict")      # 참 / 거짓은 통과 · 실패 글이 더 남음
        assert a == b
    assert old.stats["fact_hit"]["valid"] == 40 and round(old.summary["fact_hit"], 3) == 0.298 and old.summary["answered"] == 0.825
