"""모델을 부르지 않는 코드 지표. 실행마다 같은 값이라 회귀 테스트에 씁니다."""
from __future__ import annotations

import re

from ..citation import broken_ranges, extract_citations
from ..schema import EvalSample, MetricResult
from .base import Metric
from .fact_coverage import facts_problem, read_facts


class FactHit(Metric):
    """필수사실이 답에 **글자 그대로**(공백 무시) 있는 비율. 표현을 바꿔 쓰면 놓치므로 정답률이 아닙니다."""
    name = "fact_hit"
    needs_llm = False

    def score(self, sample: EvalSample, llm=None) -> MetricResult:
        facts, problem = read_facts(sample)
        if problem or not facts:
            return facts_problem(self.name, problem)
        answer = re.sub(r"\s+", "", sample.response)
        hits = [f for f in facts if re.sub(r"\s+", "", f) in answer]
        score = len(hits) / len(facts)
        return MetricResult(metric=self.name, score=score, verdict=MetricResult.verdict_from_score(score),
                            details={"hits": hits, "missing": [f for f in facts if f not in hits]})


class CitedRetrieved(Metric):
    """답에 적은 출처 가운데 그때 실제로 검색된 자료의 비율. 출처가 없으면 점수를 내지 않습니다.

    검색된 자료 ID 는 sample.metadata[ids_key] 에 목록으로 넣습니다(숫자 · 문자열 ID 를 같은 것으로 봄).
    출처 범위([p.1-3])는 펼쳐서 셉니다. 같은 출처를 두 번 적으면 두 번 셉니다.
    """
    name = "cited_retrieved"
    needs_llm = False

    def __init__(self, ids_key: str = "retrieved_ids"):
        self.ids_key = ids_key

    def score(self, sample: EvalSample, llm=None) -> MetricResult:
        ids = sample.metadata.get(self.ids_key)
        if not isinstance(ids, (list, tuple, set, frozenset)):
            problem = f"metadata['{self.ids_key}'] 에 검색된 자료 ID 목록이 없습니다."
            return MetricResult(metric=self.name, reason=problem, error=problem, failed_component="input",
                                reason_code="invalid_input")
        broken = broken_ranges(sample.response)               # 잘못 적은 범위도 출처로 세고, 맞지 않은 것으로 봄
        cites = extract_citations(sample.response) + broken
        if not cites:
            return MetricResult(metric=self.name, reason="답에 출처가 없습니다.", reason_code="no_citation")
        retrieved = {str(i) for i in ids}
        outside = [c for c in cites if c not in retrieved]
        score = 1 - len(outside) / len(cites)
        return MetricResult(metric=self.name, score=score, verdict=MetricResult.verdict_from_score(score),
                            reason=f"검색되지 않은 출처: {outside}" if outside else "모두 검색된 자료",
                            failed_component="generator" if outside else None,
                            details={"citations": cites, "outside": outside, "broken": broken})
