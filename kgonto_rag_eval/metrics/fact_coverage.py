"""FactCoverage: 사람이 정한 필수사실 가운데 답에 들어 있는 비율. 무엇이 빠졌는지 · 어긋났는지를 남깁니다.

필수사실은 sample.metadata["expected_facts"] 에 문자열 목록으로 넣습니다.
사실 하나에는 대상과 조건을 함께 적으세요. 예: "수급 가구의 19세 이상 30세 미만 미혼 자녀가 대상".
필수사실 밖의 틀린 내용은 보지 않으므로 AnswerCorrectness · Faithfulness 와 함께 읽습니다.
"""
from __future__ import annotations

import re
from typing import Literal, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from ..schema import EvalSample, MetricResult
from .base import Metric


class _FactVerdict(BaseModel):
    index: int = Field(description="필수사실 번호(1부터)")
    reason: str = Field(description="한국어 한 문장, 판단 이유를 먼저")
    label: Literal["present", "missing", "contradicted"]
    quotes: list[str] = Field(default_factory=list, description="present · contradicted 이면 답에서 그대로 옮긴 짧은 문구(고치지 말 것)")


class _FactVerdicts(BaseModel):
    items: list[_FactVerdict] = Field(description="필수사실마다 하나씩")


_SYSTEM = """당신은 필수사실 채점자입니다. 필수사실마다 답에 그 내용이 있는지 판정합니다.
- present: 답이 그 사실을 모두 말합니다. 표현 · 순서 · 숫자 표기가 달라도 같은 뜻이면 present.
- missing: 답에 없거나 일부만 있습니다.
- contradicted: 답이 그 사실과 다른 값 · 대상 · 조건을 말합니다.
- 질문이 묻는 바로 그 대상에 대한 내용이어야 합니다. 다른 제도 · 다른 대상의 같은 금액이나 비슷한 나이는 present 가 아닙니다.
- 필수사실에 질문이 이미 정해 준 조건이 섞여 있으면, 답이 그 조건을 전제로 나머지를 말했을 때 present 입니다. 질문 조건만으로 이루어진 사실도 답이 그 대상에 대해 답했으면 present 입니다.
- 답이 질문에 답하지 않았으면(자료에 없다고 함) present 로 보지 않습니다.
- quotes 에는 판단에 쓴 답의 문구를 글자 그대로 옮깁니다. 필수사실의 문구를 옮기지 마세요.
- 모든 번호(1..N)를 한 번씩 판정합니다."""


def _squash(text: str) -> str:
    """공백 · 문장부호 · 마크다운 표시를 뺀 글자 - 문구가 답에 실제로 있는지 볼 때 씀"""
    return re.sub(r"[^\w%]", "", text)


def read_facts(sample: EvalSample) -> tuple[list[str], Optional[str]]:
    """metadata 의 필수사실을 읽음 -> (목록, 문제). 문제가 있으면 채점하지 않음."""
    if "expected_facts" not in sample.metadata:
        return [], "metadata['expected_facts'] 가 없습니다."
    facts = sample.metadata["expected_facts"]
    if not isinstance(facts, (list, tuple)) or not all(isinstance(f, str) and f.strip() for f in facts):
        return [], "expected_facts 는 비어 있지 않은 문자열의 목록이어야 합니다."
    if len({f.strip() for f in facts}) != len(facts):
        return [], "expected_facts 에 같은 사실이 두 번 있습니다."
    return list(facts), None


def facts_problem(metric: str, problem: Optional[str]) -> MetricResult:
    """필수사실을 채점에 쓸 수 없을 때의 결과 - 잘못 넣었으면 오류, 비었으면 채점 대상 아님"""
    if problem:
        return MetricResult(metric=metric, reason=problem, error=problem, failed_component="input", reason_code="invalid_input")
    return MetricResult(metric=metric, reason="필수사실이 없습니다.", failed_component="reference", reason_code="no_required_facts")


class FactCoverage(Metric):
    name = "fact_coverage"

    def score(self, sample: EvalSample, llm) -> MetricResult:
        facts, problem = read_facts(sample)
        if problem or not facts:
            return facts_problem(self.name, problem)
        if not sample.response.strip():
            return self._result([{"fact": f, "label": "missing", "quotes": [], "reason": "답변이 비어 있습니다."} for f in facts])
        numbered = "\n".join(f"{i}. {f}" for i, f in enumerate(facts, 1))
        out = llm.with_structured_output(_FactVerdicts).invoke([
            SystemMessage(content=_SYSTEM),
            HumanMessage(content=f"질문: {sample.user_input}\n\n답변:\n{sample.response}\n\n필수사실:\n{numbered}"),
        ])
        verdicts = sorted(out.items, key=lambda v: v.index)
        if [v.index for v in verdicts] != list(range(1, len(facts) + 1)):
            return self.judge_error(f"필수사실 번호가 1..{len(facts)} 과 맞지 않습니다: {[v.index for v in verdicts]}",
                                    raw=out.model_dump())
        answer = _squash(sample.response)
        items = []
        for fact, v in zip(facts, verdicts):
            quotes = [q for q in v.quotes if _squash(q)]
            if v.label == "present" and (not quotes or any(_squash(q) not in answer for q in quotes)):
                return self.judge_error(f"present 인데 근거 문구가 답에 없습니다: {fact} / {v.quotes}", raw=out.model_dump())
            items.append({"fact": fact, "label": v.label, "quotes": quotes, "reason": v.reason})
        return self._result(items)

    def _result(self, items: list[dict]) -> MetricResult:
        """사실별 판정 -> 점수 · 빠진 사실 · 어긋난 사실"""
        present = [i for i in items if i["label"] == "present"]
        missing = [i["fact"] for i in items if i["label"] == "missing"]
        contradicted = [i["fact"] for i in items if i["label"] == "contradicted"]
        score = len(present) / len(items)
        verdict = MetricResult.verdict_from_score(score)
        parts = [f"있음 {len(present)} / {len(items)}"]
        if missing:
            parts.append("빠짐: " + ", ".join(missing))
        if contradicted:
            parts.append("어긋남: " + ", ".join(contradicted))
        return MetricResult(metric=self.name, score=score, verdict=verdict, reason=" | ".join(parts),
                            failed_component="generator" if verdict == "fail" else None,
                            details={"items": items, "missing": missing, "contradicted": contradicted})
