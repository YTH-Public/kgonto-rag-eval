"""기본 평가자(judge) LLM. 호출자가 직접 LLM 을 주입하는 것을 권장합니다."""
from __future__ import annotations

from functools import lru_cache

from langchain.chat_models import init_chat_model

DEFAULT_MODEL = "openai:gpt-6-luna"


@lru_cache(maxsize=4)
def default_judge(model: str = DEFAULT_MODEL):
    """기본 평가자 LLM 을 반환합니다.

    gpt-6 계열은 temperature 를 받지 않으므로 지정하지 않습니다.
    채점 품질이 중요하면 더 강한 모델(예: gpt-6.1-sol, use_responses_api=True)을
    init_chat_model 로 만들어 evaluate(llm=...) 로 넣고, 일부 표본은 사람 판정과 맞춰 보세요.
    """
    return init_chat_model(model, reasoning_effort="none")
