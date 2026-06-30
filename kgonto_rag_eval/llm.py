"""기본 평가자(judge) LLM. 호출자가 직접 LLM 을 주입할 수도 있습니다."""
from __future__ import annotations

from functools import lru_cache

from langchain.chat_models import init_chat_model

DEFAULT_MODEL = "openai:gpt-5.4-mini"


@lru_cache(maxsize=4)
def default_judge(model: str = DEFAULT_MODEL):
    """기본 평가자 LLM 을 반환합니다.

    gpt-5.4-mini 는 temperature 를 지원하지 않으므로 지정하지 않습니다.
    더 강한 judge 가 필요하면 호출부에서 init_chat_model 로 만든 LLM 을 evaluate(llm=...) 로 주입하세요.
    """
    return init_chat_model(model)
