"""LLM-judge 평가 지표."""
from .answer_correctness import AnswerCorrectness
from .answer_relevancy import AnswerRelevancy
from .base import Metric
from .context_precision import ContextPrecision
from .context_recall import ContextRecall
from .context_relevance import ContextRelevance
from .faithfulness import Faithfulness

__all__ = [
    "Metric",
    "Faithfulness",
    "ContextRelevance",
    "ContextPrecision",
    "ContextRecall",
    "AnswerCorrectness",
    "AnswerRelevancy",
]

# 자주 쓰는 기본 묶음
CORE_METRICS = [Faithfulness(), ContextRelevance(), ContextRecall(), AnswerCorrectness()]
ALL_METRICS = [
    Faithfulness(), ContextRelevance(), ContextPrecision(),
    ContextRecall(), AnswerCorrectness(), AnswerRelevancy(),
]
