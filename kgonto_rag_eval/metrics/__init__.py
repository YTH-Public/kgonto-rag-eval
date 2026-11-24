"""LLM-judge 평가 지표."""
from .answer_correctness import AnswerCorrectness
from .answer_relevancy import AnswerRelevancy
from .answered_relevancy import AnsweredRelevancy
from .base import Metric
from .code import CitedRetrieved, FactHit
from .context_precision import ContextPrecision
from .context_recall import ContextRecall
from .context_relevance import ContextRelevance
from .fact_coverage import FactCoverage
from .faithfulness import Faithfulness
from .function import judge_metric, metric
from .similarity import AnswerSimilarity

__all__ = [
    "Metric",
    "Faithfulness",
    "ContextRelevance",
    "ContextPrecision",
    "ContextRecall",
    "AnswerCorrectness",
    "AnswerRelevancy",
    # 0.3: 필요할 때 골라 쓰는 지표 (CORE · ALL 에는 넣지 않음)
    "AnsweredRelevancy",
    "FactCoverage",
    "FactHit",
    "CitedRetrieved",
    "AnswerSimilarity",
    "metric",
    "judge_metric",
]

# 자주 쓰는 기본 묶음
CORE_METRICS = [Faithfulness(), ContextRelevance(), ContextRecall(), AnswerCorrectness()]
ALL_METRICS = [
    Faithfulness(), ContextRelevance(), ContextPrecision(),
    ContextRecall(), AnswerCorrectness(), AnswerRelevancy(),
]
