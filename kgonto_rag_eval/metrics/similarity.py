"""AnswerSimilarity: 답과 기준 답이 뜻으로 얼마나 가까운가 (임베딩 코사인 유사도)."""
from __future__ import annotations

from ..schema import EvalSample, MetricResult
from .base import Metric


class AnswerSimilarity(Metric):
    """임베딩 두 개의 코사인 유사도(음수는 0). 점수가 높아도 숫자 · 부정이 틀릴 수 있어 합격선을 두지 않습니다.

        evaluate(dataset, [AnswerSimilarity(embeddings=OpenAIEmbeddings(model="text-embedding-3-small"))])

    scikit-learn 이 필요합니다(pip install "kgonto-rag-eval[similarity]"). 문항마다 임베딩을 한 번 부릅니다(비용 있음).
    """
    name = "answer_similarity"
    needs_llm = False
    requires_reference = True

    def __init__(self, embeddings):
        if not hasattr(embeddings, "embed_documents"):
            raise TypeError("embeddings 에는 embed_documents 가 있는 임베딩 모델을 넣으세요.")
        self.embeddings = embeddings
        self.model = str(getattr(embeddings, "model", None) or getattr(embeddings, "model_name", None) or type(embeddings).__name__)
        self.dimensions = getattr(embeddings, "dimensions", None)

    def score(self, sample: EvalSample, llm=None) -> MetricResult:
        from sklearn.metrics.pairwise import cosine_similarity

        if not (sample.reference or "").strip():
            return MetricResult(metric=self.name, reason="기준 답이 없습니다.", reason_code="no_reference")
        if not sample.response.strip():
            return MetricResult(metric=self.name, score=0.0, reason="답변이 비어 있습니다.", failed_component="generator")
        answer, reference = self.embeddings.embed_documents([sample.response, sample.reference])
        if not any(answer) or not any(reference):
            raise ValueError("임베딩이 모두 0 입니다.")
        raw = float(cosine_similarity([answer], [reference])[0][0])
        return MetricResult(metric=self.name, score=min(1.0, max(0.0, raw)), details={"raw_cosine": raw, "model": self.model})
