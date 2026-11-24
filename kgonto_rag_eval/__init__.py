"""kgonto-rag-eval: KG·온톨로지 친화 RAG 평가 프레임워크 (LangChain v1)."""
from __future__ import annotations

from importlib import import_module

__all__ = [
    "EvalSample",
    "EvaluationDataset",
    "MetricResult",
    "EvalReport",
    "evaluate",
    "to_dataset",
    "default_judge",
    "metrics",
    "pairwise_judge",
    "save_baseline",
    "compare_baseline",
    "citation_support",
    "split_cited_sentences",
    "resolve_citations",
    "quote_in_source",
    "extract_citations",
    "diagnose",
]
__version__ = "0.3.0"

_LAZY_EXPORTS = {
    "EvalSample": ("kgonto_rag_eval.schema", "EvalSample"),
    "EvaluationDataset": ("kgonto_rag_eval.schema", "EvaluationDataset"),
    "MetricResult": ("kgonto_rag_eval.schema", "MetricResult"),
    "EvalReport": ("kgonto_rag_eval.schema", "EvalReport"),
    "to_dataset": ("kgonto_rag_eval.inputs", "to_dataset"),
    "evaluate": ("kgonto_rag_eval.evaluate", "evaluate"),
    "default_judge": ("kgonto_rag_eval.llm", "default_judge"),
    "metrics": ("kgonto_rag_eval.metrics", None),
    "pairwise_judge": ("kgonto_rag_eval.judge", "pairwise_judge"),
    "save_baseline": ("kgonto_rag_eval.baseline", "save_baseline"),
    "compare_baseline": ("kgonto_rag_eval.baseline", "compare_baseline"),
    "citation_support": ("kgonto_rag_eval.citation", "citation_support"),
    "split_cited_sentences": ("kgonto_rag_eval.citation", "split_cited_sentences"),
    "resolve_citations": ("kgonto_rag_eval.citation", "resolve_citations"),
    "quote_in_source": ("kgonto_rag_eval.citation", "quote_in_source"),
    "extract_citations": ("kgonto_rag_eval.citation", "extract_citations"),
    "diagnose": ("kgonto_rag_eval.diagnose", "diagnose"),
}


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(name)
    module_name, attr = _LAZY_EXPORTS[name]
    module = import_module(module_name)
    value = module if attr is None else getattr(module, attr)
    globals()[name] = value
    return value
