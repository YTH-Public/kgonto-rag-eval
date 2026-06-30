"""KG 기반 평가셋 생성 + KG/온톨로지 평가. (networkx 필요: pip install kgonto-rag-eval[kg])"""
from __future__ import annotations

from importlib import import_module

from .graph_eval import (
    check_ontology,
    compare_kg,
    non_isolated_node_ratio,
    schema_violation_rate,
)
from .io import load_kg
from .retrieval_eval import path_recall, path_triples, triple_coverage
from .traceability import source_traceability

__all__ = [
    "load_kg",
    "PersonaSpec",
    "DEFAULT_PERSONAS",
    "auto_personas_from_graph",
    "PathEvidence",
    "TestItem",
    "enumerate_paths",
    "generate_testset_from_kg",
    "validate_item",
    "to_eval_dataset",
    "save_jsonl",
    "load_jsonl",
    "compare_kg",
    "check_ontology",
    # 신규: 검색/근거/구조 결정적 지표 (L1~L2)
    "schema_violation_rate",
    "non_isolated_node_ratio",
    "path_recall",
    "triple_coverage",
    "path_triples",
    "source_traceability",
]

_LAZY_EXPORTS = {
    "PersonaSpec": ("kgonto_rag_eval.kg.personas", "PersonaSpec"),
    "DEFAULT_PERSONAS": ("kgonto_rag_eval.kg.personas", "DEFAULT_PERSONAS"),
    "auto_personas_from_graph": ("kgonto_rag_eval.kg.personas", "auto_personas_from_graph"),
    "PathEvidence": ("kgonto_rag_eval.kg.testset", "PathEvidence"),
    "TestItem": ("kgonto_rag_eval.kg.testset", "TestItem"),
    "enumerate_paths": ("kgonto_rag_eval.kg.testset", "enumerate_paths"),
    "generate_testset_from_kg": ("kgonto_rag_eval.kg.testset", "generate_testset_from_kg"),
    "validate_item": ("kgonto_rag_eval.kg.testset", "validate_item"),
    "to_eval_dataset": ("kgonto_rag_eval.kg.testset", "to_eval_dataset"),
    "save_jsonl": ("kgonto_rag_eval.kg.testset", "save_jsonl"),
    "load_jsonl": ("kgonto_rag_eval.kg.testset", "load_jsonl"),
}


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(name)
    module_name, attr = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name), attr)
    globals()[name] = value
    return value
