"""GraphRAG 평가: 구축(build) -> 검색(search) -> 한 표(report). 모델 · DB 연결 없음, 표를 넣고 표를 받습니다.

    pip install "kgonto-rag-eval[graphrag]"
"""
from .build import break_graph, check_build, check_load, why_missed
from .report import stage_table
from .search import compare_result_rows, entity_link_recall, evaluate_cypher, evidence_bundle_coverage, write_word

__all__ = [
    "check_build", "why_missed", "check_load", "break_graph",
    "entity_link_recall", "compare_result_rows", "evidence_bundle_coverage", "evaluate_cypher", "write_word",
    "stage_table",
]
