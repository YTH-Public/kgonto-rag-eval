"""근거 추적성(provenance) 평가 (결정적, LLM 불필요).

source_traceability: KG 사실 또는 검색 결과가 원문 출처(page/chunk 등) 칸을 갖고 있는 비율.
"출처 칸이 있는가" 만 봅니다. 그 출처가 실제로 주장을 뒷받침하는지는 citation.citation_support 로 봅니다.

근거: ALCE (Gao et al. 2023, arXiv:2305.14627) 의 attribution 아이디어에서 "출처가 붙었는가" 부분만 떼어 낸 것입니다.
"""
from __future__ import annotations

from typing import Any, Iterable

# 출처로 인정하는 메타데이터 키 (하나라도 값이 있으면 추적 가능으로 봅니다)
_SOURCE_KEYS = ("provenance", "source_id", "source", "source_page", "page", "chunk_id", "chunk",
                "doc_id", "url", "block_id", "rcept_no")


def _is_relation(d: dict) -> bool:
    return "relation" in d and ("target" in d or "object" in d)


def _has_source(item: Any) -> bool:
    if hasattr(item, "metadata") and isinstance(getattr(item, "metadata"), dict):
        meta = item.metadata
    elif isinstance(item, dict):
        meta = item.get("metadata", item)
    else:
        return False
    keys = _SOURCE_KEYS
    if isinstance(meta, dict) and _is_relation(meta):
        keys = tuple(k for k in keys if k != "source")   # 관계의 source 는 주어이지 문헌 출처가 아님
    return any(k in meta and meta[k] not in (None, "") for k in keys)


def source_traceability(items: Iterable[Any]) -> dict:
    """검색/KG 항목 중 원문 출처 칸(page/chunk/provenance 등)을 가진 비율.

    items: langchain Document 리스트(metadata) 또는 dict 리스트(metadata 포함 또는 키 직접 보유).
    관계 dict({source, relation, target})의 source 는 주어로 보고 출처로 세지 않습니다.
    """
    items = list(items)
    traced = [it for it in items if _has_source(it)]
    rate = len(traced) / len(items) if items else 0.0
    return {
        "traceability_rate": round(rate, 3),
        "traced": len(traced),
        "total": len(items),
        "untraced": [str(getattr(it, "page_content", it))[:40] for it in items if not _has_source(it)],
    }
