"""근거 추적성(provenance) 평가 (결정적, LLM 불필요).

source_traceability: KG 사실 또는 검색 결과가 원문 출처(page/chunk/source 등)로
되짚어갈 수 있는 비율. "근거가 원문으로 돌아가는가" 를 봅니다.

근거: ALCE (Gao et al. 2023, "Enabling LLMs to Generate Text with Citations",
arXiv:2305.14627) 의 attribution/citation-support 아이디어를, NLI 판정 대신
출처 메타데이터 유무로 본 결정적 축소판입니다.
"""
from __future__ import annotations

from typing import Any, Iterable

# 출처로 인정하는 메타데이터 키 (하나라도 값이 있으면 추적 가능으로 봅니다)
_SOURCE_KEYS = ("source_id", "source", "source_page", "page", "chunk_id", "chunk", "doc_id", "url")


def _has_source(item: Any) -> bool:
    if hasattr(item, "metadata") and isinstance(getattr(item, "metadata"), dict):
        meta = item.metadata
    elif isinstance(item, dict):
        meta = item.get("metadata", item)
    else:
        return False
    return any(k in meta and meta[k] not in (None, "") for k in _SOURCE_KEYS)


def source_traceability(items: Iterable[Any]) -> dict:
    """검색/KG 항목 중 원문 출처(page/chunk/source)를 가진 비율.

    items: langchain Document 리스트(metadata) 또는 dict 리스트(metadata 포함 또는 키 직접 보유).
    근거: ALCE (Gao et al. 2023, arXiv:2305.14627) 의 attribution 을 KG/검색 provenance 로 축소 적용.
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
