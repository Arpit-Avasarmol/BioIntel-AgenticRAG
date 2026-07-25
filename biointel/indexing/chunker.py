"""Structure-aware chunking: Document -> list[Chunk] with full provenance.

Strategy:
- Treat title+abstract and each Section as logical units.
- Split each unit into token-bounded windows with overlap (so retrieval returns
  coherent passages, not arbitrary fragments).
- Every chunk carries a deterministic ID derived from (doc_id, section, index),
  so re-indexing is idempotent and citations are stable.
"""

from __future__ import annotations

from biointel.common.schemas import Chunk, Document

# tiktoken encoder reused via common.text
from biointel.common.text import (
    _encoder,  # noqa: PLC2701 (intentional reuse)
    clean_text,
    count_tokens,
)


def _split_by_tokens(text: str, max_tokens: int, overlap: int) -> list[str]:
    """Split text into windows of <= max_tokens with `overlap` token overlap."""
    enc = _encoder()
    ids = enc.encode(text)
    if len(ids) <= max_tokens:
        return [text]
    windows: list[str] = []
    step = max(1, max_tokens - overlap)
    for start in range(0, len(ids), step):
        window_ids = ids[start : start + max_tokens]
        if not window_ids:
            break
        windows.append(enc.decode(window_ids))
        if start + max_tokens >= len(ids):
            break
    return windows


def chunk_document(
    doc: Document,
    max_tokens: int = 512,
    overlap: int = 64,
    min_tokens: int = 16,
) -> list[Chunk]:
    """Chunk a Document into retrievable units.

    ``max_tokens`` bounds each chunk; ``overlap`` preserves context across
    window boundaries; ``min_tokens`` drops trivially short fragments.
    """
    units: list[tuple[str, str]] = []  # (section_name, text)

    # Title + abstract form the lead unit (most information-dense).
    lead_parts = [p for p in (doc.title, doc.abstract) if p and p.strip()]
    if lead_parts:
        units.append(("Summary", "\n\n".join(lead_parts)))

    # Each section is its own unit (already section-aware from connectors).
    for sec in sorted(doc.sections, key=lambda s: s.order):
        txt = clean_text(sec.text)
        if txt:
            units.append((sec.name or "Body", txt))

    chunks: list[Chunk] = []
    index = 0
    for section_name, text in units:
        for window in _split_by_tokens(text, max_tokens, overlap):
            tok = count_tokens(window)
            if tok < min_tokens:
                continue
            chunk = Chunk(
                chunk_id=Chunk.make_id(doc.doc_id, section_name, index),
                doc_id=doc.doc_id,
                source=doc.source,
                doc_type=doc.doc_type,
                title=doc.title,
                text=window,
                section=section_name,
                chunk_index=index,
                source_url=doc.source_url,
                license=doc.license,
                ids=doc.ids,
                date=doc.date,
                token_count=tok,
                extra={
                    k: doc.extra.get(k)
                    for k in (
                        "phases",
                        "status",
                        "lead_sponsor",
                        "target_symbol",
                        "association_score",
                        "assignees",
                        "journal",
                    )
                    if k in doc.extra
                },
            )
            chunks.append(chunk)
            index += 1
    return chunks


def chunk_document_from_settings(doc: Document) -> list[Chunk]:
    """Convenience wrapper (kept simple; token params are stable defaults)."""
    return chunk_document(doc)
