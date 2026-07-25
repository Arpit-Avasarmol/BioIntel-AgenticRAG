"""Acceptance test 2: chunking is deterministic, bounded, and provenance-preserving."""

from __future__ import annotations

from biointel.common.schemas import DocType, Document, DocumentIds, Section, SourceType
from biointel.indexing.chunker import chunk_document


def _doc(text_sections: dict[str, str]) -> Document:
    return Document(
        doc_id="pubmed:999",
        source=SourceType.PUBMED,
        doc_type=DocType.PAPER,
        title="A study of IL-23 inhibition in Crohn disease",
        abstract="Background abstract sentence one. Abstract sentence two.",
        sections=[
            Section(name=name, text=body, order=i + 1)
            for i, (name, body) in enumerate(text_sections.items())
        ],
        ids=DocumentIds(pmid="999"),
        source_url="https://pubmed.ncbi.nlm.nih.gov/999/",
        license="public-domain",
    )


def test_chunks_respect_token_bounds():
    long_body = " ".join(f"sentence number {i} about interleukin signaling." for i in range(200))
    doc = _doc({"Methods": long_body})
    chunks = chunk_document(doc, max_tokens=128, overlap=16, min_tokens=16)
    assert chunks, "expected chunks"
    for c in chunks:
        assert c.token_count <= 128 + 16, f"chunk exceeds bound: {c.token_count}"
        assert c.token_count >= 1


def test_chunk_ids_deterministic_and_unique():
    body = " ".join(f"token{i}" for i in range(300))
    doc = _doc({"Results": body})
    a = chunk_document(doc, max_tokens=64, overlap=8, min_tokens=8)
    b = chunk_document(doc, max_tokens=64, overlap=8, min_tokens=8)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b], "chunking must be deterministic"
    ids = [c.chunk_id for c in a]
    assert len(ids) == len(set(ids)), "chunk_ids must be unique within a document"


def test_provenance_preserved_on_every_chunk():
    doc = _doc({"Discussion": " ".join(["word"] * 100)})
    chunks = chunk_document(doc, max_tokens=64, overlap=8, min_tokens=8)
    for c in chunks:
        assert c.doc_id == doc.doc_id
        assert c.source == doc.source
        assert c.doc_type == doc.doc_type
        assert c.ids.pmid == "999"
        assert c.source_url == doc.source_url
        assert c.license == "public-domain"
        assert c.title == doc.title


def test_short_sections_below_min_tokens_dropped():
    doc = _doc({"Tiny": "three word section"})
    # min_tokens high enough to drop the tiny section but title/abstract remain
    chunks = chunk_document(doc, max_tokens=64, overlap=8, min_tokens=16)
    sections = {c.section for c in chunks}
    assert "Tiny" not in sections


def test_make_id_stable_formula():
    from biointel.common.schemas import Chunk

    a = Chunk.make_id("pubmed:1", "Methods", 0)
    b = Chunk.make_id("pubmed:1", "Methods", 0)
    c = Chunk.make_id("pubmed:1", "Methods", 1)
    assert a == b and a != c
    assert len(a) == 40  # sha1 hexdigest
