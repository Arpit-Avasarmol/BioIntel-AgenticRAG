"""Acceptance tests 3 & 4: RRF fusion + reranking, and vector-store interface.

RRF is validated against hand-computed values; the reranker's graceful-degradation
path and the hybrid orchestration are validated with fakes; the store payload
round-trip and point-id stability validate the pluggable ``VectorStore`` contract.
"""

from __future__ import annotations

import pytest

from biointel.common.schemas import Chunk, DocType, DocumentIds, RetrievedChunk, SourceType
from biointel.retrieval.base import chunk_to_payload, payload_to_chunk
from biointel.retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion
from biointel.retrieval.reranker import Reranker


def _rc(cid, dense=None, sparse=None, fused=None):
    c = Chunk(
        chunk_id=cid,
        doc_id="d:" + cid,
        source=SourceType.PUBMED,
        doc_type=DocType.PAPER,
        text="text " + cid,
    )
    return RetrievedChunk(chunk=c, dense_score=dense, sparse_score=sparse, fused_score=fused)


# ------------------------------------------------------------------ RRF fusion
def test_rrf_scores_match_hand_computation():
    k = 60
    dense = [_rc("A", dense=0.9), _rc("B", dense=0.8), _rc("C", dense=0.7)]
    sparse = [_rc("B", sparse=5.1), _rc("D", sparse=4.0), _rc("A", sparse=3.2)]
    fused = reciprocal_rank_fusion([dense, sparse], k=k)
    got = {rc.chunk.chunk_id: rc.fused_score for rc in fused}
    expected = {
        "A": 1 / (k + 1) + 1 / (k + 3),
        "B": 1 / (k + 2) + 1 / (k + 1),
        "C": 1 / (k + 3),
        "D": 1 / (k + 2),
    }
    assert set(got) == set(expected)
    for cid, exp in expected.items():
        assert abs(got[cid] - exp) < 1e-9


def test_rrf_orders_and_ranks():
    dense = [_rc("A", dense=0.9), _rc("B", dense=0.8), _rc("C", dense=0.7)]
    sparse = [_rc("B", sparse=5.1), _rc("D", sparse=4.0), _rc("A", sparse=3.2)]
    fused = reciprocal_rank_fusion([dense, sparse], k=60)
    assert [rc.chunk.chunk_id for rc in fused] == ["B", "A", "D", "C"]
    assert [rc.rank for rc in fused] == [1, 2, 3, 4]


def test_rrf_merges_per_modality_scores():
    dense = [_rc("A", dense=0.9)]
    sparse = [_rc("A", sparse=3.2)]
    fused = reciprocal_rank_fusion([dense, sparse], k=60)
    a = fused[0]
    assert a.dense_score == 0.9 and a.sparse_score == 3.2


def test_rrf_weighted_and_edgecases():
    dense = [_rc("A", dense=0.9), _rc("B", dense=0.8)]
    sparse = [_rc("B", sparse=5.1), _rc("A", sparse=3.2)]
    fw = reciprocal_rank_fusion([dense, sparse], k=60, weights=[1.0, 2.0])
    gw = {rc.chunk.chunk_id: rc.fused_score for rc in fw}
    assert abs(gw["B"] - (1 / 62 + 2 * (1 / 61))) < 1e-9
    assert reciprocal_rank_fusion([[], []], k=60) == []
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([dense], k=60, weights=[1.0, 2.0])


# -------------------------------------------------------------------- reranker
def test_reranker_degrades_gracefully_without_model(monkeypatch):
    """Without a loadable cross-encoder, reranker preserves fused order + fills scores."""
    monkeypatch.setattr("biointel.retrieval.reranker._load_reranker", lambda: None)
    rr = Reranker()
    cands = [_rc("A", fused=0.01), _rc("B", fused=0.03), _rc("C", fused=0.02)]
    out = rr.rerank("q", cands, top_k=2)
    assert [c.chunk.chunk_id for c in out] == ["B", "C"]
    assert all(c.rerank_score is not None for c in out)
    assert [c.rank for c in out] == [1, 2]
    assert rr.rerank("q", []) == []


# ------------------------------------------------------- hybrid orchestration
class _FakeVS:
    def search(self, qv, k, filters=None):
        return [_rc("A", dense=0.9), _rc("B", dense=0.8), _rc("C", dense=0.7)][:k]


class _FakeKS:
    def search(self, q, k, filters=None):
        return [_rc("B", sparse=5.1), _rc("D", sparse=4.0), _rc("A", sparse=3.2)][:k]


class _FakeEmb:
    def encode_query(self, q):
        return [0.1, 0.2, 0.3]


class _FakeRR:
    def rerank(self, q, cands, top_k=None):
        order = {"A": 3.0, "B": 2.0, "D": 1.0, "C": 0.5}
        for c in cands:
            c.rerank_score = order.get(c.chunk.chunk_id, 0.0)
        ranked = sorted(cands, key=lambda c: c.rerank_score, reverse=True)[: (top_k or len(cands))]
        for i, c in enumerate(ranked):
            c.rank = i + 1
        return ranked


def test_hybrid_end_to_end_with_fakes():
    hr = HybridRetriever(
        vector_store=_FakeVS(), keyword_store=_FakeKS(), embedder=_FakeEmb(), reranker=_FakeRR()
    )
    res = hr.retrieve("q", top_k=3)
    assert [c.chunk.chunk_id for c in res] == ["A", "B", "D"]
    assert all(c.fused_score is not None and c.rerank_score is not None for c in res)


def test_hybrid_rerank_false_returns_rrf_order():
    hr = HybridRetriever(
        vector_store=_FakeVS(), keyword_store=_FakeKS(), embedder=_FakeEmb(), reranker=_FakeRR()
    )
    res = hr.retrieve("q", top_k=2, rerank=False)
    assert [c.chunk.chunk_id for c in res] == ["B", "A"]


def test_hybrid_survives_one_engine_failure():
    class BrokenKS:
        def search(self, q, k, filters=None):
            raise RuntimeError("opensearch down")

    hr = HybridRetriever(
        vector_store=_FakeVS(), keyword_store=BrokenKS(), embedder=_FakeEmb(), reranker=_FakeRR()
    )
    res = hr.retrieve("q", top_k=3)
    assert len(res) == 3
    assert {c.chunk.chunk_id for c in res} <= {"A", "B", "C"}


# ----------------------------------------------------- vector-store interface
def test_payload_round_trip_preserves_chunk():
    c = Chunk(
        chunk_id="abc123",
        doc_id="pubmed:1",
        source=SourceType.PUBMED,
        doc_type=DocType.PAPER,
        title="T",
        text="body text",
        section="Methods",
        chunk_index=2,
        source_url="http://x/1",
        license="public-domain",
        ids=DocumentIds(pmid="1", doi="10.1/x"),
        token_count=42,
    )
    payload = chunk_to_payload(c)
    back = payload_to_chunk(payload)
    assert back.chunk_id == c.chunk_id
    assert back.doc_id == c.doc_id
    assert back.source == c.source
    assert back.doc_type == c.doc_type
    assert back.section == c.section
    assert back.chunk_index == c.chunk_index
    assert back.ids.pmid == "1" and back.ids.doi == "10.1/x"
    assert back.token_count == 42


def test_qdrant_point_id_is_stable_uuid():
    """The pluggable store must map chunk_id -> a stable point id deterministically."""
    from biointel.retrieval.qdrant_store import _point_id

    pid1 = _point_id("chunk-xyz")
    pid2 = _point_id("chunk-xyz")
    pid3 = _point_id("chunk-abc")
    assert pid1 == pid2 and pid1 != pid3
