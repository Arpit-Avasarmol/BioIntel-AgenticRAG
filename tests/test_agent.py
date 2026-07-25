"""Acceptance tests 5 & 6: citation verification, contradiction detection, agent flow.

These are the audit-critical guardrails. Citation verification is a pure,
deterministic, offline check; the agent flow is exercised end-to-end with a fake
LLM and fake retriever (the ``linear`` runner, identical to the LangGraph path).
"""

from __future__ import annotations

from biointel.agent.graph import run_agent
from biointel.agent.nodes import AgentDeps, contradiction_node
from biointel.agent.structures import (
    ContradictionItem,
    ContradictionReport,
    QueryPlan,
    SubQuestion,
    TrialRecord,
)
from biointel.agent.verify import overlap_ratio, parse_markers, verify_answer
from biointel.common.schemas import (
    Chunk,
    Contradiction,
    DocType,
    DocumentIds,
    RetrievedChunk,
    SourceType,
)


def _rc(cid, text, **ids):
    c = Chunk(
        chunk_id=cid,
        doc_id="doc:" + cid,
        source=SourceType.PUBMED,
        doc_type=DocType.PAPER,
        text=text,
        ids=DocumentIds(**ids),
        source_url="http://x/" + cid,
        title="T" + cid,
    )
    return RetrievedChunk(chunk=c)


SOURCES = [
    _rc(
        "c1",
        "Ustekinumab is a monoclonal antibody IL-23 inhibitor approved for Crohn disease.",
        pmid="111",
    ),
    _rc(
        "c2",
        "Risankizumab selectively inhibits IL-23 and improved endoscopic "
        "remission in Crohn disease.",
        pmid="222",
    ),
]


# ------------------------------------------------------ citation verification
def test_marker_parsing_and_overlap():
    assert parse_markers("A [1] B [2][3].") == [1, 2, 3]
    good = "Ustekinumab is an IL-23 inhibitor approved for Crohn disease [1]."
    assert overlap_ratio(good, SOURCES[0].chunk.text) > 0.6
    assert overlap_ratio(good, "totally unrelated weather text") < 0.2


def test_fully_supported_answer_verifies():
    ans = (
        "Ustekinumab is an IL-23 inhibitor approved for Crohn disease [1]. "
        "Risankizumab improved endoscopic remission in Crohn disease [2]."
    )
    r = verify_answer(ans, SOURCES)
    assert r.verified is True
    assert not r.unsupported_sentences
    assert {c.marker for c in r.citations} == {"[1]", "[2]"}
    assert all(c.quote for c in r.citations)


def test_uncited_sentence_flagged():
    ans = (
        "Ustekinumab is an IL-23 inhibitor approved for Crohn disease [1]. "
        "This drug also cures every other disease completely."
    )
    r = verify_answer(ans, SOURCES)
    assert r.verified is False
    assert any("cures every other disease" in s for s in r.unsupported_sentences)


def test_citation_on_unrelated_claim_fails():
    ans = "The mitochondria is the powerhouse of the cell and makes ATP efficiently [1]."
    r = verify_answer(ans, SOURCES)
    assert r.verified is False


def test_out_of_range_citation_caught():
    ans = "Ustekinumab is an IL-23 inhibitor approved for Crohn disease [9]."
    r = verify_answer(ans, SOURCES)
    assert r.verified is False
    assert any("out of range" in w for w in r.warnings)


def test_non_substantive_sentence_exempt():
    ans = "In summary: Ustekinumab is an IL-23 inhibitor approved for Crohn disease [1]."
    r = verify_answer(ans, SOURCES)
    assert r.verified is True


# --------------------------------------------------------- contradiction node
def test_contradiction_node_maps_sources_to_labels():
    class LLM:
        model = "fake"

        def generate_json(self, messages, schema, **kw):
            return ContradictionReport(
                contradictions=[
                    ContradictionItem(
                        statement_a="Drug X improved remission.",
                        statement_b="Drug X showed no benefit.",
                        source_a="1",
                        source_b="2",
                        explanation="Opposite efficacy conclusions.",
                    )
                ]
            )

    deps = AgentDeps(llm=LLM(), retriever=None)
    out = contradiction_node({"retrieved": SOURCES}, deps)
    cons = out["contradictions"]
    assert len(cons) == 1
    assert isinstance(cons[0], Contradiction)
    # '1' -> label of SOURCES[0]
    assert "pmid=111" in cons[0].source_a
    assert "pmid=222" in cons[0].source_b


def test_contradiction_node_no_conflict_returns_empty():
    class LLM:
        model = "fake"

        def generate_json(self, messages, schema, **kw):
            return ContradictionReport(contradictions=[])

    deps = AgentDeps(llm=LLM(), retriever=None)
    out = contradiction_node({"retrieved": SOURCES}, deps)
    assert out["contradictions"] == []


# --------------------------------------------------------------- full agent
class _AgentLLM:
    model = "fake-llm"

    def __init__(self, chat_seq):
        self._chat = list(chat_seq)
        self.n_chat = 0

    def generate_json(self, messages, schema, **kw):
        if schema is QueryPlan:
            return QueryPlan(sub_questions=[SubQuestion(question="q1"), SubQuestion(question="q2")])
        if schema is ContradictionReport:
            return ContradictionReport(contradictions=[])
        if schema is TrialRecord:
            return TrialRecord(nct_id="NCT1")
        return schema()

    def chat(self, messages, **kw):
        self.n_chat += 1
        return self._chat[min(self.n_chat - 1, len(self._chat) - 1)]


class _AgentRetriever:
    def retrieve(self, q, top_k=None, filters=None):
        for i, r in enumerate(SOURCES):
            r.rerank_score = 1.0 - i * 0.1
        return SOURCES


def test_agent_happy_path_verified():
    good = (
        "Ustekinumab is an IL-23 inhibitor approved for Crohn disease [1]. "
        "Risankizumab improved endoscopic remission in Crohn disease [2]."
    )
    deps = AgentDeps(llm=_AgentLLM([good]), retriever=_AgentRetriever())
    ans = run_agent("Which IL-23 inhibitors work in Crohn disease?", deps=deps)
    assert ans.verified is True
    assert len(ans.citations) == 2
    assert ans.sub_questions == ["q1", "q2"]
    assert ans.used_chunks == ["c1", "c2"]
    assert ans.contradictions == []


def test_agent_regenerates_once_then_verifies():
    bad = (
        "Ustekinumab is an IL-23 inhibitor approved for Crohn disease [1]. "
        "It also cures all cancers instantly."
    )
    good = (
        "Ustekinumab is an IL-23 inhibitor approved for Crohn disease [1]. "
        "Risankizumab improved endoscopic remission in Crohn disease [2]."
    )
    llm = _AgentLLM([bad, good])
    deps = AgentDeps(llm=llm, retriever=_AgentRetriever())
    ans = run_agent("q", deps=deps)
    assert llm.n_chat == 2, "should regenerate exactly once"
    assert ans.verified is True


def test_agent_no_results_is_honest():
    class Empty:
        def retrieve(self, q, top_k=None, filters=None):
            return []

    deps = AgentDeps(llm=_AgentLLM(["unused"]), retriever=Empty())
    ans = run_agent("obscure", deps=deps)
    assert "could not find" in ans.answer.lower()
    assert ans.used_chunks == []
    assert ans.verified is False
