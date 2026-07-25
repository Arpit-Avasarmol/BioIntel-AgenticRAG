"""Citation verification — the audit-critical guardrail against hallucination.

The synthesis LLM is instructed to end every factual sentence with ``[n]`` markers
that reference numbered sources. This module *checks that it actually did*, and
that the cited source text lexically supports the sentence:

1. Split the answer into sentences (reusing the shared sentence splitter).
2. For each sentence, parse its ``[n]`` markers.
3. A sentence is **supported** if it has at least one valid marker AND the cited
   chunk's text shares at least ``citation_min_overlap`` of the sentence's content
   tokens (a cheap, dependency-free lexical grounding check that catches citations
   pasted onto unrelated claims).
4. Sentences with no markers or with markers that fail the overlap check are
   reported as ``unsupported`` so the agent can regenerate or flag the answer.

This is intentionally lexical (not another LLM call): it is fast, deterministic,
fully offline-testable, and cannot itself hallucinate. It is a *necessary* support
signal, not a claim of semantic correctness.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from biointel.common.config import settings
from biointel.common.schemas import Citation, RetrievedChunk
from biointel.common.text import split_sentences

_MARKER_RE = re.compile(r"\[(\d+)\]")
_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Common English stopwords excluded from the overlap denominator so that
# grounding is judged on content words, not filler.
_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "of",
        "to",
        "in",
        "on",
        "for",
        "and",
        "or",
        "but",
        "with",
        "without",
        "within",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "as",
        "at",
        "by",
        "from",
        "into",
        "than",
        "then",
        "thus",
        "we",
        "our",
        "they",
        "their",
        "he",
        "she",
        "his",
        "her",
        "can",
        "may",
        "might",
        "will",
        "would",
        "should",
        "could",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "not",
        "no",
        "nor",
        "so",
        "such",
        "also",
        "more",
        "most",
        "less",
        "least",
        "very",
        "much",
        "many",
        "few",
        "some",
        "any",
        "all",
        "each",
        "per",
        "via",
        "using",
        "use",
        "used",
        "based",
        "show",
        "shown",
        "showed",
        "suggest",
        "suggests",
        "report",
        "reports",
        "reported",
        "found",
        "find",
        "between",
        "among",
        "which",
        "who",
        "whom",
        "whose",
        "what",
        "when",
        "where",
        "why",
        "how",
    ]
)


def _content_tokens(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS and len(t) > 1}


def parse_markers(sentence: str) -> list[int]:
    """Return the list of 1-based citation numbers referenced in a sentence."""
    return [int(m) for m in _MARKER_RE.findall(sentence)]


def strip_markers(sentence: str) -> str:
    """Remove ``[n]`` markers to get the bare claim text for overlap scoring."""
    return _MARKER_RE.sub("", sentence).strip()


def overlap_ratio(sentence: str, chunk_text: str) -> float:
    """Fraction of the sentence's content tokens also present in the chunk."""
    s_tokens = _content_tokens(strip_markers(sentence))
    if not s_tokens:
        return 1.0  # nothing substantive to support (e.g. a transition sentence)
    c_tokens = _content_tokens(chunk_text)
    if not c_tokens:
        return 0.0
    return len(s_tokens & c_tokens) / len(s_tokens)


@dataclass
class SentenceCheck:
    sentence: str
    markers: list[int]
    supported: bool
    best_overlap: float
    best_source: int | None = None


@dataclass
class VerificationResult:
    verified: bool
    citations: list[Citation] = field(default_factory=list)
    sentence_checks: list[SentenceCheck] = field(default_factory=list)
    unsupported_sentences: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def support_ratio(self) -> float:
        checks = [c for c in self.sentence_checks if _content_tokens(strip_markers(c.sentence))]
        if not checks:
            return 1.0
        return sum(c.supported for c in checks) / len(checks)


def verify_answer(
    answer: str,
    retrieved: list[RetrievedChunk],
    min_overlap: float | None = None,
) -> VerificationResult:
    """Verify that every factual sentence in ``answer`` is supported by a citation.

    ``retrieved`` must be the exact ordered list that was rendered into the SOURCES
    block (index i -> marker i+1), so markers map back to real chunks.
    """
    threshold = settings.citation_min_overlap if min_overlap is None else min_overlap
    sentences = split_sentences(answer)
    n_sources = len(retrieved)

    checks: list[SentenceCheck] = []
    citations: dict[str, Citation] = {}
    unsupported: list[str] = []
    warnings: list[str] = []

    for sent in sentences:
        markers = parse_markers(sent)
        substantive = bool(_content_tokens(strip_markers(sent)))

        # Non-substantive sentences (headers, transitions) don't need citations.
        if not substantive:
            checks.append(SentenceCheck(sent, markers, True, 1.0, None))
            continue

        if not markers:
            checks.append(SentenceCheck(sent, markers, False, 0.0, None))
            unsupported.append(sent)
            continue

        best_overlap = 0.0
        best_source: int | None = None
        valid_marker_found = False
        for m in markers:
            if m < 1 or m > n_sources:
                warnings.append(f"Citation [{m}] is out of range (1..{n_sources}).")
                continue
            valid_marker_found = True
            rc = retrieved[m - 1]
            ov = overlap_ratio(sent, rc.chunk.text)
            if ov > best_overlap:
                best_overlap = ov
                best_source = m

        supported = valid_marker_found and best_overlap >= threshold
        checks.append(SentenceCheck(sent, markers, supported, best_overlap, best_source))
        if not supported:
            unsupported.append(sent)
            continue

        # Record a verified citation for every valid, supporting marker.
        for m in markers:
            if 1 <= m <= n_sources:
                rc = retrieved[m - 1]
                if overlap_ratio(sent, rc.chunk.text) >= threshold or m == best_source:
                    c = rc.chunk
                    key = f"[{m}]::{c.chunk_id}"
                    if key not in citations:
                        citations[key] = Citation(
                            marker=f"[{m}]",
                            chunk_id=c.chunk_id,
                            doc_id=c.doc_id,
                            source=c.source,
                            source_url=c.source_url,
                            label=c.citation_label(),
                            quote=_best_quote(sent, c.text),
                        )

    verified = len(unsupported) == 0 and not any("out of range" in w for w in warnings)
    if unsupported:
        warnings.append(f"{len(unsupported)} sentence(s) lack supporting citations.")

    return VerificationResult(
        verified=verified,
        citations=list(citations.values()),
        sentence_checks=checks,
        unsupported_sentences=unsupported,
        warnings=warnings,
    )


def _best_quote(sentence: str, chunk_text: str, window: int = 240) -> str:
    """Pick a short supporting snippet from the chunk for display.

    Finds the chunk sentence with the highest token overlap to the claim; falls
    back to the chunk's head. Purely cosmetic (shown under the citation).
    """
    candidates = split_sentences(chunk_text) or [chunk_text]
    best = max(candidates, key=lambda cs: overlap_ratio(sentence, cs), default="")
    best = best.strip()
    if len(best) > window:
        best = best[:window].rsplit(" ", 1)[0] + "…"
    return best
