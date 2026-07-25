"""PMC Open Access connector — full text for the OA subset only.

Uses NCBI E-utilities to find PMC IDs, then efetch (db=pmc) for JATS XML full
text. We only keep articles carrying an Open Access license (CC-BY family / CC0);
non-OA full text is never fetched or stored, per PMC terms.

Refs: https://www.ncbi.nlm.nih.gov/pmc/tools/openftlist/
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Iterator
from typing import Any

from biointel.common.config import settings
from biointel.common.logging import get_logger
from biointel.common.schemas import DocType, Document, DocumentIds, Section, SourceType
from biointel.common.text import clean_text
from biointel.ingestion.base import BaseConnector, HttpClient

logger = get_logger(__name__)

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# Section names we keep from JATS body (skip refs, acknowledgments, etc.).
_KEEP_SECTIONS = {
    "introduction",
    "background",
    "methods",
    "materials and methods",
    "results",
    "discussion",
    "conclusion",
    "conclusions",
}


class PMCConnector(BaseConnector):
    source = SourceType.PMC
    license = "CC-BY (OA subset)"

    def __init__(self, query: str, max_records: int = 60, **kw: Any):
        super().__init__(**kw)
        # Force the OA filter so we never pull restricted full text.
        self.query = query if "open access" in query.lower() else f"{query} AND open access[filter]"
        self.max_records = max_records
        rps = 10.0 if settings.ncbi_api_key else 3.0
        self.http = HttpClient(base_url=EUTILS, rps=rps, timeout=90)

    def _common(self) -> dict[str, str]:
        p = {"db": "pmc", "tool": settings.ncbi_tool, "email": settings.ncbi_email}
        if settings.ncbi_api_key:
            p["api_key"] = settings.ncbi_api_key
        return p

    def fetch(self) -> Iterator[dict[str, Any]]:
        search = self.http.get(
            "/esearch.fcgi",
            params={
                **self._common(),
                "term": self.query,
                "retmax": str(self.max_records),
                "retmode": "json",
            },
        ).json()
        ids = search.get("esearchresult", {}).get("idlist", [])[: self.max_records]
        logger.info("[pmc] esearch matched %d OA ids", len(ids))
        for i in range(0, len(ids), 20):
            batch = ids[i : i + 20]
            xml = self.http.get(
                "/efetch.fcgi",
                params={**self._common(), "id": ",".join(batch), "retmode": "xml"},
            ).text
            root = ET.fromstring(xml)
            for art in root.findall(".//article"):
                yield {"xml": ET.tostring(art, encoding="unicode")}

    @staticmethod
    def _text(el: ET.Element | None) -> str:
        return clean_text("".join(el.itertext())) if el is not None else ""

    def _is_open_access(self, art: ET.Element) -> tuple[bool, str]:
        """Check the JATS <permissions>/<license> for an OA license."""
        for lic in art.findall(".//permissions/license"):
            href = lic.get("{http://www.w3.org/1999/xlink}href", "") or ""
            lic_type = (lic.get("license-type") or "").lower()
            text = self._text(lic).lower()
            blob = f"{href} {lic_type} {text}"
            is_oa = (
                "creativecommons.org" in blob
                or "cc-by" in blob
                or "cc0" in blob
                or "open access" in blob
            )
            if is_oa:
                # Extract a compact license label.
                if "cc0" in blob:
                    return True, "CC0-1.0"
                if "by-nc-nd" in blob:
                    return True, "CC-BY-NC-ND"
                if "by-nc" in blob:
                    return True, "CC-BY-NC"
                if "by" in blob:
                    return True, "CC-BY"
                return True, "open-access"
        return False, ""

    def normalize(self, raw: dict[str, Any]) -> Document | None:
        try:
            art = ET.fromstring(raw["xml"])
        except ET.ParseError:
            return None

        is_oa, lic = self._is_open_access(art)
        if not is_oa:
            logger.debug("[pmc] skipping non-OA article")
            return None

        # IDs
        pmcid = pmid = doi = None
        for aid in art.findall(".//article-id"):
            t = aid.get("pub-id-type")
            if t == "pmc":
                if aid.text and not aid.text.startswith("PMC"):
                    pmcid = f"PMC{aid.text}"
                else:
                    pmcid = aid.text
            elif t == "pmid":
                pmid = aid.text
            elif t == "doi":
                doi = aid.text
        if not pmcid:
            return None

        title = self._text(art.find(".//title-group/article-title"))
        abstract = self._text(art.find(".//abstract"))

        # Body sections
        sections: list[Section] = []
        order = 0
        for sec in art.findall(".//body//sec"):
            name = self._text(sec.find("title"))
            if not name:
                continue
            if name.lower() not in _KEEP_SECTIONS:
                # Keep top-level named sections even if not in the canonical set,
                # but skip clearly non-content ones.
                _skip = ("reference", "acknowledg", "funding", "conflict")
                if any(k in name.lower() for k in _skip):
                    continue
            paras = " ".join(self._text(p) for p in sec.findall("p"))
            if paras.strip():
                order += 1
                sections.append(Section(name=name, text=paras, order=order))

        authors = []
        for c in art.findall(".//contrib[@contrib-type='author']"):
            surname = self._text(c.find(".//surname"))
            given = self._text(c.find(".//given-names"))
            name = f"{surname} {given}".strip()
            if name:
                authors.append(name)

        return Document(
            doc_id=f"pmc:{pmcid}",
            source=self.source,
            doc_type=DocType.PAPER,
            title=title,
            abstract=abstract,
            sections=sections,
            authors=authors,
            ids=DocumentIds(pmcid=pmcid, pmid=pmid, doi=doi),
            source_url=f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/",
            license=lic,
            extra={"journal": self._text(art.find(".//journal-title"))},
        )
