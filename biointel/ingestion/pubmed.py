"""PubMed connector via NCBI E-utilities (esearch + efetch).

Official API — https://www.ncbi.nlm.nih.gov/books/NBK25501/ . No HTML scraping.
An optional NCBI_API_KEY raises the rate limit from 3 to 10 requests/second.
PubMed metadata/abstracts are in the public domain.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Iterator
from datetime import date
from typing import Any

from biointel.common.config import settings
from biointel.common.logging import get_logger
from biointel.common.schemas import DocType, Document, DocumentIds, Section, SourceType
from biointel.common.text import clean_text
from biointel.ingestion.base import BaseConnector, HttpClient

logger = get_logger(__name__)

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


class PubMedConnector(BaseConnector):
    source = SourceType.PUBMED
    license = "public-domain"

    def __init__(self, query: str, max_records: int = 200, batch_size: int = 100, **kw: Any):
        super().__init__(**kw)
        self.query = query
        self.max_records = max_records
        self.batch_size = min(batch_size, 200)
        rps = 10.0 if settings.ncbi_api_key else 3.0
        self.http = HttpClient(base_url=EUTILS, rps=rps, timeout=60)

    # ---- helpers ----
    def _common_params(self) -> dict[str, str]:
        p = {"db": "pubmed", "tool": settings.ncbi_tool, "email": settings.ncbi_email}
        if settings.ncbi_api_key:
            p["api_key"] = settings.ncbi_api_key
        return p

    def _esearch(self) -> list[str]:
        params = {
            **self._common_params(),
            "term": self.query,
            "retmax": str(self.max_records),
            "retmode": "json",
        }
        data = self.http.get("/esearch.fcgi", params=params).json()
        ids = data.get("esearchresult", {}).get("idlist", [])
        logger.info("[pubmed] esearch matched %s ids (capped at %d)", len(ids), self.max_records)
        return ids[: self.max_records]

    def fetch(self) -> Iterator[dict[str, Any]]:
        pmids = self._esearch()
        for i in range(0, len(pmids), self.batch_size):
            batch = pmids[i : i + self.batch_size]
            params = {
                **self._common_params(),
                "id": ",".join(batch),
                "retmode": "xml",
                "rettype": "abstract",
            }
            xml = self.http.get("/efetch.fcgi", params=params).text
            root = ET.fromstring(xml)
            for article in root.findall(".//PubmedArticle"):
                yield {"xml": ET.tostring(article, encoding="unicode")}

    # ---- normalization ----
    @staticmethod
    def _text(el: ET.Element | None) -> str:
        if el is None:
            return ""
        return clean_text("".join(el.itertext()))

    def normalize(self, raw: dict[str, Any]) -> Document | None:
        try:
            art = ET.fromstring(raw["xml"])
        except ET.ParseError:
            return None
        pmid_el = art.find(".//PMID")
        pmid = pmid_el.text if pmid_el is not None else None
        if not pmid:
            return None

        title = self._text(art.find(".//ArticleTitle"))
        # Abstract can have multiple labeled sections.
        abstract_parts, sections = [], []
        for ab in art.findall(".//Abstract/AbstractText"):
            label = ab.get("Label") or ""
            txt = self._text(ab)
            if not txt:
                continue
            abstract_parts.append(txt)
            if label:
                sections.append(Section(name=label.title(), text=txt, order=len(sections) + 1))
        abstract = "\n".join(abstract_parts)

        # DOI
        doi = None
        for aid in art.findall(".//ArticleId"):
            if aid.get("IdType") == "doi":
                doi = aid.text
        # PMCID (if the article is also in PMC)
        pmcid = None
        for aid in art.findall(".//ArticleId"):
            if aid.get("IdType") == "pmc":
                pmcid = aid.text

        # Authors
        authors = []
        for a in art.findall(".//Author"):
            last = a.findtext("LastName") or ""
            init = a.findtext("Initials") or ""
            name = f"{last} {init}".strip()
            if name:
                authors.append(name)

        # Publication date (year at minimum)
        pub_date = None
        y = art.findtext(".//PubDate/Year")
        if y and y.isdigit():
            m = art.findtext(".//PubDate/Month") or "1"
            d = art.findtext(".//PubDate/Day") or "1"
            month = _month_to_int(m)
            day = int(d) if d.isdigit() else 1
            try:
                pub_date = date(int(y), month, min(day, 28))
            except ValueError:
                pub_date = date(int(y), 1, 1)

        # Entities: MeSH major topics (coarse but useful filters).
        mesh = [self._text(mh.find("DescriptorName")) for mh in art.findall(".//MeshHeading")]
        mesh = [m for m in mesh if m]

        return Document(
            doc_id=f"pubmed:{pmid}",
            source=self.source,
            doc_type=DocType.PAPER,
            title=title,
            abstract=abstract,
            sections=sections,
            authors=authors,
            date=pub_date,
            ids=DocumentIds(pmid=pmid, doi=doi, pmcid=pmcid),
            source_url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            license=self.license,
            extra={"mesh_terms": mesh, "journal": art.findtext(".//Journal/Title") or ""},
        )


_MONTHS = {
    m: i
    for i, m in enumerate(
        ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1
    )
}


def _month_to_int(m: str) -> int:
    m = m.strip().lower()
    if m.isdigit():
        return max(1, min(12, int(m)))
    return _MONTHS.get(m[:3], 1)
