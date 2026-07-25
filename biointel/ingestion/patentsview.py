"""USPTO patents connector via the PatentsView PatentSearch API (no scraping).

PatentsView (https://search.patentsview.org/) is the official, free USPTO data
API. We query the patents endpoint for a text theme, retrieving title, abstract,
assignees, inventors, dates, and CPC classes. US government public data.

Note: PatentsView exposes granted-patent bibliographic + abstract text (not full
claims body). That is the licensing-clean surface; full claim/description text is
available via the optional Google Patents BigQuery connector.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import date
from typing import Any

from biointel.common.config import settings
from biointel.common.logging import get_logger
from biointel.common.schemas import DocType, Document, DocumentIds, Section, SourceType
from biointel.common.text import clean_text
from biointel.ingestion.base import BaseConnector, HttpClient

logger = get_logger(__name__)

# PatentsView "PatentSearch API" (v1) endpoint for granted patents.
API = "https://search.patentsview.org/api/v1"


def _patentsearch_headers() -> dict[str, str]:
    headers = {
        "User-Agent": f"{settings.ncbi_tool}/0.1 (mailto:{settings.ncbi_email})",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    if settings.patentsview_api_key:
        headers["X-Api-Key"] = settings.patentsview_api_key
    return headers


class PatentsViewConnector(BaseConnector):
    source = SourceType.PATENTSVIEW
    license = "public (USPTO / PatentsView)"

    def __init__(
        self,
        query_text: str,
        date_from: str | None = None,
        max_records: int = 100,
        page_size: int = 100,
        **kw: Any,
    ):
        super().__init__(**kw)
        self.query_text = query_text
        self.date_from = date_from
        self.max_records = max_records
        self.page_size = min(page_size, 1000)
        self.http = HttpClient(
            base_url=API,
            rps=2.0,
            timeout=60,
            headers=_patentsearch_headers(),
        )

    def _build_query(self) -> dict[str, Any]:
        # PatentSearch API text search across title + abstract.
        terms = [t for t in self.query_text.split() if t]
        if not terms:
            terms = [self.query_text]
        clauses: list[dict[str, Any]] = [
            {"patent_title": {"_text_all": terms}},
            {"patent_abstract": {"_text_all": terms}},
        ]
        q: dict[str, Any] = {"_or": clauses}
        if self.date_from:
            q = {"_and": [q, {"patent_date": {"_gte": self.date_from}}]}
        return q

    def _parse_response(self, resp: dict[str, Any]) -> list[dict[str, Any]]:
        patents = resp.get("patents")
        if patents is None and isinstance(resp.get("data"), dict):
            patents = resp["data"].get("patents")
        return patents or []

    def fetch(self) -> Iterator[dict[str, Any]]:
        fields = [
            "patent_id",
            "patent_number",
            "patent_title",
            "patent_abstract",
            "patent_date",
            "assignees.assignee_organization",
            "inventors.inventor_name_last",
            "inventors.inventor_name_first",
            "cpc_current.cpc_group_id",
        ]
        fetched = 0
        after: str | None = None
        while fetched < self.max_records:
            options: dict[str, Any] = {"size": self.page_size, "per_page": self.page_size}
            if after:
                options["after"] = after
            body = {
                "q": self._build_query(),
                "f": fields,
                "o": options,
                "s": [{"patent_id": "asc"}],
            }
            resp = self.http.post("/patent", content=json.dumps(body)).json()
            patents = self._parse_response(resp)
            if not patents:
                break
            for p in patents:
                yield p
                fetched += 1
                after = p.get("patent_id") or p.get("patent_number")
                if fetched >= self.max_records:
                    return
            if len(patents) < self.page_size:
                break
        logger.info("[patentsview] fetched %d patents", fetched)

    def normalize(self, raw: dict[str, Any]) -> Document | None:
        pid = raw.get("patent_id") or raw.get("patent_number")
        if not pid:
            return None
        pid = str(pid).replace("US", "")
        title = clean_text(raw.get("patent_title") or "")
        abstract = clean_text(raw.get("patent_abstract") or "")

        assignees = [
            a.get("assignee_organization")
            for a in (raw.get("assignees") or [])
            if a.get("assignee_organization")
        ]
        inventors = [
            f"{i.get('inventor_name_last', '')} {i.get('inventor_name_first', '')}".strip()
            for i in (raw.get("inventors") or [])
        ]
        inventors = [i for i in inventors if i]
        cpc = [
            c.get("cpc_group_id") for c in (raw.get("cpc_current") or []) if c.get("cpc_group_id")
        ]

        pub_date = _parse_date(raw.get("patent_date"))

        sections = []
        if assignees:
            sections.append(Section(name="Assignees", text="; ".join(assignees), order=1))

        return Document(
            doc_id=f"uspatent:{pid}",
            source=self.source,
            doc_type=DocType.PATENT,
            title=title,
            abstract=abstract,
            sections=sections,
            authors=inventors,
            date=pub_date,
            ids=DocumentIds(patent_no=f"US{pid}"),
            source_url=f"https://patents.google.com/patent/US{pid}",
            license=self.license,
            extra={"assignees": assignees, "cpc_classes": cpc},
        )


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        y, m, d = value.split("-")[:3]
        return date(int(y), int(m), int(d))
    except (ValueError, IndexError):
        return None
