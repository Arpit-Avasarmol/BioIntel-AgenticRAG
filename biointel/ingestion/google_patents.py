"""OPTIONAL: Google Patents Public Data connector via BigQuery.

Google Patents raw HTML must NOT be scraped. Instead, Google publishes a public
BigQuery dataset (`patents-public-data.patents.publications`) with full text
(titles, abstracts, claims) under permissive terms. BigQuery usage is billed by
Google, so this connector is **disabled by default** and only activates when
GOOGLE_APPLICATION_CREDENTIALS + GCP_PROJECT_ID are set.

Install extra: `uv pip install -e ".[gcp]"`
Docs: https://console.cloud.google.com/marketplace/product/google_patents_public_datasets/google-patents-public-data
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date, datetime
from typing import Any

from biointel.common.config import settings
from biointel.common.logging import get_logger
from biointel.common.schemas import DocType, Document, DocumentIds, Section, SourceType
from biointel.common.text import clean_text
from biointel.ingestion.base import BaseConnector

logger = get_logger(__name__)


class GooglePatentsConnector(BaseConnector):
    source = SourceType.GOOGLE_PATENTS
    license = "Google Patents Public Data (see dataset terms)"

    def __init__(self, query_text: str, max_records: int = 100, **kw: Any):
        super().__init__(**kw)
        self.query_text = query_text
        self.max_records = max_records
        if not settings.google_patents_enabled:
            raise RuntimeError(
                "Google Patents connector requires GOOGLE_APPLICATION_CREDENTIALS "
                "and GCP_PROJECT_ID. It is optional and disabled by default; use "
                "the PatentsView connector for no-auth USPTO patents."
            )

    def _client(self):
        from google.cloud import bigquery  # imported lazily (optional dep)

        return bigquery.Client(project=settings.gcp_project_id)

    def fetch(self) -> Iterator[dict[str, Any]]:
        from google.cloud import bigquery

        client = self._client()
        # Parameterized query over English titles/abstracts. LIMIT bounds cost.
        sql = """
        SELECT
          publication_number,
          (SELECT text FROM UNNEST(title_localized) WHERE language = 'en' LIMIT 1) AS title,
          (SELECT text FROM UNNEST(abstract_localized) WHERE language = 'en' LIMIT 1) AS abstract,
          (SELECT text FROM UNNEST(claims_localized) WHERE language = 'en' LIMIT 1) AS claims,
          assignee_harmonized,
          inventor_harmonized,
          publication_date
        FROM `patents-public-data.patents.publications`
        WHERE country_code = 'US'
          AND EXISTS (
            SELECT 1 FROM UNNEST(abstract_localized) a
            WHERE a.language = 'en' AND LOWER(a.text) LIKE @needle
          )
        LIMIT @max_records
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("needle", "STRING", f"%{self.query_text.lower()}%"),
                bigquery.ScalarQueryParameter("max_records", "INT64", self.max_records),
            ]
        )
        logger.info("[google_patents] running BigQuery (LIMIT %d)", self.max_records)
        for row in client.query(sql, job_config=job_config).result():
            yield dict(row)

    def normalize(self, raw: dict[str, Any]) -> Document | None:
        pub = raw.get("publication_number")
        if not pub:
            return None
        title = clean_text(raw.get("title") or "")
        abstract = clean_text(raw.get("abstract") or "")
        claims = clean_text(raw.get("claims") or "")

        sections = []
        if claims:
            # Claims can be very long; keep the leading portion.
            sections.append(Section(name="Claims", text=claims[:20000], order=1))

        assignees = [a.get("name") for a in (raw.get("assignee_harmonized") or []) if a.get("name")]
        inventors = [i.get("name") for i in (raw.get("inventor_harmonized") or []) if i.get("name")]

        pub_date = _parse_int_date(raw.get("publication_date"))

        return Document(
            doc_id=f"gpatent:{pub}",
            source=self.source,
            doc_type=DocType.PATENT,
            title=title,
            abstract=abstract,
            sections=sections,
            authors=inventors,
            date=pub_date,
            ids=DocumentIds(patent_no=pub),
            source_url=f"https://patents.google.com/patent/{pub}",
            license=self.license,
            extra={"assignees": assignees},
        )


def _parse_int_date(value: Any) -> date | None:
    """BigQuery publication_date is often an INT like 20200115."""
    if not value:
        return None
    try:
        s = str(value)
        return datetime.strptime(s, "%Y%m%d").date()
    except ValueError:
        return None
