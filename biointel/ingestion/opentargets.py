"""Open Targets connector via the GraphQL v4 API (no key, no scraping).

Endpoint: https://api.platform.opentargets.org/api/v4/graphql
Pattern (per Open Targets best practice): for each disease EFO/MONDO ID, pull the
top associated targets + known drugs in a single graph traversal, then emit one
Document per target-disease association. Data is CC0 1.0. We record the data
release version for citation.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from biointel.common.logging import get_logger
from biointel.common.schemas import DocType, Document, DocumentIds, Section, SourceType
from biointel.common.text import clean_text
from biointel.ingestion.base import BaseConnector, HttpClient

logger = get_logger(__name__)

URL = "https://api.platform.opentargets.org/api/v4/graphql"

_DISEASE_QUERY = """
query Disease($efoId: String!, $size: Int!) {
  meta { dataVersion { year month } }
  disease(efoId: $efoId) {
    id
    name
    knownDrugs { uniqueDrugs rows { drug { id name isApproved } mechanismOfAction } }
    associatedTargets(page: { index: 0, size: $size }, enableIndirect: true) {
      rows {
        target { id approvedSymbol approvedName }
        score
        datatypeScores { id score }
      }
    }
  }
}
"""


class OpenTargetsConnector(BaseConnector):
    source = SourceType.OPENTARGETS
    license = "CC0-1.0"

    def __init__(self, diseases: list[str], max_targets: int = 50, **kw: Any):
        super().__init__(**kw)
        self.diseases = diseases
        self.max_targets = max_targets
        self.http = HttpClient(base_url="", rps=2.0, timeout=60)
        self.data_version = ""

    def _query(self, query: str, variables: dict[str, Any]) -> dict[str, Any]:
        resp = self.http.post(URL, json={"query": query, "variables": variables})
        payload = resp.json()
        if "errors" in payload:
            raise RuntimeError(payload["errors"])
        return payload["data"]

    def fetch(self) -> Iterator[dict[str, Any]]:
        for efo in self.diseases:
            data = self._query(_DISEASE_QUERY, {"efoId": efo, "size": self.max_targets})
            meta = data.get("meta", {}).get("dataVersion", {})
            if meta:
                self.data_version = f"{meta.get('year')}.{meta.get('month')}"
            disease = data.get("disease")
            if not disease:
                logger.warning("[opentargets] no disease for %s", efo)
                continue
            drugs = disease.get("knownDrugs", {}).get("rows", [])
            for row in disease.get("associatedTargets", {}).get("rows", []):
                yield {
                    "disease_id": disease["id"],
                    "disease_name": disease["name"],
                    "assoc": row,
                    "known_drugs": drugs,
                }

    def normalize(self, raw: dict[str, Any]) -> Document | None:
        assoc = raw.get("assoc", {})
        target = assoc.get("target", {})
        ensembl = target.get("id")
        if not ensembl:
            return None
        symbol = target.get("approvedSymbol", "")
        tname = target.get("approvedName", "")
        disease_id = raw["disease_id"]
        disease_name = raw["disease_name"]
        score = assoc.get("score", 0.0)

        # Datatype breakdown
        dt = assoc.get("datatypeScores", [])
        dt_txt = ", ".join(f"{d['id']}={d['score']:.3f}" for d in dt)

        # Known drugs mentioning this target's disease (coarse linkage).
        drug_names = []
        for d in raw.get("known_drugs", [])[:15]:
            drug = d.get("drug", {})
            nm = drug.get("name")
            if nm:
                approved = "approved" if drug.get("isApproved") else "investigational"
                drug_names.append(f"{nm} ({approved})")

        abstract = (
            f"Open Targets association between {symbol} ({tname}) and {disease_name}. "
            f"Overall association score: {score:.3f}."
        )
        sections = [
            Section(name="Evidence by datatype", text=clean_text(dt_txt), order=1),
        ]
        if drug_names:
            sections.append(
                Section(
                    name="Known drugs for disease",
                    text="; ".join(drug_names),
                    order=2,
                )
            )

        return Document(
            doc_id=f"ot:{ensembl}:{disease_id}",
            source=self.source,
            doc_type=DocType.TARGET,
            title=f"{symbol} × {disease_name} (Open Targets)",
            abstract=abstract,
            sections=sections,
            ids=DocumentIds(ensembl_id=ensembl, efo_id=disease_id),
            source_url=f"https://platform.opentargets.org/evidence/{ensembl}/{disease_id}",
            license=self.license,
            extra={
                "association_score": score,
                "target_symbol": symbol,
                "disease_name": disease_name,
                "data_version": self.data_version,
            },
        )
