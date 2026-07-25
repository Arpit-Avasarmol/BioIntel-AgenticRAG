"""ChEMBL connector via the official REST API (no key, no scraping).

Docs: https://www.ebi.ac.uk/chembl/api/data/docs
Flow: gene symbol -> ChEMBL target(s) -> bioactivities -> one Document per
compound summarizing its activity against the target.

License: ChEMBL data is CC-BY-SA 3.0 — attribution is surfaced in the Document
``license`` field and in the README license table.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from typing import Any

from biointel.common.logging import get_logger
from biointel.common.schemas import DocType, Document, DocumentIds, Section, SourceType
from biointel.common.text import clean_text
from biointel.ingestion.base import BaseConnector, HttpClient

logger = get_logger(__name__)

API = "https://www.ebi.ac.uk/chembl/api/data"


class ChEMBLConnector(BaseConnector):
    source = SourceType.CHEMBL
    license = "CC-BY-SA-3.0"

    def __init__(self, targets: list[str], max_records: int = 150, **kw: Any):
        super().__init__(**kw)
        self.target_symbols = targets
        self.max_records = max_records
        self.http = HttpClient(base_url=API, rps=3.0, timeout=60)

    def _resolve_targets(self, symbol: str) -> list[dict[str, Any]]:
        """Find ChEMBL targets for a gene symbol (Homo sapiens)."""
        resp = self.http.get(
            "/target/search",
            params={"q": symbol, "format": "json"},
        ).json()
        targets = resp.get("targets", [])
        human = [t for t in targets if t.get("organism") == "Homo sapiens"]
        return human or targets[:1]

    def fetch(self) -> Iterator[dict[str, Any]]:
        seen_targets: set[str] = set()
        per_compound: dict[str, dict[str, Any]] = {}
        count = 0
        for symbol in self.target_symbols:
            for tgt in self._resolve_targets(symbol):
                chembl_tid = tgt.get("target_chembl_id")
                if not chembl_tid or chembl_tid in seen_targets:
                    continue
                seen_targets.add(chembl_tid)
                # Pull bioactivities with standard values for this target.
                offset = 0
                while count < self.max_records:
                    acts = self.http.get(
                        "/activity",
                        params={
                            "target_chembl_id": chembl_tid,
                            "standard_type__in": "IC50,Ki,Kd,EC50,Potency",
                            "limit": 200,
                            "offset": offset,
                            "format": "json",
                        },
                    ).json()
                    activities = acts.get("activities", [])
                    if not activities:
                        break
                    for act in activities:
                        cid = act.get("molecule_chembl_id")
                        if not cid:
                            continue
                        rec = per_compound.setdefault(
                            cid,
                            {
                                "molecule_chembl_id": cid,
                                "pref_name": act.get("molecule_pref_name"),
                                "target_symbol": symbol,
                                "target_chembl_id": chembl_tid,
                                "target_name": tgt.get("pref_name"),
                                "activities": [],
                            },
                        )
                        rec["activities"].append(
                            {
                                "type": act.get("standard_type"),
                                "value": act.get("standard_value"),
                                "units": act.get("standard_units"),
                                "assay": act.get("assay_description"),
                            }
                        )
                    offset += 200
                    if acts.get("page_meta", {}).get("next") is None:
                        break
        # Emit one raw record per compound (bounded).
        for rec in per_compound.values():
            yield rec
            count += 1
            if count >= self.max_records:
                break
        logger.info("[chembl] compiled %d compound records", min(count, len(per_compound)))

    def normalize(self, raw: dict[str, Any]) -> Document | None:
        cid = raw.get("molecule_chembl_id")
        if not cid:
            return None
        name = raw.get("pref_name") or cid
        target_symbol = raw.get("target_symbol", "")
        target_name = raw.get("target_name", "")

        # Summarize activities into a readable block.
        by_type: dict[str, list[str]] = defaultdict(list)
        for a in raw.get("activities", [])[:50]:
            if a.get("value") and a.get("type"):
                by_type[a["type"]].append(f"{a['value']} {a.get('units') or ''}".strip())
        act_lines = [f"{t}: {', '.join(vals[:10])}" for t, vals in by_type.items()]
        activity_txt = "\n".join(act_lines)

        abstract = (
            f"{name} ({cid}) bioactivity against {target_name} ({target_symbol}). "
            f"Measured endpoints: {', '.join(by_type.keys()) or 'N/A'}."
        )
        sections = [Section(name="Bioactivity Summary", text=clean_text(activity_txt), order=1)]

        return Document(
            doc_id=f"chembl:{cid}",
            source=self.source,
            doc_type=DocType.COMPOUND,
            title=f"{name} — activity vs {target_symbol}",
            abstract=abstract,
            sections=sections,
            ids=DocumentIds(chembl_id=cid),
            source_url=f"https://www.ebi.ac.uk/chembl/compound_report_card/{cid}/",
            license=self.license,
            extra={
                "target_symbol": target_symbol,
                "target_chembl_id": raw.get("target_chembl_id"),
                "n_activities": len(raw.get("activities", [])),
            },
        )
