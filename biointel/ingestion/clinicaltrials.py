"""ClinicalTrials.gov connector via the official API v2 (no key, no scraping).

Docs: https://clinicaltrials.gov/data-api/api
Pulls studies by condition + optional intervention/term query, paginates with
``pageToken``, and flattens the nested protocol section into a Document with
sections for design, eligibility, arms, and outcomes.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date
from typing import Any

from biointel.common.config import settings
from biointel.common.logging import get_logger
from biointel.common.schemas import DocType, Document, DocumentIds, Section, SourceType
from biointel.common.text import clean_text
from biointel.ingestion.base import BaseConnector, HttpClient, _default_user_agent

logger = get_logger(__name__)

API = "https://clinicaltrials.gov/api/v2"


class ClinicalTrialsConnector(BaseConnector):
    source = SourceType.CLINICALTRIALS
    license = "public (ClinicalTrials.gov)"

    def __init__(
        self,
        conditions: list[str],
        query_terms: str | None = None,
        statuses: list[str] | None = None,
        max_records: int = 150,
        page_size: int = 100,
        **kw: Any,
    ):
        super().__init__(**kw)
        self.conditions = conditions
        self.query_terms = query_terms
        self.statuses = statuses
        self.max_records = max_records
        self.page_size = min(page_size, 1000)
        self.http = HttpClient(
            base_url=API,
            rps=1.5,
            timeout=60,
            headers={
                "User-Agent": _default_user_agent(),
                "Accept": "application/json",
            },
        )

    def fetch(self) -> Iterator[dict[str, Any]]:
        # API v2 accepts Essie query syntax via query.cond / query.term.
        cond_expr = " OR ".join(f'"{c}"' for c in self.conditions)
        params: dict[str, Any] = {
            "query.cond": cond_expr,
            "pageSize": self.page_size,
            "format": "json",
        }
        if self.query_terms:
            params["query.term"] = self.query_terms
        if self.statuses:
            params["filter.overallStatus"] = ",".join(self.statuses)

        fetched = 0
        page_token: str | None = None
        while True:
            if page_token:
                params["pageToken"] = page_token
            data = self.http.get("/studies", params=params).json()
            studies = data.get("studies", [])
            if not studies:
                break
            for st in studies:
                yield st
                fetched += 1
                if fetched >= self.max_records:
                    return
            page_token = data.get("nextPageToken")
            if not page_token:
                break
        logger.info("[clinicaltrials] fetched %d studies", fetched)

    def normalize(self, raw: dict[str, Any]) -> Document | None:
        ps = raw.get("protocolSection", {})
        ident = ps.get("identificationModule", {})
        nct = ident.get("nctId")
        if not nct:
            return None

        title = ident.get("officialTitle") or ident.get("briefTitle") or ""
        status_mod = ps.get("statusModule", {})
        design_mod = ps.get("designModule", {})
        desc_mod = ps.get("descriptionModule", {})
        elig_mod = ps.get("eligibilityModule", {})
        arms_mod = ps.get("armsInterventionsModule", {})
        outcomes_mod = ps.get("outcomesModule", {})
        sponsor_mod = ps.get("sponsorCollaboratorsModule", {})
        conditions_mod = ps.get("conditionsModule", {})

        abstract = clean_text(desc_mod.get("briefSummary", ""))
        sections: list[Section] = []
        order = 0

        def add(name: str, text: str) -> None:
            nonlocal order
            text = clean_text(text)
            if text:
                order += 1
                sections.append(Section(name=name, text=text, order=order))

        add("Detailed Description", desc_mod.get("detailedDescription", ""))

        phases = design_mod.get("phases", [])
        study_type = design_mod.get("studyType", "")
        design_info = design_mod.get("designInfo", {})
        design_txt = (
            f"Study type: {study_type}. Phases: {', '.join(phases) or 'N/A'}. "
            f"Allocation: {design_info.get('allocation', 'N/A')}. "
            f"Primary purpose: {design_info.get('primaryPurpose', 'N/A')}. "
            f"Masking: {design_info.get('maskingInfo', {}).get('masking', 'N/A')}."
        )
        add("Study Design", design_txt)

        # Interventions / arms
        interventions = arms_mod.get("interventions", [])
        drug_names: list[str] = []
        int_txt_parts = []
        for iv in interventions:
            nm = iv.get("name", "")
            if nm:
                drug_names.append(nm)
            int_txt_parts.append(f"{iv.get('type', '')}: {nm} — {iv.get('description', '')}")
        add("Interventions", " | ".join(int_txt_parts))

        # Eligibility
        elig_txt = (
            f"{elig_mod.get('eligibilityCriteria', '')}\n"
            f"Sex: {elig_mod.get('sex', 'N/A')}. Ages: {elig_mod.get('minimumAge', 'N/A')}"
            f"–{elig_mod.get('maximumAge', 'N/A')}."
        )
        add("Eligibility", elig_txt)

        # Outcomes
        prim = outcomes_mod.get("primaryOutcomes", [])
        out_txt = " | ".join(f"{o.get('measure', '')} [{o.get('timeFrame', '')}]" for o in prim)
        add("Primary Outcomes", out_txt)

        # Date (study start)
        start = status_mod.get("startDateStruct", {}).get("date")
        pub_date = _parse_partial_date(start)

        lead_sponsor = sponsor_mod.get("leadSponsor", {}).get("name", "")
        conditions = conditions_mod.get("conditions", [])

        return Document(
            doc_id=f"nct:{nct}",
            source=self.source,
            doc_type=DocType.TRIAL,
            title=title,
            abstract=abstract,
            sections=sections,
            date=pub_date,
            ids=DocumentIds(nct_id=nct),
            source_url=f"https://clinicaltrials.gov/study/{nct}",
            license=self.license,
            extra={
                "phases": phases,
                "status": status_mod.get("overallStatus", ""),
                "study_type": study_type,
                "lead_sponsor": lead_sponsor,
                "conditions": conditions,
                "interventions": drug_names,
            },
        )


def _parse_partial_date(value: str | None) -> date | None:
    """Handle 'YYYY', 'YYYY-MM', or 'YYYY-MM-DD'."""
    if not value:
        return None
    parts = value.split("-")
    try:
        y = int(parts[0])
        m = int(parts[1]) if len(parts) > 1 else 1
        d = int(parts[2]) if len(parts) > 2 else 1
        return date(y, m, d)
    except (ValueError, IndexError):
        return None
