"""Acceptance test 1: connector normalization against offline fixtures.

Each connector's ``normalize()`` must turn a raw payload into a valid ``Document``
with the right doc_type, stable doc_id, license, and provenance — with no network.

* PubMed / PMC ``normalize`` consume ``{"xml": <single-article-xml>}`` (the shape
  their ``fetch()`` yields). The fixtures are single-article XML documents.
* ClinicalTrials / ChEMBL / OpenTargets / PatentsView ``normalize`` consume the
  parsed JSON object for one record.
"""

from __future__ import annotations

import pytest

from biointel.common.schemas import DocType, Document, SourceType
from tests.conftest import load_fixture, load_json_fixture


def test_pubmed_normalize():
    from biointel.ingestion.pubmed import PubMedConnector

    xml = load_fixture("pubmed_article.xml")
    conn = PubMedConnector(query="test")
    doc = conn.normalize({"xml": xml})
    assert isinstance(doc, Document)
    assert doc.source == SourceType.PUBMED
    assert doc.doc_type == DocType.PAPER
    assert doc.doc_id.startswith("pubmed:")
    assert doc.ids.pmid
    assert doc.title
    assert doc.license == "public-domain"
    assert doc.date is not None and doc.date.year >= 1900


def test_pmc_normalize_oa_only():
    from biointel.ingestion.pmc import PMCConnector

    xml = load_fixture("pmc_article.xml")
    conn = PMCConnector(query="test")
    doc = conn.normalize({"xml": xml})
    assert doc is not None, "OA fixture should normalize (non-OA returns None)"
    assert doc.source == SourceType.PMC
    assert doc.doc_type == DocType.PAPER
    assert doc.doc_id.startswith("pmc:")
    assert "CC" in doc.license.upper() or doc.license == "open-access"
    # excluded boilerplate sections must not be present
    names = {s.name.lower() for s in doc.sections}
    assert not any(n in names for n in ("references", "acknowledgements", "funding"))


def test_clinicaltrials_normalize():
    from biointel.ingestion.clinicaltrials import ClinicalTrialsConnector

    data = load_json_fixture("clinicaltrials_study.json")
    conn = ClinicalTrialsConnector(conditions=["Crohn Disease"])
    doc = conn.normalize(data)
    assert doc.source == SourceType.CLINICALTRIALS
    assert doc.doc_type == DocType.TRIAL
    assert doc.doc_id.startswith("nct:")
    assert doc.ids.nct_id
    assert doc.title


def test_chembl_normalize():
    from biointel.ingestion.chembl import ChEMBLConnector

    data = load_json_fixture("chembl_compound.json")
    conn = ChEMBLConnector(targets=["IL23A"])
    doc = conn.normalize(data)
    assert doc.source == SourceType.CHEMBL
    assert doc.doc_type == DocType.COMPOUND
    assert doc.doc_id.startswith("chembl:")
    assert doc.license == "CC-BY-SA-3.0"


def test_opentargets_normalize():
    from biointel.ingestion.opentargets import OpenTargetsConnector

    data = load_json_fixture("opentargets_assoc.json")
    conn = OpenTargetsConnector(diseases=["EFO_0003767"])
    doc = conn.normalize(data)
    assert doc.source == SourceType.OPENTARGETS
    assert doc.doc_type == DocType.TARGET
    assert doc.doc_id.startswith("ot:")
    assert doc.license == "CC0-1.0"


def test_patentsview_normalize():
    from biointel.ingestion.patentsview import PatentsViewConnector

    data = load_json_fixture("patentsview_patent.json")
    conn = PatentsViewConnector(query_text="IL-23 inhibitor")
    doc = conn.normalize(data)
    assert doc.source == SourceType.PATENTSVIEW
    assert doc.doc_type == DocType.PATENT
    assert doc.doc_id.startswith("uspatent:")
    assert doc.ids.patent_no


def test_all_connectors_registered():
    """The runner registry must expose every implemented connector."""
    from biointel.ingestion.runner import CONNECTORS

    for key in (
        "pubmed",
        "pmc",
        "clinicaltrials",
        "chembl",
        "opentargets",
        "patentsview",
    ):
        assert key in CONNECTORS, f"{key} missing from CONNECTORS registry"


def test_google_patents_requires_credentials():
    """The optional BigQuery connector must fail clearly without GCP creds.

    Construction itself raises when GCP credentials are not configured (the
    default), which is the documented, safe behavior.
    """
    from biointel.ingestion.google_patents import GooglePatentsConnector

    with pytest.raises(Exception):  # noqa: B017 - any clear failure is acceptable
        GooglePatentsConnector(query_text="IL-23 inhibitor")
