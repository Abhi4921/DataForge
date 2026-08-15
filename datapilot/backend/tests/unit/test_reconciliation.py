from __future__ import annotations

import pytest

from app.services.reconciliation_service import ReconciliationService
from app.schemas.project import (
    KeywordMatchEvidence,
    LLMProjectAnalysis,
    LLMProjectUnderstanding,
    LLMConfidenceField,
    LLMMLTask,
    LLMKeywords,
    LLMDatasetRequirements,
    LLMFeatureRequirement,
    MatchType,
)


@pytest.fixture
def recon() -> ReconciliationService:
    return ReconciliationService()


def _make_dict_match(
    canonical: str = "phishing",
    text: str = "phishing",
    domain: str = "cybersecurity",
    subdomain: str = "web_security",
) -> KeywordMatchEvidence:
    return KeywordMatchEvidence(
        canonical_term=canonical,
        matched_text=text,
        match_type=MatchType.EXACT,
        dictionary_entry_id=f"cybersecurity.{canonical}",
        domain=domain,
        subdomain=subdomain,
    )


def _make_llm_output(
    domain: str = "cybersecurity",
    subdomain: str = "web security",
    needs_clarification: bool = False,
    concepts: list[str] | None = None,
    related: list[str] | None = None,
) -> LLMProjectAnalysis:
    return LLMProjectAnalysis(
        project_understanding=LLMProjectUnderstanding(
            domain=LLMConfidenceField(value=domain, confidence=0.9, status="inferred"),
            subdomain=LLMConfidenceField(value=subdomain, confidence=0.85, status="inferred"),
            problem_type=LLMConfidenceField(value="Machine Learning", confidence=0.9, status="inferred"),
            ml_tasks=[LLMMLTask(task="classification", confidence=0.85)],
            objective="Detect phishing websites",
            target=LLMConfidenceField(value="phishing/legitimate", confidence=0.7, status="inferred"),
        ),
        keywords=LLMKeywords(
            inferred_concepts=concepts or [],
            related_terms=related or [],
        ),
        dataset_requirements=LLMDatasetRequirements(
            required_features=[
                LLMFeatureRequirement(name="URL features", description="URL characteristics"),
                LLMFeatureRequirement(name="domain info", description="Domain information"),
            ],
            target_description="phishing vs legitimate label",
            label_description="binary label",
            data_type_notes="Tabular data with URL features",
        ),
        ambiguities=[],
        missing_information=[],
        needs_clarification=needs_clarification,
        overall_confidence=0.85,
    )


class TestReconciliationKeywords:
    def test_merges_dictionary_and_llm(self, recon: ReconciliationService):
        dict_matches = [_make_dict_match("phishing", "phishing websites")]
        llm = _make_llm_output(concepts=["website classification", "URL analysis"])
        result = recon.reconcile("detect phishing websites", dict_matches, llm, 3)

        canonicals = set(c.lower() for c in result.keywords.canonical)
        assert "phishing" in canonicals
        assert "website classification" in canonicals
        assert "url analysis" in canonicals

    def test_deduplicates(self, recon: ReconciliationService):
        dict_matches = [_make_dict_match("phishing", "phishing")]
        llm = _make_llm_output(concepts=["phishing"])
        result = recon.reconcile("detect phishing", dict_matches, llm, 2)

        assert result.keywords.canonical.count("phishing") == 1

    def test_provenance_tracking(self, recon: ReconciliationService):
        dict_matches = [_make_dict_match("phishing", "phishing")]
        llm = _make_llm_output(concepts=["phishing"])
        result = recon.reconcile("detect phishing", dict_matches, llm, 2)

        phishing_prov = next(
            (p for p in result.keywords.provenance if p.canonical_term == "phishing"),
            None,
        )
        assert phishing_prov is not None
        assert "dictionary" in phishing_prov.sources
        assert "llm" in phishing_prov.sources

    def test_related_concepts_included(self, recon: ReconciliationService):
        dict_matches = []
        llm = _make_llm_output(related=["website classification", "URL analysis"])
        result = recon.reconcile("detect phishing", dict_matches, llm, 2)

        assert "website classification" in result.keywords.related_concepts


class TestReconciliationUnderstanding:
    def test_domain_from_dict_when_no_llm(self, recon: ReconciliationService):
        dict_matches = [_make_dict_match("phishing", "phishing", "cybersecurity", "web_security")]
        llm = _make_llm_output(domain=None, subdomain=None)
        llm.project_understanding.domain = LLMConfidenceField(value=None, confidence=0, status="unknown")
        llm.project_understanding.subdomain = LLMConfidenceField(value=None, confidence=0, status="unknown")
        result = recon.reconcile("detect phishing", dict_matches, llm, 2)

        assert result.project_understanding.domain is not None
        assert "cybersecurity" in result.project_understanding.domain.value.lower()

    def test_domain_from_llm_when_no_dict(self, recon: ReconciliationService):
        dict_matches = []
        llm = _make_llm_output(domain="healthcare", subdomain="clinical")
        result = recon.reconcile("predict disease", dict_matches, llm, 2)

        assert result.project_understanding.domain is not None
        assert "healthcare" in result.project_understanding.domain.value.lower()

    def test_ml_tasks_preserved(self, recon: ReconciliationService):
        dict_matches = []
        llm = _make_llm_output()
        result = recon.reconcile("test", dict_matches, llm, 1)

        assert len(result.project_understanding.ml_tasks) > 0
        assert result.project_understanding.ml_tasks[0].task == "classification"

    def test_objective_preserved(self, recon: ReconciliationService):
        dict_matches = []
        llm = _make_llm_output()
        result = recon.reconcile("test", dict_matches, llm, 1)

        assert result.project_understanding.objective == "Detect phishing websites"


class TestReconciliationAnalysis:
    def test_overall_confidence_bounded(self, recon: ReconciliationService):
        dict_matches = [_make_dict_match()]
        llm = _make_llm_output()
        llm.overall_confidence = 0.95
        result = recon.reconcile("test", dict_matches, llm, 1)

        assert 0.0 <= result.analysis.overall_confidence <= 1.0

    def test_needs_clarification_preserved(self, recon: ReconciliationService):
        dict_matches = []
        llm = _make_llm_output(needs_clarification=True)
        result = recon.reconcile("test", dict_matches, llm, 1)

        assert result.analysis.needs_clarification is True


class TestReconciliationDatasetRequirements:
    def test_features_from_llm(self, recon: ReconciliationService):
        dict_matches = []
        llm = _make_llm_output()
        result = recon.reconcile("test", dict_matches, llm, 1)

        assert len(result.dataset_requirements.feature_requirements) > 0

    def test_target_from_llm(self, recon: ReconciliationService):
        dict_matches = []
        llm = _make_llm_output()
        result = recon.reconcile("test", dict_matches, llm, 1)

        assert len(result.dataset_requirements.target_requirements) > 0
