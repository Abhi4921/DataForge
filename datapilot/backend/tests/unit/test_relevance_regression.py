"""Regression tests for the student-academic-performance relevance defects.

These reproduce the three problems reported from a real
``POST /api/v1/datasets/recommend`` run:

1. the Kaggle query was flooded with generic AI concepts
   ("artificial intelligence", "success", "achievement"),
2. a Gemini candidate whose metadata clearly listed the required features was
   scored ``feature_match = 0/5``,
3. AI candidates lost their topic domain even when their own metadata made the
   domain unambiguous.

Everything here is offline: no Kaggle or Gemini call is ever made. External
services are represented by the fixtures in ``tests.conftest``.
"""

from __future__ import annotations

import pytest

from app.schemas.dataset import RankingFactorName, SourceType, VerificationStatus
from app.services.datasets.normalizer import (
    infer_domain_from_text,
    normalize_genai_candidate,
)
from app.services.datasets.query_builder import build_dataset_search_query
from app.services.datasets.ranking import DatasetRankingEngine
from tests.conftest import (
    TEST_NOW,
    make_candidate,
    make_genai_candidate,
    make_requirements,
    make_settings,
)

# Generic analytic filler that must never crowd out concrete concepts.
GENERIC_TERMS = (
    "artificial",
    "intelligence",
    "science",
    "success",
    "achievement",
    "analytics",
)


def _rank_one(candidate, requirements=None):
    engine = DatasetRankingEngine(settings=make_settings(), now=TEST_NOW)
    return engine.rank([candidate], requirements or make_requirements())[0]


def _factor(candidate, name: RankingFactorName):
    return next(f for f in candidate.ranking_factors if f.name is name)


# ---------------------------------------------------------------------------
# Issue 1: the registry query
# ---------------------------------------------------------------------------


class TestQueryPrioritisesConcreteConcepts:
    def test_explicit_features_beat_generic_ai_keywords(self):
        requirements = make_requirements(
            keywords=[
                "artificial intelligence",
                "data science",
                "success",
                "academic achievement",
            ],
            features=["attendance", "study hours"],
            target="academic performance",
        )
        tokens = build_dataset_search_query(requirements).split()
        assert "attendance" in tokens
        assert "study" in tokens and "hours" in tokens
        for generic in GENERIC_TERMS:
            assert generic not in tokens

    def test_student_description_yields_useful_terms(self):
        requirements = make_requirements(
            keywords=["academic achievement", "artificial intelligence", "success"],
            features=[
                "attendance",
                "previous examination marks",
                "assignment scores",
                "study hours",
                "class participation",
            ],
            target="academic performance",
        )
        tokens = build_dataset_search_query(requirements).split()
        assert "academic" in tokens
        assert "performance" in tokens
        assert "attendance" in tokens
        assert "intelligence" not in tokens
        assert "success" not in tokens

    def test_query_stays_within_the_term_budget(self):
        requirements = make_requirements(
            features=[
                "attendance",
                "previous examination marks",
                "assignment scores",
                "study hours",
                "class participation",
            ]
        )
        assert len(build_dataset_search_query(requirements).split()) <= 6

    def test_verbose_model_target_is_reduced_and_leads(self):
        """The LLM sometimes writes the target as a whole sentence. It must be
        resolved to its canonical concept, not dumped into the query."""
        requirements = make_requirements(
            features=["attendance", "study hours"],
            target="Student performance metrics and support requirement status",
        )
        tokens = build_dataset_search_query(requirements).split()
        assert tokens[:2] == ["student", "performance"]
        assert "metrics" not in tokens
        assert "support" not in tokens
        assert "requirement" not in tokens

    def test_temporal_qualifiers_are_dropped(self):
        requirements = make_requirements(
            features=["previous examination marks", "attendance"],
            target="student performance",
        )
        tokens = build_dataset_search_query(requirements).split()
        assert "previous" not in tokens
        assert "examination" in tokens


# ---------------------------------------------------------------------------
# Issue 2: required-feature matching
# ---------------------------------------------------------------------------


class TestFeatureTermMatching:
    @pytest.mark.parametrize(
        "requirement,metadata",
        [
            ("attendance", "attendance"),
            ("assignment_scores", "assignment scores"),
            ("study_hours", "study hours"),
            ("class_participation", "class participation"),
        ],
    )
    def test_underscore_and_space_forms_are_equivalent(self, requirement, metadata):
        requirements = make_requirements(features=[requirement], target=None)
        candidate = make_candidate(description=metadata, feature_description=None)
        factor = _factor(_rank_one(candidate, requirements), RankingFactorName.FEATURE_MATCH)
        assert factor.score == 1.0

    @pytest.mark.parametrize(
        "requirement,metadata",
        [
            ("attendance", "attendance"),
            ("assignment scores", "assignment_scores"),
            ("study hours", "study_hours"),
            ("class participation", "class_participation"),
        ],
    )
    def test_spaced_requirements_match_underscored_metadata(self, requirement, metadata):
        requirements = make_requirements(features=[requirement], target=None)
        candidate = make_candidate(description=None, feature_description=metadata)
        factor = _factor(_rank_one(candidate, requirements), RankingFactorName.FEATURE_MATCH)
        assert factor.score == 1.0

    def test_previous_examination_marks_matches_reordered_wording(self):
        requirements = make_requirements(features=["previous_examination_marks"], target=None)
        candidate = make_candidate(
            description="Marks obtained in previous examinations",
            feature_description=None,
        )
        factor = _factor(_rank_one(candidate, requirements), RankingFactorName.FEATURE_MATCH)
        assert factor.score == 1.0

    def test_previous_examination_marks_matches_dictionary_synonym(self):
        requirements = make_requirements(features=["previous examination marks"], target=None)
        candidate = make_candidate(description="exam scores", feature_description=None)
        factor = _factor(_rank_one(candidate, requirements), RankingFactorName.FEATURE_MATCH)
        assert factor.score == 1.0

    def test_model_written_feature_descriptions_do_not_zero_the_match(self):
        """The original 0/5 came from concatenating each feature's description
        into the match term, so a dataset naming just "attendance" could never
        match "attendance <paragraph>". Matching uses the feature name."""
        requirements = make_requirements(features=["attendance"], target=None)
        requirements.dataset_requirements.feature_requirements[0].description = (
            "the percentage of classes a learner attended during the term"
        )
        candidate = make_candidate(description="attendance records", feature_description=None)
        factor = _factor(_rank_one(candidate, requirements), RankingFactorName.FEATURE_MATCH)
        assert factor.score == 1.0

    def test_reporting_feature_scores_a_full_match_for_listed_features(self):
        requirements = make_requirements(
            features=[
                "attendance",
                "previous_examination_marks",
                "assignment_scores",
                "study_hours",
                "class_participation",
            ],
            target=None,
            tasks=[],
        )
        candidate = make_candidate(
            description=None,
            feature_description="attendance; assignment_scores; class_participation; study_hours",
        )
        factor = _factor(_rank_one(candidate, requirements), RankingFactorName.FEATURE_MATCH)
        assert factor.score == pytest.approx(0.8)  # 4 of 5, as reported
        assert "previous_examination_marks" not in factor.matched_terms


# ---------------------------------------------------------------------------
# Task 3: target matching via dictionary-recognised equivalents
# ---------------------------------------------------------------------------


class TestTargetAliasMatching:
    @pytest.mark.parametrize(
        "phrase",
        [
            "student performance",
            "academic achievement",
            "student achievement",
            "academic outcome",
            "final grade",
            "GPA",
        ],
    )
    def test_academic_performance_matches_recognised_equivalents(self, phrase):
        candidate = make_candidate(description=phrase, target_description=None)
        factor = _factor(_rank_one(candidate), RankingFactorName.TARGET_MATCH)
        assert factor.score == 1.0

    def test_unrelated_target_stays_zero(self):
        candidate = make_candidate(
            description="engine temperature readings",
            target_description=None,
            domain=None,
            tags=[],
        )
        assert _factor(_rank_one(candidate), RankingFactorName.TARGET_MATCH).score == 0.0

    def test_verbose_model_target_is_reduced_to_canonical_for_matching(self):
        requirements = make_requirements(
            target="Student performance metrics and support requirement status"
        )
        candidate = make_candidate(description="academic performance records")
        factor = _factor(_rank_one(candidate, requirements), RankingFactorName.TARGET_MATCH)
        assert factor.score == 1.0


# ---------------------------------------------------------------------------
# Task 4: conservative domain inference for AI candidates
# ---------------------------------------------------------------------------


class TestAIDomainInference:
    def test_ai_candidate_domain_inferred_from_its_own_metadata(self):
        ai = normalize_genai_candidate(
            make_genai_candidate(
                name="Students Academic Performance Dataset",
                description="Records of secondary school students",
                features=["attendance", "study hours"],
                target_description="final grade",
            )
        )
        assert ai.domain == "education"
        factor = _factor(_rank_one(ai), RankingFactorName.DOMAIN_RELEVANCE)
        assert factor.score >= 0.85

    def test_ai_candidate_without_signal_keeps_no_domain(self):
        ai = normalize_genai_candidate(
            make_genai_candidate(
                name="Penguin Colony Census",
                description="Beak lengths from field surveys",
                features=["beak length"],
                target_description="colony size",
            )
        )
        assert ai.domain is None

    def test_inference_returns_none_for_unmapped_text(self):
        assert infer_domain_from_text("qwerty zxcvbn") is None
        assert infer_domain_from_text("") is None
        assert infer_domain_from_text(None) is None

    def test_inference_uses_only_candidate_metadata_not_provenance(self):
        ai = normalize_genai_candidate(
            make_genai_candidate(
                name="Bare Measurements",
                description="Anonymised numeric columns",
                features=["x", "y"],
                target_description="z",
            )
        )
        assert ai.domain is None


# ---------------------------------------------------------------------------
# Task 5: verification distinction is preserved
# ---------------------------------------------------------------------------


class TestVerificationPreserved:
    def test_ai_candidate_remains_unverified_after_ranking(self):
        ai = normalize_genai_candidate(
            make_genai_candidate(name="Students Academic Performance Dataset")
        )
        ranked = _rank_one(ai)
        assert ranked.source_type is SourceType.AI_SUGGESTED
        assert ranked.verification_status is VerificationStatus.UNVERIFIED
        assert _factor(ranked, RankingFactorName.VERIFICATION).score == (
            make_settings().ranking_trust_unverified
        )

    def test_verified_candidate_keeps_verified_external(self):
        ranked = _rank_one(make_candidate())
        assert ranked.source_type is SourceType.VERIFIED_EXTERNAL
        assert ranked.verification_status is VerificationStatus.VERIFIED
        assert _factor(ranked, RankingFactorName.VERIFICATION).score == (
            make_settings().ranking_trust_verified
        )


# ---------------------------------------------------------------------------
# Task 7: reasons come only from real matches
# ---------------------------------------------------------------------------


class TestRankingReasonsFromEvidence:
    def test_feature_reason_lists_only_matched_terms(self):
        requirements = make_requirements(
            features=["attendance", "study hours", "tuition fees"],
            target=None,
            tasks=[],
        )
        candidate = make_candidate(description="attendance and study hours", feature_description=None)
        ranked = _rank_one(candidate, requirements)
        reason = next(r for r in ranked.ranking_reasons if "required features" in r)
        lowered = reason.lower()
        assert "attendance" in lowered
        assert "study hours" in lowered
        assert "tuition fees" not in lowered

    def test_no_reason_without_a_match(self):
        candidate = make_candidate(
            description="Volcanic rock spectroscopy readings",
            target_description=None,
            feature_description=None,
            domain=None,
            tags=[],
        )
        ranked = _rank_one(candidate)
        assert not any("required features" in r.lower() for r in ranked.ranking_reasons)
        assert not any("prediction target" in r.lower() for r in ranked.ranking_reasons)


# ---------------------------------------------------------------------------
# Task 6/10: ranking weights remain valid
# ---------------------------------------------------------------------------


class TestRankingWeights:
    def test_default_weights_sum_to_one_hundred(self):
        weights = DatasetRankingEngine(settings=make_settings()).weights()
        assert sum(weights.values()) == pytest.approx(100.0)
        assert set(weights) == {factor.value for factor in RankingFactorName}
