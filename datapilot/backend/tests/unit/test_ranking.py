"""Unit tests for the deterministic dataset ranking engine.

The clock is injected, so every assertion here is reproducible.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.schemas.dataset import (
    DatasetCandidate,
    RankingFactorName,
    SourceType,
    VerificationStatus,
)
from app.services.datasets.normalizer import normalize_genai_candidate
from app.services.datasets.ranking import (
    DatasetRankingEngine,
    build_project_profile,
)
from tests.conftest import (
    TEST_NOW,
    make_candidate,
    make_genai_candidate,
    make_requirements,
    make_settings,
)


def _engine(**overrides) -> DatasetRankingEngine:
    return DatasetRankingEngine(settings=make_settings(**overrides), now=TEST_NOW)


def _factor(candidate: DatasetCandidate, name: RankingFactorName):
    for factor in candidate.ranking_factors:
        if factor.name is name:
            return factor
    raise AssertionError(f"factor {name} not present")


def _rank_one(candidate: DatasetCandidate, requirements=None, **overrides):
    engine = _engine(**overrides)
    return engine.rank([candidate], requirements or make_requirements())[0]


class TestRankingShape:
    def test_empty_input_returns_empty(self):
        assert _engine().rank([], make_requirements()) == []

    def test_every_candidate_receives_scores_and_factors(self):
        candidates = [make_candidate(), make_candidate(name="Second", source_id="owner/second")]
        ranked = _engine().rank(candidates, make_requirements())

        assert len(ranked) == 2
        for candidate in ranked:
            assert candidate.ranking_score is not None
            assert 0.0 <= candidate.ranking_score <= 100.0
            assert candidate.relevance_score is not None
            assert len(candidate.ranking_factors) == 9

    def test_all_nine_factors_present(self):
        ranked = _rank_one(make_candidate())
        names = {f.name for f in ranked.ranking_factors}
        assert names == set(RankingFactorName)

    def test_factor_contributions_sum_to_ranking_score(self):
        ranked = _rank_one(make_candidate())
        total = sum(f.contribution for f in ranked.ranking_factors)
        assert total == pytest.approx(ranked.ranking_score, abs=0.05)

    def test_relevance_excludes_verification_factor(self):
        ranked = _rank_one(make_candidate())
        assert "VERIFICATION" not in [
            f.name.value for f in ranked.ranking_factors if f.name is not RankingFactorName.VERIFICATION
        ]
        assert ranked.relevance_score <= 100.0


class TestDeterminism:
    def test_same_inputs_produce_same_order(self):
        candidates = [
            make_candidate(name="A", source_id="owner/a", download_count=100),
            make_candidate(name="B", source_id="owner/b", download_count=900000),
            make_candidate(name="C", source_id="owner/c", download_count=5000),
        ]
        first = [c.id for c in _engine().rank(candidates, make_requirements())]
        second = [c.id for c in _engine().rank(candidates, make_requirements())]
        assert first == second

    def test_ties_break_deterministically_by_name(self):
        candidates = [
            make_candidate(name="Zebra", source_id="owner/z"),
            make_candidate(name="Alpha", source_id="owner/a"),
        ]
        ranked = _engine().rank(candidates, make_requirements())
        assert [c.name for c in ranked] == ["Alpha", "Zebra"]

    def test_identical_scores_prefer_verified_over_ai(self):
        verified = make_candidate(name="Same Name", source_id="owner/v")
        # Byte-identical content, but suggested by the model rather than listed
        # by a registry: the content factors tie, so trust must decide.
        ai = verified.model_copy(
            update={
                "id": "genai:same-name",
                "source": "genai",
                "source_id": "same-name",
                "source_type": SourceType.AI_SUGGESTED,
                "verification_status": VerificationStatus.UNVERIFIED,
            }
        )
        ranked = _engine().rank([ai, verified], make_requirements())
        assert ranked[0].verification_status is VerificationStatus.VERIFIED


class TestDomainRelevance:
    def test_matching_topic_domain_scores_high(self):
        ranked = _rank_one(make_candidate(domain="education"))
        factor = _factor(ranked, RankingFactorName.DOMAIN_RELEVANCE)
        assert factor.score >= 0.85

    def test_mismatched_topic_domain_scores_low(self):
        ranked = _rank_one(
            make_candidate(domain="finance", tags=["finance"], description="Loan defaults")
        )
        assert _factor(ranked, RankingFactorName.DOMAIN_RELEVANCE).score < 0.5

    def test_domain_terms_in_text_raise_score(self):
        with_domain = _rank_one(
            make_candidate(description="Education data for universities", domain=None, tags=[])
        )
        without = _rank_one(
            make_candidate(description="Unrelated tabular measurements", domain=None, tags=[])
        )
        assert (
            _factor(with_domain, RankingFactorName.DOMAIN_RELEVANCE).score
            > _factor(without, RankingFactorName.DOMAIN_RELEVANCE).score
        )

    def test_unavailable_when_project_domain_unknown(self):
        requirements = make_requirements(domain=None, subdomain=None)
        ranked = _rank_one(make_candidate(domain=None), requirements=requirements)
        assert _factor(ranked, RankingFactorName.DOMAIN_RELEVANCE).score is None


class TestFeatureMatching:
    def test_required_features_in_metadata_raise_score(self):
        ranked = _rank_one(
            make_candidate(
                description=(
                    "Attendance, previous examination marks, assignment scores "
                    "and study hours"
                )
            )
        )
        factor = _factor(ranked, RankingFactorName.FEATURE_MATCH)
        assert factor.score == 1.0
        assert "attendance" in [m.lower() for m in factor.matched_terms]

    def test_no_feature_overlap_scores_zero(self):
        ranked = _rank_one(make_candidate(description="Volcanic rock spectroscopy readings"))
        assert _factor(ranked, RankingFactorName.FEATURE_MATCH).score == 0.0

    def test_partial_overlap_is_proportional(self):
        ranked = _rank_one(make_candidate(description="Attendance only"))
        factor = _factor(ranked, RankingFactorName.FEATURE_MATCH)
        assert 0.0 < factor.score < 1.0

    def test_unavailable_when_no_required_features(self):
        requirements = make_requirements(features=[])
        ranked = _rank_one(make_candidate(), requirements=requirements)
        assert _factor(ranked, RankingFactorName.FEATURE_MATCH).score is None

    def test_matched_terms_are_real_project_terms(self):
        ranked = _rank_one(make_candidate(description="study hours only"))
        factor = _factor(ranked, RankingFactorName.FEATURE_MATCH)
        for term in factor.matched_terms:
            assert term in {"attendance", "previous examination marks", "assignment scores", "study hours"}


class TestTargetMatching:
    def test_target_terms_raise_score(self):
        ranked = _rank_one(
            make_candidate(target_description="final academic performance score")
        )
        assert _factor(ranked, RankingFactorName.TARGET_MATCH).score > 0

    def test_unrelated_target_scores_zero(self):
        ranked = _rank_one(make_candidate(target_description="engine temperature"))
        assert _factor(ranked, RankingFactorName.TARGET_MATCH).score == 0.0

    def test_unavailable_when_no_target(self):
        requirements = make_requirements(target=None)
        ranked = _rank_one(make_candidate(), requirements=requirements)
        assert _factor(ranked, RankingFactorName.TARGET_MATCH).score is None


class TestTaskCompatibility:
    def test_declared_tasks_match(self):
        ranked = _rank_one(make_candidate(task_types=["classification", "regression"]))
        assert _factor(ranked, RankingFactorName.TASK_MATCH).score == 1.0

    def test_declared_tasks_take_priority_over_text(self):
        ranked = _rank_one(
            make_candidate(
                task_types=["clustering"],
                description="supports classification and regression",
            )
        )
        factor = _factor(ranked, RankingFactorName.TASK_MATCH)
        assert factor.score == 0.0
        assert "declared tasks" in factor.detail

    def test_text_is_used_when_no_declared_tasks(self):
        ranked = _rank_one(make_candidate(description="Great for classification and regression"))
        assert _factor(ranked, RankingFactorName.TASK_MATCH).score == 1.0

    def test_unavailable_when_project_has_no_tasks(self):
        requirements = make_requirements(tasks=[])
        ranked = _rank_one(make_candidate(), requirements=requirements)
        assert _factor(ranked, RankingFactorName.TASK_MATCH).score is None


class TestVerificationStatus:
    def test_verified_external_earns_full_trust(self):
        ranked = _rank_one(make_candidate())
        factor = _factor(ranked, RankingFactorName.VERIFICATION)
        assert factor.score == make_settings().ranking_trust_verified
        assert factor.matched_terms == ["kaggle"]

    def test_ai_suggested_earns_reduced_trust(self):
        ai = normalize_genai_candidate(make_genai_candidate())
        ranked = _rank_one(ai)
        assert _factor(ranked, RankingFactorName.VERIFICATION).score == (
            make_settings().ranking_trust_unverified
        )

    def test_unverified_external_record_cannot_claim_full_trust(self):
        candidate = make_candidate(verification_status=VerificationStatus.UNVERIFIED)
        ranked = _rank_one(candidate)
        assert _factor(ranked, RankingFactorName.VERIFICATION).score == (
            make_settings().ranking_trust_unverified
        )

    def test_perfect_ai_candidate_cannot_outrank_perfect_verified(self):
        """The core guarantee of the trust factor."""
        verified = make_candidate(
            name="Student Academic Performance",
            description="Attendance, examination marks, assignment scores, study hours, academic performance",
            task_types=["classification", "regression"],
            domain="education",
        )
        ai = normalize_genai_candidate(
            make_genai_candidate(
                name="Student Academic Performance",
                features=[
                    "attendance",
                    "previous examination marks",
                    "assignment scores",
                    "study hours",
                ],
            )
        )
        ai.target_description = "academic performance"
        ai.domain = "education"

        ranked = _engine().rank([ai, verified], make_requirements())
        assert ranked[0].source_type is SourceType.VERIFIED_EXTERNAL
        assert ranked[0].ranking_score > ranked[1].ranking_score
        assert ranked[1].ranking_score < 90.0

    def test_irrelevant_ai_candidate_scores_near_zero(self):
        ai = normalize_genai_candidate(
            make_genai_candidate(
                name="Penguin Colony Census",
                description="Beak lengths and colony counts from field surveys",
                features=["beak length"],
                tasks=["clustering"],
                why_relevant="Unrelated to the project",
            )
        )
        ai.target_description = "colony size"
        ranked = _rank_one(ai)
        assert ranked.ranking_score < 15.0

    def test_trust_weights_are_configurable(self):
        engine = _engine(ranking_trust_unverified=0.9, ranking_trust_verified=1.0)
        ai = normalize_genai_candidate(make_genai_candidate())
        ranked = engine.rank([ai], make_requirements())[0]
        assert _factor(ranked, RankingFactorName.VERIFICATION).score == 0.9


class TestQualityAndPopularity:
    def test_usability_is_used(self):
        ranked = _rank_one(make_candidate(usability=0.75))
        assert _factor(ranked, RankingFactorName.QUALITY).score == 0.75

    def test_missing_usability_is_excluded_not_zeroed(self):
        ranked = _rank_one(make_candidate(usability=None))
        factor = _factor(ranked, RankingFactorName.QUALITY)
        assert factor.score is None
        assert factor.contribution == 0.0

    def test_missing_usability_does_not_crush_the_score(self):
        with_quality = _rank_one(make_candidate(usability=0.0))
        without_quality = _rank_one(make_candidate(usability=None))
        assert without_quality.ranking_score > with_quality.ranking_score

    def test_higher_downloads_score_higher(self):
        low = _factor(_rank_one(make_candidate(download_count=10)), RankingFactorName.POPULARITY)
        high = _factor(
            _rank_one(make_candidate(download_count=900000)), RankingFactorName.POPULARITY
        )
        assert high.score > low.score

    def test_popularity_is_log_scaled(self):
        factor = _factor(
            _rank_one(make_candidate(download_count=1_000_000, vote_count=10_000)),
            RankingFactorName.POPULARITY,
        )
        assert factor.score == pytest.approx(1.0)

    def test_zero_downloads_score_zero(self):
        factor = _factor(
            _rank_one(make_candidate(download_count=0, vote_count=0)),
            RankingFactorName.POPULARITY,
        )
        assert factor.score == 0.0

    def test_missing_counts_excluded(self):
        factor = _factor(
            _rank_one(make_candidate(download_count=None, vote_count=None)),
            RankingFactorName.POPULARITY,
        )
        assert factor.score is None

    def test_downloads_alone_are_sufficient(self):
        factor = _factor(
            _rank_one(make_candidate(download_count=50000, vote_count=None)),
            RankingFactorName.POPULARITY,
        )
        assert factor.score is not None


class TestRecency:
    def test_fresh_dataset_scores_high(self):
        recent = _rank_one(make_candidate(updated_at=TEST_NOW - timedelta(days=1)))
        assert _factor(recent, RankingFactorName.RECENCY).score > 0.99

    def test_older_dataset_scores_lower(self):
        recent = _rank_one(make_candidate(updated_at=TEST_NOW - timedelta(days=10)))
        old = _rank_one(make_candidate(updated_at=TEST_NOW - timedelta(days=1000)))
        assert (
            _factor(recent, RankingFactorName.RECENCY).score
            > _factor(old, RankingFactorName.RECENCY).score
        )

    def test_very_old_dataset_floors_at_zero(self):
        old = _rank_one(make_candidate(updated_at=TEST_NOW - timedelta(days=5000)))
        assert _factor(old, RankingFactorName.RECENCY).score == 0.0

    def test_future_timestamp_does_not_exceed_one(self):
        future = _rank_one(make_candidate(updated_at=TEST_NOW + timedelta(days=10)))
        assert _factor(future, RankingFactorName.RECENCY).score == 1.0

    def test_missing_date_excluded(self):
        factor = _factor(
            _rank_one(make_candidate(updated_at=None)), RankingFactorName.RECENCY
        )
        assert factor.score is None

    def test_naive_datetime_is_treated_as_utc(self):
        naive = _rank_one(make_candidate(updated_at=datetime(2026, 9, 1, 0, 0, 0)))
        assert _factor(naive, RankingFactorName.RECENCY).score is not None


class TestWeightsAreConfigurable:
    def test_weights_sum_to_one_hundred_by_default(self):
        assert sum(_engine().weights().values()) == pytest.approx(100.0)

    def test_all_nine_weights_exposed(self):
        assert set(_engine().weights()) == {n.value for n in RankingFactorName}

    def test_raising_verification_weight_widens_the_trust_gap(self):
        verified = make_candidate()
        ai = normalize_genai_candidate(make_genai_candidate())
        low = _engine(ranking_weight_verification=15.0).rank(
            [verified.model_copy(deep=True), ai.model_copy(deep=True)], make_requirements()
        )
        high = _engine(ranking_weight_verification=60.0).rank(
            [verified.model_copy(deep=True), ai.model_copy(deep=True)], make_requirements()
        )
        gap_low = low[0].ranking_score - low[1].ranking_score
        gap_high = high[0].ranking_score - high[1].ranking_score
        assert gap_high > gap_low


class TestRankingReasons:
    def test_match_reasons_only_reference_real_matches(self):
        ranked = _rank_one(
            make_candidate(
                description="Attendance and study hours for universities",
                domain="education",
            )
        )
        match_reasons = [
            r
            for r in ranked.ranking_reasons
            if any(
                marker in r
                for marker in ("domain", "concepts", "required features", "target", "ML task")
            )
        ]
        assert match_reasons
        for reason in match_reasons:
            lowered = reason.lower()
            assert any(
                term in lowered
                for term in ("attendance", "study hours", "education", "classification")
            ), reason

    def test_no_reason_for_zero_score_factors(self):
        ranked = _rank_one(make_candidate(description="Volcanic rock spectroscopy"))
        assert not any("required features" in r.lower() for r in ranked.ranking_reasons)
        assert not any("prediction target" in r.lower() for r in ranked.ranking_reasons)

    def test_feature_reason_lists_matched_terms(self):
        ranked = _rank_one(make_candidate(description="Attendance and study hours"))
        reasons = [r for r in ranked.ranking_reasons if "required features" in r]
        assert reasons
        assert "attendance" in reasons[0].lower()

    def test_verified_reason_states_external_registry(self):
        ranked = _rank_one(make_candidate())
        assert any("Verified listing" in r for r in ranked.ranking_reasons)

    def test_ai_reason_states_it_is_not_confirmed(self):
        ai = normalize_genai_candidate(make_genai_candidate())
        ranked = _rank_one(ai)
        reasons = [r for r in ranked.ranking_reasons if "AI-suggested" in r]
        assert reasons
        assert "not yet confirmed" in reasons[0]

    def test_ai_rationale_is_not_used_as_a_reason(self):
        ai = normalize_genai_candidate(
            make_genai_candidate(why_relevant="PERFECTLY MATCHES EVERYTHING")
        )
        ranked = _rank_one(ai)
        assert all("PERFECTLY MATCHES" not in r for r in ranked.ranking_reasons)

    def test_ai_rationale_cannot_inflate_match_score(self):
        """Evidence text excludes the model's own opinion."""
        with_rationale = _rank_one(
            normalize_genai_candidate(
                make_genai_candidate(
                    name="Totally Unrelated",
                    description="Unrelated measurements",
                    features=["unrelated"],
                    why_relevant="academic performance attendance study hours",
                )
            )
        )
        without_rationale = _rank_one(
            normalize_genai_candidate(
                make_genai_candidate(
                    name="Totally Unrelated",
                    description="Unrelated measurements",
                    features=["unrelated"],
                    why_relevant="",
                )
            )
        )
        assert with_rationale.ranking_score == pytest.approx(
            without_rationale.ranking_score
        )

    def test_reason_count_is_capped(self):
        ranked = _rank_one(make_candidate())
        assert len(ranked.ranking_reasons) <= 7  # 6 match reasons + trust

    def test_unrelated_dataset_gets_only_trust_reasons(self):
        ranked = _rank_one(
            make_candidate(
                name="Penguin Colony Census",
                description="Beak lengths and colony counts",
                domain="environment",
                tags=["environment"],
            )
        )
        assert all(
            any(
                keyword in reason
                for keyword in ("Verified listing", "Adoption", "quality score", "Last updated")
            )
            for reason in ranked.ranking_reasons
        )


class TestProjectProfile:
    def test_extracts_terms_from_requirements(self):
        profile = build_project_profile(make_requirements())
        assert "Education" in profile.domain_terms
        assert "attendance" in profile.feature_terms
        assert "academic performance" in profile.target_terms
        assert "classification" in profile.task_terms
        assert profile.project_domain_key == "education"

    def test_maps_project_domain_into_dictionary_taxonomy(self):
        profile = build_project_profile(make_requirements(domain="Healthcare"))
        assert profile.project_domain_key == "healthcare"

    def test_generic_words_are_dropped_from_terms(self):
        profile = build_project_profile(
            make_requirements(features=["the system data"], keywords=["machine learning model"])
        )
        assert "the system data" not in profile.feature_terms
        assert "machine learning model" not in profile.concept_terms

    def test_duplicate_terms_are_collapsed(self):
        requirements = make_requirements(
            features=["attendance", "attendance", "study hours"]
        )
        profile = build_project_profile(requirements)
        assert profile.feature_terms.count("attendance") == 1

    def test_empty_requirements_produce_empty_profile(self):
        requirements = make_requirements(
            domain=None, subdomain=None, features=[], keywords=[], tasks=[], target=None
        )
        profile = build_project_profile(requirements)
        assert profile.domain_terms == []
        assert profile.feature_terms == []
        assert profile.task_terms == []
        assert profile.target_terms == []


class TestRankingWithoutRequirements:
    def test_engine_handles_candidate_with_minimal_metadata(self):
        candidate = DatasetCandidate(
            id="genai:bare",
            name="Bare Dataset",
            source="genai",
            source_type=SourceType.AI_SUGGESTED,
            source_id="bare",
        )
        ranked = _rank_one(candidate, requirements=make_requirements())
        assert ranked.ranking_score is not None
        assert ranked.relevance_score is not None
