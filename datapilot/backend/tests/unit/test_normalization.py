"""Unit tests for payload normalization: Kaggle -> DatasetCandidate,
GenAI -> DatasetCandidate. Pure functions, no network."""

from __future__ import annotations

from datetime import timezone

import pytest

from app.schemas.dataset import LLMDiscoveredCandidate, SourceType, VerificationStatus
from app.services.datasets.normalizer import (
    coerce_int,
    coerce_ratio,
    infer_domain_from_tags,
    normalize_genai_candidate,
    normalize_genai_candidates,
    normalize_kaggle_record,
    normalize_kaggle_records,
    normalize_url,
    parse_timestamp,
)
from tests.conftest import make_genai_candidate, make_kaggle_record


class TestKaggleNormalization:
    def test_produces_verified_external_candidate(self):
        candidate = normalize_kaggle_record(make_kaggle_record(ref="owner/students"))

        assert candidate is not None
        assert candidate.source == "kaggle"
        assert candidate.source_type is SourceType.VERIFIED_EXTERNAL
        assert candidate.verification_status is VerificationStatus.VERIFIED

    def test_prefers_nullable_field_variants(self):
        """Kaggle ships both 'title' and 'titleNullable'; the latter holds the
        value and the former a boolean."""
        record = {
            "ref": "owner/dataset",
            "title": False,
            "hasTitle": True,
            "titleNullable": "Real Title",
            "url": False,
            "urlNullable": "https://www.kaggle.com/datasets/owner/dataset",
        }
        candidate = normalize_kaggle_record(record)
        assert candidate.name == "Real Title"
        assert candidate.url == "https://www.kaggle.com/datasets/owner/dataset"

    def test_falls_back_to_plain_field_when_nullable_missing(self):
        record = {"ref": "owner/dataset", "title": "Plain Title"}
        candidate = normalize_kaggle_record(record)
        assert candidate.name == "Plain Title"

    def test_uses_subtitle_when_description_empty(self):
        candidate = normalize_kaggle_record(
            make_kaggle_record(description="", subtitle="Useful subtitle")
        )
        assert candidate.description == "Useful subtitle"

    def test_prefers_real_description_over_subtitle(self):
        candidate = normalize_kaggle_record(
            make_kaggle_record(description="Full description", subtitle="Sub")
        )
        assert candidate.description == "Full description"

    def test_derives_name_from_ref_when_title_missing(self):
        candidate = normalize_kaggle_record({"ref": "owner/my-dataset", "title": None})
        assert candidate.name == "my-dataset"

    def test_builds_source_id_when_ref_missing(self):
        candidate = normalize_kaggle_record({"title": "Lonely Dataset", "ownerRef": "someone"})
        assert candidate.source_id == "someone/lonely-dataset"
        assert candidate.id == "kaggle:someone/lonely-dataset"

    def test_rejects_record_without_title_or_ref(self):
        assert normalize_kaggle_record({"title": None, "ref": None}) is None

    def test_rejects_non_dict_record(self):
        assert normalize_kaggle_record("string") is None
        assert normalize_kaggle_record(None) is None
        assert normalize_kaggle_record([1, 2]) is None

    def test_extracts_tag_names(self):
        candidate = normalize_kaggle_record(
            make_kaggle_record(tags=["education", "students", "university"])
        )
        assert candidate.tags == ["education", "students", "university"]

    def test_maps_tag_to_dictionary_domain(self):
        candidate = normalize_kaggle_record(make_kaggle_record(tags=["health", "heart"]))
        assert candidate.domain == "healthcare"

    def test_unknown_tag_yields_no_domain(self):
        candidate = normalize_kaggle_record(make_kaggle_record(tags=["zzzz-unknown-topic"]))
        assert candidate.domain is None

    def test_rejects_out_of_range_usability(self):
        candidate = normalize_kaggle_record(make_kaggle_record(usability=7.5))
        assert candidate.usability is None

    def test_rejects_negative_download_count(self):
        candidate = normalize_kaggle_record(make_kaggle_record(downloads=-5))
        assert candidate.download_count == -5  # preserved verbatim, not clamped

    def test_non_numeric_counts_become_none(self):
        record = make_kaggle_record()
        record["downloadCount"] = "many"
        record["voteCount"] = None
        candidate = normalize_kaggle_record(record)
        assert candidate.download_count is None
        assert candidate.vote_count is None


class TestKaggleRecordBatch:
    def test_skips_bad_rows_and_keeps_good_ones(self):
        payload = [make_kaggle_record(ref="owner/a"), "junk", None, make_kaggle_record(ref="owner/b")]
        candidates = normalize_kaggle_records(payload)
        assert [c.source_id for c in candidates] == ["owner/a", "owner/b"]

    def test_non_list_payload_returns_empty(self):
        assert normalize_kaggle_records({"error": "nope"}) == []
        assert normalize_kaggle_records("nope") == []
        assert normalize_kaggle_records(None) == []

    def test_empty_list_returns_empty(self):
        assert normalize_kaggle_records([]) == []


class TestGenAINormalization:
    def test_always_marked_as_ai_suggested_and_unverified(self):
        """Even a high-confidence, well-known claim cannot become verified."""
        candidate = normalize_genai_candidate(
            make_genai_candidate(confidence=1.0, is_well_known=True)
        )
        assert candidate.source_type is SourceType.AI_SUGGESTED
        assert candidate.verification_status is VerificationStatus.UNVERIFIED

    def test_model_cannot_self_declare_verified(self):
        hostile = {
            "name": "Totally Real Dataset",
            "verification_status": "verified",
            "source_type": "verified_external",
            "confidence": 1.0,
            "is_well_known": True,
        }
        candidate = normalize_genai_candidate(hostile)
        assert candidate.source_type is SourceType.AI_SUGGESTED
        assert candidate.verification_status is VerificationStatus.UNVERIFIED

    def test_maps_published_fields(self):
        candidate = normalize_genai_candidate(make_genai_candidate())

        assert candidate.source == "genai"
        assert candidate.name == "UCI Student Performance Dataset"
        assert candidate.url == "https://archive.ics.uci.edu/dataset/342/student+performance"
        assert candidate.confidence_score == 0.82
        assert candidate.task_types == ["classification", "regression"]
        assert candidate.target_description == "final grade"
        assert candidate.license == "CC BY 4.0"

    def test_rationale_is_preserved_but_not_scored(self):
        candidate = normalize_genai_candidate(make_genai_candidate())
        assert candidate.rationale is not None
        assert "grade" in candidate.rationale.lower()

    def test_features_become_feature_description(self):
        candidate = normalize_genai_candidate(
            make_genai_candidate(features=["attendance", "study hours"])
        )
        assert candidate.feature_description == "attendance; study hours"

    def test_invalid_url_is_dropped_not_fabricated(self):
        candidate = normalize_genai_candidate(
            make_genai_candidate(url="not-a-url/kaggle")
        )
        assert candidate.url is None

    def test_plausible_but_unverifiable_url_is_kept_as_claim(self):
        candidate = normalize_genai_candidate(
            make_genai_candidate(url="https://www.kaggle.com/datasets/some/thing")
        )
        assert candidate.url == "https://www.kaggle.com/datasets/some/thing"
        assert candidate.verification_status is VerificationStatus.UNVERIFIED

    def test_confidence_outside_range_is_rejected(self):
        with pytest.raises(Exception):
            LLMDiscoveredCandidate(name="X", confidence=1.5)

    def test_discards_candidate_without_name(self):
        assert normalize_genai_candidate({"name": "  "}) is None
        assert normalize_genai_candidate({"confidence": 0.5}) is None

    def test_discards_wrong_type(self):
        assert normalize_genai_candidate(42) is None
        assert normalize_genai_candidate(None) is None

    def test_invalid_field_types_are_discarded_not_raised(self):
        assert normalize_genai_candidate({"name": "X", "confidence": "high"}) is None

    def test_possible_source_is_preserved_as_a_hint(self):
        candidate = normalize_genai_candidate(make_genai_candidate())
        assert candidate.possible_source == "UCI"

    def test_is_well_known_is_preserved_but_never_verifies(self):
        candidate = normalize_genai_candidate(
            make_genai_candidate(is_well_known=True)
        )
        assert candidate.is_well_known is True
        assert candidate.verification_status is VerificationStatus.UNVERIFIED
        assert candidate.source_type is SourceType.AI_SUGGESTED

    def test_absent_is_well_known_means_speculative(self):
        """The model contract defines an omitted flag as "not well known"."""
        item = make_genai_candidate().model_dump()
        item.pop("is_well_known")
        candidate = normalize_genai_candidate(item)
        assert candidate is not None
        assert candidate.is_well_known is False

    def test_possible_source_never_counts_as_evidence(self):
        """A provenance hint is not evidence that the dataset fits the project."""
        item = make_genai_candidate().model_dump()
        item["possible_source"] = "Heart Surgery Data"
        candidate = normalize_genai_candidate(item)
        assert "heart surgery" not in candidate.evidence_text()

    def test_kaggle_candidate_has_no_ai_provenance(self):
        candidate = normalize_kaggle_record(make_kaggle_record())
        assert candidate.possible_source is None
        assert candidate.is_well_known is None

    def test_empty_list_produces_no_candidates(self):
        assert normalize_genai_candidates([]) == []

    def test_non_list_produces_no_candidates(self):
        assert normalize_genai_candidates({"candidates": []}) == []

    def test_duplicates_within_one_response_collapse(self):
        payload = [make_genai_candidate(), make_genai_candidate()]
        assert len(normalize_genai_candidates(payload)) == 1

    def test_mixed_valid_and_invalid_keeps_valid(self):
        payload = [make_genai_candidate(), {"name": ""}, "junk", make_genai_candidate(name="Other Dataset")]
        candidates = normalize_genai_candidates(payload)
        assert [c.name for c in candidates] == [
            "UCI Student Performance Dataset",
            "Other Dataset",
        ]


class TestCoercionHelpers:
    def test_coerce_int(self):
        assert coerce_int(5) == 5
        assert coerce_int("7") == 7
        assert coerce_int(None) is None
        assert coerce_int("abc") is None
        assert coerce_int(True) is None

    def test_coerce_ratio_clamps_to_unit_range_only(self):
        assert coerce_ratio(0.5) == 0.5
        assert coerce_ratio(0.0) == 0.0
        assert coerce_ratio(1.0) == 1.0
        assert coerce_ratio(1.2) is None
        assert coerce_ratio(-0.1) is None
        assert coerce_ratio("bad") is None

    def test_parse_timestamp_variants(self):
        assert parse_timestamp("2025-04-12T10:49:08.663Z").tzinfo == timezone.utc
        assert parse_timestamp("2025-04-12T10:49:08Z").year == 2025
        assert parse_timestamp("2025-04-12").month == 4
        assert parse_timestamp("2025-04-12T10:49:08+02:00").tzinfo == timezone.utc
        assert parse_timestamp("2025-04-12T10:49:08+02:00").hour == 8
        assert parse_timestamp("2025-04-12T10:49:08-05:00").hour == 15
        assert parse_timestamp("not a date") is None
        assert parse_timestamp(None) is None
        assert parse_timestamp("2025-13-45T99:99:99Z") is None

    def test_parse_timestamp_naive_datetime_gets_utc(self):
        from datetime import datetime

        parsed = parse_timestamp(datetime(2025, 1, 1, 12, 0, 0))
        assert parsed.tzinfo is not None

    def test_normalize_url_only_accepts_http(self):
        assert normalize_url("https://x.com/a") == "https://x.com/a"
        assert normalize_url("http://x.com/a") == "http://x.com/a"
        assert normalize_url("ftp://x.com/a") is None
        assert normalize_url("javascript:alert(1)") is None
        assert normalize_url("") is None
        assert normalize_url(None) is None

    def test_infer_domain_returns_none_without_tags(self):
        assert infer_domain_from_tags([]) is None

    def test_infer_domain_honours_explicit_domain_set(self):
        assert infer_domain_from_tags(["robotics"], {"robotics"}) == "robotics"
        assert infer_domain_from_tags(["robotics"], {"finance"}) is None
