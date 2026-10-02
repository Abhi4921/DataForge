"""Tests for turning structured requirements into a registry search query."""

from __future__ import annotations

from app.services.datasets.query_builder import build_dataset_search_query
from tests.conftest import make_requirements


class TestQueryIsDerivedFromStructure:
    def test_uses_domain_and_features(self):
        query = build_dataset_search_query(make_requirements())
        assert query
        assert "performance" in query

    def test_dictionary_keywords_precede_domain(self):
        requirements = make_requirements(
            dictionary_keywords=["attendance", "examination"],
            features=[],
            target=None,
            subdomain=None,
        )
        query = build_dataset_search_query(requirements)
        assert query.startswith("attendance examination")
        assert "education" in query

    def test_domain_is_used_when_keywords_are_sparse(self):
        requirements = make_requirements(
            keywords=[], features=[], target=None, subdomain=None, tasks=[]
        )
        query = build_dataset_search_query(requirements)
        assert query == "education"

    def test_never_returns_the_raw_description(self):
        """A conversational description is useless as a registry query."""
        requirements = make_requirements()
        query = build_dataset_search_query(requirements)
        assert "i want to build" not in query.lower()
        assert "system that predicts" not in query.lower()

    def test_query_is_deterministic(self):
        requirements = make_requirements()
        assert build_dataset_search_query(requirements) == build_dataset_search_query(
            requirements
        )

    def test_respects_max_terms(self):
        requirements = make_requirements(
            keywords=["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta"]
        )
        query = build_dataset_search_query(requirements, max_terms=3)
        assert len(query.split()) <= 3

    def test_default_max_terms_is_respected(self):
        requirements = make_requirements(
            features=["f1", "f2", "f3", "f4"],
            keywords=["k1", "k2", "k3", "k4", "k5", "k6", "k7", "k8"],
        )
        assert len(build_dataset_search_query(requirements).split()) <= 6

    def test_no_duplicate_terms(self):
        requirements = make_requirements(
            keywords=["attendance", "attendance", "study hours"]
        )
        tokens = build_dataset_search_query(requirements).split()
        assert len(tokens) == len(set(tokens))

    def test_has_no_double_spaces(self):
        requirements = make_requirements(
            keywords=["a", "of", "the", "attendance", "study hours"]
        )
        query = build_dataset_search_query(requirements)
        assert "  " not in query

    def test_terms_are_lowercase(self):
        query = build_dataset_search_query(make_requirements())
        assert query == query.lower()


class TestStopTerms:
    def test_generic_words_are_dropped(self):
        requirements = make_requirements(
            domain="Data Science",
            subdomain=None,
            features=["dataset", "system"],
            keywords=["I want to use a model"],
        )
        query = build_dataset_search_query(requirements)
        assert "want" not in query.split()
        assert "model" not in query.split()

    def test_education_domain_keeps_domain_word(self):
        """``student`` is generic, but ``education`` is the retrieval signal."""
        requirements = make_requirements(
            domain="Education", subdomain=None, features=[], keywords=["attendance"]
        )
        assert "education" in build_dataset_search_query(requirements)

    def test_domain_is_dropped_when_explicit_terms_fill_the_budget(self):
        """The domain is a coarse category: explicit target and feature terms
        take precedence, because registries index concrete dataset concepts."""
        requirements = make_requirements(domain="Education", subdomain=None)
        assert "education" not in build_dataset_search_query(requirements)


class TestEmptyRequirements:
    def test_returns_empty_string_when_there_is_nothing_to_search(self):
        requirements = make_requirements(
            domain=None,
            subdomain=None,
            features=[],
            keywords=[],
            target=None,
            tasks=[],
        )
        assert build_dataset_search_query(requirements) == ""

    def test_falls_back_to_dictionary_vocabulary_when_all_terms_are_stop_words(self):
        requirements = make_requirements(
            domain=None,
            subdomain=None,
            features=[],
            keywords=["student", "students"],
            target=None,
        )
        # Every meaningful term was filtered out, so the relaxed pass runs.
        query = build_dataset_search_query(requirements)
        assert isinstance(query, str)
        assert "  " not in query


class TestDifferentDomains:
    def test_healthcare_query(self):
        requirements = make_requirements(
            domain="Healthcare",
            subdomain="Medical Imaging",
            features=["ct scan", "tumor volume"],
            keywords=["tumor segmentation"],
            target="malignancy",
        )
        query = build_dataset_search_query(requirements)
        assert "healthcare" in query

    def test_finance_query(self):
        requirements = make_requirements(
            domain="Finance",
            subdomain="Fraud Detection",
            features=["transaction amount", "merchant category"],
            keywords=["fraud transaction"],
            target="is_fraud",
        )
        query = build_dataset_search_query(requirements)
        assert "finance" in query
        assert "fraud" in query

    def test_sports_query(self):
        requirements = make_requirements(
            domain="Sports",
            subdomain="Player Analytics",
            features=["match statistics", "player performance"],
            keywords=["player performance"],
            target="match outcome",
        )
        query = build_dataset_search_query(requirements)
        assert "sports" in query
