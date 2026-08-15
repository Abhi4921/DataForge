from __future__ import annotations

import pytest

from app.services.keyword_service import KeywordService
from app.schemas.project import MatchType


@pytest.fixture(scope="module")
def ks() -> KeywordService:
    return KeywordService()


class TestKeywordServiceLoading:
    def test_loads_successfully(self, ks: KeywordService):
        assert ks.version == "0.2.0"

    def test_has_canonical_terms(self, ks: KeywordService):
        terms = ks.get_all_canonical_terms()
        assert len(terms) > 50


class TestExactMatching:
    def test_exact_phishing(self, ks: KeywordService):
        matches = ks.match("I want to detect phishing")
        assert any(m.canonical_term == "phishing" for m in matches)

    def test_exact_classification(self, ks: KeywordService):
        matches = ks.match("I need to do classification")
        assert any(m.canonical_term == "classification" for m in matches)

    def test_exact_anomaly_detection(self, ks: KeywordService):
        matches = ks.match("build an anomaly detection system")
        assert any(m.canonical_term == "anomaly detection" for m in matches)


class TestSynonymMatching:
    def test_fake_website_maps_to_phishing(self, ks: KeywordService):
        matches = ks.match("identify fake websites")
        assert any(m.canonical_term == "phishing" for m in matches)

    def test_malicious_url_maps_to_phishing(self, ks: KeywordService):
        matches = ks.match("detect malicious URLs")
        assert any(m.canonical_term == "phishing" for m in matches)

    def test_opinion_mining_maps_to_sentiment(self, ks: KeywordService):
        matches = ks.match("perform opinion mining on reviews")
        assert any(m.canonical_term == "sentiment analysis" for m in matches)

    def test_outlier_detection_maps_to_anomaly(self, ks: KeywordService):
        matches = ks.match("find outlier detection in data")
        assert any(m.canonical_term == "anomaly detection" for m in matches)


class TestAbbreviationMatching:
    def test_ner_abbreviation(self, ks: KeywordService):
        matches = ks.match("use NER for entity extraction")
        assert any(m.canonical_term == "named entity recognition" for m in matches)

    def test_cnn_abbreviation(self, ks: KeywordService):
        matches = ks.match("train a CNN for image classification")
        assert any(m.canonical_term == "convolutional neural network" for m in matches)


class TestCaseNormalization:
    def test_case_insensitive(self, ks: KeywordService):
        matches = ks.match("PHISHING detection")
        assert any(m.canonical_term == "phishing" for m in matches)

    def test_mixed_case(self, ks: KeywordService):
        matches = ks.match("Sentiment Analysis of tweets")
        assert any(m.canonical_term == "sentiment analysis" for m in matches)


class TestMultiWordConcepts:
    def test_image_classification(self, ks: KeywordService):
        matches = ks.match("computer vision image classification")
        assert any(m.canonical_term == "image classification" for m in matches)

    def test_computer_vision(self, ks: KeywordService):
        matches = ks.match("computer vision image classification")
        assert any(m.canonical_term == "image classification" for m in matches)

    def test_neural_networks(self, ks: KeywordService):
        matches = ks.match("build neural networks for deep learning")
        assert any(m.canonical_term == "neural networks" for m in matches)


class TestMultipleConcepts:
    def test_multiple_matches(self, ks: KeywordService):
        matches = ks.match(
            "I want to detect phishing websites and also perform spam filtering"
        )
        canonicals = {m.canonical_term for m in matches}
        assert "phishing" in canonicals
        assert "spam detection" in canonicals


class TestMatchEvidence:
    def test_match_type_is_correct(self, ks: KeywordService):
        matches = ks.match("detect phishing")
        phishing_match = next((m for m in matches if m.canonical_term == "phishing"), None)
        assert phishing_match is not None
        assert phishing_match.match_type == MatchType.EXACT

    def test_synonym_match_type(self, ks: KeywordService):
        matches = ks.match("identify fake websites")
        phishing_match = next((m for m in matches if m.canonical_term == "phishing"), None)
        assert phishing_match is not None
        assert phishing_match.match_type == MatchType.SYNONYM

    def test_dictionary_entry_id_present(self, ks: KeywordService):
        matches = ks.match("detect phishing")
        for m in matches:
            assert m.dictionary_entry_id  # non-empty
            assert "." in m.dictionary_entry_id  # has hierarchical format


class TestNoMatch:
    def test_unrelated_text(self, ks: KeywordService):
        matches = ks.match("the weather is nice today")
        assert len(matches) == 0

    def test_empty_text(self, ks: KeywordService):
        matches = ks.match("")
        assert len(matches) == 0
