"""Tests for keyword service matching - without Gemini dependency.

These tests prove that deterministic dictionary matching works
independently of any LLM service.
"""
from __future__ import annotations

import pytest
from app.services.keyword_service import KeywordService


@pytest.fixture(scope="module")
def ks() -> KeywordService:
    return KeywordService()


# ======================================================================
# TEST A: Cybersecurity / IoT project
# ======================================================================
class TestCybersecurityIoTProject:
    """Tests for: 'I want to develop a machine learning system that detects
    malicious network activity in IoT devices by analyzing network traffic
    and identifying abnormal behavior.'
    """

    TEXT = (
        "I want to develop a machine learning system that detects "
        "malicious network activity in IoT devices by analyzing "
        "network traffic and identifying abnormal behavior."
    )

    def test_machine_learning_matches(self, ks: KeywordService):
        matches = ks.match(self.TEXT)
        canonicals = {m.canonical_term.lower() for m in matches}
        assert "machine learning" in canonicals, (
            f"'machine learning' not found. Got: {canonicals}"
        )

    def test_iot_matches(self, ks: KeywordService):
        matches = ks.match(self.TEXT)
        canonicals = {m.canonical_term.lower() for m in matches}
        assert "iot" in canonicals or "internet of things" in canonicals, (
            f"IoT concept not found. Got: {canonicals}"
        )

    def test_network_traffic_matches(self, ks: KeywordService):
        matches = ks.match(self.TEXT)
        canonicals = {m.canonical_term.lower() for m in matches}
        assert "network traffic" in canonicals or "network traffic analysis" in canonicals, (
            f"Network traffic concept not found. Got: {canonicals}"
        )

    def test_anomaly_detection_matches(self, ks: KeywordService):
        matches = ks.match(self.TEXT)
        canonicals = {m.canonical_term.lower() for m in matches}
        # "abnormal behavior" is a concept that maps to anomaly detection domain
        has_anomaly_concept = (
            "anomaly detection" in canonicals
            or "abnormal behavior" in canonicals
            or "abnormal behaviour" in canonicals
        )
        assert has_anomaly_concept, (
            f"Anomaly-related concept not found. Got: {canonicals}"
        )

    def test_multiple_relevant_matches(self, ks: KeywordService):
        matches = ks.match(self.TEXT)
        assert len(matches) >= 3, (
            f"Expected at least 3 matches, got {len(matches)}: "
            f"{[m.canonical_term for m in matches]}"
        )

    def test_domain_is_cybersecurity(self, ks: KeywordService):
        matches = ks.match(self.TEXT)
        domains = {m.domain for m in matches}
        assert "cybersecurity" in domains or "iot" in domains, (
            f"Expected cybersecurity or IoT domain. Got: {domains}"
        )


# ======================================================================
# TEST B: Education project
# ======================================================================
class TestEducationProject:
    """Tests for: 'I want to build a machine learning system that predicts
    student academic performance using attendance, previous examination
    marks, assignment scores, study hours, and participation in class.'
    """

    TEXT = (
        "I want to build a machine learning system that predicts "
        "student academic performance using attendance, previous "
        "examination marks, assignment scores, study hours, and "
        "participation in class."
    )

    def test_machine_learning_matches(self, ks: KeywordService):
        matches = ks.match(self.TEXT)
        canonicals = {m.canonical_term.lower() for m in matches}
        assert "machine learning" in canonicals, (
            f"'machine learning' not found. Got: {canonicals}"
        )

    def test_student_performance_matches(self, ks: KeywordService):
        matches = ks.match(self.TEXT)
        canonicals = {m.canonical_term.lower() for m in matches}
        assert "student performance" in canonicals or "student performance prediction" in canonicals, (
            f"Student performance concept not found. Got: {canonicals}"
        )

    def test_education_domain(self, ks: KeywordService):
        matches = ks.match(self.TEXT)
        domains = {m.domain for m in matches}
        assert "education" in domains, (
            f"Expected education domain. Got: {domains}"
        )

    def test_multiple_relevant_matches(self, ks: KeywordService):
        matches = ks.match(self.TEXT)
        assert len(matches) >= 2, (
            f"Expected at least 2 matches, got {len(matches)}: "
            f"{[m.canonical_term for m in matches]}"
        )

    def test_at_least_three_matches(self, ks: KeywordService):
        matches = ks.match(self.TEXT)
        assert len(matches) >= 3, (
            f"Expected at least 3 matches, got {len(matches)}: "
            f"{[m.canonical_term for m in matches]}"
        )


# ======================================================================
# TEST C: Phishing project
# ======================================================================
class TestPhishingProject:
    """Tests for: 'I want to identify fake websites using machine learning.'
    """

    TEXT = "I want to identify fake websites using machine learning."

    def test_phishing_matches(self, ks: KeywordService):
        matches = ks.match(self.TEXT)
        canonicals = {m.canonical_term.lower() for m in matches}
        assert "phishing" in canonicals, (
            f"'phishing' not found. Got: {canonicals}"
        )

    def test_machine_learning_matches(self, ks: KeywordService):
        matches = ks.match(self.TEXT)
        canonicals = {m.canonical_term.lower() for m in matches}
        assert "machine learning" in canonicals, (
            f"'machine learning' not found. Got: {canonicals}"
        )

    def test_cybersecurity_domain(self, ks: KeywordService):
        matches = ks.match(self.TEXT)
        domains = {m.domain for m in matches}
        assert "cybersecurity" in domains, (
            f"Expected cybersecurity domain. Got: {domains}"
        )

    def test_fake_website_maps_to_phishing(self, ks: KeywordService):
        matches = ks.match(self.TEXT)
        phishing_match = next(
            (m for m in matches if m.canonical_term.lower() == "phishing"), None
        )
        assert phishing_match is not None, "No phishing match found"
        assert phishing_match.match_type.value in ("synonym", "exact", "phrase"), (
            f"Expected synonym/exact/phrase match, got {phishing_match.match_type.value}"
        )


# ======================================================================
# TEST D: Energy forecasting project
# ======================================================================
class TestEnergyForecastingProject:
    """Tests for: 'I want to forecast electricity consumption using
    historical energy usage data.'
    """

    TEXT = (
        "I want to forecast electricity consumption using "
        "historical energy usage data."
    )

    def test_energy_matches(self, ks: KeywordService):
        matches = ks.match(self.TEXT)
        canonicals = {m.canonical_term.lower() for m in matches}
        has_energy = any("energy" in c or "electricity" in c for c in canonicals)
        assert has_energy, (
            f"Energy concept not found. Got: {canonicals}"
        )

    def test_forecasting_matches(self, ks: KeywordService):
        matches = ks.match(self.TEXT)
        canonicals = {m.canonical_term.lower() for m in matches}
        has_forecast = any("forecast" in c for c in canonicals)
        assert has_forecast, (
            f"Forecasting concept not found. Got: {canonicals}"
        )

    def test_multiple_matches(self, ks: KeywordService):
        matches = ks.match(self.TEXT)
        assert len(matches) >= 2, (
            f"Expected at least 2 matches, got {len(matches)}: "
            f"{[m.canonical_term for m in matches]}"
        )


# ======================================================================
# TEST E: Fraud detection project
# ======================================================================
class TestFraudDetectionProject:
    """Tests for: 'I want to detect fraudulent credit card transactions.'
    """

    TEXT = "I want to detect fraudulent credit card transactions."

    def test_fraud_detection_matches(self, ks: KeywordService):
        matches = ks.match(self.TEXT)
        canonicals = {m.canonical_term.lower() for m in matches}
        has_fraud = any("fraud" in c or "credit" in c for c in canonicals)
        assert has_fraud, (
            f"Fraud/credit concept not found. Got: {canonicals}"
        )

    def test_finance_domain(self, ks: KeywordService):
        matches = ks.match(self.TEXT)
        domains = {m.domain for m in matches}
        assert "finance" in domains or "cybersecurity" in domains, (
            f"Expected finance or cybersecurity domain. Got: {domains}"
        )

    def test_multiple_matches(self, ks: KeywordService):
        matches = ks.match(self.TEXT)
        assert len(matches) >= 2, (
            f"Expected at least 2 matches, got {len(matches)}: "
            f"{[m.canonical_term for m in matches]}"
        )


# ======================================================================
# VARIATION TESTS
# ======================================================================
class TestVariations:
    """Test that variations of the same concept are handled correctly."""

    def test_iot_case_insensitive(self, ks: KeywordService):
        for text in ["IoT", "iot", "IOT", "IoT devices"]:
            matches = ks.match(text)
            assert len(matches) > 0, f"'{text}' produced no matches"

    def test_network_traffic_with_hyphen(self, ks: KeywordService):
        matches_hyphen = ks.match("analyze network-traffic")
        matches_space = ks.match("analyze network traffic")
        assert len(matches_hyphen) == len(matches_space), (
            f"Hyphen mismatch: hyphen={len(matches_hyphen)}, space={len(matches_space)}"
        )

    def test_abnormal_behavior_both_spellings(self, ks: KeywordService):
        matches_us = ks.match("abnormal behavior detection")
        matches_uk = ks.match("abnormal behaviour detection")
        # At least one should match something
        total = len(matches_us) + len(matches_uk)
        assert total > 0, "Neither 'behavior' nor 'behaviour' produced matches"

    def test_academic_performance(self, ks: KeywordService):
        matches = ks.match("academic performance")
        assert len(matches) > 0, "'academic performance' produced no matches"

    def test_student_performance(self, ks: KeywordService):
        matches = ks.match("student performance")
        assert len(matches) > 0, "'student performance' produced no matches"

    def test_classification(self, ks: KeywordService):
        matches = ks.match("classification")
        assert any(m.canonical_term.lower() == "classification" for m in matches)

    def test_anomaly_detection(self, ks: KeywordService):
        matches = ks.match("anomaly detection")
        assert any(m.canonical_term.lower() == "anomaly detection" for m in matches)

    def test_sentiment_analysis(self, ks: KeywordService):
        matches = ks.match("sentiment analysis")
        assert any(m.canonical_term.lower() == "sentiment analysis" for m in matches)

    def test_phishing_exact(self, ks: KeywordService):
        matches = ks.match("phishing")
        assert any(m.canonical_term.lower() == "phishing" for m in matches)

    def test_spam_detection(self, ks: KeywordService):
        matches = ks.match("spam detection")
        assert any(m.canonical_term.lower() == "spam detection" for m in matches)


# ======================================================================
# NO GEMINI DEPENDENCY TEST
# ======================================================================
class TestNoGeminiDependency:
    """Prove keyword matching works without Gemini."""

    def test_works_without_api_key(self, ks: KeywordService):
        """This test runs without any Gemini API key."""
        import os
        # Ensure no API key is used by keyword service
        original_key = os.environ.get("GEMINI_API_KEY")
        try:
            os.environ.pop("GEMINI_API_KEY", None)
            # Re-create keyword service - should work fine
            ks2 = KeywordService()
            matches = ks2.match("detect phishing websites")
            assert len(matches) > 0, "Keyword service failed without API key"
        finally:
            if original_key:
                os.environ["GEMINI_API_KEY"] = original_key

    def test_dictionary_loads_independently(self):
        """Dictionary loads without any external dependencies."""
        ks = KeywordService()
        assert ks.version is not None
        assert len(ks._canonical_index) > 0
        assert len(ks._all_phrases) > 0
