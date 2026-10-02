from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

# Ensure the backend package is importable
_backend_dir = Path(__file__).resolve().parent.parent
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

os.environ.setdefault("APP_ENV", "testing")
os.environ.setdefault("LOG_LEVEL", "WARNING")

from app.main import app
from app.schemas.dataset import (
    DatasetCandidate,
    LLMDiscoveredCandidate,
    SourceType,
    VerificationStatus,
)
from app.services.keyword_service import KeywordService
from app.schemas.project import (
    Analysis,
    DatasetRequirements,
    DomainInfo,
    FeatureRequirement,
    InputInfo,
    KeywordMatchEvidence,
    Keywords,
    LLMConfidenceField,
    LLMDatasetRequirements,
    LLMKeywords,
    LLMMLTask,
    LLMProjectAnalysis,
    LLMProjectUnderstanding,
    MatchType,
    MLTask,
    ProblemTypeInfo,
    ProjectAnalysisData,
    ProjectUnderstanding,
    SubdomainInfo,
    TargetInfo,
    TargetRequirement,
)



@pytest.fixture(scope="session")
def keyword_service() -> KeywordService:
    return KeywordService()


@pytest.fixture(scope="session")
def client() -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


def make_mock_llm_output(
    domain: str = "artificial intelligence",
    subdomain: str = "machine learning",
    tasks: list[str] | None = None,
    target: str | None = "classification labels",
    target_status: str = "inferred",
    needs_clarification: bool = False,
    confidence: float = 0.85,
    concepts: list[str] | None = None,
) -> LLMProjectAnalysis:
    if tasks is None:
        tasks = ["classification"]
    if concepts is None:
        concepts = []

    return LLMProjectAnalysis(
        project_understanding=LLMProjectUnderstanding(
            domain=LLMConfidenceField(value=domain, confidence=0.9, status="inferred"),
            subdomain=LLMConfidenceField(value=subdomain, confidence=0.85, status="inferred"),
            problem_type=LLMConfidenceField(value="Machine Learning", confidence=0.9, status="inferred"),
            ml_tasks=[LLMMLTask(task=t, confidence=0.85) for t in tasks],
            objective="Build a system to solve the described problem",
            target=LLMConfidenceField(value=target, confidence=0.7, status=target_status),
        ),
        keywords=LLMKeywords(
            inferred_concepts=concepts,
            related_terms=[],
        ),
        dataset_requirements=LLMDatasetRequirements(
            required_features=[],
            target_description=target,
            label_description=None,
            data_type_notes="Tabular data",
        ),
        ambiguities=[],
        missing_information=[],
        needs_clarification=needs_clarification,
        overall_confidence=confidence,
    )

# ---------------------------------------------------------------------------
# Phase 2 fixtures: dataset discovery (no live Kaggle or Gemini calls)
# ---------------------------------------------------------------------------

# Fixed clock so recency scoring is reproducible across runs.
TEST_NOW = datetime(2026, 9, 28, 12, 0, 0, tzinfo=timezone.utc)


def make_requirements(
    domain: Optional[str] = "Education",
    subdomain: Optional[str] = "Educational Analytics",
    tasks: list[str] | None = None,
    target: Optional[str] = "academic performance",
    features: list[str] | None = None,
    keywords: list[str] | None = None,
    dictionary_keywords: list[str] | None = None,
) -> ProjectAnalysisData:
    """Build structured requirements as the Project Analyzer would emit them."""
    if tasks is None:
        tasks = ["classification", "regression"]
    if features is None:
        features = [
            "attendance",
            "previous examination marks",
            "assignment scores",
            "study hours",
        ]
    if keywords is None:
        keywords = [
            "student performance",
            "attendance",
            "examination",
            "assignment scores",
            "study hours",
        ]
    if dictionary_keywords is None:
        dictionary_keywords = []

    return ProjectAnalysisData(
        input=InputInfo(word_count=28),
        project_understanding=ProjectUnderstanding(
            domain=DomainInfo(value=domain, confidence=0.9) if domain else None,
            subdomain=SubdomainInfo(value=subdomain, confidence=0.8) if subdomain else None,
            problem_type=ProblemTypeInfo(value="Machine Learning", confidence=0.9),
            ml_tasks=[MLTask(task=t, confidence=0.8) for t in tasks],
            objective="Identify students who may need academic support",
            target=TargetInfo(value=target, confidence=0.7, status="inferred")
            if target
            else None,
        ),
        keywords=Keywords(
            canonical=keywords,
            dictionary_matches=[
                KeywordMatchEvidence(
                    canonical_term=term,
                    matched_text=term,
                    match_type=MatchType.EXACT,
                    dictionary_entry_id=f"test/{term}",
                    domain=domain or "",
                    subdomain=subdomain or "",
                )
                for term in dictionary_keywords
            ],
            llm_inferred=[],
            related_concepts=[],
            provenance=[],
            conflicts=[],
        ),
        dataset_requirements=DatasetRequirements(
            feature_requirements=[
                FeatureRequirement(name=f, description="", importance="required")
                for f in features
            ],
            target_requirements=[
                TargetRequirement(name=target, description="Outcome to predict")
            ]
            if target
            else [],
        ),
        analysis=Analysis(overall_confidence=0.85),
    )



def make_kaggle_record(
    ref: str = "owner/student-performance",
    title: str = "Student Academic Performance",
    subtitle: str = "Attendance, marks and study hours for university students",
    description: str = "",
    owner: str = "Owner Name",
    tags: list[str] | None = None,
    downloads: int = 50000,
    votes: int = 400,
    usability: float = 0.9,
    license_name: str = "CC0-1.0",
    total_bytes: int = 204800,
    last_updated: str = "2026-08-01T00:00:00.000Z",
) -> dict:
    """A realistic ``datasets/list`` row as returned by the Kaggle API."""
    if tags is None:
        tags = ["education", "students", "university"]
    return {
        "ref": ref,
        "title": title,
        "subtitle": subtitle,
        "description": description,
        "url": f"https://www.kaggle.com/datasets/{ref}",
        "ownerRef": owner.lower().replace(" ", ""),
        "ownerName": owner,
        "licenseName": license_name,
        "totalBytes": total_bytes,
        "lastUpdated": last_updated,
        "downloadCount": downloads,
        "voteCount": votes,
        "usabilityRating": usability,
        "tags": [
            {"name": t, "ref": t.replace(" ", "-"), "fullPath": f"subject > {t}"}
            for t in tags
        ],
        "isPrivate": False,
        "isFeatured": False,
    }


def make_kaggle_payload(count: int = 3, **kwargs) -> list[dict]:
    """Several distinct Kaggle rows, used to exercise paging and limits."""
    titles = [
        ("owner/student-performance", "Student Academic Performance"),
        ("owner/student-habits", "Student Habits Survey Data"),
        ("owner/university-enrollment", "University Enrollment Statistics"),
        ("owner/school-attendance", "School Attendance Records"),
        ("owner/exam-results", "Examination Results Archive"),
    ]
    rows = []
    for index in range(min(count, len(titles))):
        ref, title = titles[index]
        rows.append(make_kaggle_record(ref=ref, title=title, **kwargs))
    return rows


def make_genai_candidate(
    name: str = "UCI Student Performance Dataset",
    url: str = "https://archive.ics.uci.edu/dataset/342/student+performance",
    confidence: float = 0.82,
    is_well_known: bool = True,
    features: list[str] | None = None,
    tasks: list[str] | None = None,
    why_relevant: str = "Contains the grade and attendance variables this project needs",
    description: str = "Secondary school student records with grades and attendance",
    target_description: str = "final grade",
    license: str = "CC BY 4.0",
) -> LLMDiscoveredCandidate:
    if features is None:
        features = ["attendance", "study hours", "previous grades"]
    if tasks is None:
        tasks = ["classification", "regression"]
    return LLMDiscoveredCandidate(
        name=name,
        possible_source="UCI",
        url=url,
        description=description,
        why_relevant=why_relevant,
        confidence=confidence,
        is_well_known=is_well_known,
        suggested_features=features,
        target_description=target_description,
        ml_tasks=tasks,
        license=license,
    )



def make_candidate(
    name: str = "Verified Student Dataset",
    source: str = "kaggle",
    source_type: SourceType = SourceType.VERIFIED_EXTERNAL,
    source_id: str = "owner/student-performance",
    **kwargs,
) -> DatasetCandidate:
    """A DatasetCandidate with sensible defaults for ranking tests."""
    payload = {
        "id": f"{source}:{source_id}",
        "name": name,
        "source": source,
        "source_type": source_type,
        "source_id": source_id,
        "verification_status": (
            VerificationStatus.VERIFIED
            if source_type is SourceType.VERIFIED_EXTERNAL
            else VerificationStatus.UNVERIFIED
        ),
        "url": f"https://www.kaggle.com/datasets/{source_id}",
        "description": "Attendance, examination marks, assignment scores and study hours",
        "owner": "owner",
        "tags": ["education"],
        "domain": "education",
        "license": "CC0-1.0",
        "download_count": 10000,
        "vote_count": 200,
        "usability": 0.8,
        "updated_at": datetime(2026, 8, 1, tzinfo=timezone.utc),
    }
    payload.update(kwargs)
    return DatasetCandidate(**payload)


def make_settings(**overrides):
    """Settings with deterministic, network-free Phase 2 configuration."""
    from app.core.config import Settings

    payload = {
        "gemini_api_key": "",
        "kaggle_api_token": "",
        "kaggle_enabled": True,
        "kaggle_max_pages": 1,
        "dataset_discovery_genai_enabled": True,
        "app_env": "testing",
    }
    payload.update(overrides)
    return Settings(**payload)


def make_mock_transport(responses: list[httpx.Response] | httpx.Response) -> httpx.MockTransport:
    """A MockTransport returning the given responses in order.

    Exception instances are raised instead of returned, which is how a real
    transport surfaces timeouts and connection failures.
    """
    queue = list(responses) if isinstance(responses, list) else [responses]

    def handler(request: httpx.Request) -> httpx.Response:
        if not queue:
            raise AssertionError("Kaggle source made more requests than expected")
        nxt = queue.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt

    return httpx.MockTransport(handler)



@pytest.fixture
def settings():
    return make_settings()


@pytest.fixture
def requirements():
    return make_requirements()


@pytest.fixture
def fixed_now():
    return TEST_NOW

