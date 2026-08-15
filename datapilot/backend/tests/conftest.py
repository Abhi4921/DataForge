from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Ensure the backend package is importable
_backend_dir = Path(__file__).resolve().parent.parent
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

os.environ.setdefault("APP_ENV", "testing")
os.environ.setdefault("LOG_LEVEL", "WARNING")

from app.main import app
from app.services.keyword_service import KeywordService
from app.schemas.project import (
    LLMProjectAnalysis,
    LLMProjectUnderstanding,
    LLMConfidenceField,
    LLMMLTask,
    LLMKeywords,
    LLMDatasetRequirements,
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
