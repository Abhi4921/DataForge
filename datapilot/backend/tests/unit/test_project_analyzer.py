from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from app.services.project_analyzer import ProjectAnalyzer
from app.services.llm_service import LLMServiceError
from app.schemas.project import (
    ProjectAnalysisRequest,
    ProjectAnalysisResponse,
)
from tests.conftest import make_mock_llm_output


@pytest.fixture
def analyzer_with_mock_llm() -> ProjectAnalyzer:
    mock_llm = MagicMock()
    mock_llm.analyze_project = AsyncMock(return_value=make_mock_llm_output(
        domain="cybersecurity",
        subdomain="network security",
        tasks=["classification", "anomaly_detection"],
        target="attack/benign",
        target_status="inferred",
        concepts=["intrusion detection", "network traffic analysis"],
    ))
    return ProjectAnalyzer(llm_service=mock_llm)


@pytest.mark.asyncio
class TestProjectAnalyzerValidation:
    async def test_valid_project(self, analyzer_with_mock_llm: ProjectAnalyzer):
        req = ProjectAnalysisRequest(
            description="I want to build a phishing detection system using ML"
        )
        result = await analyzer_with_mock_llm.analyze(req)
        assert result.success is True
        assert result.data.input.word_count > 0

    async def test_empty_description_rejected(self, analyzer_with_mock_llm: ProjectAnalyzer):
        from fastapi import HTTPException
        req = ProjectAnalysisRequest(description="   ")
        with pytest.raises(HTTPException) as exc_info:
            await analyzer_with_mock_llm.analyze(req)
        assert exc_info.value.status_code == 422

    async def test_null_description_rejected(self, analyzer_with_mock_llm: ProjectAnalyzer):
        from fastapi import HTTPException
        req = ProjectAnalysisRequest(description=None)
        with pytest.raises(HTTPException) as exc_info:
            await analyzer_with_mock_llm.analyze(req)
        assert exc_info.value.status_code == 422

    async def test_150_words_accepted(self, analyzer_with_mock_llm: ProjectAnalyzer):
        words = " ".join(["word"] * 150)
        req = ProjectAnalysisRequest(description=words)
        result = await analyzer_with_mock_llm.analyze(req)
        assert result.success is True
        assert result.data.input.word_count == 150

    async def test_151_words_rejected(self, analyzer_with_mock_llm: ProjectAnalyzer):
        from fastapi import HTTPException
        words = " ".join(["word"] * 151)
        req = ProjectAnalysisRequest(description=words)
        with pytest.raises(HTTPException) as exc_info:
            await analyzer_with_mock_llm.analyze(req)
        assert exc_info.value.status_code == 422


@pytest.mark.asyncio
class TestProjectAnalyzerLLMErrors:
    async def test_llm_auth_error(self):
        mock_llm = MagicMock()
        mock_llm.analyze_project = AsyncMock(
            side_effect=LLMServiceError("LLM_AUTHENTICATION_ERROR", "Auth failed")
        )
        analyzer = ProjectAnalyzer(llm_service=mock_llm)
        req = ProjectAnalysisRequest(description="build a classifier")
        with pytest.raises(LLMServiceError) as exc_info:
            await analyzer.analyze(req)
        assert exc_info.value.code == "LLM_AUTHENTICATION_ERROR"

    async def test_llm_rate_limit(self):
        mock_llm = MagicMock()
        mock_llm.analyze_project = AsyncMock(
            side_effect=LLMServiceError("LLM_RATE_LIMITED", "Rate limited")
        )
        analyzer = ProjectAnalyzer(llm_service=mock_llm)
        req = ProjectAnalysisRequest(description="build a classifier")
        with pytest.raises(LLMServiceError) as exc_info:
            await analyzer.analyze(req)
        assert exc_info.value.code == "LLM_RATE_LIMITED"

    async def test_llm_timeout(self):
        mock_llm = MagicMock()
        mock_llm.analyze_project = AsyncMock(
            side_effect=LLMServiceError("LLM_TIMEOUT", "Timeout")
        )
        analyzer = ProjectAnalyzer(llm_service=mock_llm)
        req = ProjectAnalysisRequest(description="build a classifier")
        with pytest.raises(LLMServiceError) as exc_info:
            await analyzer.analyze(req)
        assert exc_info.value.code == "LLM_TIMEOUT"


@pytest.mark.asyncio
class TestProjectAnalyzerResponse:
    async def test_response_structure(self, analyzer_with_mock_llm: ProjectAnalyzer):
        req = ProjectAnalysisRequest(
            description="I want to detect malicious network activity in IoT devices"
        )
        result = await analyzer_with_mock_llm.analyze(req)

        assert result.success is True
        assert result.request_id
        assert result.api_version == "v1"
        assert result.data.input.word_count > 0
        assert result.data.project_understanding is not None
        assert result.data.keywords is not None
        assert result.data.dataset_requirements is not None
        assert result.data.analysis is not None
        assert result.meta.model
        assert result.meta.processing_time_ms >= 0

    async def test_request_id_is_uuid(self, analyzer_with_mock_llm: ProjectAnalyzer):
        import uuid
        req = ProjectAnalysisRequest(description="test project")
        result = await analyzer_with_mock_llm.analyze(req)
        uuid.UUID(result.request_id)  # should not raise
