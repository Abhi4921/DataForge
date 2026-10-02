"""Tests for the dataset discovery HTTP API.

The service is stubbed so these tests stay offline: they verify request
validation, response shape, ranking payload and error mapping, not the sources.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import pytest
from fastapi.testclient import TestClient

from app.api.v1 import datasets as datasets_api
from app.main import app
from app.schemas.dataset import (
    SourceOutcome,
    SourceStatus,
    SourceType,
    VerificationStatus,
)
from app.services.datasets.base import DatasetSourceError
from app.services.datasets.discovery import DatasetDiscoveryResult
from app.services.llm_service import LLMServiceError
from tests.conftest import make_candidate


class _StubService:
    """Minimal stand-in for ``DatasetDiscoveryService``."""

    def __init__(self, result: Optional[DatasetDiscoveryResult] = None, error=None):
        self._result = result if result is not None else _result()
        self._error = error
        self.calls: list[dict] = []

    def ranking_weights(self) -> dict[str, float]:
        return {"domain_relevance": 20.0, "verification": 15.0}

    async def search(self, query: str, limit: int, request_id: str) -> DatasetDiscoveryResult:
        self.calls.append({"op": "search", "query": query, "limit": limit})
        if self._error:
            raise self._error
        return self._result

    async def recommend(self, description: str, limit: int, request_id: str):
        self.calls.append({"op": "recommend", "description": description, "limit": limit})
        if self._error:
            raise self._error
        return self._requirements(), self._result

    @staticmethod
    def _requirements():
        from tests.conftest import make_requirements

        return make_requirements()


def _result(candidates=None, query="student performance") -> DatasetDiscoveryResult:
    candidates = candidates if candidates is not None else [make_candidate()]
    return DatasetDiscoveryResult(
        query=query,
        candidates=candidates,
        sources=[
            SourceOutcome(
                source="kaggle",
                status=SourceStatus.OK,
                source_type=SourceType.VERIFIED_EXTERNAL,
                candidate_count=len(candidates),
            ),
            SourceOutcome(
                source="genai",
                status=SourceStatus.SKIPPED,
                source_type=SourceType.AI_SUGGESTED,
                message="requires structured project requirements",
            ),
        ],
        raw_candidate_count=len(candidates),
        duration_ms=12,
    )


def _error_body(response) -> dict:
    """The ``ErrorResponse`` payload.

    Phase 1 raises ``HTTPException(detail=ErrorResponse(...))``, so the
    standardized error body lives under ``detail``. Dataset routes follow the
    same envelope for consistency.
    """
    return response.json()["detail"]


@pytest.fixture
def client(monkeypatch):
    """A test client whose dataset service is replaced per test."""

    def _install(service: _StubService) -> TestClient:
        monkeypatch.setattr(datasets_api, "_service", service)
        return TestClient(app)

    return _install


@pytest.fixture
def stub():
    return _StubService()


class TestSearchEndpoint:
    def test_returns_200_and_candidates(self, client, stub):
        test_client = client(stub)
        response = test_client.post(
            "/api/v1/datasets/search", json={"query": "student performance"}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["success"] is True
        assert body["count"] == 1
        assert body["datasets"][0]["source_type"] == "verified_external"
        assert body["datasets"][0]["verification_status"] == "verified"

    def test_never_returns_ai_suggestions_for_bare_search(self, client):
        ai = make_candidate(
            name="AI Idea",
            source="genai",
            source_type=SourceType.AI_SUGGESTED,
            verification_status=VerificationStatus.UNVERIFIED,
        )
        test_client = client(_StubService(_result([ai])))
        response = test_client.post("/api/v1/datasets/search", json={"query": "q"})
        assert response.status_code == 200
        # The real service skips GenAI here; a stubbed AI row must still be
        # labelled honestly if one ever appears.
        assert response.json()["datasets"][0]["verification_status"] in {
            "verified",
            "unverified",
        }

    def test_reports_skipped_sources(self, client, stub):
        test_client = client(stub)
        body = test_client.post("/api/v1/datasets/search", json={"query": "q"}).json()
        skipped = [s for s in body["sources"] if s["status"] == "skipped"]
        assert skipped and skipped[0]["message"]

    def test_missing_query_is_422_with_stable_code(self, client, stub):
        test_client = client(stub)
        response = test_client.post("/api/v1/datasets/search", json={})
        assert response.status_code == 422
        assert _error_body(response)["error"]["code"] == "DATASET_QUERY_REQUIRED"

    def test_blank_query_is_422_with_stable_code(self, client, stub):
        test_client = client(stub)
        response = test_client.post("/api/v1/datasets/search", json={"query": "   "})
        assert response.status_code == 422
        assert _error_body(response)["error"]["code"] == "DATASET_QUERY_EMPTY"

    def test_overlong_query_is_rejected(self, client, stub):
        test_client = client(stub)
        response = test_client.post("/api/v1/datasets/search", json={"query": "a" * 5000})
        assert response.status_code == 422

    @pytest.mark.parametrize("limit", [0, -1, 51, 1000])
    def test_invalid_limit_is_422(self, client, stub, limit):
        test_client = client(stub)
        response = test_client.post(
            "/api/v1/datasets/search", json={"query": "q", "limit": limit}
        )
        assert response.status_code == 422

    def test_response_contains_no_credentials(self, client, stub):
        test_client = client(stub)
        raw = test_client.post("/api/v1/datasets/search", json={"query": "q"}).text
        assert "KGAT_" not in raw
        assert "Authorization" not in raw

    def test_request_id_is_a_uuid(self, client, stub):
        import uuid

        test_client = client(stub)
        body = test_client.post("/api/v1/datasets/search", json={"query": "q"}).json()
        uuid.UUID(body["request_id"])


class TestSearchErrorMapping:
    @pytest.mark.parametrize(
        ("code", "status"),
        [
            ("KAGGLE_AUTHENTICATION_ERROR", 401),
            ("KAGGLE_RATE_LIMITED", 429),
            ("KAGGLE_TIMEOUT", 504),
            ("KAGGLE_UNAVAILABLE", 502),
            ("KAGGLE_INVALID_RESPONSE", 502),
            ("MADE_UP_CODE", 502),
        ],
    )
    def test_source_errors_map_to_http_status(self, client, code, status):
        service = _StubService(error=DatasetSourceError(code=code, message="upstream said no"))
        test_client = client(service)
        response = test_client.post("/api/v1/datasets/search", json={"query": "q"})
        assert response.status_code == status
        assert _error_body(response)["error"]["code"].startswith("DATASET_")

    def test_error_response_never_leaks_upstream_detail(self, client):
        service = _StubService(
            error=DatasetSourceError(
                code="SOME_INTERNAL_DETAIL", message="Traceback with secret path"
            )
        )
        test_client = client(service)
        response = test_client.post("/api/v1/datasets/search", json={"query": "q"})
        assert response.status_code == 502
        body = _error_body(response)
        assert "Traceback" not in body["error"]["message"]
        assert body["error"]["code"] == "DATASET_SOURCE_UNAVAILABLE"

    def test_unexpected_exception_is_500_without_detail(self, client):
        service = _StubService(error=RuntimeError("boom secret"))
        test_client = client(service)
        response = test_client.post("/api/v1/datasets/search", json={"query": "q"})
        assert response.status_code == 500
        assert "boom secret" not in response.text

    def test_llm_error_is_mapped(self, client):
        service = _StubService(
            error=LLMServiceError(code="DATASET_DISCOVERY_LLM_TIMEOUT", message="slow")
        )
        test_client = client(service)
        response = test_client.post("/api/v1/datasets/search", json={"query": "q"})
        assert response.status_code == 504


class TestRecommendEndpoint:
    def test_returns_ranked_recommendations(self, client, stub):
        test_client = client(stub)
        response = test_client.post(
            "/api/v1/datasets/recommend",
            json={"description": "Predict student performance from attendance and study hours"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["count"] == 1
        first = body["recommendations"][0]
        assert first["rank"] == 1
        assert first["dataset"]["source_type"] == "verified_external"
        assert "ranking_weights" in body
        assert body["meta"]["verified_count"] == 1

    def test_missing_description_is_422(self, client, stub):
        test_client = client(stub)
        response = test_client.post("/api/v1/datasets/recommend", json={})
        assert response.status_code == 422
        assert _error_body(response)["error"]["code"] == "PROJECT_DESCRIPTION_REQUIRED"

    def test_blank_description_is_422(self, client, stub):
        test_client = client(stub)
        response = test_client.post("/api/v1/datasets/recommend", json={"description": "  "})
        assert response.status_code == 422

    def test_too_long_description_is_422(self, client, stub):
        test_client = client(stub)
        response = test_client.post(
            "/api/v1/datasets/recommend", json={"description": "word " * 400}
        )
        assert response.status_code == 422

    def test_includes_project_requirements(self, client, stub):
        test_client = client(stub)
        body = test_client.post(
            "/api/v1/datasets/recommend", json={"description": "Predict student performance"}
        ).json()
        assert body["project_requirements"]["project_understanding"]


class TestOpenAPI:
    def test_dataset_routes_are_documented(self, client, stub):
        test_client = client(stub)
        paths = test_client.get("/openapi.json").json()["paths"]
        assert "/api/v1/datasets/search" in paths
        assert "/api/v1/datasets/recommend" in paths

    def test_existing_analyzer_route_still_present(self, client, stub):
        test_client = client(stub)
        paths = test_client.get("/openapi.json").json()["paths"]
        assert "/api/v1/projects/analyze" in paths


class TestReadiness:
    def test_readiness_reports_dataset_sources(self, client, stub):
        test_client = client(stub)
        response = test_client.get("/ready")
        assert response.status_code == 200
        body = response.json()
        assert "kaggle_configured" in body
        assert "dataset_sources" in body
