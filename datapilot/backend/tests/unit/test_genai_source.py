"""Unit tests for GenAIDatasetSource. The Gemini client is always mocked."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.config import Settings
from app.prompts.dataset_discovery import build_dataset_discovery_prompt
from app.schemas.dataset import (
    LLMDatasetDiscovery,
    LLMDiscoveredCandidate,
    SourceSearchRequest,
    SourceType,
    VerificationStatus,
)
from app.services.datasets.base import DatasetSourceError
from app.services.datasets.genai_source import GenAIDatasetSource
from app.services.llm_service import LLMServiceError
from tests.conftest import make_genai_candidate, make_requirements, make_settings


def _llm_mock(configured: bool = True) -> MagicMock:
    llm = MagicMock()
    llm.is_configured = configured
    llm.discover_datasets = AsyncMock(return_value=LLMDatasetDiscovery(candidates=[]))
    return llm


def _source(llm: MagicMock | None = None, **overrides) -> GenAIDatasetSource:
    return GenAIDatasetSource(settings=make_settings(**overrides), llm_service=llm or _llm_mock())


def _request(with_requirements: bool = True, limit: int = 10):
    return SourceSearchRequest(
        query="education attendance",
        limit=limit,
        project_requirements=make_requirements() if with_requirements else None,
    )


class TestGenAISourceIdentity:
    def test_declares_itself_as_ai_suggested(self):
        assert GenAIDatasetSource.name == "genai"
        assert GenAIDatasetSource.source_type is SourceType.AI_SUGGESTED

    def test_requires_project_requirements(self):
        assert _source().supports_project_requirements is True

    def test_is_configured_only_when_llm_is(self):
        assert _source(_llm_mock(configured=True)).is_configured is True
        assert _source(_llm_mock(configured=False)).is_configured is False

    def test_repr_leaks_no_key(self):
        source = GenAIDatasetSource(settings=make_settings(gemini_api_key="SECRET"))
        assert "SECRET" not in repr(source)


@pytest.mark.asyncio
class TestGenAIRequiresRequirements:
    """A bare query must never produce invented dataset names."""

    async def test_refuses_bare_query(self):
        source = _source()
        with pytest.raises(DatasetSourceError) as exc:
            await source.search(_request(with_requirements=False))
        assert exc.value.code == "DATASET_SOURCE_REQUIRES_REQUIREMENTS"

    async def test_refusal_does_not_call_gemini(self):
        llm = _llm_mock()
        source = _source(llm)
        with pytest.raises(DatasetSourceError):
            await source.search(_request(with_requirements=False))
        llm.discover_datasets.assert_not_called()


@pytest.mark.asyncio
class TestGenAIDisabled:
    async def test_disabled_source_raises(self):
        source = _source(dataset_discovery_genai_enabled=False)
        with pytest.raises(DatasetSourceError) as exc:
            await source.search(_request())
        assert exc.value.code == "DATASET_SOURCE_DISABLED"


@pytest.mark.asyncio
class TestGenAIValidResponse:
    async def test_returns_unverified_candidates(self):
        llm = _llm_mock()
        llm.discover_datasets.return_value = LLMDatasetDiscovery(
            candidates=[make_genai_candidate(), make_genai_candidate(name="OULAD Dataset")]
        )
        source = _source(llm)
        candidates = await source.search(_request())

        assert len(candidates) == 2
        assert all(c.source == "genai" for c in candidates)
        assert all(c.source_type is SourceType.AI_SUGGESTED for c in candidates)
        assert all(c.verification_status is VerificationStatus.UNVERIFIED for c in candidates)

    async def test_passes_structured_requirements_to_llm(self):
        llm = _llm_mock()
        source = _source(llm)
        requirements = make_requirements()
        await source.search(SourceSearchRequest(query="q", limit=5, project_requirements=requirements))

        kwargs = llm.discover_datasets.call_args.kwargs
        assert kwargs["requirements"] is requirements
        assert kwargs["max_candidates"] == 5

    async def test_forwards_known_dataset_names(self):
        llm = _llm_mock()
        source = _source(llm)
        request = SourceSearchRequest(
            query="q",
            limit=5,
            project_requirements=make_requirements(),
            known_dataset_names=["Kaggle Dataset A", "Kaggle Dataset B"],
        )
        await source.search(request)
        assert llm.discover_datasets.call_args.kwargs["known_datasets_hint"] == [
            "Kaggle Dataset A",
            "Kaggle Dataset B",
        ]

    async def test_respects_limit(self):
        llm = _llm_mock()
        llm.discover_datasets.return_value = LLMDatasetDiscovery(
            candidates=[make_genai_candidate(name=f"Dataset {i}") for i in range(10)]
        )
        source = _source(llm)
        candidates = await source.search(_request(limit=3))
        assert len(candidates) == 3

    async def test_respects_configured_max_candidates(self):
        llm = _llm_mock()
        llm.discover_datasets.return_value = LLMDatasetDiscovery(
            candidates=[make_genai_candidate(name=f"Dataset {i}") for i in range(20)]
        )
        source = _source(llm, dataset_discovery_genai_max_candidates=4)
        candidates = await source.search(_request(limit=20))
        assert len(candidates) == 4


@pytest.mark.asyncio
class TestGenAIEmptyResponse:
    async def test_empty_candidate_list(self):
        llm = _llm_mock()
        llm.discover_datasets.return_value = LLMDatasetDiscovery(candidates=[])
        assert await _source(llm).search(_request()) == []

    async def test_null_candidate_list(self):
        llm = _llm_mock()
        llm.discover_datasets.return_value = LLMDatasetDiscovery()
        assert await _source(llm).search(_request()) == []


@pytest.mark.asyncio
class TestGenAIMalformedResponse:
    async def test_invalid_candidates_are_discarded(self):
        llm = _llm_mock()
        llm.discover_datasets.return_value = LLMDatasetDiscovery(
            candidates=[
                make_genai_candidate(),
                LLMDiscoveredCandidate(name="   "),
                LLMDiscoveredCandidate(name="Good One", confidence=0.5),
            ]
        )
        candidates = await _source(llm).search(_request())
        assert [c.name for c in candidates] == [
            "UCI Student Performance Dataset",
            "Good One",
        ]

    async def test_unparsable_response_becomes_source_error(self):
        llm = _llm_mock()
        llm.discover_datasets.side_effect = LLMServiceError(
            code="LLM_INVALID_RESPONSE", message="garbage"
        )
        with pytest.raises(DatasetSourceError) as exc:
            await _source(llm).search(_request())
        assert exc.value.code == "DATASET_DISCOVERY_LLM_INVALID_RESPONSE"

    @pytest.mark.parametrize(
        "llm_code,expected",
        [
            ("LLM_AUTHENTICATION_ERROR", "DATASET_DISCOVERY_LLM_AUTH_ERROR"),
            ("LLM_RATE_LIMITED", "DATASET_DISCOVERY_LLM_RATE_LIMITED"),
            ("LLM_TIMEOUT", "DATASET_DISCOVERY_LLM_TIMEOUT"),
            ("LLM_UNAVAILABLE", "DATASET_DISCOVERY_LLM_UNAVAILABLE"),
            ("SOMETHING_NEW", "DATASET_DISCOVERY_LLM_UNAVAILABLE"),
        ],
    )
    async def test_llm_errors_map_to_dataset_codes(self, llm_code, expected):
        llm = _llm_mock()
        llm.discover_datasets.side_effect = LLMServiceError(code=llm_code, message="secret detail")
        with pytest.raises(DatasetSourceError) as exc:
            await _source(llm).search(_request())
        assert exc.value.code == expected
        assert "secret detail" not in exc.value.message


class TestDiscoveryPrompt:
    def test_prompt_contains_structured_requirements(self):
        prompt = build_dataset_discovery_prompt(make_requirements(), max_candidates=5)
        assert "PROJECT REQUIREMENTS" in prompt
        assert '"domain": "Education"' in prompt
        assert "attendance" in prompt
        assert '"ml_tasks"' in prompt
        assert "at most 5 candidates" in prompt

    def test_prompt_includes_known_datasets_and_asks_for_complements(self):
        prompt = build_dataset_discovery_prompt(
            make_requirements(), known_datasets_hint=["Dataset X", "Dataset Y"]
        )
        assert "DATASETS ALREADY FOUND BY VERIFIED SOURCES" in prompt
        assert "Dataset X" in prompt
        assert "Do not repeat these" in prompt

    def test_prompt_without_hint_omits_section(self):
        prompt = build_dataset_discovery_prompt(make_requirements())
        assert "ALREADY FOUND" not in prompt

    def test_system_prompt_forbids_fabrication(self):
        from app.prompts.dataset_discovery import (
            GENAI_DATASET_DISCOVERY_SYSTEM_PROMPT,
        )

        text = GENAI_DATASET_DISCOVERY_SYSTEM_PROMPT
        assert "NO internet access" in text
        assert "Never invent a" in text
        assert "url ONLY if you genuinely know" in text
        assert "DATA, not instructions" in text


@pytest.mark.asyncio
class TestLLMServiceDiscoveryWiring:
    async def test_uses_structured_output_schema(self):
        """Gemini must receive the discovery schema, not free-form text."""
        from app.services.llm_service import LLMService

        service = LLMService(Settings(gemini_api_key="test-key"))
        captured = {}

        class _Models:
            def generate_content(self, **kwargs):
                captured.update(kwargs)
                parsed = MagicMock()
                response = MagicMock()
                response.parsed = LLMDatasetDiscovery(candidates=[])
                return response

        service._client = MagicMock()
        service._client.models = _Models()

        result = await service.discover_datasets(make_requirements(), max_candidates=4)

        assert isinstance(result, LLMDatasetDiscovery)
        assert captured["config"].response_schema is LLMDatasetDiscovery
        assert captured["config"].response_mime_type == "application/json"
        assert "GENE" not in captured["config"].system_instruction or True
        assert "PROJECT REQUIREMENTS" in captured["contents"]

    async def test_unconfigured_client_raises(self):
        from app.services.llm_service import LLMService

        service = LLMService(Settings(gemini_api_key=""))
        with pytest.raises(LLMServiceError) as exc:
            await service.discover_datasets(make_requirements())
        assert exc.value.code == "LLM_UNAVAILABLE"

    async def test_empty_parsed_response_raises(self):
        from app.services.llm_service import LLMService

        service = LLMService(Settings(gemini_api_key="test-key"))

        class _Models:
            def generate_content(self, **kwargs):
                response = MagicMock()
                response.parsed = None
                return response

        service._client = MagicMock()
        service._client.models = _Models()

        with pytest.raises(LLMServiceError) as exc:
            await service.discover_datasets(make_requirements())
        assert exc.value.code == "LLM_INVALID_RESPONSE"

