"""Tests for the discovery orchestrator: source orchestration, partial failure
handling and the ranking handoff. Sources themselves are stubbed, so these tests
never touch the network.
"""

from __future__ import annotations

from typing import Optional

import pytest

from app.schemas.dataset import SourceSearchRequest, SourceType, VerificationStatus
from app.services.datasets.base import DatasetSource, DatasetSourceError
from app.services.datasets.discovery import DatasetDiscoveryService
from tests.conftest import make_candidate, make_requirements, make_settings


class _StubSource(DatasetSource):
    """A source that returns canned candidates or raises a canned error."""

    def __init__(
        self,
        name: str,
        source_type: SourceType,
        candidates: Optional[list] = None,
        error: Optional[DatasetSourceError] = None,
        requires_requirements: bool = False,
        configured: bool = True,
    ) -> None:
        self._name = name
        self._source_type = source_type
        self._candidates = candidates or []
        self._error = error
        self._requires_requirements = requires_requirements
        self._configured = configured
        self.calls: list[SourceSearchRequest] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def source_type(self) -> SourceType:
        return self._source_type

    @property
    def supports_project_requirements(self) -> bool:
        return self._requires_requirements

    @property
    def is_configured(self) -> bool:
        return self._configured

    async def search(self, request: SourceSearchRequest) -> list:
        self.calls.append(request)
        if self._error:
            raise self._error
        return [c.model_copy(deep=True) for c in self._candidates]

    async def healthcheck(self) -> bool:
        return self._configured


def _kaggle(candidates=None, error=None) -> _StubSource:
    return _StubSource(
        "kaggle",
        SourceType.VERIFIED_EXTERNAL,
        candidates=candidates if candidates is not None else [make_candidate()],
        error=error,
    )


def _genai(candidates=None, error=None) -> _StubSource:
    return _StubSource(
        "genai",
        SourceType.AI_SUGGESTED,
        candidates=candidates or [],
        error=error,
        requires_requirements=True,
    )


def _service(*sources) -> DatasetDiscoveryService:
    return DatasetDiscoveryService(settings=make_settings(), sources=list(sources))


def _ai(name="AI Student Performance", url="https://example.org/ai"):
    return make_candidate(
        name=name,
        source="genai",
        source_type=SourceType.AI_SUGGESTED,
        verification_status=VerificationStatus.UNVERIFIED,
        source_id=name,
        url=url,
    )


class TestSearch:
    @pytest.mark.asyncio
    async def test_returns_candidates(self):
        service = _service(_kaggle())
        result = await service.search("student performance", limit=10)
        assert len(result.candidates) == 1
        assert result.query == "student performance"

    @pytest.mark.asyncio
    async def test_genai_source_is_skipped_without_requirements(self):
        genai = _genai([_ai()])
        service = _service(_kaggle(), genai)
        result = await service.search("student performance", limit=10)
        assert genai.calls == [], "bare search must not call Gemini"
        outcome = next(o for o in result.sources if o.source == "genai")
        assert outcome.status.value == "skipped"
        assert all(c.source_type is SourceType.VERIFIED_EXTERNAL for c in result.candidates)

    @pytest.mark.asyncio
    async def test_query_is_normalized(self):
        service = _service(_kaggle())
        result = await service.search("  student   performance  ", limit=10)
        assert result.query == "student performance"

    @pytest.mark.asyncio
    async def test_empty_query_raises(self):
        service = _service(_kaggle())
        with pytest.raises(DatasetSourceError) as exc:
            await service.search("   ", limit=10)
        assert exc.value.code == "DATASET_QUERY_EMPTY"

    @pytest.mark.asyncio
    async def test_limit_is_respected(self):
        service = _service(_kaggle([make_candidate(name="A", source_id="o/a")]))
        result = await service.search("q", limit=1)
        assert len(result.candidates) == 1

    @pytest.mark.asyncio
    async def test_bare_search_does_not_rank(self):
        """Without requirements there is nothing to score against."""
        service = _service(_kaggle())
        result = await service.search("q", limit=10)
        assert all(c.ranking_score is None for c in result.candidates)


class TestDiscover:
    @pytest.mark.asyncio
    async def test_uses_structured_requirements_for_genai(self):
        genai = _genai([_ai()])
        service = _service(_kaggle(), genai)
        await service.discover(make_requirements(), limit=10)
        assert len(genai.calls) == 1
        assert genai.calls[0].project_requirements is not None

    @pytest.mark.asyncio
    async def test_tells_genai_what_verified_sources_already_found(self):
        genai = _genai([_ai()])
        service = _service(
            _kaggle([make_candidate(name="Student Academic Performance")]), genai
        )
        await service.discover(make_requirements(), limit=10)
        assert genai.calls[0].known_dataset_names == ["Student Academic Performance"]

    @pytest.mark.asyncio
    async def test_ranks_candidates_when_requirements_exist(self):
        service = _service(_kaggle())
        result = await service.discover(make_requirements(), limit=10)
        assert all(c.ranking_score is not None for c in result.candidates)
        assert all(c.ranking_factors for c in result.candidates)

    @pytest.mark.asyncio
    async def test_results_are_sorted_by_score(self):
        ai = _ai()
        ai.ranking_score = 0.0
        verified = make_candidate(name="Perfect Match", source_id="o/perfect")
        verified.ranking_score = 99.0
        service = _service(_kaggle([verified]), _genai([ai]))
        result = await service.discover(make_requirements(), limit=10)
        scores = [c.ranking_score for c in result.candidates]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_verified_and_ai_are_returned_together(self):
        service = _service(_kaggle(), _genai([_ai()]))
        result = await service.discover(make_requirements(), limit=10)
        assert result.verified_count == 1
        assert result.unverified_count == 1

    @pytest.mark.asyncio
    async def test_ai_candidates_are_never_verified(self):
        service = _service(_kaggle(), _genai([_ai()]))
        result = await service.discover(make_requirements(), limit=10)
        for candidate in result.candidates:
            if candidate.source_type is SourceType.AI_SUGGESTED:
                assert candidate.verification_status is VerificationStatus.UNVERIFIED


class TestDeduplicationHandoff:
    @pytest.mark.asyncio
    async def test_duplicates_from_two_sources_collapse(self):
        verified = make_candidate(
            name="Student Performance", source_id="o/one", url="https://example.org/x"
        )
        ai = _ai(name="Student Performance", url="https://example.org/x")
        service = _service(_kaggle([verified]), _genai([ai]))
        result = await service.discover(make_requirements(), limit=10)
        assert len(result.candidates) == 1
        assert result.raw_candidate_count == 2

    @pytest.mark.asyncio
    async def test_dedup_state_does_not_leak_between_requests(self):
        verified = make_candidate(
            name="Student Performance", source_id="o/one", url="https://example.org/x"
        )
        service = _service(_kaggle([verified]))
        first = await service.discover(make_requirements(), limit=10)
        second = await service.discover(make_requirements(), limit=10)
        assert len(first.candidates) == 1
        assert len(second.candidates) == 1
        assert second.candidates[0].merged_candidate_count == 1


class TestPartialFailure:
    @pytest.mark.asyncio
    async def test_one_failing_source_does_not_break_the_result(self):
        failing = _kaggle(
            error=DatasetSourceError(code="KAGGLE_TIMEOUT", message="timed out")
        )
        service = _service(failing, _genai([_ai()]))
        result = await service.discover(make_requirements(), limit=10)
        assert len(result.candidates) == 1
        statuses = {o.source: o.status.value for o in result.sources}
        assert statuses["kaggle"] == "failed"

    @pytest.mark.asyncio
    async def test_failure_is_reported_not_silently_dropped(self):
        failing = _kaggle(
            error=DatasetSourceError(code="KAGGLE_RATE_LIMITED", message="slow down")
        )
        # A second working source keeps the run partially successful, so the
        # failure is reported in the outcome list instead of raised.
        service = _service(failing, _genai([_ai()]))
        result = await service.discover(make_requirements(), limit=10)
        outcome = next(o for o in result.sources if o.source == "kaggle")
        assert outcome.error_code == "KAGGLE_RATE_LIMITED"
        assert result.candidates

    @pytest.mark.asyncio
    async def test_all_sources_failing_raises_the_first_cause(self):
        failing = _kaggle(
            error=DatasetSourceError(code="KAGGLE_AUTHENTICATION_ERROR", message="nope")
        )
        service = _service(failing)
        with pytest.raises(DatasetSourceError) as exc:
            await service.discover(make_requirements(), limit=10)
        assert exc.value.code == "KAGGLE_AUTHENTICATION_ERROR"

    @pytest.mark.asyncio
    async def test_skipped_source_alone_does_not_raise(self):
        service = _service(_genai([]))
        result = await service.search("q", limit=10)
        assert result.candidates == []

    @pytest.mark.asyncio
    async def test_unexpected_source_exception_is_contained(self):
        class _Boom(_StubSource):
            async def search(self, request):
                raise RuntimeError("kaboom")

        boom = _Boom("kaggle", SourceType.VERIFIED_EXTERNAL)
        service = _service(boom, _genai([_ai()]))
        result = await service.discover(make_requirements(), limit=10)
        assert len(result.candidates) == 1

    @pytest.mark.asyncio
    async def test_error_message_never_contains_a_credential(self):
        failing = _kaggle(
            error=DatasetSourceError(
                code="KAGGLE_AUTHENTICATION_ERROR",
                message="Token KGAT_secret_value rejected",
            )
        )
        service = _service(failing, _genai([_ai()]))
        result = await service.discover(make_requirements(), limit=10)
        assert "KGAT_secret_value" not in (result.sources[0].message or "")


class TestSourceCallDiscipline:
    @pytest.mark.asyncio
    async def test_each_source_is_queried_once(self):
        kaggle = _kaggle()
        genai = _genai([_ai()])
        service = _service(kaggle, genai)
        await service.discover(make_requirements(), limit=10)
        assert len(kaggle.calls) == 1
        assert len(genai.calls) == 1

    @pytest.mark.asyncio
    async def test_verified_sources_run_before_suggestions(self):
        order: list[str] = []

        class _Ordered(_StubSource):
            async def search(self, request):
                order.append(self.name)
                return []

        service = _service(
            _Ordered("kaggle", SourceType.VERIFIED_EXTERNAL),
            _Ordered("genai", SourceType.AI_SUGGESTED, requires_requirements=True),
        )
        await service.discover(make_requirements(), limit=10)
        assert order == ["kaggle", "genai"]


class TestMetadata:
    @pytest.mark.asyncio
    async def test_duration_is_recorded(self):
        service = _service(_kaggle())
        result = await service.search("q", limit=10)
        assert result.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_source_outcomes_cover_every_source(self):
        service = _service(_kaggle(), _genai([_ai()]))
        result = await service.discover(make_requirements(), limit=10)
        assert {o.source for o in result.sources} == {"kaggle", "genai"}

    def test_ranking_weights_are_exposed(self):
        weights = _service(_kaggle()).ranking_weights()
        assert weights
        assert round(sum(weights.values())) == 100
