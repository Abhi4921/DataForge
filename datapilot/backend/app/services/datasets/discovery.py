"""Unified, source-agnostic dataset discovery.

    Project Requirements
            |
    DatasetDiscoveryService
        /          \\
  Kaggle          Gemini
  (real results)  (AI suggestions)
        \\          /
          Normalize  (done inside each source)
               |
          Deduplicate
               |
               Rank  (deterministic)
               |
       Unified ranked result

The service never branches on which source produced a candidate. Adding UCI,
Hugging Face or research-paper discovery means registering one more
:class:`~app.services.datasets.base.DatasetSource` in
``build_default_sources`` and changing nothing else.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Optional, Sequence

from app.core.config import Settings, get_settings
from app.core.logging_config import get_logger
from app.schemas.dataset import (
    DatasetCandidate,
    SourceOutcome,
    SourceSearchRequest,
    SourceStatus,
    SourceType,
)
from app.schemas.project import ProjectAnalysisData, ProjectAnalysisRequest
from app.services.datasets.base import DatasetSource, DatasetSourceError, redact_secrets
from app.services.datasets.deduplicator import DatasetDeduplicator
from app.services.datasets.query_builder import (
    build_dataset_search_query,
    build_dataset_search_queries,
)
from app.services.datasets.ranking import DatasetRankingEngine
from app.services.project_analyzer import ProjectAnalyzer

logger = get_logger(__name__)


@dataclass
class DatasetDiscoveryResult:
    """Outcome of one discovery run."""

    query: str
    candidates: list[DatasetCandidate] = field(default_factory=list)
    sources: list[SourceOutcome] = field(default_factory=list)
    requirements: Optional[ProjectAnalysisData] = None
    raw_candidate_count: int = 0
    duration_ms: float = 0.0

    def count_by_source_type(self, source_type: SourceType) -> int:
        return sum(1 for c in self.candidates if c.source_type is source_type)

    @property
    def verified_count(self) -> int:
        return self.count_by_source_type(SourceType.VERIFIED_EXTERNAL)

    @property
    def unverified_count(self) -> int:
        return self.count_by_source_type(SourceType.AI_SUGGESTED)


class DatasetDiscoveryService:
    """Queries every registered source, then dedupes and ranks the results."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        sources: Optional[Sequence[DatasetSource]] = None,
        ranking_engine: Optional[DatasetRankingEngine] = None,
        project_analyzer: Optional[ProjectAnalyzer] = None,
    ) -> None:
        from app.services.datasets import build_default_sources

        self._settings = settings or get_settings()
        self._sources: list[DatasetSource] = (
            list(sources) if sources is not None else build_default_sources(self._settings)
        )
        self._ranking = ranking_engine or DatasetRankingEngine(self._settings)
        self._project_analyzer = project_analyzer

    @property
    def sources(self) -> list[DatasetSource]:
        return list(self._sources)

    def ranking_weights(self) -> dict[str, float]:
        """Active ranking weights, so a response documents its own scoring."""
        return self._ranking.weights()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        limit: int,
        request_id: Optional[str] = None,
    ) -> DatasetDiscoveryResult:
        """Discover datasets for a bare text query.

        No project requirements exist, so sources that require them (GenAI) are
        reported as ``skipped`` rather than being called with an empty context.
        This is what keeps ``/datasets/search`` free of Gemini calls.
        """
        return await self._discover(
            query=query,
            limit=limit,
            requirements=None,
            request_id=request_id,
        )

    async def discover(
        self,
        requirements: ProjectAnalysisData,
        limit: int,
        request_id: Optional[str] = None,
        query: Optional[str] = None,
    ) -> DatasetDiscoveryResult:
        """Discover datasets for already-structured project requirements."""
        return await self._discover(
            query=query
            or build_dataset_search_query(
                requirements,
                max_terms=self._settings.dataset_discovery_max_query_terms,
            ),
            limit=limit,
            requirements=requirements,
            request_id=request_id,
        )

    async def recommend(
        self,
        description: str,
        limit: int,
        request_id: Optional[str] = None,
    ) -> tuple[ProjectAnalysisData, DatasetDiscoveryResult]:
        """Full DataPilot workflow: analyze -> discover -> rank.

        Returns the structured requirements together with the ranked dataset
        result so the caller can present both.
        """
        rid = request_id or str(uuid.uuid4())
        analyzer = self._project_analyzer or ProjectAnalyzer(self._settings)
        analysis = await analyzer.analyze(
            ProjectAnalysisRequest(description=description),
            request_id=rid,
        )
        requirements = analysis.data
        result = await self.discover(requirements, limit=limit, request_id=rid)
        return requirements, result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _discover(
        self,
        query: str,
        limit: int,
        requirements: Optional[ProjectAnalysisData],
        request_id: Optional[str],
    ) -> DatasetDiscoveryResult:
        rid = request_id or str(uuid.uuid4())
        start = time.monotonic()
        normalized_query = " ".join((query or "").split()).strip()

        if not normalized_query:
            raise DatasetSourceError(
                code="DATASET_QUERY_EMPTY",
                message="A non-empty search query is required.",
            )

        limit = max(1, min(int(limit), self._settings.kaggle_max_limit))
        base_request = SourceSearchRequest(
            query=normalized_query,
            limit=limit,
            project_requirements=requirements,
        )
        if requirements is not None:
            pool_limit = min(
                50,
                max(
                    limit,
                    int(self._settings.dataset_discovery_candidate_pool_size),
                ),
            )
            base_request.candidate_pool_limit = pool_limit
            base_request.query_variants = build_dataset_search_queries(
                requirements,
                max_terms=self._settings.dataset_discovery_max_query_terms,
            )[1:]

        verified_sources, suggestion_sources = self._partition_sources()
        outcomes: list[SourceOutcome] = []
        collected: list[DatasetCandidate] = []

        # Pass 1: verified external sources. Each source is queried exactly once.
        verified_names: list[str] = []
        for source in verified_sources:
            outcome, candidates = await self._run_source(source, base_request, rid)
            outcomes.append(outcome)
            collected.extend(candidates)
            verified_names.extend(c.name for c in candidates)

        # Pass 2: suggestion-based sources, told what verified sources already
        # found so they propose complements rather than repeats. Still one call
        # per source.
        suggestion_request = base_request.model_copy(
            update={"known_dataset_names": verified_names[:20]}
        )
        for source in suggestion_sources:
            outcome, candidates = await self._run_source(source, suggestion_request, rid)
            outcomes.append(outcome)
            collected.extend(candidates)

        raw_count = len(collected)

        # A fresh deduplicator per run: the index holds candidate references, so
        # reusing one across requests would leak results between callers.
        deduplicated = DatasetDeduplicator().deduplicate(collected)
        ranked = (
            self._ranking.rank(deduplicated, requirements)
            if requirements is not None
            else list(deduplicated)
        )
        final = ranked[:limit]

        duration_ms = round((time.monotonic() - start) * 1000, 2)
        logger.info(
            "[%s] Discovery for %r: %d raw -> %d unique -> %d returned "
            "(%d verified, %d unverified) in %.1fms",
            rid,
            normalized_query,
            raw_count,
            len(deduplicated),
            len(final),
            sum(1 for c in final if c.source_type is SourceType.VERIFIED_EXTERNAL),
            sum(1 for c in final if c.source_type is SourceType.AI_SUGGESTED),
            duration_ms,
        )

        if not final and outcomes:
            failed = [o for o in outcomes if o.status is SourceStatus.FAILED]
            if failed and all(o.status is not SourceStatus.OK for o in outcomes):
                # Every source failed: surface the real cause instead of an
                # empty list that reads like "no datasets exist".
                raise DatasetSourceError(
                    code=failed[0].error_code or "DATASET_DISCOVERY_FAILED",
                    message=failed[0].message or "All dataset sources failed.",
                )

        return DatasetDiscoveryResult(
            query=normalized_query,
            candidates=final,
            sources=outcomes,
            requirements=requirements,
            raw_candidate_count=raw_count,
            duration_ms=duration_ms,
        )

    def _partition_sources(
        self,
    ) -> tuple[list[DatasetSource], list[DatasetSource]]:
        verified = [s for s in self._sources if s.source_type is SourceType.VERIFIED_EXTERNAL]
        suggestions = [s for s in self._sources if s.source_type is SourceType.AI_SUGGESTED]
        return verified, suggestions

    async def _run_source(
        self,
        source: DatasetSource,
        request: SourceSearchRequest,
        rid: str,
    ) -> tuple[SourceOutcome, list[DatasetCandidate]]:
        """Query one source, converting any failure into a reported outcome."""
        if not source.is_configured:
            return (
                SourceOutcome(
                    source=source.name,
                    source_type=source.source_type,
                    status=SourceStatus.SKIPPED,
                    message="Source is not configured",
                ),
                [],
            )

        if source.supports_project_requirements and request.project_requirements is None:
            return (
                SourceOutcome(
                    source=source.name,
                    source_type=source.source_type,
                    status=SourceStatus.SKIPPED,
                    message="No structured project requirements supplied for this query",
                ),
                [],
            )

        start = time.monotonic()
        try:
            candidates = await source.search(request)
        except DatasetSourceError as exc:
            logger.warning("[%s] Source %s failed: %s", rid, source.name, exc.code)
            outcome = SourceOutcome(
                source=source.name,
                source_type=source.source_type,
                status=SourceStatus.FAILED,
                error_code=exc.code,
                message=redact_secrets(exc.message),
            )
        except Exception as exc:
            # Never leak an upstream exception body into a response.
            logger.error("[%s] Source %s raised %s", rid, source.name, type(exc).__name__)
            outcome = SourceOutcome(
                source=source.name,
                source_type=source.source_type,
                status=SourceStatus.FAILED,
                error_code="DATASET_SOURCE_UNEXPECTED_ERROR",
                message="The dataset source failed unexpectedly.",
            )
        else:
            outcome = SourceOutcome(
                source=source.name,
                source_type=source.source_type,
                status=SourceStatus.OK,
                candidate_count=len(candidates),
            )
        outcome.duration_ms = round((time.monotonic() - start) * 1000, 2)
        return outcome, candidates if outcome.status is SourceStatus.OK else []
