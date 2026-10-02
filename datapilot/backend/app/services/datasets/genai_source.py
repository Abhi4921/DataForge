"""GenAI dataset candidate discovery.

This source is deliberately NOT modelled as "the internet". It is a
*suggestion* channel: it receives the structured project requirements produced
by the Project Requirement Analyzer and asks Gemini which public datasets might
be relevant.

Every candidate it returns is forced to ``source_type=ai_suggested`` and
``verification_status=unverified`` in the normalizer. The model's own
``is_well_known`` flag and ``confidence`` are recorded as advisory metadata but
never used to claim that a dataset exists.

The source is skipped (not called) when no structured requirements are
available, which is what keeps ``/datasets/search`` from spending a Gemini call.
"""

from __future__ import annotations

from typing import ClassVar, Optional

from app.core.logging_config import get_logger
from app.schemas.dataset import (
    DatasetCandidate,
    SourceSearchRequest,
    SourceType,
    VerificationStatus,
)
from app.services.datasets.base import DatasetSource, DatasetSourceError
from app.services.datasets.normalizer import normalize_genai_candidates
from app.services.llm_service import LLMService, LLMServiceError

logger = get_logger(__name__)

# LLMServiceError codes that must not surface as a dataset-source failure code.
_LLM_ERROR_MAP = {
    "LLM_AUTHENTICATION_ERROR": "DATASET_DISCOVERY_LLM_AUTH_ERROR",
    "LLM_RATE_LIMITED": "DATASET_DISCOVERY_LLM_RATE_LIMITED",
    "LLM_TIMEOUT": "DATASET_DISCOVERY_LLM_TIMEOUT",
    "LLM_UNAVAILABLE": "DATASET_DISCOVERY_LLM_UNAVAILABLE",
    "LLM_INVALID_RESPONSE": "DATASET_DISCOVERY_LLM_INVALID_RESPONSE",
    "INTERNAL_ERROR": "DATASET_DISCOVERY_LLM_UNAVAILABLE",
}


class GenAIDatasetSource(DatasetSource):
    """Suggests candidate datasets via Gemini based on project requirements."""

    name: ClassVar[str] = "genai"
    source_type: ClassVar[SourceType] = SourceType.AI_SUGGESTED
    default_limit: ClassVar[int] = 8
    max_limit: ClassVar[int] = 20

    def __init__(self, settings=None, llm_service: Optional[LLMService] = None) -> None:
        super().__init__(settings)
        self._llm_service = llm_service or LLMService(self._settings)

    @property
    def is_enabled(self) -> bool:
        return bool(self._settings.dataset_discovery_genai_enabled)

    @property
    def is_configured(self) -> bool:
        return bool(self._settings.dataset_discovery_genai_enabled) and (
            self._llm_service.is_configured
        )

    @property
    def supports_project_requirements(self) -> bool:
        # Without structured requirements this source has nothing to reason
        # from, so it must be skipped rather than queried with an empty context.
        return True

    async def search(self, request: SourceSearchRequest) -> list[DatasetCandidate]:
        if not self.is_enabled:
            raise DatasetSourceError(
                code="DATASET_SOURCE_DISABLED",
                message="GenAI dataset discovery is disabled.",
            )

        requirements = request.project_requirements
        if requirements is None:
            # Defensive: the discovery service skips us, but a direct caller
            # must not receive fabricated generic dataset names.
            raise DatasetSourceError(
                code="DATASET_SOURCE_REQUIRES_REQUIREMENTS",
                message=(
                    "GenAI dataset discovery requires structured project "
                    "requirements and cannot answer a bare text query."
                ),
            )

        max_candidates = max(1, min(int(request.limit), self.max_limit))
        max_candidates = min(max_candidates, self._settings.dataset_discovery_genai_max_candidates)

        try:
            discovery = await self._llm_service.discover_datasets(
                requirements=requirements,
                max_candidates=max_candidates,
                known_datasets_hint=request.known_dataset_names,
            )
        except LLMServiceError as exc:
            logger.warning("GenAI dataset discovery failed (%s)", exc.code)
            raise DatasetSourceError(
                code=_LLM_ERROR_MAP.get(exc.code, "DATASET_DISCOVERY_LLM_UNAVAILABLE"),
                message="GenAI dataset discovery could not be completed.",
            ) from exc

        candidates = normalize_genai_candidates(discovery.candidates)
        if len(candidates) < len(discovery.candidates):
            logger.info(
                "Discarded %d invalid GenAI candidates",
                len(discovery.candidates) - len(candidates),
            )

        unverified = [
            c for c in candidates if c.verification_status is VerificationStatus.UNVERIFIED
        ]
        logger.info(
            "GenAI returned %d candidates (%d verified, %d unverified)",
            len(candidates),
            len(candidates) - len(unverified),
            len(unverified),
        )
        return candidates[:max_candidates]

    async def healthcheck(self) -> bool:
        return self.is_configured

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<GenAIDatasetSource enabled={self.is_enabled} configured={self.is_configured}>"
