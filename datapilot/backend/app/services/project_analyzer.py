from __future__ import annotations

import re
import time
import uuid
from typing import Optional

from app.core.config import Settings, get_settings
from app.core.logging_config import get_logger
from app.prompts.project_analysis import build_dictionary_context
from app.schemas.project import (
    ErrorCode,
    KeywordMatchEvidence,
    LLMProjectAnalysis,
    ProjectAnalysisData,
    ProjectAnalysisResponse,
    ProjectAnalysisRequest,
    ResponseMeta,
)
from app.services.keyword_service import KeywordService
from app.services.llm_service import LLMService, LLMServiceError
from app.services.reconciliation_service import ReconciliationService

logger = get_logger(__name__)

_WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)


def _count_words(text: str) -> int:
    return len(_WORD_RE.findall(text))


class ProjectAnalyzer:
    """Orchestrates dictionary matching, LLM analysis, and reconciliation."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        keyword_service: Optional[KeywordService] = None,
        llm_service: Optional[LLMService] = None,
        reconciliation_service: Optional[ReconciliationService] = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._keyword_service = keyword_service or KeywordService()
        self._llm_service = llm_service or LLMService(self._settings)
        self._reconciliation = reconciliation_service or ReconciliationService()

    @property
    def keyword_service(self) -> KeywordService:
        return self._keyword_service

    @property
    def llm_service(self) -> LLMService:
        return self._llm_service

    async def analyze(
        self,
        request: ProjectAnalysisRequest,
        request_id: Optional[str] = None,
    ) -> ProjectAnalysisResponse:
        """Full pipeline: validate -> dictionary match -> LLM -> reconcile."""
        rid = request_id or str(uuid.uuid4())
        start = time.monotonic()

        self._validate_input(request)

        description = request.description  # type: ignore
        word_count = _count_words(description)

        logger.info(
            "[%s] Analyzing project description (%d words)", rid, word_count
        )

        dict_matches = self._keyword_service.match(description)
        logger.info(
            "[%s] Dictionary matched %d concepts", rid, len(dict_matches)
        )

        dictionary_context = build_dictionary_context(
            self._keyword_service, dict_matches
        )

        try:
            llm_output = await self._llm_service.analyze_project(
                description=description,
                dictionary_context=dictionary_context,
            )
            logger.info("[%s] LLM analysis complete", rid)
        except LLMServiceError:
            raise
        except Exception as exc:
            logger.error("[%s] Unexpected analysis error: %s", rid, exc)
            raise LLMServiceError(
                code="INTERNAL_ERROR",
                message="An unexpected error occurred during analysis.",
            ) from exc

        data = self._reconciliation.reconcile(
            description=description,
            dictionary_matches=dict_matches,
            llm_output=llm_output,
            word_count=word_count,
        )

        elapsed_ms = (time.monotonic() - start) * 1000
        logger.info("[%s] Analysis complete in %.1fms", rid, elapsed_ms)

        return ProjectAnalysisResponse(
            request_id=rid,
            data=data,
            meta=ResponseMeta(
                model=self._settings.gemini_model,
                dictionary_version=self._keyword_service.version,
                processing_time_ms=round(elapsed_ms, 2),
                timestamp=time.strftime(
                    "%Y-%m-%dT%H:%M:%S", time.gmtime()
                ),
            ),
        )

    def _validate_input(self, request: ProjectAnalysisRequest) -> None:
        from app.schemas.project import ErrorResponse, ErrorDetail
        from fastapi import HTTPException

        if request.description is None:
            raise HTTPException(
                status_code=422,
                detail=ErrorResponse(
                    error=ErrorDetail(
                        code=ErrorCode.PROJECT_DESCRIPTION_REQUIRED,
                        message="Project description is required.",
                    ),
                ).model_dump(),
            )

        desc = request.description.strip()
        if not desc:
            raise HTTPException(
                status_code=422,
                detail=ErrorResponse(
                    error=ErrorDetail(
                        code=ErrorCode.PROJECT_DESCRIPTION_EMPTY,
                        message="Project description must not be empty.",
                    ),
                ).model_dump(),
            )

        word_count = _count_words(desc)
        if word_count > self._settings.max_project_description_words:
            raise HTTPException(
                status_code=422,
                detail=ErrorResponse(
                    error=ErrorDetail(
                        code=ErrorCode.PROJECT_DESCRIPTION_TOO_LONG,
                        message=(
                            f"Project description must not exceed "
                            f"{self._settings.max_project_description_words} words."
                        ),
                        details={
                            "word_count": word_count,
                            "maximum_allowed": self._settings.max_project_description_words,
                        },
                    ),
                ).model_dump(),
            )

        if len(desc) > self._settings.max_project_description_chars:
            raise HTTPException(
                status_code=422,
                detail=ErrorResponse(
                    error=ErrorDetail(
                        code=ErrorCode.PROJECT_DESCRIPTION_TOO_LONG,
                        message="Project description exceeds maximum character limit.",
                        details={
                            "char_count": len(desc),
                            "maximum_allowed": self._settings.max_project_description_chars,
                        },
                    ),
                ).model_dump(),
            )
