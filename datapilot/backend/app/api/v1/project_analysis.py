from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, Request

from app.core.logging_config import get_logger
from app.schemas.project import (
    ErrorCode,
    ErrorDetail,
    ErrorResponse,
    ProjectAnalysisRequest,
    ProjectAnalysisResponse,
)
from app.services.llm_service import LLMServiceError
from app.services.project_analyzer import ProjectAnalyzer

logger = get_logger(__name__)

router = APIRouter(prefix="/projects", tags=["Project Analysis"])

_analyzer: ProjectAnalyzer | None = None


def get_analyzer() -> ProjectAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = ProjectAnalyzer()
    return _analyzer


@router.post(
    "/analyze",
    response_model=ProjectAnalysisResponse,
    status_code=200,
    summary="Analyze a project description",
    description=(
        "Takes a natural-language ML/AI project description (max 150 words) "
        "and returns structured project requirements including domain, "
        "subdomain, ML tasks, keywords, dataset requirements, ambiguities, "
        "and missing information."
    ),
    responses={
        200: {"description": "Successful analysis"},
        422: {"description": "Validation error"},
        500: {"description": "Internal server error"},
    },
)
async def analyze_project(
    body: ProjectAnalysisRequest,
    request: Request,
) -> ProjectAnalysisResponse:
    request_id = str(uuid.uuid4())
    logger.info("[%s] POST /api/v1/projects/analyze", request_id)

    analyzer = get_analyzer()

    try:
        return await analyzer.analyze(body, request_id=request_id)
    except HTTPException:
        raise
    except LLMServiceError as exc:
        status_map = {
            "LLM_AUTHENTICATION_ERROR": 401,
            "LLM_RATE_LIMITED": 429,
            "LLM_TIMEOUT": 504,
            "LLM_UNAVAILABLE": 502,
            "LLM_INVALID_RESPONSE": 502,
            "INTERNAL_ERROR": 500,
        }
        status = status_map.get(exc.code, 500)
        raise HTTPException(
            status_code=status,
            detail=ErrorResponse(
                request_id=request_id,
                error=ErrorDetail(
                    code=ErrorCode(exc.code),
                    message=exc.message,
                ),
            ).model_dump(),
        ) from exc
    except Exception as exc:
        logger.error("[%s] Unhandled error: %s", request_id, exc)
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                request_id=request_id,
                error=ErrorDetail(
                    code=ErrorCode.INTERNAL_ERROR,
                    message="An internal error occurred.",
                ),
            ).model_dump(),
        ) from exc
