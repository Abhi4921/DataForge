"""Dataset discovery and recommendation endpoints.

* ``POST /api/v1/datasets/search``    - direct metadata search.
* ``POST /api/v1/datasets/recommend``  - the full DataPilot workflow:
  project description -> requirements -> discovery -> ranking.
"""

from __future__ import annotations

import uuid
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request

from app.core.logging_config import get_logger
from app.schemas.dataset import (
    DatasetRecommendationRequest,
    DatasetRecommendationResponse,
    DatasetSearchRequest,
    DatasetSearchResponse,
    RankedDataset,
)
from app.schemas.project import (
    MAX_PROJECT_DESCRIPTION_WORDS,
    ErrorCode,
    ErrorDetail,
    ErrorResponse,
    count_words,
)
from app.services.datasets.base import DatasetSourceError, redact_secrets
from app.services.datasets.discovery import DatasetDiscoveryService
from app.services.llm_service import LLMServiceError

logger = get_logger(__name__)

router = APIRouter(prefix="/datasets", tags=["Dataset Discovery"])

_service: Optional[DatasetDiscoveryService] = None

# Source-level failure codes mapped to HTTP statuses. Codes are stable
# identifiers; the accompanying messages are safe to return.
_STATUS_MAP: dict[str, int] = {
    "DATASET_QUERY_EMPTY": 422,
    "DATASET_QUERY_REQUIRED": 422,
    "DATASET_QUERY_TOO_LONG": 422,
    "DATASET_SOURCE_REQUIRES_REQUIREMENTS": 422,
    "DATASET_SOURCE_DISABLED": 503,
    "KAGGLE_AUTHENTICATION_ERROR": 401,
    "DATASET_AUTHENTICATION_ERROR": 401,
    "DATASET_DISCOVERY_LLM_AUTH_ERROR": 401,
    "LLM_AUTHENTICATION_ERROR": 401,
    "KAGGLE_RATE_LIMITED": 429,
    "DATASET_RATE_LIMITED": 429,
    "DATASET_DISCOVERY_LLM_RATE_LIMITED": 429,
    "LLM_RATE_LIMITED": 429,
    "KAGGLE_TIMEOUT": 504,
    "DATASET_SOURCE_TIMEOUT": 504,
    "DATASET_DISCOVERY_LLM_TIMEOUT": 504,
    "LLM_TIMEOUT": 504,
    "KAGGLE_UNAVAILABLE": 502,
    "KAGGLE_INVALID_RESPONSE": 502,
    "KAGGLE_NOT_FOUND": 404,
    "DATASET_SOURCE_INVALID_ID": 422,
    "DATASET_SOURCE_UNSUPPORTED_OPERATION": 501,
    "DATASET_DISCOVERY_LLM_UNAVAILABLE": 502,
    "DATASET_DISCOVERY_LLM_INVALID_RESPONSE": 502,
    "DATASET_DISCOVERY_FAILED": 502,
    "DATASET_SOURCE_UNEXPECTED_ERROR": 502,
    "LLM_UNAVAILABLE": 502,
    "LLM_INVALID_RESPONSE": 502,
}

_PUBLIC_CODE_MAP: dict[str, ErrorCode] = {
    "DATASET_QUERY_EMPTY": ErrorCode.DATASET_QUERY_EMPTY,
    "DATASET_QUERY_REQUIRED": ErrorCode.DATASET_QUERY_REQUIRED,
    "DATASET_QUERY_TOO_LONG": ErrorCode.DATASET_QUERY_TOO_LONG,
    "DATASET_SOURCE_REQUIRES_REQUIREMENTS": ErrorCode.DATASET_QUERY_REQUIRED,
    "DATASET_SOURCE_DISABLED": ErrorCode.DATASET_SOURCE_UNAVAILABLE,
    "KAGGLE_AUTHENTICATION_ERROR": ErrorCode.DATASET_AUTHENTICATION_ERROR,
    "DATASET_AUTHENTICATION_ERROR": ErrorCode.DATASET_AUTHENTICATION_ERROR,
    "DATASET_DISCOVERY_LLM_AUTH_ERROR": ErrorCode.DATASET_AUTHENTICATION_ERROR,
    "LLM_AUTHENTICATION_ERROR": ErrorCode.DATASET_AUTHENTICATION_ERROR,
    "KAGGLE_RATE_LIMITED": ErrorCode.DATASET_RATE_LIMITED,
    "DATASET_RATE_LIMITED": ErrorCode.DATASET_RATE_LIMITED,
    "DATASET_DISCOVERY_LLM_RATE_LIMITED": ErrorCode.DATASET_RATE_LIMITED,
    "LLM_RATE_LIMITED": ErrorCode.DATASET_RATE_LIMITED,
    "KAGGLE_TIMEOUT": ErrorCode.DATASET_SOURCE_TIMEOUT,
    "DATASET_SOURCE_TIMEOUT": ErrorCode.DATASET_SOURCE_TIMEOUT,
    "DATASET_DISCOVERY_LLM_TIMEOUT": ErrorCode.DATASET_SOURCE_TIMEOUT,
    "LLM_TIMEOUT": ErrorCode.DATASET_SOURCE_TIMEOUT,
    "KAGGLE_UNAVAILABLE": ErrorCode.DATASET_SOURCE_UNAVAILABLE,
    "KAGGLE_INVALID_RESPONSE": ErrorCode.DATASET_INVALID_RESPONSE,
    "KAGGLE_NOT_FOUND": ErrorCode.DATASET_INVALID_RESPONSE,
    "DATASET_DISCOVERY_LLM_INVALID_RESPONSE": ErrorCode.DATASET_INVALID_RESPONSE,
    "LLM_INVALID_RESPONSE": ErrorCode.DATASET_INVALID_RESPONSE,
    "DATASET_SOURCE_INVALID_ID": ErrorCode.DATASET_QUERY_REQUIRED,
    "DATASET_DISCOVERY_LLM_UNAVAILABLE": ErrorCode.DATASET_SOURCE_UNAVAILABLE,
    "LLM_UNAVAILABLE": ErrorCode.DATASET_SOURCE_UNAVAILABLE,
    "DATASET_DISCOVERY_FAILED": ErrorCode.DATASET_DISCOVERY_FAILED,
    "DATASET_SOURCE_UNEXPECTED_ERROR": ErrorCode.DATASET_SOURCE_UNAVAILABLE,
    "DATASET_SOURCE_UNSUPPORTED_OPERATION": ErrorCode.DATASET_SOURCE_UNSUPPORTED_OPERATION,
}


def get_service() -> DatasetDiscoveryService:
    global _service
    if _service is None:
        _service = DatasetDiscoveryService()
    return _service


def _error(request_id: str, exc: DatasetSourceError | LLMServiceError) -> HTTPException:
    status = _STATUS_MAP.get(exc.code, 502)
    code = _PUBLIC_CODE_MAP.get(exc.code)
    if code is None:
        # Unknown internal code: expose the class of failure, not the detail.
        code = ErrorCode.DATASET_SOURCE_UNAVAILABLE
        message = "A dataset source could not be reached."
    else:
        message = redact_secrets(exc.message)
    return HTTPException(
        status_code=status,
        detail=ErrorResponse(
            request_id=request_id,
            error=ErrorDetail(code=code, message=message),
        ).model_dump(),
    )


@router.post(
    "/search",
    response_model=DatasetSearchResponse,
    status_code=200,
    summary="Search for datasets",
    description=(
        "Searches the configured external dataset sources (currently Kaggle) "
        "for dataset metadata matching a free-text query. Metadata discovery "
        "only: no dataset is downloaded.\n\n"
        "Results carry `source_type=verified_external` and "
        "`verification_status=verified` because they are real published "
        "datasets. Sources that require structured project requirements (such "
        "as GenAI suggestions) are reported as `skipped` and are not called."
    ),
    responses={
        200: {"description": "Search completed"},
        401: {"description": "Dataset source rejected the credentials"},
        422: {"description": "Invalid query"},
        429: {"description": "Dataset source rate limit exceeded"},
        502: {"description": "Dataset source unavailable or returned an invalid response"},
        504: {"description": "Dataset source timed out"},
    },
)
async def search_datasets(
    body: DatasetSearchRequest,
    request: Request,
) -> DatasetSearchResponse:
    request_id = str(uuid.uuid4())
    logger.info("[%s] POST /api/v1/datasets/search (limit=%s)", request_id, body.limit)

    if body.query is None:
        raise HTTPException(
            status_code=422,
            detail=ErrorResponse(
                request_id=request_id,
                error=ErrorDetail(
                    code=ErrorCode.DATASET_QUERY_REQUIRED,
                    message="A 'query' field is required.",
                ),
            ).model_dump(),
        )
    if not body.query:
        raise HTTPException(
            status_code=422,
            detail=ErrorResponse(
                request_id=request_id,
                error=ErrorDetail(
                    code=ErrorCode.DATASET_QUERY_EMPTY,
                    message="The 'query' field must not be empty.",
                ),
            ).model_dump(),
        )

    service = get_service()
    try:
        result = await service.search(query=body.query, limit=body.limit, request_id=request_id)
    except (DatasetSourceError, LLMServiceError) as exc:
        raise _error(request_id, exc) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("[%s] Unhandled dataset search error: %s", request_id, type(exc).__name__)
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

    return DatasetSearchResponse(
        request_id=request_id,
        query=result.query,
        count=len(result.candidates),
        datasets=result.candidates,
        sources=result.sources,
        meta={
            "raw_candidates": result.raw_candidate_count,
            "verified_count": result.verified_count,
            "ai_suggested_count": result.unverified_count,
            "processing_time_ms": result.duration_ms,
        },
    )


@router.post(
    "/recommend",
    response_model=DatasetRecommendationResponse,
    status_code=200,
    summary="Recommend datasets for a project description",
    description=(
        "The full DataPilot workflow.\n\n"
        "1. The existing Project Requirement Analyzer converts the description "
        "into structured requirements.\n"
        "2. Those requirements drive a real dataset search (Kaggle) and a "
        "GenAI candidate-suggestion pass.\n"
        "3. Candidates from every source are normalized into one model, "
        "deduplicated and scored with a deterministic ranking engine.\n\n"
        "Verified external datasets and AI-suggested candidates are returned "
        "together with explicit `verification_status` and evidence-based "
        "reasons. AI suggestions are never presented as confirmed datasets."
    ),
    responses={
        200: {"description": "Recommendations produced"},
        401: {"description": "A source rejected the configured credentials"},
        422: {"description": "Invalid project description or limit"},
        429: {"description": "A source rate limit was exceeded"},
        502: {"description": "A source was unavailable or returned an invalid response"},
        504: {"description": "A source timed out"},
    },
)
async def recommend_datasets(
    body: DatasetRecommendationRequest,
    request: Request,
) -> DatasetRecommendationResponse:
    request_id = str(uuid.uuid4())
    logger.info("[%s] POST /api/v1/datasets/recommend (limit=%s)", request_id, body.limit)

    if body.description is None:
        raise HTTPException(
            status_code=422,
            detail=ErrorResponse(
                request_id=request_id,
                error=ErrorDetail(
                    code=ErrorCode.PROJECT_DESCRIPTION_REQUIRED,
                    message="Project description is required.",
                ),
            ).model_dump(),
        )
    if not body.description:
        raise HTTPException(
            status_code=422,
            detail=ErrorResponse(
                request_id=request_id,
                error=ErrorDetail(
                    code=ErrorCode.PROJECT_DESCRIPTION_EMPTY,
                    message="Project description must not be empty.",
                ),
            ).model_dump(),
        )
    word_count = count_words(body.description)
    if word_count > MAX_PROJECT_DESCRIPTION_WORDS:
        raise HTTPException(
            status_code=422,
            detail=ErrorResponse(
                request_id=request_id,
                error=ErrorDetail(
                    code=ErrorCode.PROJECT_DESCRIPTION_TOO_LONG,
                    message=(
                        f"Project description must be at most "
                        f"{MAX_PROJECT_DESCRIPTION_WORDS} words; got {word_count}."
                    ),
                ),
            ).model_dump(),
        )

    service = get_service()
    try:
        requirements, result = await service.recommend(
            description=body.description,
            limit=body.limit,
            request_id=request_id,
        )
    except (DatasetSourceError, LLMServiceError) as exc:
        raise _error(request_id, exc) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("[%s] Unhandled recommendation error: %s", request_id, type(exc).__name__)
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

    recommendations = [
        RankedDataset(
            rank=index,
            dataset=candidate,
            ranking_score=float(candidate.ranking_score or 0.0),
            verification_status=candidate.verification_status,
            source_type=candidate.source_type,
            reasons=candidate.ranking_reasons,
            ranking_factors=candidate.ranking_factors,
        )
        for index, candidate in enumerate(result.candidates, start=1)
    ]

    return DatasetRecommendationResponse(
        request_id=request_id,
        project_requirements=requirements,
        search_query=result.query,
        count=len(recommendations),
        recommendations=recommendations,
        sources=result.sources,
        ranking_weights=service.ranking_weights(),
        meta=_meta(result),
    )


def _meta(result: Any) -> dict[str, Any]:
    return {
        "raw_candidates": result.raw_candidate_count,
        "verified_count": result.verified_count,
        "ai_suggested_count": result.unverified_count,
        "processing_time_ms": result.duration_ms,
        "source_types": {
            "verified_external": "Real dataset published by an external registry",
            "ai_suggested": "Model suggestion, not yet confirmed externally",
        },
    }
