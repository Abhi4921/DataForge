from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.project_analysis import router as project_analysis_router
from app.core.config import get_settings
from app.core.logging_config import get_logger, setup_logging
from app.schemas.project import ErrorCode, ErrorDetail, ErrorResponse, HealthResponse, ReadyResponse
from app.services.keyword_service import KeywordService

logger = get_logger(__name__)

_keyword_service: KeywordService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _keyword_service
    setup_logging()
    settings = get_settings()
    logger.info("Starting DataPilot backend v%s", settings.app_version)
    logger.info("Environment: %s", settings.app_env)

    try:
        _keyword_service = KeywordService()
        logger.info(
            "Keyword dictionary loaded (version=%s)", _keyword_service.version
        )
    except Exception as exc:
        logger.error("Failed to load keyword dictionary: %s", exc)
        _keyword_service = None

    yield

    logger.info("Shutting down DataPilot backend")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_title,
        version=settings.app_version,
        description=(
            "DataPilot Backend - Project Requirement Analyzer. "
            "Analyzes natural-language ML/AI project descriptions and "
            "extracts structured requirements for dataset discovery."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(project_analysis_router, prefix=settings.api_v1_prefix)

    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    async def health_check() -> HealthResponse:
        return HealthResponse()

    @app.get("/ready", response_model=ReadyResponse, tags=["Health"])
    async def readiness_check() -> ReadyResponse:
        settings = get_settings()
        return ReadyResponse(
            gemini_configured=bool(settings.gemini_api_key),
            dictionary_loaded=_keyword_service is not None,
            model=settings.gemini_model,
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error("Unhandled exception: %s", exc)
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error=ErrorDetail(
                    code=ErrorCode.INTERNAL_ERROR,
                    message="An internal error occurred.",
                ),
            ).model_dump(),
        )

    return app


app = create_app()
