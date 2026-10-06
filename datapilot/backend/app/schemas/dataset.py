"""Source-independent dataset models for Phase 2 (dataset discovery + ranking).

Design rules enforced here:

* A ``DatasetCandidate`` is the ONLY internal representation of a dataset.
  Nothing downstream (dedup, ranking, API) may depend on a source-specific type.
* Real, externally listed datasets use ``SourceType.VERIFIED_EXTERNAL``.
  Model-generated suggestions use ``SourceType.AI_SUGGESTED`` and can never be
  promoted to ``VerificationStatus.VERIFIED`` by a normalizer.
* Metadata that a source does not provide stays ``None``. Values are never
  invented, defaulted to a plausible number, or copied from another candidate.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator

from app.schemas.project import ProjectAnalysisData

# Upper bound on a free-text search query. Kaggle's search endpoint rejects very
# long queries, and a project description is never a good search query anyway.
MAX_SEARCH_QUERY_LENGTH = 200

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SourceType(str, Enum):
    """How a candidate entered the system.

    ``VERIFIED_EXTERNAL`` - returned by a real external dataset registry.
    ``AI_SUGGESTED`` - produced by a generative model and NOT confirmed by any
    external registry. May still be correct, but carries no external evidence.
    """

    VERIFIED_EXTERNAL = "verified_external"
    AI_SUGGESTED = "ai_suggested"


class VerificationStatus(str, Enum):
    """Whether the existence of this dataset has been confirmed externally."""

    VERIFIED = "verified"
    UNVERIFIED = "unverified"


class SourceStatus(str, Enum):
    """Per-source outcome recorded in a discovery result."""

    OK = "ok"
    FAILED = "failed"
    SKIPPED = "skipped"


class RankingFactorName(str, Enum):
    DOMAIN_RELEVANCE = "domain_relevance"
    KEYWORD_OVERLAP = "keyword_overlap"
    FEATURE_MATCH = "feature_match"
    TARGET_MATCH = "target_match"
    TASK_MATCH = "task_match"
    QUALITY = "quality"
    POPULARITY = "popularity"
    RECENCY = "recency"
    VERIFICATION = "verification"


# ---------------------------------------------------------------------------
# Internal models
# ---------------------------------------------------------------------------


class RankingFactor(BaseModel):
    """One machine-readable scoring dimension for a single candidate."""

    name: RankingFactorName
    weight: float = Field(..., ge=0.0, description="Configured weight of this factor.")
    score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description=(
            "Normalized 0-1 score, or None when the source supplied no "
            "evidence for this dimension. Missing evidence is excluded from "
            "the weighted average rather than scored as zero."
        ),
    )
    contribution: float = Field(
        default=0.0, description="Points this factor added to the final score."
    )
    matched_terms: list[str] = Field(
        default_factory=list,
        description="Actual project terms found in the dataset evidence.",
    )
    detail: Optional[str] = Field(
        None, description="Short human-readable justification for the score."
    )


class DatasetCandidate(BaseModel):
    """Source-independent dataset candidate.

    Every field except ``name``, ``source`` and ``source_type`` is optional
    because no single source populates the full set.
    """

    id: str = Field(..., description="Stable internal id, '<source>:<source_id>'.")

    name: str
    source: str = Field(
        ...,
        description="Originating source name, e.g. 'kaggle' or 'genai'.",
    )
    source_type: SourceType
    source_id: str = Field(
        ...,
        description="Identifier assigned by the source, e.g. Kaggle 'owner/slug'.",
    )

    verification_status: VerificationStatus = VerificationStatus.UNVERIFIED

    url: Optional[str] = None
    description: Optional[str] = None
    owner: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    domain: Optional[str] = Field(
        None, description="Topic domain of the dataset, if the source states one."
    )
    task_types: list[str] = Field(
        default_factory=list,
        description="ML tasks the dataset is advertised as supporting.",
    )
    target_description: Optional[str] = None
    feature_description: Optional[str] = None
    rationale: Optional[str] = Field(
        None,
        description=(
            "Free-text rationale stated by the source. For AI suggestions this "
            "is the model's own reasoning and is NOT matched evidence; the "
            "ranking engine never counts it as a score contribution."
        ),
    )
    license: Optional[str] = None

    # -- AI-suggestion provenance -------------------------------------------
    # Kept so a well-known-but-unconfirmed suggestion can be told apart from a
    # speculative one. This is provenance, NOT verification: it never upgrades
    # ``verification_status``, which stays ``unverified`` for AI candidates.
    possible_source: Optional[str] = Field(
        None,
        description=(
            "Platform an AI suggestion probably lives on (e.g. 'UCI', "
            "'Kaggle'). A hint from the model, not a confirmed host."
        ),
    )
    is_well_known: Optional[bool] = Field(
        None,
        description=(
            "Whether the model believes this dataset is widely known. This is "
            "the model's own belief, so it carries no verification weight."
        ),
    )

    updated_at: Optional[datetime] = None
    created_at: Optional[datetime] = None

    size_bytes: Optional[int] = Field(
        None, description="Total dataset size in bytes, when published."
    )
    file_count: Optional[int] = None
    file_types: list[str] = Field(default_factory=list)

    download_count: Optional[int] = None
    vote_count: Optional[int] = None
    usability: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Publisher quality score, when published."
    )

    # -- ranking output -----------------------------------------------------
    relevance_score: Optional[float] = Field(None, ge=0.0, le=100.0)
    confidence_score: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Source-reported certainty."
    )
    ranking_score: Optional[float] = Field(None, ge=0.0, le=100.0)
    ranking_factors: list[RankingFactor] = Field(default_factory=list)
    ranking_reasons: list[str] = Field(default_factory=list)

    # -- deduplication ------------------------------------------------------
    duplicate_sources: list[str] = Field(
        default_factory=list,
        description="Other sources that also pointed at this same dataset.",
    )
    merged_candidate_count: int = Field(
        default=1, description="How many raw candidates collapsed into this one."
    )

    @field_validator("id")
    @classmethod
    def id_must_be_prefixed(cls, v: str) -> str:
        if ":" not in v:
            raise ValueError("DatasetCandidate.id must be '<source>:<source_id>'")
        return v

    # -- helpers used by the ranker and the API layer ------------------------

    def evidence_text(self) -> str:
        """All textual evidence available about this dataset, lowercased.

        This is the ONLY text the ranker is allowed to match project terms
        against. It contains nothing but values the source actually published.

        ``rationale`` is deliberately excluded: it is the source's opinion
        about the project, not evidence about the dataset. Including it would
        let an AI suggestion inflate its own match score.
        """
        parts: list[str] = [
            self.name,
            self.description or "",
            self.owner or "",
            self.domain or "",
            " ".join(self.tags),
            self.target_description or "",
            self.feature_description or "",
            " ".join(self.task_types),
        ]
        return " ".join(p for p in parts if p).lower()

    def merge(self, other: "DatasetCandidate") -> "DatasetCandidate":
        """Absorb another candidate describing the same dataset.

        Rules: verified external evidence always wins over AI suggestions,
        missing values are filled in, and neither side invents metadata.
        """
        if (
            other.source_type is SourceType.VERIFIED_EXTERNAL
            and self.source_type is SourceType.AI_SUGGESTED
        ):
            primary, secondary = other, self
        else:
            primary, secondary = self, other

        for field_name in (
            "url",
            "description",
            "owner",
            "domain",
            "target_description",
            "feature_description",
            "rationale",
            "license",
            "updated_at",
            "created_at",
            "size_bytes",
            "file_count",
            "download_count",
            "vote_count",
            "usability",
            "confidence_score",
            "possible_source",
            "is_well_known",
        ):
            if getattr(primary, field_name) is None:
                setattr(primary, field_name, getattr(secondary, field_name))

        primary.tags = _merge_unique(primary.tags + secondary.tags)
        primary.file_types = _merge_unique(primary.file_types + secondary.file_types)
        primary.task_types = _merge_unique(primary.task_types + secondary.task_types)
        primary.duplicate_sources = _merge_unique(
            [
                s
                for s in (
                    primary.duplicate_sources
                    + [secondary.source]
                    + secondary.duplicate_sources
                )
                if s != primary.source
            ]
        )
        primary.merged_candidate_count += secondary.merged_candidate_count
        return primary


def _merge_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for v in values:
        key = v.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(v.strip())
    return out


# ---------------------------------------------------------------------------
# Source-facing request object (identical for every DatasetSource)
# ---------------------------------------------------------------------------


class SourceSearchRequest(BaseModel):
    """What a source needs in order to discover datasets.

    Both a plain text query and full project requirements travel together so
    that every source sees one uniform input, and each source decides what it
    can use. Sources that cannot work without requirements report ``SKIPPED``.
    """

    query: str
    limit: int
    candidate_pool_limit: Optional[int] = Field(
        None,
        ge=1,
        le=50,
        description=(
            "Optional larger pool for verified sources when candidates will be "
            "ranked and trimmed by the discovery service."
        ),
    )
    query_variants: list[str] = Field(
        default_factory=list,
        description=(
            "Bounded alternate registry queries for improving recall. "
            "Suggestion sources continue to receive the primary query only."
        ),
    )
    project_requirements: Optional[ProjectAnalysisData] = None
    known_dataset_names: list[str] = Field(
        default_factory=list,
        description=(
            "Names already discovered by real sources, passed to suggestion "
            "based sources so they can propose complementary candidates."
        ),
    )


class SourceOutcome(BaseModel):
    """Per-source status, safe to return from the API (no secrets, no tracebacks)."""

    source: str
    source_type: SourceType
    status: SourceStatus
    candidate_count: int = 0
    duration_ms: float = 0.0
    error_code: Optional[str] = None
    message: Optional[str] = None


# ---------------------------------------------------------------------------
# Gemini structured-output schema for dataset candidate discovery
# ---------------------------------------------------------------------------


class LLMDiscoveredCandidate(BaseModel):
    """One dataset suggestion produced by Gemini.

    NOTE: ``is_well_known`` and ``confidence`` are advisory only. The GenAI
    normalizer always emits ``SourceType.AI_SUGGESTED`` and
    ``VerificationStatus.UNVERIFIED`` regardless of what the model claims,
    because a language model cannot verify external dataset existence.
    """

    name: str = Field(..., description="Dataset name as commonly known.")
    possible_source: Optional[str] = Field(
        None,
        description="Platform most likely hosting it, e.g. 'UCI', 'Kaggle'.",
    )
    url: Optional[str] = Field(
        None, description="Canonical URL ONLY if genuinely known, else null."
    )
    description: Optional[str] = Field(None, description="What the dataset contains.")
    why_relevant: Optional[str] = Field(
        None, description="Why this dataset may fit the described project."
    )
    confidence: float = Field(
        0.0, ge=0.0, le=1.0, description="Model's own confidence in this suggestion."
    )
    is_well_known: bool = Field(
        False,
        description=(
            "True only for widely documented public datasets the model is "
            "confident exist. False means speculative."
        ),
    )
    suggested_features: list[str] = Field(
        default_factory=list, description="Features the dataset is expected to contain."
    )
    target_description: Optional[str] = None
    ml_tasks: list[str] = Field(default_factory=list)
    license: Optional[str] = None


class LLMDatasetDiscovery(BaseModel):
    """Top-level Gemini response for dataset candidate discovery."""

    candidates: list[LLMDiscoveredCandidate] = Field(default_factory=list)
    notes: Optional[str] = Field(
        None, description="Caveats about the quality of these suggestions."
    )


# ---------------------------------------------------------------------------
# Public API schemas
# ---------------------------------------------------------------------------


class DatasetSearchRequest(BaseModel):
    """Body of ``POST /api/v1/datasets/search``."""

    query: Optional[str] = Field(
        None,
        max_length=MAX_SEARCH_QUERY_LENGTH,
        description=(
            "Free-text dataset search query. Maximum "
            f"{MAX_SEARCH_QUERY_LENGTH} characters."
        ),
        examples=["student academic performance"],
    )
    limit: int = Field(
        10,
        ge=1,
        le=50,
        description="Maximum number of candidates to return (1-50).",
    )

    @field_validator("query")
    @classmethod
    def normalize_query(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        return " ".join(v.split()).strip()


class DatasetSearchResponse(BaseModel):
    """Body of ``POST /api/v1/datasets/search``."""

    success: bool = True
    request_id: str
    api_version: str = "v1"
    query: str
    count: int
    datasets: list[DatasetCandidate]
    sources: list[SourceOutcome] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)


class DatasetRecommendationRequest(BaseModel):
    """Body of ``POST /api/v1/datasets/recommend``."""

    description: Optional[str] = Field(
        None,
        description=(
            "Natural-language project description (max 150 words). The existing "
            "Project Requirement Analyzer converts this into structured "
            "requirements before dataset discovery runs."
        ),
        examples=[
            "I want to build a machine learning system that predicts student "
            "academic performance using attendance, previous examination marks, "
            "assignment scores, study hours, and participation in class."
        ],
    )
    limit: int = Field(
        10, ge=1, le=50, description="Maximum number of recommendations (1-50)."
    )

    @field_validator("description")
    @classmethod
    def normalize_description(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        return " ".join(v.split()).strip()


class RankedDataset(BaseModel):
    """One entry of the ranked recommendation list."""

    rank: int = Field(..., ge=1)
    dataset: DatasetCandidate
    ranking_score: float = Field(..., ge=0.0, le=100.0)
    verification_status: VerificationStatus
    source_type: SourceType
    reasons: list[str] = Field(default_factory=list)
    ranking_factors: list[RankingFactor] = Field(default_factory=list)


class DatasetRecommendationResponse(BaseModel):
    """Body of ``POST /api/v1/datasets/recommend``."""

    success: bool = True
    request_id: str
    api_version: str = "v1"
    project_requirements: ProjectAnalysisData
    search_query: str
    count: int
    recommendations: list[RankedDataset]
    sources: list[SourceOutcome] = Field(default_factory=list)
    ranking_weights: dict[str, float] = Field(default_factory=dict)
    meta: dict[str, Any] = Field(default_factory=dict)
