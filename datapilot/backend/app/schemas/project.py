from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Word-counting utility
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)


def count_words(text: str) -> int:
    """Deterministic whitespace/token-based word count.

    Policy: a 'word' is any contiguous sequence of Unicode word characters
    (letters, digits, underscores).  This is consistent across runs and
    locales.  Punctuation-only tokens are ignored.
    """
    return len(_WORD_RE.findall(text))


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class InfoSource(str, Enum):
    EXPLICIT = "explicit"
    INFERRED = "inferred"
    UNKNOWN = "unknown"


class MatchType(str, Enum):
    EXACT = "exact"
    SYNONYM = "synonym"
    ALIAS = "alias"
    ABBREVIATION = "abbreviation"
    PHRASE = "phrase"
    RELATED = "related"


class ErrorCode(str, Enum):
    INVALID_REQUEST = "INVALID_REQUEST"
    PROJECT_DESCRIPTION_REQUIRED = "PROJECT_DESCRIPTION_REQUIRED"
    PROJECT_DESCRIPTION_EMPTY = "PROJECT_DESCRIPTION_EMPTY"
    PROJECT_DESCRIPTION_TOO_LONG = "PROJECT_DESCRIPTION_TOO_LONG"
    LLM_AUTHENTICATION_ERROR = "LLM_AUTHENTICATION_ERROR"
    LLM_RATE_LIMITED = "LLM_RATE_LIMITED"
    LLM_TIMEOUT = "LLM_TIMEOUT"
    LLM_UNAVAILABLE = "LLM_UNAVAILABLE"
    LLM_INVALID_RESPONSE = "LLM_INVALID_RESPONSE"
    INTERNAL_ERROR = "INTERNAL_ERROR"


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------

class ProjectAnalysisRequest(BaseModel):
    description: Optional[str] = Field(
        None,
        description="Natural-language project description (max 150 words).",
        examples=[
            "I want to develop a machine learning system that detects "
            "malicious network activity in IoT devices by analyzing "
            "network traffic and identifying abnormal behavior."
        ],
    )

    @field_validator("description")
    @classmethod
    def normalize_description(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        v = v.strip()
        v = re.sub(r"\s+", " ", v)
        return v


# ---------------------------------------------------------------------------
# Sub-models used in the structured response
# ---------------------------------------------------------------------------

class ConfidenceValue(BaseModel):
    value: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(0.0, ge=0.0, le=1.0)


class DomainInfo(BaseModel):
    value: str
    confidence: float = Field(0.0, ge=0.0, le=1.0)


class SubdomainInfo(BaseModel):
    value: str
    confidence: float = Field(0.0, ge=0.0, le=1.0)


class ProblemTypeInfo(BaseModel):
    value: str
    confidence: float = Field(0.0, ge=0.0, le=1.0)


class MLTask(BaseModel):
    task: str
    confidence: float = Field(0.0, ge=0.0, le=1.0)


class TargetInfo(BaseModel):
    value: Optional[str] = None
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    status: InfoSource = InfoSource.UNKNOWN


class ProjectUnderstanding(BaseModel):
    domain: Optional[DomainInfo] = None
    subdomain: Optional[SubdomainInfo] = None
    problem_type: Optional[ProblemTypeInfo] = None
    ml_tasks: list[MLTask] = Field(default_factory=list)
    objective: Optional[str] = None
    target: Optional[TargetInfo] = None


class KeywordMatchEvidence(BaseModel):
    canonical_term: str
    matched_text: str
    match_type: MatchType
    dictionary_entry_id: str
    domain: str
    subdomain: str


class KeywordProvenance(BaseModel):
    canonical_term: str
    sources: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)


class ConceptConflict(BaseModel):
    type: Literal["concept_conflict"] = "concept_conflict"
    dictionary_interpretation: str
    llm_interpretation: str
    resolution: str


class Keywords(BaseModel):
    canonical: list[str] = Field(default_factory=list)
    dictionary_matches: list[KeywordMatchEvidence] = Field(default_factory=list)
    llm_inferred: list[str] = Field(default_factory=list)
    related_concepts: list[str] = Field(default_factory=list)
    provenance: list[KeywordProvenance] = Field(default_factory=list)
    conflicts: list[ConceptConflict] = Field(default_factory=list)


class DatasetRequirement(BaseModel):
    description: str
    importance: str = "recommended"


class FeatureRequirement(BaseModel):
    name: str
    description: str = ""
    importance: str = "recommended"


class TargetRequirement(BaseModel):
    name: str
    description: str = ""
    possible_values: list[str] = Field(default_factory=list)


class LabelRequirement(BaseModel):
    description: str
    possible_values: list[str] = Field(default_factory=list)


class DataTypeRequirement(BaseModel):
    type: str
    description: str = ""


class DatasetRequirements(BaseModel):
    required_properties: list[DatasetRequirement] = Field(default_factory=list)
    preferred_properties: list[DatasetRequirement] = Field(default_factory=list)
    feature_requirements: list[FeatureRequirement] = Field(default_factory=list)
    target_requirements: list[TargetRequirement] = Field(default_factory=list)
    data_type_requirements: list[DataTypeRequirement] = Field(default_factory=list)
    label_requirements: list[LabelRequirement] = Field(default_factory=list)


class MissingInformation(BaseModel):
    field: str
    description: str
    severity: Literal["info", "warning", "critical"] = "warning"


class Ambiguity(BaseModel):
    field: str
    description: str
    possible_interpretations: list[str] = Field(default_factory=list)


class Analysis(BaseModel):
    overall_confidence: float = Field(0.0, ge=0.0, le=1.0)
    ambiguities: list[Ambiguity] = Field(default_factory=list)
    missing_information: list[MissingInformation] = Field(default_factory=list)
    needs_clarification: bool = False


class InputInfo(BaseModel):
    word_count: int


class ProjectAnalysisData(BaseModel):
    input: InputInfo
    project_understanding: ProjectUnderstanding
    keywords: Keywords
    dataset_requirements: DatasetRequirements
    analysis: Analysis


class ResponseMeta(BaseModel):
    model: str
    dictionary_version: str
    processing_time_ms: float
    timestamp: str


class ProjectAnalysisResponse(BaseModel):
    success: bool = True
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    api_version: str = "v1"
    data: ProjectAnalysisData
    meta: ResponseMeta


# ---------------------------------------------------------------------------
# Error response
# ---------------------------------------------------------------------------

class ErrorDetail(BaseModel):
    code: ErrorCode
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    success: bool = False
    error: ErrorDetail
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


# ---------------------------------------------------------------------------
# Health / readiness
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "datapilot-backend"
    version: str = "0.1.0"


class ReadyResponse(BaseModel):
    status: str = "ok"
    gemini_configured: bool
    dictionary_loaded: bool
    model: str


# ---------------------------------------------------------------------------
# LLM structured output schema (what Gemini must return)
# ---------------------------------------------------------------------------

class LLMConfidenceField(BaseModel):
    value: Optional[str] = None
    confidence: float = 0.0
    status: str = "unknown"


class LLMMLTask(BaseModel):
    task: str
    confidence: float = 0.0


class LLMProjectUnderstanding(BaseModel):
    domain: LLMConfidenceField = Field(
        default_factory=LLMConfidenceField,
        description="The primary domain of the project.",
    )
    subdomain: LLMConfidenceField = Field(
        default_factory=LLMConfidenceField,
        description="The specific subdomain within the domain.",
    )
    problem_type: LLMConfidenceField = Field(
        default_factory=LLMConfidenceField,
        description="The type of problem: machine learning, deep learning, NLP, computer vision, etc.",
    )
    ml_tasks: list[LLMMLTask] = Field(
        default_factory=list,
        description="Potential ML tasks relevant to this project.",
    )
    objective: Optional[str] = Field(
        None,
        description="The stated or inferred objective of the project.",
    )
    target: LLMConfidenceField = Field(
        default_factory=LLMConfidenceField,
        description="The target variable or prediction goal, if identifiable.",
    )


class LLMKeywords(BaseModel):
    inferred_concepts: list[str] = Field(
        default_factory=list,
        description="Key concepts inferred from the description.",
    )
    related_terms: list[str] = Field(
        default_factory=list,
        description="Related terms that may be useful for dataset discovery.",
    )


class LLMFeatureRequirement(BaseModel):
    name: str
    description: str = ""


class LLMDatasetRequirements(BaseModel):
    required_features: list[LLMFeatureRequirement] = Field(
        default_factory=list,
        description="Features that a dataset should contain.",
    )
    target_description: Optional[str] = Field(
        None,
        description="Description of the target variable.",
    )
    label_description: Optional[str] = Field(
        None,
        description="Description of the labels or classes.",
    )
    data_type_notes: Optional[str] = Field(
        None,
        description="Notes on expected data types (e.g., tabular, time-series, text, images).",
    )


class LLMAmbiguity(BaseModel):
    field: str
    description: str
    possible_interpretations: list[str] = Field(default_factory=list)


class LLMMissingInfo(BaseModel):
    field: str
    description: str
    severity: str = "warning"


class LLMProjectAnalysis(BaseModel):
    project_understanding: LLMProjectUnderstanding
    keywords: LLMKeywords
    dataset_requirements: LLMDatasetRequirements
    ambiguities: list[LLMAmbiguity] = Field(default_factory=list)
    missing_information: list[LLMMissingInfo] = Field(default_factory=list)
    needs_clarification: bool = False
    overall_confidence: float = Field(0.0, ge=0.0, le=1.0)
