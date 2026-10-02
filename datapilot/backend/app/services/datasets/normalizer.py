"""Source-specific payload -> :class:`DatasetCandidate` normalization.

These are pure functions with no network access so they can be unit tested
directly against captured payloads.

Rules that must not be broken:

* Never fabricate. A field the payload does not contain stays ``None``.
* Never trust a model's self-assessment. GenAI output is always
  ``ai_suggested`` / ``unverified`` regardless of what the model claims.
* Malformed individual records are skipped, not fatal: one bad row must not
  discard an otherwise good search result.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from app.core.logging_config import get_logger
from app.schemas.dataset import (
    DatasetCandidate,
    LLMDiscoveredCandidate,
    SourceType,
    VerificationStatus,
)

logger = get_logger(__name__)

# Kaggle returns ISO-8601 UTC timestamps such as "2025-04-12T10:49:08.663Z".
_ISO_RE = re.compile(
    r"^(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2}):(\d{2})(?:\.(\d+))?"
    r"(Z|[+-]\d{2}:?\d{2})?$"
)
_DATE_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")

_SLUG_RE = re.compile(r"[^a-z0-9]+")

# Kaggle topic tags that unambiguously map onto a DataPilot dictionary domain.
# Deliberately conservative: an unmapped tag yields domain=None rather than a
# guess. Extend as new topics are observed.
KAGGLE_TAG_DOMAIN_ALIASES: dict[str, str] = {
    "education": "education",
    "school": "education",
    "university": "education",
    "student": "education",
    "students": "education",
    "academic": "education",
    "health": "healthcare",
    "healthcare": "healthcare",
    "medical": "healthcare",
    "medicine": "healthcare",
    "disease": "healthcare",
    "heart": "healthcare",
    "diabetes": "healthcare",
    "cancer": "healthcare",
    "heart disease": "healthcare",
    "mental health": "healthcare",
    "finance": "finance",
    "financial": "finance",
    "banking": "finance",
    "credit": "finance",
    "fraud": "finance",
    "insurance": "finance",
    "stock": "finance",
    "trading": "finance",
    "loan": "finance",
    "cybersecurity": "cybersecurity",
    "security": "cybersecurity",
    "network": "cybersecurity",
    "malware": "cybersecurity",
    "intrusion detection": "cybersecurity",
    "phishing": "cybersecurity",
    "energy": "energy",
    "electricity": "energy",
    "power": "energy",
    "solar": "energy",
    "wind": "energy",
    "iot": "iot",
    "internet of things": "iot",
    "agriculture": "agriculture",
    "agricultural": "agriculture",
    "farming": "agriculture",
    "environment": "environment",
    "climate": "environment",
    "weather": "environment",
    "retail": "retail",
    "ecommerce": "retail",
    "shopping": "retail",
    "sports": "sports",
    "football": "sports",
    "nba": "sports",
    "real estate": "real_estate",
    "housing": "real_estate",
    "telecom": "telecom",
    "telecommunications": "telecom",
    "manufacturing": "manufacturing",
    "transportation": "transportation",
    "vehicle": "transportation",
    "traffic": "transportation",
    "geospatial": "geospatial_analytics",
    "satellite": "geospatial_analytics",
    "robotics": "robotics",
    "human resources": "human_resources",
    "recruitment": "human_resources",
    "social media": "social_media",
    "sentiment": "social_media",
    "entertainment": "entertainment",
    "movies": "entertainment",
    "public safety": "public_safety",
    "crime": "public_safety",
}


# ---------------------------------------------------------------------------
# Coercion helpers
# ---------------------------------------------------------------------------


def normalize_text(value: Any) -> Optional[str]:
    """Return a clean non-empty string, or None for anything unusable."""
    if value is None or isinstance(value, (dict, list, bool)):
        return None
    text = str(value).strip()
    if not text:
        return None
    return " ".join(text.split())


def coerce_int(value: Any) -> Optional[int]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def coerce_float(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def coerce_ratio(value: Any) -> Optional[float]:
    """Clamp a publisher score into 0.0-1.0, rejecting out-of-range junk."""
    number = coerce_float(value)
    if number is None:
        return None
    if number < 0.0 or number > 1.0:
        return None
    return number


def parse_timestamp(value: Any) -> Optional[datetime]:
    """Parse an ISO-8601 or YYYY-MM-DD timestamp into an aware UTC datetime."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = normalize_text(value)
    if not text:
        return None

    match = _ISO_RE.match(text)
    if match:
        year, month, day, hour, minute, second, _, offset = match.groups()
        micro = 0
        if match.group(7):
            micro = int((match.group(7) + "000000")[:6])
        # The wall-clock values belong to `offset`, so label the datetime with
        # that offset before converting to UTC. Labelling it UTC first would
        # silently discard the offset.
        source_tz = timezone.utc
        if offset and offset != "Z":
            sign = 1 if offset[0] == "+" else -1
            digits = offset[1:].replace(":", "")
            delta_hours = int(digits[:2])
            delta_minutes = int(digits[2:4]) if len(digits) >= 4 else 0
            source_tz = timezone(
                sign * timedelta(hours=delta_hours, minutes=delta_minutes)
            )
        try:
            stamp = datetime(
                int(year), int(month), int(day),
                int(hour), int(minute), int(second), micro,
                tzinfo=source_tz,
            )
        except ValueError:
            return None
        return stamp.astimezone(timezone.utc)

    match = _DATE_RE.match(text)
    if match:
        year, month, day = match.groups()
        try:
            return datetime(int(year), int(month), int(day), tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


def normalize_url(value: Any) -> Optional[str]:
    """Accept only well-formed absolute http(s) URLs; reject anything else."""
    text = normalize_text(value)
    if not text:
        return None
    lowered = text.lower()
    if not (lowered.startswith("http://") or lowered.startswith("https://")):
        return None
    if " " in text:
        return None
    return text


def slugify(value: str) -> str:
    return _SLUG_RE.sub("-", value.lower()).strip("-")[:80]


def extract_tags(raw_tags: Any) -> list[str]:
    """Kaggle returns tags as [{name, ref, fullPath, description}, ...]."""
    tags: list[str] = []
    if not isinstance(raw_tags, list):
        return tags
    for entry in raw_tags[:50]:
        if isinstance(entry, str):
            label = normalize_text(entry)
        elif isinstance(entry, dict):
            label = (
                normalize_text(entry.get("name"))
                or normalize_text(entry.get("ref"))
                or normalize_text(entry.get("fullPath"))
            )
        else:
            label = None
        if label and label not in tags:
            tags.append(label)
    return tags


def infer_domain_from_tags(
    tags: list[str], known_domains: Optional[set[str]] = None
) -> Optional[str]:
    """Map external topic tags onto a DataPilot dictionary domain, or None."""
    if known_domains is None:
        known_domains = _default_domain_keys()
    if not known_domains:
        return None

    normalized_known = {_normalize_key(d) for d in known_domains}
    for tag in tags:
        key = _normalize_key(tag)
        if not key:
            continue
        if key in normalized_known:
            original = next(d for d in known_domains if _normalize_key(d) == key)
            return original
        alias = KAGGLE_TAG_DOMAIN_ALIASES.get(key)
        if alias and _normalize_key(alias) in normalized_known:
            return alias
    return None


def infer_domain_from_text(
    text: Optional[str], known_domains: Optional[set[str]] = None
) -> Optional[str]:
    """Infer a domain from candidate-published free text, or None.

    Used for AI-suggested candidates, which carry no topic tags. Only the
    candidate's own metadata is inspected (never the model's rationale or a
    provenance hint), and only via the same conservative alias table used for
    Kaggle tags. Single significant words and adjacent two-word windows are
    tested; anything unmapped yields None rather than a guess, so a candidate
    is never assigned a domain without real evidence.
    """
    normalized = _normalize_key(text or "")
    if not normalized:
        return None
    words = normalized.split()
    candidates: list[str] = list(words)
    candidates.extend(f"{words[i]} {words[i + 1]}" for i in range(len(words) - 1))
    return infer_domain_from_tags(candidates, known_domains)


def _normalize_key(value: str) -> str:
    return " ".join(value.lower().replace("_", " ").split())


_DOMAIN_KEY_CACHE: Optional[set[str]] = None


def _default_domain_keys() -> set[str]:
    global _DOMAIN_KEY_CACHE
    if _DOMAIN_KEY_CACHE is None:
        try:
            from app.services.keyword_service import KeywordService

            _DOMAIN_KEY_CACHE = set(KeywordService().get_domain_keys())
        except Exception as exc:  # pragma: no cover - dictionary is bundled
            logger.warning("Could not load dictionary domains for normalization: %s", exc)
            _DOMAIN_KEY_CACHE = set()
    return _DOMAIN_KEY_CACHE


def reset_domain_cache() -> None:
    """Test hook: clears the memoized dictionary domain keys."""
    global _DOMAIN_KEY_CACHE
    _DOMAIN_KEY_CACHE = None


# ---------------------------------------------------------------------------
# Kaggle
# ---------------------------------------------------------------------------

# Kaggle answers with a "<field>" and a "<field>Nullable" variant of every
# field. The nullable variant holds the real value; the plain one holds False.
def _kaggle_field(raw: dict, field: str) -> Any:
    value = raw.get(f"{field}Nullable")
    if value is None:
        value = raw.get(field)
    return value


def normalize_kaggle_record(
    raw: Any, known_domains: Optional[set[str]] = None
) -> Optional[DatasetCandidate]:
    """Convert one Kaggle ``datasets/list`` row into a DatasetCandidate.

    Returns None when the row lacks both a title and a ``ref``, which means it
    cannot be identified at all.
    """
    if not isinstance(raw, dict):
        logger.debug("Skipping non-dict Kaggle record: %s", type(raw).__name__)
        return None

    ref = normalize_text(_kaggle_field(raw, "ref"))
    title = normalize_text(_kaggle_field(raw, "title"))
    if not ref and not title:
        logger.debug("Skipping Kaggle record with no title and no ref")
        return None

    source_id = ref or f"{normalize_text(_kaggle_field(raw, 'ownerRef')) or 'unknown'}/{slugify(title or 'dataset')}"
    name = title or source_id.split("/")[-1]

    subtitle = normalize_text(_kaggle_field(raw, "subtitle"))
    description = normalize_text(_kaggle_field(raw, "description"))
    if not description:
        # Kaggle frequently ships an empty description but a useful subtitle.
        description = subtitle

    tags = extract_tags(_kaggle_field(raw, "tags"))
    owner = normalize_text(_kaggle_field(raw, "ownerRef")) or normalize_text(
        _kaggle_field(raw, "ownerName")
    )

    return DatasetCandidate(
        id=f"kaggle:{source_id}",
        name=name,
        source="kaggle",
        source_type=SourceType.VERIFIED_EXTERNAL,
        source_id=source_id,
        verification_status=VerificationStatus.VERIFIED,
        url=normalize_url(_kaggle_field(raw, "url")),
        description=description,
        owner=owner,
        tags=tags,
        domain=infer_domain_from_tags(tags, known_domains),
        license=normalize_text(_kaggle_field(raw, "licenseName")),
        updated_at=parse_timestamp(_kaggle_field(raw, "lastUpdated")),
        # Kaggle's list endpoint does not expose a creation timestamp, and the
        # file list is empty there, so these stay None instead of being guessed.
        created_at=None,
        size_bytes=coerce_int(_kaggle_field(raw, "totalBytes")),
        file_count=None,
        file_types=[],
        download_count=coerce_int(_kaggle_field(raw, "downloadCount")),
        vote_count=coerce_int(_kaggle_field(raw, "voteCount")),
        usability=coerce_ratio(_kaggle_field(raw, "usabilityRating")),
    )


def normalize_kaggle_records(
    raw_items: Any, known_domains: Optional[set[str]] = None
) -> list[DatasetCandidate]:
    """Normalize a Kaggle list response, skipping unusable rows."""
    if not isinstance(raw_items, list):
        logger.warning(
            "Kaggle returned %s instead of a list; returning no candidates",
            type(raw_items).__name__,
        )
        return []

    candidates: list[DatasetCandidate] = []
    for item in raw_items:
        candidate = normalize_kaggle_record(item, known_domains)
        if candidate is not None:
            candidates.append(candidate)
    logger.debug("Normalized %d/%d Kaggle records", len(candidates), len(raw_items))
    return candidates


# ---------------------------------------------------------------------------
# GenAI
# ---------------------------------------------------------------------------


def normalize_genai_candidate(
    raw: Any, index: int = 0
) -> Optional[DatasetCandidate]:
    """Convert one Gemini suggestion into an unverified DatasetCandidate.

    The model cannot verify that a dataset exists, so ``source_type`` and
    ``verification_status`` are forced here and are not read from the payload.
    """
    if isinstance(raw, LLMDiscoveredCandidate):
        item = raw
    elif isinstance(raw, dict):
        try:
            item = LLMDiscoveredCandidate(**raw)
        except Exception as exc:
            logger.warning("Discarding invalid GenAI candidate %d: %s", index, exc)
            return None
    else:
        logger.warning("Discarding GenAI candidate %d of type %s", index, type(raw).__name__)
        return None

    name = normalize_text(item.name)
    if not name:
        logger.warning("Discarding GenAI candidate %d with no name", index)
        return None

    suggested_features = [f for f in (normalize_text(s) for s in item.suggested_features) if f]

    description = normalize_text(item.description)
    target_description = normalize_text(item.target_description)
    # The domain is inferred only from text the model actually published about
    # the dataset. The model's rationale and provenance hint are excluded: they
    # describe the project, not the dataset, and must not become evidence.
    domain = infer_domain_from_text(
        " ".join(
            part
            for part in (name, description, " ".join(suggested_features), target_description)
            if part
        )
    )

    return DatasetCandidate(
        id=f"genai:{slugify(name) or 'candidate'}",
        name=name,
        source="genai",
        source_type=SourceType.AI_SUGGESTED,
        source_id=slugify(name) or f"candidate-{index}",
        # Forced: a language model cannot externally verify a dataset.
        verification_status=VerificationStatus.UNVERIFIED,
        url=normalize_url(item.url),
        description=description,
        owner=None,
        tags=[],
        domain=domain,
        task_types=[t for t in (normalize_text(x) for x in item.ml_tasks) if t],
        target_description=target_description,
        feature_description="; ".join(suggested_features) if suggested_features else None,
        rationale=normalize_text(item.why_relevant),
        license=normalize_text(item.license),
        confidence_score=coerce_ratio(item.confidence),
        # Model provenance only. Recorded so a user can tell a well-known
        # dataset from a speculative one, never to imply verification.
        possible_source=normalize_text(item.possible_source),
        is_well_known=item.is_well_known,
    )


def normalize_genai_candidates(raw_items: Any) -> list[DatasetCandidate]:
    if not isinstance(raw_items, list):
        logger.warning(
            "GenAI returned %s instead of a list; returning no candidates",
            type(raw_items).__name__,
        )
        return []

    candidates: list[DatasetCandidate] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(raw_items):
        candidate = normalize_genai_candidate(item, index)
        if candidate is None:
            continue
        if candidate.id in seen_ids:
            continue
        seen_ids.add(candidate.id)
        candidates.append(candidate)
    return candidates
