"""Deterministic dataset ranking engine.

No LLM participates in the numerical ranking. Every score comes from
structured, inspectable evidence: project/dataset term overlap, declared
metadata, and the candidate's verification status. The same inputs always
produce the same ranking, which is what makes the result testable and
reproducible.

Scoring model
-------------
Eight *content* factors measure how well a dataset fits the project, and one
*trust* factor measures whether the dataset is externally confirmed::

    relevance_score = weighted mean of content factors          (0-100)
    ranking_score   = weighted mean of all factors             (0-100)

Default weights (all configurable in ``Settings``)::

    domain_relevance  20    target_match     11
    keyword_overlap   11    task_match       11
    feature_match     17    quality           8
                         popularity          5
                         recency             2
                         verification       15   <- trust
                          -------------------
                          content total      85
                          total             100

Two consequences are intentional:

* **Missing evidence is excluded, not zeroed.** A candidate with no published
  usability score is not penalised for it; the factor is dropped from the
  weighted mean and reported as ``score=None``. Only ``verification`` is
  always available.
* **AI suggestions can never equal verified datasets.** The trust factor
  returns ``ranking_trust_verified`` (1.0) for externally listed datasets and
  ``ranking_trust_unverified`` (0.25) for AI suggestions, capping an otherwise
  perfect AI suggestion at ~86.8/100 while a verified one reaches 100.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Sequence

from app.core.config import Settings, get_settings
from app.core.logging_config import get_logger
from app.schemas.dataset import (
    DatasetCandidate,
    RankingFactor,
    RankingFactorName,
    SourceType,
    VerificationStatus,
)
from app.schemas.project import MatchType, ProjectAnalysisData
from app.services.datasets.normalizer import infer_domain_from_tags
from app.services.keyword_service import KeywordService

logger = get_logger(__name__)

_PUNCT = re.compile(r"[^a-z0-9]+")

# Generic English and generic-ML words. A project term made only of these
# carries no retrieval signal. Note this list is deliberately different from the
# query builder's stop list: "student" is noise when forming a registry query
# but is strong ranking evidence when it appears in a dataset description.
_TERM_STOPWORDS = frozenset(
    {
        "a", "an", "and", "any", "are", "as", "at", "based", "be", "been",
        "build", "building", "but", "by", "can", "could", "create", "data",
        "dataset", "datasets", "deep", "detect", "detection", "develop",
        "each", "for", "from", "give", "has", "have", "how", "identify", "in",
        "into", "is", "it", "its", "learning", "like", "machine", "make",
        "method", "model", "need", "network", "of", "on", "or", "other",
        "our", "output", "predict", "prediction", "project", "should", "such",
        "supervised", "system", "than", "that", "the", "their", "then",
        "there", "these", "they", "this", "to", "unsupervised", "use", "used",
        "using", "via", "want", "was", "were", "what", "when", "which", "who",
        "will", "with", "would", "you",
    }
)

MAX_REASONS = 6


# ---------------------------------------------------------------------------
# Text matching
# ---------------------------------------------------------------------------


def _normalize_phrase(text: Optional[str]) -> str:
    if not text:
        return ""
    return _PUNCT.sub(" ", text.lower().replace("_", " ")).strip()


def _singularize(word: str) -> str:
    """Collapse a simple English plural to its singular form.

    Matching should not fail just because a dataset says "scores" while the
    requirement says "score". Only regular plurals are folded; irregular words
    are returned unchanged so nothing is invented.
    """
    if len(word) > 3 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _term_tokens(term: str) -> list[str]:
    return [t for t in _normalize_phrase(term).split() if t]


def _match_tokens(text: str) -> list[str]:
    return [_singularize(t) for t in _normalize_phrase(text).split() if t]


def _is_meaningful(term: str) -> bool:
    tokens = _term_tokens(term)
    if not tokens:
        return False
    return any(len(t) > 2 and t not in _TERM_STOPWORDS for t in tokens)


class _Evidence:
    """Normalized searchable text belonging to one candidate.

    Term matching is deliberately tolerant but not loose:

    * the exact phrase (after normalization and plural folding) is found, or
    * every significant token of the term appears somewhere in the evidence
      (so "class participation" also matches "participation in class"), or
    * a single token appears as a whole word.

    Requiring *all* tokens of a multi-word term keeps the relaxation from
    producing false positives; synonym gaps are covered separately by the
    dictionary-backed alias expansion.
    """

    __slots__ = ("phrase", "tokens")

    def __init__(self, text: str) -> None:
        words = _match_tokens(text)
        self.phrase = " ".join(words)
        self.tokens = set(words)

    def contains(self, term: str) -> bool:
        tokens = [_singularize(t) for t in _term_tokens(term)]
        if not tokens:
            return False
        if len(tokens) == 1:
            return tokens[0] in self.tokens
        if " ".join(tokens) in self.phrase:
            return True
        return all(token in self.tokens for token in tokens)


def _dedupe_terms(terms: Sequence[str], limit: int) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for term in terms:
        if not term:
            continue
        key = _normalize_phrase(term)
        if not key or key in seen or not _is_meaningful(term):
            continue
        seen.add(key)
        out.append(term.strip())
        if len(out) >= limit:
            break
    return out


# The dictionary is loaded once per process and reused across ranking runs.
_KEYWORD_SERVICE: Optional[KeywordService] = None


def _keyword_service() -> KeywordService:
    global _KEYWORD_SERVICE
    if _KEYWORD_SERVICE is None:
        _KEYWORD_SERVICE = KeywordService()
    return _KEYWORD_SERVICE


def _expand_aliases(terms: Sequence[str]) -> dict[str, list[str]]:
    """Map each base term to dictionary-backed equivalent phrases.

    A synonym only matches when the *project knowledge dictionary* already
    treats it as the same concept (canonical term, synonym, alias, or declared
    target requirement). This keeps the relaxation evidence-based and
    generalises across domains instead of hardcoding per-project synonyms.
    """
    aliases: dict[str, list[str]] = {}
    service = _keyword_service()
    for term in terms:
        key = _normalize_phrase(term)
        if not key:
            continue
        matches = service.match(term)
        if not matches:
            continue
        # A RELATED hit means the dictionary lists *this term* as a related
        # concept of another concept; it is the reverse direction, not an
        # equivalent, so it must not contribute aliases (otherwise "student
        # performance" would inherit "attendance", "examination", ...).
        matches = [m for m in matches if m.match_type is not MatchType.RELATED]
        if not matches:
            continue
        # Use only the most specific dictionary phrase found for this term. A
        # short partial hit ("academic" -> the broad "education" concept) must
        # not drag in that concept's unrelated vocabulary; "academic
        # performance" should resolve to the "student performance" concept.
        best_len = max(len(_normalize_phrase(m.matched_text).split()) for m in matches)
        extra: list[str] = []
        seen: set[str] = set()
        for match in matches:
            if len(_normalize_phrase(match.matched_text).split()) != best_len:
                continue
            concept = service.get_concept(match.dictionary_entry_id)
            if not concept:
                continue
            candidates = (
                [concept.get("canonical_term", "")]
                + list(concept.get("synonyms", []))
                + list(concept.get("aliases", []))
                + list(concept.get("target_requirements", []))
            )
            for candidate in candidates:
                normalized = _normalize_phrase(candidate)
                if normalized and normalized != key and normalized not in seen:
                    seen.add(normalized)
                    extra.append(str(candidate).strip())
        if extra:
            aliases[key] = extra
    return aliases


def _concise_target(value: Optional[str]) -> str:
    """Reduce a model-written target to a short, matchable concept.

    A concise value is kept. A verbose, sentence-like value is resolved through
    the knowledge dictionary to its canonical concept ("Student performance
    metrics and support requirement status" -> "student performance"); when no
    concept matches, the value is dropped. Matching a whole sentence instead
    would make the target factor fire on almost any education dataset (every
    significant token appears somewhere), so the long form is never used.
    """
    if not value:
        return ""
    if len(_normalize_phrase(value).split()) <= 4:
        return value.strip()
    return _keyword_service().resolve_canonical_phrase(value) or ""


# ---------------------------------------------------------------------------
# Project profile
# ---------------------------------------------------------------------------


@dataclass
class ProjectProfile:
    """Structured project terms used for matching. Built once per ranking run."""

    domain_label: Optional[str] = None
    domain_terms: list[str] = field(default_factory=list)
    concept_terms: list[str] = field(default_factory=list)
    feature_terms: list[str] = field(default_factory=list)
    target_terms: list[str] = field(default_factory=list)
    task_terms: list[str] = field(default_factory=list)
    project_domain_key: Optional[str] = None
    # base term (normalized) -> dictionary-backed equivalent phrases
    feature_aliases: dict[str, list[str]] = field(default_factory=dict)
    target_aliases: dict[str, list[str]] = field(default_factory=dict)


def build_project_profile(requirements: ProjectAnalysisData) -> ProjectProfile:
    understanding = requirements.project_understanding
    keywords = requirements.keywords
    dataset_requirements = requirements.dataset_requirements

    domain_label = understanding.domain.value if understanding.domain else None
    domain_terms = _dedupe_terms(
        [
            understanding.domain.value if understanding.domain else "",
            understanding.subdomain.value if understanding.subdomain else "",
            understanding.problem_type.value if understanding.problem_type else "",
        ],
        limit=10,
    )

    concept_terms = _dedupe_terms(
        list(keywords.canonical)
        + list(keywords.related_concepts)
        + [m.matched_text for m in keywords.dictionary_matches],
        limit=25,
    )

    # Match against the feature *name* (a short, stable label) rather than
    # "name description". Concatenating a model-written description turns the
    # term into a long phrase that no real dataset description ever contains,
    # which silently zeroes the feature factor.
    feature_terms = _dedupe_terms(
        [f.name for f in dataset_requirements.feature_requirements],
        limit=20,
    )

    target_terms = _dedupe_terms(
        [_concise_target(t.name) for t in dataset_requirements.target_requirements]
        + ([_concise_target(understanding.target.value)] if understanding.target else []),
        limit=10,
    )

    task_terms = _dedupe_terms([t.task for t in understanding.ml_tasks], limit=8)

    return ProjectProfile(
        domain_label=domain_label,
        domain_terms=domain_terms,
        concept_terms=concept_terms,
        feature_terms=feature_terms,
        target_terms=target_terms,
        task_terms=task_terms,
        # Reuses the normalizer's tag->domain mapping so the candidate's topic
        # domain and the project's domain are compared in one taxonomy.
        project_domain_key=infer_domain_from_tags([domain_label]) if domain_label else None,
        feature_aliases=_expand_aliases(feature_terms),
        target_aliases=_expand_aliases(target_terms),
    )


# ---------------------------------------------------------------------------
# Ranking engine
# ---------------------------------------------------------------------------


class DatasetRankingEngine:
    """Scores and orders candidates deterministically."""

    def __init__(self, settings: Optional[Settings] = None, now: Optional[datetime] = None) -> None:
        self._settings = settings or get_settings()
        # Injectable clock keeps ranking reproducible in tests.
        self._now = now

    # -- weights -----------------------------------------------------------

    def _weight(self, name: RankingFactorName) -> float:
        return {
            RankingFactorName.DOMAIN_RELEVANCE: self._settings.ranking_weight_domain_relevance,
            RankingFactorName.KEYWORD_OVERLAP: self._settings.ranking_weight_keyword_overlap,
            RankingFactorName.FEATURE_MATCH: self._settings.ranking_weight_feature_match,
            RankingFactorName.TARGET_MATCH: self._settings.ranking_weight_target_match,
            RankingFactorName.TASK_MATCH: self._settings.ranking_weight_task_match,
            RankingFactorName.QUALITY: self._settings.ranking_weight_quality,
            RankingFactorName.POPULARITY: self._settings.ranking_weight_popularity,
            RankingFactorName.RECENCY: self._settings.ranking_weight_recency,
            RankingFactorName.VERIFICATION: self._settings.ranking_weight_verification,
        }[name]

    def weights(self) -> dict[str, float]:
        return {name.value: self._weight(name) for name in RankingFactorName}

    # -- entry point -------------------------------------------------------

    def rank(
        self,
        candidates: Sequence[DatasetCandidate],
        requirements: ProjectAnalysisData,
    ) -> list[DatasetCandidate]:
        """Score every candidate in place and return them best-first."""
        if not candidates:
            return []

        profile = build_project_profile(requirements)
        scored = [self._score(candidate, profile) for candidate in candidates]

        scored.sort(
            key=lambda c: (
                -float(c.ranking_score or 0.0),
                0 if c.verification_status is VerificationStatus.VERIFIED else 1,
                c.name.lower(),
                c.id,
            )
        )
        logger.debug("Ranked %d candidates", len(scored))
        return scored

    def _score(
        self, candidate: DatasetCandidate, profile: ProjectProfile
    ) -> DatasetCandidate:
        evidence = _Evidence(candidate.evidence_text())
        task_evidence = _Evidence(" ".join(candidate.task_types)) if candidate.task_types else None

        factors: list[RankingFactor] = [
            self._domain_relevance(candidate, profile, evidence),
            self._keyword_overlap(profile, evidence),
            self._feature_match(profile, evidence),
            self._target_match(profile, evidence),
            self._task_match(profile, evidence, task_evidence),
            self._quality(candidate),
            self._popularity(candidate),
            self._recency(candidate),
            self._verification(candidate),
        ]

        candidate.ranking_factors = factors
        candidate.ranking_score = self._weighted_score(factors)
        candidate.relevance_score = self._content_score(factors)
        candidate.ranking_reasons = self._build_reasons(factors, candidate)
        return candidate

    # -- aggregation -------------------------------------------------------

    @staticmethod
    def _contributions(factors: list[RankingFactor]) -> tuple[float, float]:
        """Return (available_weight, weighted_sum) over scorable factors."""
        available = [f for f in factors if f.score is not None]
        total_weight = sum(f.weight for f in available)
        weighted = sum(f.weight * float(f.score) for f in available)  # type: ignore[arg-type]
        return total_weight, weighted

    def _weighted_score(self, factors: list[RankingFactor]) -> float:
        total_weight, weighted = self._contributions(factors)
        if total_weight <= 0:
            return 0.0
        score = weighted / total_weight * 100.0

        # Report each factor's actual point contribution so contributions sum
        # to the final score.
        for factor in factors:
            if factor.score is None or total_weight <= 0:
                factor.contribution = 0.0
            else:
                factor.contribution = round(factor.weight * factor.score / total_weight * 100.0, 2)
        return round(score, 2)

    def _content_score(self, factors: list[RankingFactor]) -> float:
        content = [f for f in factors if f.name is not RankingFactorName.VERIFICATION]
        total_weight, weighted = self._contributions(content)
        if total_weight <= 0:
            return 0.0
        return round(weighted / total_weight * 100.0, 2)

    # -- individual factors ------------------------------------------------

    @staticmethod
    def _coverage(
        terms: list[str],
        evidence: _Evidence,
        aliases: Optional[dict[str, list[str]]] = None,
    ) -> tuple[Optional[float], list[str]]:
        """Fraction of base terms found in the evidence.

        A base term counts as matched when it appears directly *or* when one of
        its dictionary-backed equivalent phrases appears. The denominator stays
        the base term count, so alias matching rewards recognised synonyms
        without diluting the score.
        """
        if not terms:
            return None, []
        matched: list[str] = []
        for term in terms:
            if evidence.contains(term):
                matched.append(term)
                continue
            if aliases:
                extra = aliases.get(_normalize_phrase(term))
                if extra and any(evidence.contains(alias) for alias in extra):
                    matched.append(term)
        return len(matched) / len(terms), matched

    def _domain_relevance(
        self, candidate: DatasetCandidate, profile: ProjectProfile, evidence: _Evidence
    ) -> RankingFactor:
        factor = RankingFactor(
            name=RankingFactorName.DOMAIN_RELEVANCE,
            weight=self._weight(RankingFactorName.DOMAIN_RELEVANCE),
        )
        if not profile.domain_terms and not profile.project_domain_key:
            factor.detail = "Project domain unknown"
            return factor

        coverage, matched = self._coverage(profile.domain_terms, evidence)

        domain_match = False
        if (
            profile.project_domain_key
            and candidate.domain
            and _normalize_phrase(candidate.domain)
            == _normalize_phrase(profile.project_domain_key)
        ):
            domain_match = True

        score = coverage if coverage is not None else 0.0
        if domain_match:
            score = max(score, 0.85)
            matched = matched + [f"domain:{profile.project_domain_key}"]
        if not profile.domain_terms and domain_match:
            score = 0.85

        factor.score = round(min(score, 1.0), 4)
        factor.matched_terms = matched
        factor.detail = (
            f"topic domain '{candidate.domain}' matches project domain "
            f"'{profile.project_domain_key}'"
            if domain_match
            else f"matched {len(matched)}/{len(profile.domain_terms)} domain terms"
        )
        return factor

    def _keyword_overlap(
        self, profile: ProjectProfile, evidence: _Evidence
    ) -> RankingFactor:
        factor = RankingFactor(
            name=RankingFactorName.KEYWORD_OVERLAP,
            weight=self._weight(RankingFactorName.KEYWORD_OVERLAP),
        )
        if not profile.concept_terms:
            factor.detail = "Project concepts unknown"
            return factor
        score, matched = self._coverage(profile.concept_terms, evidence)
        factor.score = round(score, 4) if score is not None else None
        factor.matched_terms = matched
        factor.detail = f"matched {len(matched)}/{len(profile.concept_terms)} project concepts"
        return factor

    def _feature_match(
        self, profile: ProjectProfile, evidence: _Evidence
    ) -> RankingFactor:
        factor = RankingFactor(
            name=RankingFactorName.FEATURE_MATCH,
            weight=self._weight(RankingFactorName.FEATURE_MATCH),
        )
        if not profile.feature_terms:
            factor.detail = "No required features identified"
            return factor
        score, matched = self._coverage(profile.feature_terms, evidence, profile.feature_aliases)
        factor.score = round(score, 4) if score is not None else None
        factor.matched_terms = matched
        factor.detail = (
            f"dataset metadata references {len(matched)}/{len(profile.feature_terms)} "
            "required feature terms"
        )
        return factor

    def _target_match(
        self, profile: ProjectProfile, evidence: _Evidence
    ) -> RankingFactor:
        factor = RankingFactor(
            name=RankingFactorName.TARGET_MATCH,
            weight=self._weight(RankingFactorName.TARGET_MATCH),
        )
        if not profile.target_terms:
            factor.detail = "No target variable identified"
            return factor
        score, matched = self._coverage(profile.target_terms, evidence, profile.target_aliases)
        factor.score = round(score, 4) if score is not None else None
        factor.matched_terms = matched
        factor.detail = (
            f"dataset metadata references {len(matched)}/{len(profile.target_terms)} target terms"
        )
        return factor

    def _task_match(
        self,
        profile: ProjectProfile,
        evidence: _Evidence,
        task_evidence: Optional[_Evidence],
    ) -> RankingFactor:
        factor = RankingFactor(
            name=RankingFactorName.TASK_MATCH,
            weight=self._weight(RankingFactorName.TASK_MATCH),
        )
        if not profile.task_terms:
            factor.detail = "No ML tasks identified"
            return factor

        # A candidate that declares its supported tasks is matched against that
        # declaration; otherwise fall back to its descriptive text.
        target_evidence = task_evidence or evidence
        score, matched = self._coverage(profile.task_terms, target_evidence)
        factor.score = round(score, 4) if score is not None else None
        factor.matched_terms = matched
        basis = "declared tasks" if task_evidence is not None else "dataset text"
        factor.detail = f"matched {len(matched)}/{len(profile.task_terms)} ML tasks in {basis}"
        return factor

    def _quality(self, candidate: DatasetCandidate) -> RankingFactor:
        factor = RankingFactor(
            name=RankingFactorName.QUALITY,
            weight=self._weight(RankingFactorName.QUALITY),
        )
        if candidate.usability is None:
            factor.detail = "No published quality score"
            return factor
        factor.score = candidate.usability
        factor.detail = f"publisher quality score {candidate.usability:.2f}"
        return factor

    def _popularity(self, candidate: DatasetCandidate) -> RankingFactor:
        factor = RankingFactor(
            name=RankingFactorName.POPULARITY,
            weight=self._weight(RankingFactorName.POPULARITY),
        )
        parts: list[tuple[float, float]] = []
        if candidate.download_count is not None and candidate.download_count >= 0:
            parts.append(
                (
                    0.7,
                    self._log_scale(
                        candidate.download_count, self._settings.ranking_download_reference
                    ),
                )
            )
        if candidate.vote_count is not None and candidate.vote_count >= 0:
            parts.append(
                (
                    0.3,
                    self._log_scale(
                        candidate.vote_count, self._settings.ranking_vote_reference
                    ),
                )
            )
        if not parts:
            factor.detail = "No download or vote counts published"
            return factor

        total_weight = sum(w for w, _ in parts)
        score = sum(w * v for w, v in parts) / total_weight
        factor.score = round(min(score, 1.0), 4)
        factor.detail = f"{candidate.download_count} downloads, {candidate.vote_count} votes"
        return factor

    @staticmethod
    def _log_scale(value: int, reference: int) -> float:
        if reference <= 1:
            return 1.0 if value > 0 else 0.0
        return min(1.0, math.log10(1 + max(0, value)) / math.log10(1 + reference))

    def _recency(self, candidate: DatasetCandidate) -> RankingFactor:
        factor = RankingFactor(
            name=RankingFactorName.RECENCY,
            weight=self._weight(RankingFactorName.RECENCY),
        )
        if candidate.updated_at is None:
            factor.detail = "No last-updated date published"
            return factor

        now = self._now or datetime.now(timezone.utc)
        updated = candidate.updated_at
        if updated.tzinfo is None:
            updated = updated.replace(tzinfo=timezone.utc)

        age_days = (now - updated).total_seconds() / 86400.0
        max_age = max(1, self._settings.ranking_recency_max_age_days)
        if age_days <= 0:
            score = 1.0
        else:
            score = max(0.0, 1.0 - age_days / max_age)
        factor.score = round(min(score, 1.0), 4)
        factor.detail = f"last updated {updated.date().isoformat()}"
        return factor

    def _verification(self, candidate: DatasetCandidate) -> RankingFactor:
        factor = RankingFactor(
            name=RankingFactorName.VERIFICATION,
            weight=self._weight(RankingFactorName.VERIFICATION),
        )
        trust = (
            self._settings.ranking_trust_verified
            if candidate.source_type is SourceType.VERIFIED_EXTERNAL
            else self._settings.ranking_trust_unverified
        )
        # Defence in depth: an unverified record can never earn full trust even
        # if it claims an external source.
        if candidate.verification_status is VerificationStatus.UNVERIFIED:
            trust = min(trust, self._settings.ranking_trust_unverified)

        factor.score = max(0.0, min(1.0, trust))
        if candidate.verification_status is VerificationStatus.VERIFIED:
            factor.matched_terms = [candidate.source]
            factor.detail = f"listed by external source '{candidate.source}'"
        else:
            factor.matched_terms = [candidate.source]
            factor.detail = (
                f"suggested by '{candidate.source}' and not confirmed by an external registry"
            )
        return factor

    # -- explanations ------------------------------------------------------

    def _build_reasons(
        self, factors: list[RankingFactor], candidate: DatasetCandidate
    ) -> list[str]:
        """Build reasons strictly from factors with real matched evidence.

        A factor only produces a reason when its score is above zero, so no
        reason can be emitted for evidence that was not found. The verification
        reason is always appended last and is never dropped by the cap: telling
        the user that a result is an unconfirmed AI suggestion matters more than
        an extra match detail.
        """
        reasons: list[str] = []
        for factor in factors:
            if factor.name is RankingFactorName.VERIFICATION:
                continue
            if factor.score is None or factor.score <= 0:
                continue
            reason = self._reason_for(factor, candidate)
            if reason:
                reasons.append(reason)
        reasons = reasons[:MAX_REASONS]

        verification = next(
            (f for f in factors if f.name is RankingFactorName.VERIFICATION), None
        )
        if verification is not None and verification.score is not None:
            trust_reason = self._reason_for(verification, candidate)
            if trust_reason:
                reasons.append(trust_reason)
        return reasons

    @staticmethod
    def _join(terms: list[str], limit: int = 5) -> str:
        shown = terms[:limit]
        suffix = ", ..." if len(terms) > limit else ""
        return ", ".join(shown) + suffix

    def _reason_for(
        self, factor: RankingFactor, candidate: DatasetCandidate
    ) -> Optional[str]:
        name = factor.name
        terms = factor.matched_terms

        if name is RankingFactorName.DOMAIN_RELEVANCE:
            domain_terms = [t for t in terms if not t.startswith("domain:")]
            if any(t.startswith("domain:") for t in terms):
                return f"Published topic domain matches the project domain: {candidate.domain}"
            if domain_terms:
                return f"Dataset metadata matches project domain terms: {self._join(domain_terms)}"
            return None

        if name is RankingFactorName.KEYWORD_OVERLAP:
            if terms:
                return f"Dataset metadata matches project concepts: {self._join(terms)}"
            return None

        if name is RankingFactorName.FEATURE_MATCH:
            if terms:
                return f"Dataset metadata references required features: {self._join(terms)}"
            return None

        if name is RankingFactorName.TARGET_MATCH:
            if terms:
                return f"Dataset metadata references the prediction target: {self._join(terms)}"
            return None

        if name is RankingFactorName.TASK_MATCH:
            if terms:
                return f"Compatible with the required ML task(s): {self._join(terms)}"
            return None

        if name is RankingFactorName.QUALITY:
            if candidate.usability is not None:
                return f"Publisher quality score {candidate.usability:.2f} / 1.00"
            return None

        if name is RankingFactorName.POPULARITY:
            bits = []
            if candidate.download_count:
                bits.append(f"{candidate.download_count:,} downloads")
            if candidate.vote_count:
                bits.append(f"{candidate.vote_count:,} votes")
            if not bits:
                bits.append("Published download and vote counts")
            return f"Adoption signal: {', '.join(bits)}"

        if name is RankingFactorName.RECENCY:
            if candidate.updated_at is None:
                return None
            return f"Last updated {candidate.updated_at.date().isoformat()}"

        if name is RankingFactorName.VERIFICATION:
            if candidate.verification_status is VerificationStatus.VERIFIED:
                return f"Verified listing from an external dataset registry ({candidate.source})"
            return (
                f"AI-suggested candidate from {candidate.source}; not yet confirmed "
                "by an external dataset registry"
            )

        return None
