from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from app.core.logging_config import get_logger
from app.schemas.project import (
    KeywordMatchEvidence,
    KeywordProvenance,
    MatchType,
)

logger = get_logger(__name__)

DictionaryPath = Path(__file__).resolve().parent.parent.parent / "data" / "keyword_dictionary.json"


class _NormalizedPhrase:
    __slots__ = ("original", "normalized")

    def __init__(self, original: str, normalized: str) -> None:
        self.original = original
        self.normalized = normalized


def _normalize(text: str) -> str:
    """Normalize text for matching.

    Policy:
    - Lowercase
    - Strip whitespace
    - Convert underscores to spaces (so 'iot_sensors' matches 'iot sensors')
    - Remove punctuation except word chars and spaces
    - Collapse multiple spaces
    """
    text = text.lower().strip()
    text = text.replace("_", " ")
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class KeywordService:
    """Deterministic keyword/concept matching engine.

    Loads the dictionary once and builds indexes for:
      - canonical term lookup (exact phrase)
      - synonym lookup
      - alias lookup
      - abbreviation lookup
      - related concept lookup
    """

    def __init__(self, dictionary_path: Optional[Path] = None) -> None:
        self._path = dictionary_path or DictionaryPath
        self._raw: dict = {}
        self._version: str = "unknown"

        self._canonical_index: dict[str, list[dict]] = {}
        self._synonym_index: dict[str, list[dict]] = {}
        self._alias_index: dict[str, list[dict]] = {}
        self._abbreviation_index: dict[str, list[dict]] = {}
        self._related_index: dict[str, list[dict]] = {}
        self._all_phrases: list[_NormalizedPhrase] = []

        self._load()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self._path.exists():
            logger.error("Keyword dictionary not found at %s", self._path)
            raise FileNotFoundError(f"Dictionary not found: {self._path}")

        with open(self._path, "r", encoding="utf-8") as f:
            self._raw = json.load(f)

        meta = self._raw.get("metadata", {})
        self._version = meta.get("version", "unknown")
        logger.info("Loaded keyword dictionary v%s", self._version)

        domains = self._raw.get("domains", {})
        count = 0
        for domain_key, domain_data in domains.items():
            subdomains = domain_data.get("subdomains", {})
            for sub_key, sub_data in subdomains.items():
                for concept in sub_data.get("concepts", []):
                    concept["domain"] = domain_key
                    concept["subdomain"] = sub_key
                    self._index_concept(concept)
                    count += 1

        logger.info("Indexed %d concepts across %d domains", count, len(domains))

    def _index_concept(self, concept: dict) -> None:
        domain = concept["domain"]
        subdomain = concept["subdomain"]
        canonical = _normalize(concept["canonical_term"])
        entry = {
            "id": concept["id"],
            "canonical_term": concept["canonical_term"],
            "domain": domain,
            "subdomain": subdomain,
            "importance": concept.get("importance", 0.5),
            "potential_ml_tasks": concept.get("potential_ml_tasks", []),
            "dataset_requirements": concept.get("dataset_requirements", []),
            "feature_requirements": concept.get("feature_requirements", []),
            "target_requirements": concept.get("target_requirements", []),
        }

        self._canonical_index.setdefault(canonical, []).append(entry)
        self._all_phrases.append(_NormalizedPhrase(concept["canonical_term"], canonical))

        for syn in concept.get("synonyms", []):
            normed = _normalize(syn)
            if normed:
                self._synonym_index.setdefault(normed, []).append(
                    {**entry, "_matched_synonym": syn}
                )
                self._all_phrases.append(_NormalizedPhrase(syn, normed))

        for alias in concept.get("aliases", []):
            normed = _normalize(alias)
            if normed:
                self._alias_index.setdefault(normed, []).append(
                    {**entry, "_matched_alias": alias}
                )
                self._all_phrases.append(_NormalizedPhrase(alias, normed))

        for abbr in concept.get("abbreviations", []):
            normed = abbr.lower().strip()
            if normed:
                self._abbreviation_index.setdefault(normed, []).append(
                    {**entry, "_matched_abbr": abbr}
                )

        for related in concept.get("related_concepts", []):
            normed = _normalize(related)
            # Skip single-word related concepts (too broad, cause false positives)
            if normed and len(normed.split()) > 1:
                self._related_index.setdefault(normed, []).append(
                    {**entry, "_matched_related": related}
                )
                self._all_phrases.append(_NormalizedPhrase(related, normed))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def version(self) -> str:
        return self._version

    def match(
        self, text: str, *, case_sensitive: bool = False
    ) -> list[KeywordMatchEvidence]:
        """Match text against the knowledge base.

        Returns de-duplicated, evidence-tracked matches sorted by importance.

        Matching strategy:
        - Multi-word phrases: substring matching (e.g., 'machine learning' in text)
        - Single-word phrases: word-boundary matching to prevent false positives
          (e.g., 'learning' won't match inside 'machine learning')
        """
        if not case_sensitive:
            text = _normalize(text)
        else:
            text = text.strip()

        matched: dict[str, KeywordMatchEvidence] = {}

        # Pre-compile single-word boundary patterns
        single_word_pattern_cache: dict[str, re.Pattern] = {}

        for phrase_obj in self._all_phrases:
            words = phrase_obj.normalized.split()
            is_single_word = len(words) == 1

            if is_single_word:
                # Word-boundary matching for single words
                pattern = single_word_pattern_cache.get(phrase_obj.normalized)
                if pattern is None:
                    pattern = re.compile(
                        r"\b" + re.escape(phrase_obj.normalized) + r"\b",
                        re.IGNORECASE if not case_sensitive else 0,
                    )
                    single_word_pattern_cache[phrase_obj.normalized] = pattern
                if not pattern.search(text):
                    continue
            else:
                # Substring matching for multi-word phrases
                if phrase_obj.normalized not in text:
                    continue

            concept_entries = (
                self._canonical_index.get(phrase_obj.normalized, [])
                + self._synonym_index.get(phrase_obj.normalized, [])
                + self._alias_index.get(phrase_obj.normalized, [])
                + self._related_index.get(phrase_obj.normalized, [])
            )
            for entry in concept_entries:
                canonical = entry["canonical_term"]
                if canonical not in matched:
                    matched[canonical] = KeywordMatchEvidence(
                        canonical_term=canonical,
                        matched_text=phrase_obj.original,
                        match_type=self._determine_match_type(
                            phrase_obj.normalized, entry
                        ),
                        dictionary_entry_id=entry["id"],
                        domain=entry["domain"],
                        subdomain=entry["subdomain"],
                    )

        abbrev_pattern = re.compile(
            r"\b("
            + "|".join(re.escape(a) for a in self._abbreviation_index.keys())
            + r")\b",
            re.IGNORECASE,
        )
        for m in abbrev_pattern.finditer(text if case_sensitive else text):
            abbr_key = m.group(0).lower()
            for entry in self._abbreviation_index.get(abbr_key, []):
                canonical = entry["canonical_term"]
                if canonical not in matched:
                    matched[canonical] = KeywordMatchEvidence(
                        canonical_term=canonical,
                        matched_text=m.group(0),
                        match_type=MatchType.ABBREVIATION,
                        dictionary_entry_id=entry["id"],
                        domain=entry["domain"],
                        subdomain=entry["subdomain"],
                    )

        return sorted(matched.values(), key=lambda e: e.canonical_term)

    def get_concept(self, concept_id: str) -> Optional[dict]:
        domains = self._raw.get("domains", {})
        for domain_data in domains.values():
            for sub_data in domain_data.get("subdomains", {}).values():
                for concept in sub_data.get("concepts", []):
                    if concept["id"] == concept_id:
                        return concept
        return None

    def get_all_canonical_terms(self) -> list[str]:
        return sorted(
            set(e["canonical_term"] for entries in self._canonical_index.values() for e in entries)
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _determine_match_type(self, normalized: str, entry: dict) -> MatchType:
        if normalized == _normalize(entry["canonical_term"]):
            return MatchType.EXACT
        if "_matched_synonym" in entry:
            return MatchType.SYNONYM
        if "_matched_alias" in entry:
            return MatchType.ALIAS
        if "_matched_related" in entry:
            return MatchType.RELATED
        return MatchType.PHRASE
