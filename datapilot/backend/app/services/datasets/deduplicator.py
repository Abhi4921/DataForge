"""Deduplication of dataset candidates arriving from different sources.

Policy: merge only when there is strong evidence that two records describe the
same dataset. When uncertain, records stay separate. A wrong merge silently
drops a dataset from the results, which is worse than showing one near-duplicate.

Merge signals, strongest first:

1. Identical ``source`` + ``source_id``  - the same record seen twice.
2. Identical normalized URL               - the same dataset, same location.
3. Identical normalized name + identical owner.
4. Identical normalized name token set    - e.g. "Student Performance" vs
   "Student Performance Dataset", across any source.

Near-matches that are not exact after removing generic noise words are never
merged, so "Student Habits vs Academic Performance" and
"Student Academic Performance" stay distinct.
"""

from __future__ import annotations

import re
from typing import Optional

from app.core.logging_config import get_logger
from app.schemas.dataset import DatasetCandidate

logger = get_logger(__name__)

_NON_ALNUM = re.compile(r"[^a-z0-9]+")

# Words that carry no identifying information in a dataset title.
_GENERIC_NAME_TOKENS = frozenset(
    {"dataset", "datasets", "data", "csv", "tsv", "collection", "set", "corpus"}
)


def normalize_url_key(url: Optional[str]) -> Optional[str]:
    """Canonical form of a URL for equality checks."""
    if not url:
        return None
    key = url.strip().lower()
    key = re.sub(r"^https?://", "", key)
    key = re.sub(r"^www\.", "", key)
    key = key.split("#", 1)[0]
    key = key.split("?", 1)[0]
    key = key.rstrip("/")
    return key or None


def normalize_name_key(name: Optional[str]) -> Optional[str]:
    """Lowercased, punctuation-free name."""
    if not name:
        return None
    key = _NON_ALNUM.sub(" ", name.lower()).strip()
    return key or None


def name_token_key(name: Optional[str]) -> Optional[str]:
    """Name reduced to its identifying tokens, sorted, generic words removed.

    ``"Student Performance Dataset"`` and ``"student-performance"`` both map to
    ``"performance student"``.
    """
    key = normalize_name_key(name)
    if not key:
        return None
    tokens = [t for t in key.split() if t not in _GENERIC_NAME_TOKENS]
    if not tokens:
        return None
    return " ".join(sorted(tokens))


class DatasetDeduplicator:
    """Collapses duplicate candidates using conservative merge rules."""

    def __init__(self) -> None:
        self._source_key_index: dict[tuple[str, str], DatasetCandidate] = {}
        self._url_index: dict[str, DatasetCandidate] = {}
        self._owner_name_index: dict[tuple[str, str], DatasetCandidate] = {}
        self._name_index: dict[str, DatasetCandidate] = {}

    def deduplicate(self, candidates: list[DatasetCandidate]) -> list[DatasetCandidate]:
        """Return de-duplicated candidates, preserving first-seen order."""
        unique: list[DatasetCandidate] = []

        for candidate in candidates:
            existing = self._find_match(candidate)
            if existing is None:
                self._index(candidate)
                unique.append(candidate)
                continue

            merged = existing.merge(candidate)
            # Drop the index entries that pointed at the losing record, then
            # re-index the survivor. This must happen even when the survivor is
            # the same object we started with, otherwise a third duplicate
            # could no longer be matched against it.
            self._reindex(existing, merged)
            self._index(merged)
            if merged is not existing:
                # The verified record won, so it must take the verified
                # record's place in the output ordering.
                position = next(i for i, c in enumerate(unique) if c is existing)
                unique[position] = merged
            logger.debug(
                "Merged candidate %s into %s (%d sources)",
                candidate.id,
                merged.id,
                len(merged.duplicate_sources) + 1,
            )

        if len(unique) < len(candidates):
            logger.info(
                "Deduplication collapsed %d candidates into %d",
                len(candidates),
                len(unique),
            )
        return unique

    def reset(self) -> None:
        self._source_key_index.clear()
        self._url_index.clear()
        self._owner_name_index.clear()
        self._name_index.clear()

    # -- internals ---------------------------------------------------------

    def _find_match(self, candidate: DatasetCandidate) -> Optional[DatasetCandidate]:
        source_key = (candidate.source.lower(), candidate.source_id.lower())
        if source_key in self._source_key_index:
            return self._source_key_index[source_key]

        url_key = normalize_url_key(candidate.url)
        if url_key and url_key in self._url_index:
            return self._url_index[url_key]

        name_key = normalize_name_key(candidate.name)
        owner_key = normalize_name_key(candidate.owner)
        if name_key and owner_key and (owner_key, name_key) in self._owner_name_index:
            return self._owner_name_index[(owner_key, name_key)]

        token_key = name_token_key(candidate.name)
        if token_key and token_key in self._name_index:
            return self._name_index[token_key]

        return None

    def _index(self, candidate: DatasetCandidate) -> None:
        self._source_key_index[(candidate.source.lower(), candidate.source_id.lower())] = candidate
        url_key = normalize_url_key(candidate.url)
        if url_key:
            self._url_index[url_key] = candidate
        name_key = normalize_name_key(candidate.name)
        owner_key = normalize_name_key(candidate.owner)
        if name_key and owner_key:
            self._owner_name_index[(owner_key, name_key)] = candidate
        token_key = name_token_key(candidate.name)
        if token_key:
            self._name_index[token_key] = candidate

    def _reindex(self, old: DatasetCandidate, new: DatasetCandidate) -> None:
        """Repoint indexes from a replaced record to the surviving one."""
        old_source = (old.source.lower(), old.source_id.lower())
        if self._source_key_index.get(old_source) is old:
            self._source_key_index.pop(old_source, None)
        old_url = normalize_url_key(old.url)
        if old_url and self._url_index.get(old_url) is old:
            self._url_index.pop(old_url, None)
        old_name = normalize_name_key(old.name)
        old_owner = normalize_name_key(old.owner)
        if old_name and old_owner and self._owner_name_index.get((old_owner, old_name)) is old:
            self._owner_name_index.pop((old_owner, old_name), None)
        old_token = name_token_key(old.name)
        if old_token and self._name_index.get(old_token) is old:
            self._name_index.pop(old_token, None)
