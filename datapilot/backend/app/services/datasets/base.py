"""Source-agnostic dataset discovery interface.

Every dataset source implements this contract. The discovery service, the
deduplicator, the ranking engine and the API layer only ever see
:class:`~app.schemas.dataset.DatasetCandidate`, so adding a source later is a
matter of writing one class and registering it.

Adding a new source
-------------------
1. Subclass :class:`DatasetSource` and set ``name`` / ``source_type``.
2. Implement :meth:`search` returning normalized candidates.
3. Optionally implement :meth:`get_metadata` and :meth:`healthcheck`.
4. Register it in ``app/services/datasets/__init__.py:build_default_sources``.

For example, a future research-paper source would implement ``search`` by
finding papers about the project topic, extracting dataset names and URLs
mentioned in those papers, and normalizing them to ``DatasetCandidate`` with
``source_type=VERIFIED_EXTERNAL`` only when a resolvable external link exists.
"""

from __future__ import annotations

import abc
import re
from typing import ClassVar, Optional

from app.core.config import Settings
from app.schemas.dataset import DatasetCandidate, SourceSearchRequest, SourceType

#: Anything that looks like a credential is removed from outbound messages.
#: Sources sanitize their own messages, and the discovery layer redacts again as
#: defense in depth: a source bug must never turn into a leaked token.
_SECRET_PATTERNS = (
    re.compile(r"(?i)\bbearer\s+\S+"),
    re.compile(r"\bKAGGLE_[A-Z_]*TOKEN\S*", re.IGNORECASE),
    re.compile(r"\bKGAT_\S+"),
    re.compile(r"\bAIza\S+"),
    re.compile(r"(?i)\b(authorization|x-api-key|api[_-]?key)\b\s*[:=]\s*\S+"),
)

REDACTED = "[redacted]"


def redact_secrets(message: Optional[str]) -> Optional[str]:
    """Strip credential-shaped substrings from a user-visible message."""
    if not message:
        return message
    cleaned = message
    for pattern in _SECRET_PATTERNS:
        cleaned = pattern.sub(REDACTED, cleaned)
    return cleaned


class DatasetSourceError(Exception):
    """Raised by a source when discovery fails.

    ``code`` is a stable machine-readable identifier (never a raw upstream
    message, which could leak credentials or internals) that the API layer maps
    to an HTTP status.
    """

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(message)


class DatasetSource(abc.ABC):
    """Base class for all dataset discovery sources."""

    #: Short, stable identifier stored on every candidate, e.g. ``"kaggle"``.
    name: ClassVar[str] = "abstract"

    #: Whether this source yields externally verified results or AI suggestions.
    source_type: ClassVar[SourceType] = SourceType.VERIFIED_EXTERNAL

    #: Default maximum candidates returned by :meth:`search`.
    default_limit: ClassVar[int] = 10

    #: Hard cap this source must never exceed.
    max_limit: ClassVar[int] = 50

    def __init__(self, settings: Optional[Settings] = None) -> None:
        from app.core.config import get_settings

        self._settings = settings or get_settings()

    @property
    def settings(self) -> Settings:
        return self._settings

    @property
    def is_configured(self) -> bool:
        """Whether the source has the credentials it needs to be queried."""
        return True

    @property
    def supports_project_requirements(self) -> bool:
        """Whether this source needs structured requirements to be useful.

        Sources that do not (e.g. a plain keyword registry search) are still
        called with ``project_requirements=None``; sources that do are skipped
        instead of being queried with an empty context.
        """
        return False

    @abc.abstractmethod
    async def search(self, request: SourceSearchRequest) -> list[DatasetCandidate]:
        """Discover dataset candidates.

        Implementations must:

        * return normalized :class:`DatasetCandidate` objects,
        * never exceed ``request.candidate_pool_limit`` when it is supplied,
          otherwise never exceed ``request.limit``,
        * never download dataset contents,
        * raise :class:`DatasetSourceError` on failure rather than returning
          partial results silently.
        """

    async def get_metadata(self, source_id: str) -> DatasetCandidate:
        """Fetch detailed metadata for one dataset. Optional for a source."""
        raise DatasetSourceError(
            code="DATASET_SOURCE_UNSUPPORTED_OPERATION",
            message=f"Source '{self.name}' does not implement metadata lookup.",
        )

    async def healthcheck(self) -> bool:
        """Cheap reachability probe used by the readiness endpoint."""
        return True

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} name={self.name!r} configured={self.is_configured}>"
