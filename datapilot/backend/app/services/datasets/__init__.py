"""Dataset discovery package (Phase 2).

Public surface:

* :class:`DatasetSource` - the interface every discovery source implements.
* :class:`KaggleDatasetSource` - real, externally verified Kaggle datasets.
* :class:`GenAIDatasetSource` - unverified candidate suggestions from Gemini.
* :class:`DatasetDiscoveryService` - the orchestrator.
* :func:`build_default_sources` - the source registry.

To add a source (UCI, Hugging Face, research papers, ...) implement
:class:`DatasetSource`, then register it in :func:`build_default_sources`.
Nothing else in the codebase needs to change.
"""

from __future__ import annotations

from typing import Optional

from app.core.config import Settings
from app.services.datasets.base import DatasetSource, DatasetSourceError
from app.services.datasets.deduplicator import DatasetDeduplicator
from app.services.datasets.discovery import (
    DatasetDiscoveryResult,
    DatasetDiscoveryService,
)
from app.services.datasets.genai_source import GenAIDatasetSource
from app.services.datasets.kaggle_source import KaggleDatasetSource
from app.services.datasets.ranking import DatasetRankingEngine, build_project_profile

__all__ = [
    "DatasetDeduplicator",
    "DatasetDiscoveryResult",
    "DatasetDiscoveryService",
    "DatasetRankingEngine",
    "DatasetSource",
    "DatasetSourceError",
    "GenAIDatasetSource",
    "KaggleDatasetSource",
    "build_default_sources",
    "build_project_profile",
]


def build_default_sources(
    settings: Optional[Settings] = None,
) -> list[DatasetSource]:
    """Build the active source registry, honouring per-source enable flags.

    This is the single extension point for new dataset sources. Registering a
    source here automatically routes it through normalization, deduplication,
    ranking and both API endpoints.
    """
    sources: list[DatasetSource] = []

    kaggle = KaggleDatasetSource(settings)
    if kaggle.is_enabled:
        sources.append(kaggle)

    genai = GenAIDatasetSource(settings)
    if genai.is_enabled:
        sources.append(genai)

    # Future extension points, intentionally not implemented yet:
    #   HuggingFaceDatasetSource   -> verified_external
    #   UCIDatasetSource            -> verified_external
    #   GovernmentDatasetSource     -> verified_external
    #   GitHubDatasetSource         -> verified_external
    #   ResearchPaperDatasetSource  -> verified_external (paper -> dataset link)
    return sources


def register_source(source: DatasetSource, registry: Optional[list] = None) -> list[DatasetSource]:
    """Append an extra source to a registry list. Convenience for extensions."""
    target = registry if registry is not None else build_default_sources()
    target.append(source)
    return target
