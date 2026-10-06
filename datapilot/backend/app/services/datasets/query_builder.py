"""Build external search queries from structured project requirements.

A raw project description is far too long and conversational for a registry
search box ("I want to build a machine learning system that predicts...").
Registries match on short keyword queries, so this module reduces the
structured requirements to a compact, high-signal query.

The reduction is deterministic and explainable: the query is assembled from
terms that the dictionary/analysis stages already produced, never from raw
free-text word soup.
"""

from __future__ import annotations

from typing import Optional

from app.core.logging_config import get_logger
from app.schemas.project import ProjectAnalysisData
from app.services.keyword_service import KeywordService

logger = get_logger(__name__)

# Terms that carry no retrieval signal in a dataset registry. Beyond plain
# English stop words this deliberately drops *generic* AI/ML vocabulary
# ("artificial", "intelligence", "machine learning", "success", "achievement",
# "analytics", ...) and bare temporal qualifiers ("previous", "prior") that
# narrow a relevance search without adding a concept. Registries index concrete
# dataset concepts, and generic analytic words crowd a short query without
# narrowing the result set.
_STOP_TERMS = frozenset(
    {
        "a", "an", "and", "any", "are", "as", "at", "be", "build", "built",
        "but", "by", "can", "create", "data", "dataset", "datasets", "develop",
        "detecting", "do", "for", "from", "have", "i", "in", "into", "is", "it",
        "like", "machine", "make", "me", "model", "my", "of", "on", "or",
        "predict", "predicting", "project", "research",
        "system", "that", "the", "their", "this", "to", "use", "used", "using",
        "want", "want to", "with", "would", "you",
        # Generic AI / ML / analytics vocabulary.
        "achievement", "analysis", "analytics", "artificial", "deep",
        "foundational", "intelligence", "learning", "prediction", "science",
        "success",
        # Bare temporal/ordinal qualifiers: no concept, only a relevance hit.
        "previous", "prior", "historical", "current",
    }
)


def _tokens(text: Optional[str]) -> list[str]:
    if not text:
        return []
    cleaned = text.lower().replace("_", " ").replace("-", " ")
    return [t for t in "".join(c if c.isalnum() or c.isspace() else " " for c in cleaned).split() if t]


# The dictionary is loaded once per process; query building is read-only.
_KEYWORD_SERVICE: Optional[KeywordService] = None


def _keyword_service() -> KeywordService:
    global _KEYWORD_SERVICE
    if _KEYWORD_SERVICE is None:
        _KEYWORD_SERVICE = KeywordService()
    return _KEYWORD_SERVICE


def _concise(value: Optional[str], *, max_tokens: int = 4) -> str:
    """Reduce a model-written term to a short registry-friendly phrase.

    Short terms are kept verbatim. A verbose, sentence-like value is resolved
    through the knowledge dictionary to its canonical concept (e.g. "Student
    performance metrics and support requirement status" -> "student
    performance"); if no concept matches, the value is dropped rather than
    dumped into the query as a stream of sentence tokens.
    """
    tokens = _tokens(value)
    if not tokens:
        return ""
    if len(tokens) <= max_tokens:
        return (value or "").strip()
    return _keyword_service().resolve_canonical_phrase(value) or ""


def build_dataset_search_query(
    requirements: ProjectAnalysisData,
    max_terms: int = 6,
) -> str:
    """Reduce structured requirements to a short registry search query.

    Order is fixed and follows retrieval value, not prose order:

    1. the explicit prediction target (short, dataset-defining)
    2. explicit required feature names (concrete dataset columns)
    3. dictionary-verified user concepts (coarse domain words deferred)
    4. domain, then subdomain (coarse, but useful when the above are sparse)
    5. ML task concepts
    6. remaining LLM-inferred / related concepts, only if budget is left

    Generic AI/ML vocabulary is filtered out at every step, so it can never
    crowd out the concrete concepts a registry actually indexes. A verbose,
    model-written target is resolved to its canonical concept first, so
    "student performance" leads instead of a sentence's worth of filler. The
    query is deterministic, which keeps results reproducible and explainable.
    """
    understanding = requirements.project_understanding
    keywords = requirements.keywords
    dataset_requirements = requirements.dataset_requirements
    ordered: list[str] = []
    seen: set[str] = set()

    # Coarse category words (the project's own domain/subdomain and the
    # dictionary's domain keys) are held back for step 4 so a generic term like
    # "education" cannot consume a slot a concrete concept could use.
    coarse_domain: set[str] = set(_keyword_service().get_domain_keys())
    if understanding.domain:
        coarse_domain.add(" ".join(_tokens(understanding.domain.value)))
    if understanding.subdomain:
        coarse_domain.add(" ".join(_tokens(understanding.subdomain.value)))

    def add(value: Optional[str], drop_stop_terms: bool = True) -> None:
        for token in _tokens(value):
            if drop_stop_terms and (token in _STOP_TERMS or len(token) <= 2):
                continue
            if token not in seen:
                seen.add(token)
                ordered.append(token)

    # 1. The target variable is short and is the dataset's defining outcome.
    if len(ordered) < max_terms and understanding.target:
        add(_concise(understanding.target.value))
    for target_req in dataset_requirements.target_requirements:
        if len(ordered) >= max_terms:
            break
        add(_concise(target_req.name))

    # 2. Explicit required feature names are concrete dataset columns.
    for feature in dataset_requirements.feature_requirements:
        if len(ordered) >= max_terms:
            break
        add(feature.name)

    # 3. Dictionary-verified concepts are strong signal: they were matched
    #    against the project description itself.
    for match in keywords.dictionary_matches:
        if len(ordered) >= max_terms:
            break
        if " ".join(_tokens(match.canonical_term)) in coarse_domain:
            continue
        add(match.canonical_term)

    # 4. Domain/subdomain are coarse analytic categories: useful only once the
    #    concrete terms above are exhausted.
    if len(ordered) < max_terms and understanding.domain:
        add(understanding.domain.value)
    if len(ordered) < max_terms and understanding.subdomain:
        add(understanding.subdomain.value)

    # 5. ML task concepts.
    for task in understanding.ml_tasks:
        if len(ordered) >= max_terms:
            break
        add(task.task)

    # 6. Remaining project concepts and related terms, generic ones filtered.
    for keyword in list(keywords.canonical) + list(keywords.related_concepts):
        if len(ordered) >= max_terms:
            break
        add(keyword)

    query = " ".join(ordered[:max_terms]).strip()
    if not query:
        # Requirements exist but carried no usable terms (very short input).
        # Falling back to the raw dictionary vocabulary, stop words included,
        # keeps discovery functional instead of issuing an empty query.
        relaxed: list[str] = []
        for keyword in requirements.keywords.canonical:
            if len(relaxed) >= max_terms:
                break
            for token in _tokens(keyword):
                if len(token) > 2 and token not in relaxed:
                    relaxed.append(token)
        query = " ".join(relaxed).strip()

    logger.debug("Built dataset search query: %r", query)
    return query


def build_dataset_search_queries(
    requirements: ProjectAnalysisData,
    max_terms: int = 6,
    max_queries: int = 3,
) -> list[str]:
    """Build a small deterministic set of complementary registry queries.

    The first query preserves the established high-signal ordering. Additional
    queries emphasize domain context and project concepts so a registry's
    treatment of a long multi-term query cannot hide useful results.
    """
    if max_queries <= 0:
        return []

    primary = build_dataset_search_query(requirements, max_terms=max_terms)
    queries: list[str] = []

    def add_query(values: list[Optional[str]]) -> None:
        tokens: list[str] = []
        seen: set[str] = set()
        for value in values:
            for token in _tokens(value):
                if token in _STOP_TERMS or len(token) <= 2 or token in seen:
                    continue
                seen.add(token)
                tokens.append(token)
                if len(tokens) >= max_terms:
                    break
            if len(tokens) >= max_terms:
                break
        query = " ".join(tokens)
        if query and query not in queries and len(queries) < max_queries:
            queries.append(query)

    if primary:
        queries.append(primary)

    understanding = requirements.project_understanding
    dataset_requirements = requirements.dataset_requirements
    domain_terms = [
        understanding.domain.value if understanding.domain else None,
        understanding.subdomain.value if understanding.subdomain else None,
    ]
    feature_terms = [feature.name for feature in dataset_requirements.feature_requirements]
    target_terms = [
        understanding.target.value if understanding.target else None,
        *(target.name for target in dataset_requirements.target_requirements),
    ]
    concept_terms = list(requirements.keywords.canonical)
    if not concept_terms:
        concept_terms = [
            match.canonical_term for match in requirements.keywords.dictionary_matches
        ]

    add_query(domain_terms + concept_terms + target_terms)
    add_query(target_terms + feature_terms + domain_terms)
    return queries[:max_queries]
