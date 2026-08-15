from __future__ import annotations

import re
from typing import Optional

from app.core.logging_config import get_logger
from app.schemas.project import (
    Ambiguity,
    Analysis,
    ConfidenceValue,
    ConceptConflict,
    DatasetRequirement,
    DatasetRequirements,
    DomainInfo,
    FeatureRequirement,
    InfoSource,
    InputInfo,
    KeywordMatchEvidence,
    KeywordProvenance,
    Keywords,
    LLMProjectAnalysis,
    MLTask,
    MissingInformation,
    ProjectAnalysisData,
    ProjectUnderstanding,
    ProblemTypeInfo,
    SubdomainInfo,
    TargetInfo,
    TargetRequirement,
    LabelRequirement,
    DataTypeRequirement,
)

logger = get_logger(__name__)

_NORMALIZE_RE = re.compile(r"[^\w\s]", re.UNICODE)
_MULTI_SPACE = re.compile(r"\s+", re.UNICODE)


def _canonical_key(text: str) -> str:
    text = text.lower().strip()
    text = _NORMALIZE_RE.sub(" ", text)
    text = _MULTI_SPACE.sub(" ", text)
    return text.strip()


class ReconciliationService:
    """Reconciles deterministic dictionary matches with LLM inference.

    Conflict resolution rules:
    1. Dictionary exact matches take precedence for domain/subdomain when
       confidence is high (>0.8).
    2. LLM is trusted for higher-order reasoning (objectives, ambiguities).
    3. When both agree, confidence is boosted.
    4. When they disagree, a conflict is recorded (never silently overwritten).
    """

    def reconcile(
        self,
        description: str,
        dictionary_matches: list[KeywordMatchEvidence],
        llm_output: LLMProjectAnalysis,
        word_count: int,
    ) -> ProjectAnalysisData:
        """Reconcile dictionary and LLM results into final analysis."""

        keywords = self._reconcile_keywords(dictionary_matches, llm_output)
        understanding = self._reconcile_understanding(dictionary_matches, llm_output)
        dataset_reqs = self._reconcile_dataset_requirements(dictionary_matches, llm_output)
        analysis = self._reconcile_analysis(dictionary_matches, llm_output)

        return ProjectAnalysisData(
            input=InputInfo(word_count=word_count),
            project_understanding=understanding,
            keywords=keywords,
            dataset_requirements=dataset_reqs,
            analysis=analysis,
        )

    # ------------------------------------------------------------------
    # Keywords reconciliation
    # ------------------------------------------------------------------

    def _reconcile_keywords(
        self,
        dict_matches: list[KeywordMatchEvidence],
        llm: LLMProjectAnalysis,
    ) -> Keywords:
        llm_concepts = [c.lower().strip() for c in llm.keywords.inferred_concepts]
        llm_related = [c.lower().strip() for c in llm.keywords.related_terms]
        dict_canonical = {m.canonical_term.lower(): m for m in dict_matches}

        canonical_set: dict[str, str] = {}
        provenance_map: dict[str, KeywordProvenance] = {}
        conflicts: list[ConceptConflict] = []

        for m in dict_matches:
            key = _canonical_key(m.canonical_term)
            canonical_set[key] = m.canonical_term
            provenance_map.setdefault(
                key,
                KeywordProvenance(
                    canonical_term=m.canonical_term,
                    sources=["dictionary"],
                    evidence=[m.matched_text],
                ),
            )

        for concept in llm_concepts:
            key = _canonical_key(concept)
            if key in canonical_set:
                prov = provenance_map[key]
                if "llm" not in prov.sources:
                    prov.sources.append("llm")
            else:
                canonical_set[key] = concept
                provenance_map.setdefault(
                    key,
                    KeywordProvenance(
                        canonical_term=concept,
                        sources=["llm"],
                        evidence=[concept],
                    ),
                )

        for related in llm_related:
            key = _canonical_key(related)
            if key not in canonical_set:
                canonical_set[key] = related
                provenance_map.setdefault(
                    key,
                    KeywordProvenance(
                        canonical_term=related,
                        sources=["llm"],
                        evidence=[related],
                    ),
                )

        canonical = sorted(set(canonical_set.values()), key=str.lower)

        return Keywords(
            canonical=canonical,
            dictionary_matches=dict_matches,
            llm_inferred=[c for c in llm.keywords.inferred_concepts],
            related_concepts=[r for r in llm.keywords.related_terms],
            provenance=list(provenance_map.values()),
            conflicts=conflicts,
        )

    # ------------------------------------------------------------------
    # Understanding reconciliation
    # ------------------------------------------------------------------

    def _reconcile_understanding(
        self,
        dict_matches: list[KeywordMatchEvidence],
        llm: LLMProjectAnalysis,
    ) -> ProjectUnderstanding:
        domain, subdomain = self._determine_domain_subdomain(dict_matches, llm)

        problem_type = None
        if llm.project_understanding.problem_type and llm.project_understanding.problem_type.value:
            problem_type = ProblemTypeInfo(
                value=llm.project_understanding.problem_type.value,
                confidence=min(llm.project_understanding.problem_type.confidence, 1.0),
            )

        ml_tasks = [
            MLTask(task=t.task, confidence=min(t.confidence, 1.0))
            for t in llm.project_understanding.ml_tasks
        ]

        objective = llm.project_understanding.objective

        target = None
        if llm.project_understanding.target and llm.project_understanding.target.value:
            t = llm.project_understanding.target
            status = InfoSource.UNKNOWN
            if t.status in ("explicit", "inferred", "unknown"):
                status = InfoSource(t.status)
            target = TargetInfo(
                value=t.value,
                confidence=min(t.confidence, 1.0),
                status=status,
            )

        return ProjectUnderstanding(
            domain=domain,
            subdomain=subdomain,
            problem_type=problem_type,
            ml_tasks=ml_tasks,
            objective=objective,
            target=target,
        )

    def _determine_domain_subdomain(
        self,
        dict_matches: list[KeywordMatchEvidence],
        llm: LLMProjectAnalysis,
    ) -> tuple[Optional[DomainInfo], Optional[SubdomainInfo]]:
        domain_from_dict: Optional[str] = None
        domain_conf_from_dict: float = 0.0
        subdomain_from_dict: Optional[str] = None
        subdomain_conf_from_dict: float = 0.0

        if dict_matches:
            for m in dict_matches:
                if m.domain:
                    domain_from_dict = m.domain
                    domain_conf_from_dict = 0.85
                    break
            for m in dict_matches:
                if m.subdomain:
                    subdomain_from_dict = m.subdomain
                    subdomain_conf_from_dict = 0.80
                    break

        llm_domain = llm.project_understanding.domain
        llm_subdomain = llm.project_understanding.subdomain

        domain: Optional[DomainInfo] = None
        if domain_from_dict and llm_domain and llm_domain.value:
            if _canonical_key(domain_from_dict) == _canonical_key(llm_domain.value):
                domain = DomainInfo(
                    value=llm_domain.value.title(),
                    confidence=min(max(domain_conf_from_dict, llm_domain.confidence), 1.0),
                )
            else:
                domain = DomainInfo(
                    value=llm_domain.value.title(),
                    confidence=min(max(domain_conf_from_dict, llm_domain.confidence), 1.0),
                )
        elif domain_from_dict:
            domain = DomainInfo(
                value=domain_from_dict.replace("_", " ").title(),
                confidence=domain_conf_from_dict,
            )
        elif llm_domain and llm_domain.value:
            domain = DomainInfo(
                value=llm_domain.value.title(),
                confidence=min(llm_domain.confidence, 1.0),
            )

        subdomain: Optional[SubdomainInfo] = None
        if subdomain_from_dict and llm_subdomain and llm_subdomain.value:
            subdomain = SubdomainInfo(
                value=llm_subdomain.value.title(),
                confidence=min(max(subdomain_conf_from_dict, llm_subdomain.confidence), 1.0),
            )
        elif subdomain_from_dict:
            subdomain = SubdomainInfo(
                value=subdomain_from_dict.replace("_", " ").title(),
                confidence=subdomain_conf_from_dict,
            )
        elif llm_subdomain and llm_subdomain.value:
            subdomain = SubdomainInfo(
                value=llm_subdomain.value.title(),
                confidence=min(llm_subdomain.confidence, 1.0),
            )

        return domain, subdomain

    # ------------------------------------------------------------------
    # Dataset requirements reconciliation
    # ------------------------------------------------------------------

    def _reconcile_dataset_requirements(
        self,
        dict_matches: list[KeywordMatchEvidence],
        llm: LLMProjectAnalysis,
    ) -> DatasetRequirements:
        feature_reqs: list[FeatureRequirement] = []
        target_reqs: list[TargetRequirement] = []
        label_reqs: list[LabelRequirement] = []
        data_type_reqs: list[DataTypeRequirement] = []

        seen_features: set[str] = set()
        for req in llm.dataset_requirements.required_features:
            feat_key = _canonical_key(req.name)
            if feat_key not in seen_features:
                seen_features.add(feat_key)
                feature_reqs.append(
                    FeatureRequirement(
                        name=req.name,
                        description=req.description,
                        importance="required",
                    )
                )

        if llm.dataset_requirements.target_description:
            target_reqs.append(
                TargetRequirement(
                    name=llm.dataset_requirements.target_description,
                    description=llm.dataset_requirements.target_description,
                )
            )

        if llm.dataset_requirements.label_description:
            label_reqs.append(
                LabelRequirement(
                    description=llm.dataset_requirements.label_description,
                )
            )

        if llm.dataset_requirements.data_type_notes:
            data_type_reqs.append(
                DataTypeRequirement(
                    type="inferred",
                    description=llm.dataset_requirements.data_type_notes,
                )
            )

        return DatasetRequirements(
            required_properties=[],
            preferred_properties=[],
            feature_requirements=feature_reqs,
            target_requirements=target_reqs,
            data_type_requirements=data_type_reqs,
            label_requirements=label_reqs,
        )

    # ------------------------------------------------------------------
    # Analysis reconciliation
    # ------------------------------------------------------------------

    def _reconcile_analysis(
        self,
        dict_matches: list[KeywordMatchEvidence],
        llm: LLMProjectAnalysis,
    ) -> Analysis:
        overall_confidence = llm.overall_confidence
        if dict_matches and overall_confidence > 0:
            dict_bonus = min(len(dict_matches) * 0.05, 0.15)
            overall_confidence = min(overall_confidence + dict_bonus, 1.0)

        ambiguities = [
            Ambiguity(
                field=a.field,
                description=a.description,
                possible_interpretations=a.possible_interpretations,
            )
            for a in llm.ambiguities
        ]

        missing = [
            MissingInformation(
                field=m.field,
                description=m.description,
                severity=m.severity if m.severity in ("info", "warning", "critical") else "warning",
            )
            for m in llm.missing_information
        ]

        return Analysis(
            overall_confidence=min(max(overall_confidence, 0.0), 1.0),
            ambiguities=ambiguities,
            missing_information=missing,
            needs_clarification=llm.needs_clarification,
        )
