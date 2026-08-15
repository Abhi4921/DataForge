from __future__ import annotations

from app.services.keyword_service import KeywordService


def build_dictionary_context(keyword_service: KeywordService, matches: list) -> str:
    """Build a concise dictionary context string for the LLM prompt.

    This provides the LLM with information about what the dictionary
    already matched, so it can focus on deeper understanding rather
    than re-discovering basic terms.
    """
    if not matches:
        return ""

    lines = [
        "The following concepts were matched from the knowledge base:",
    ]
    for m in matches:
        lines.append(
            f"- {m.canonical_term} (domain: {m.domain}, "
            f"subdomain: {m.subdomain}, match_type: {m.match_type.value}, "
            f"matched_text: '{m.matched_text}')"
        )
    lines.append("")
    lines.append(
        "Use these matches as context. Do not duplicate them unless "
        "you have additional insight. Focus on deeper understanding, "
        "ambiguities, and missing information."
    )
    return "\n".join(lines)
