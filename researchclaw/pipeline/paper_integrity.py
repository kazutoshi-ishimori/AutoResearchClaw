"""Deterministic paper integrity guards for revision and export stages."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from researchclaw.pipeline._helpers import _utcnow_iso

_CITE_KEY_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_-]*\d{4}[a-zA-Z0-9_-]*")
_BRACKET_RE = re.compile(r"\[([^\]]+)\]")
_LATEX_CITE_RE = re.compile(r"\\cite\{([^}]+)\}")
_HEADING_RE = re.compile(r"^\s*#{1,4}\s+(.+?)\s*$", re.MULTILINE)

_SECTION_ALIASES = {
    "abstract": "abstract",
    "introduction": "introduction",
    "intro": "introduction",
    "background": "introduction",
    "related work": "related_work",
    "method": "methods",
    "methods": "methods",
    "methodology": "methods",
    "approach": "methods",
    "framework": "methods",
    "experimental setup": "methods",
    "experiment": "results",
    "experiments": "results",
    "results": "results",
    "evaluation": "results",
    "analysis": "results",
    "discussion": "discussion",
    "limitations": "limitations",
    "limitation": "limitations",
    "limitations and conclusion": "conclusion",
    "discussion and conclusion": "conclusion",
    "conclusion": "conclusion",
    "conclusions": "conclusion",
}
_REVISION_REQUIRED_IF_PRESENT = {
    "abstract",
    "introduction",
    "methods",
    "results",
    "discussion",
    "limitations",
    "conclusion",
}


def extract_bibtex_keys(bib_text: str) -> set[str]:
    """Return BibTeX keys declared in *bib_text*."""
    return {m.group(1).strip() for m in re.finditer(r"@\w+\{([^,]+),", bib_text)}


def extract_citation_keys(text: str) -> set[str]:
    """Return citation-like keys from markdown brackets and LaTeX cites."""
    keys: set[str] = set()
    for match in _BRACKET_RE.finditer(text):
        inner = match.group(1)
        parts = [p.strip() for p in re.split(r"[,;]\s*", inner)]
        if parts and all(_CITE_KEY_RE.fullmatch(p) for p in parts if p):
            keys.update(p for p in parts if p)
    for match in _LATEX_CITE_RE.finditer(text):
        keys.update(p.strip() for p in match.group(1).split(",") if p.strip())
    return keys


def render_revision_integrity_instruction(allowed_citation_keys: set[str]) -> str:
    """Render hard constraints for Stage 19 revision."""
    allowed_preview = ", ".join(sorted(allowed_citation_keys)[:80]) or "(none)"
    if len(allowed_citation_keys) > 80:
        allowed_preview += f", ... ({len(allowed_citation_keys)} total)"
    return (
        "\n\n## REVISION INTEGRITY GUARD (HARD CONSTRAINT)\n"
        "Revise the paper without changing its evidentiary boundary.\n"
        f"- Allowed citation keys: {allowed_preview}\n"
        "- Do NOT introduce citation keys that are absent from the upstream "
        "bibliography or original draft.\n"
        "- Do NOT add unsupported numeric, latency, dataset, repository, Docker, "
        "hardware, p-value, confidence-interval, or benchmark claims.\n"
        "- Preserve the draft's section structure. Do not collapse, summarize, "
        "or omit Methods, Results, Limitations, Discussion, or Conclusion "
        "sections that already exist.\n"
        "- If reviewer requests need missing evidence, state that the analysis "
        "was not evaluated instead of fabricating support.\n"
    )


def enforce_revision_integrity(
    *,
    draft: str,
    revised: str,
    allowed_citation_keys: set[str],
    min_word_ratio: float = 0.8,
    min_word_count: int | None = None,
) -> tuple[str, dict[str, Any]]:
    """Strip citation drift and fall back when revision drops core sections."""
    sanitized, citation_report = strip_disallowed_citations(
        revised,
        allowed_citation_keys,
    )
    draft_sections = _canonical_sections(draft)
    revised_sections = _canonical_sections(sanitized)
    required = draft_sections & _REVISION_REQUIRED_IF_PRESENT
    missing_sections = sorted(required - revised_sections)
    draft_word_count = len(draft.split())
    revised_word_count = len(sanitized.split())
    word_floor = (
        min_word_count
        if min_word_count is not None
        else int(draft_word_count * min_word_ratio)
    )
    too_short = draft_word_count > 0 and revised_word_count < word_floor
    report: dict[str, Any] = {
        "generated": _utcnow_iso(),
        "removed_citation_keys": citation_report["removed_citation_keys"],
        "missing_sections": missing_sections,
        "draft_word_count": draft_word_count,
        "revised_word_count": revised_word_count,
        "min_word_ratio": min_word_ratio,
        "min_word_count": min_word_count,
        "too_short": too_short,
        "fallback_to_draft": False,
    }
    if missing_sections or too_short:
        report["fallback_to_draft"] = True
        report["fallback_reason"] = (
            "missing_sections" if missing_sections else "revision_too_short"
        )
        return draft, report
    return sanitized, report


def strip_disallowed_citations(
    text: str,
    allowed_citation_keys: set[str],
) -> tuple[str, dict[str, Any]]:
    """Remove or normalize citation keys not present in *allowed_citation_keys*."""
    report: dict[str, Any] = {
        "generated": _utcnow_iso(),
        "removed_citation_keys": [],
    }
    if not allowed_citation_keys:
        return text, report

    removed: set[str] = set()

    def _normalized_allowed_key(key: str) -> str | None:
        if key in allowed_citation_keys:
            return key
        if key.startswith("cite_") and key[5:] in allowed_citation_keys:
            return key[5:]
        return None

    def _replace_bracket(match: re.Match[str]) -> str:
        inner = match.group(1)
        parts = [p.strip() for p in re.split(r"[,;]\s*", inner)]
        if not parts or not all(_CITE_KEY_RE.fullmatch(p) for p in parts if p):
            return match.group(0)
        kept: list[str] = []
        for part in parts:
            if not part:
                continue
            normalized = _normalized_allowed_key(part)
            if normalized:
                kept.append(normalized)
            else:
                removed.add(part)
        if not kept:
            return ""
        return "[" + ", ".join(dict.fromkeys(kept)) + "]"

    def _replace_latex(match: re.Match[str]) -> str:
        parts = [p.strip() for p in match.group(1).split(",") if p.strip()]
        kept: list[str] = []
        for part in parts:
            normalized = _normalized_allowed_key(part)
            if normalized:
                kept.append(normalized)
            else:
                removed.add(part)
        if not kept:
            return ""
        return "\\cite{" + ", ".join(dict.fromkeys(kept)) + "}"

    updated = _BRACKET_RE.sub(_replace_bracket, text)
    updated = _LATEX_CITE_RE.sub(_replace_latex, updated)
    updated = re.sub(r"  +", " ", updated)
    updated = re.sub(r" ([.,;:)])", r"\1", updated)
    report["removed_citation_keys"] = sorted(removed)
    return updated, report


def filter_contract_figures(
    chart_files: list[Path],
    contract: dict[str, Any],
) -> tuple[list[Path], list[str]]:
    """Keep only chart files explicitly allowed by paper_contract.json."""
    allowed = set(contract.get("available_figures") or [])
    if not allowed:
        return chart_files, []
    kept: list[Path] = []
    skipped: list[str] = []
    for path in chart_files:
        rel = f"charts/{path.name}"
        if rel in allowed:
            kept.append(path)
        else:
            skipped.append(rel)
    return kept, sorted(set(skipped))


def section_integrity_warnings(paper: str) -> list[str]:
    """Return lightweight structural warnings for the final paper report."""
    sections = _canonical_sections(paper)
    warnings: list[str] = []
    for required in ("abstract", "introduction", "methods", "results", "conclusion"):
        if required not in sections:
            warnings.append(f"missing_section:{required}")
    return warnings


def _canonical_sections(text: str) -> set[str]:
    sections: set[str] = set()
    for match in _HEADING_RE.finditer(text):
        heading = _normalize_heading(match.group(1))
        canonical = _SECTION_ALIASES.get(heading)
        if canonical:
            sections.add(canonical)
    return sections


def _normalize_heading(heading: str) -> str:
    heading = re.sub(r"[*_`]", "", heading).strip().lower()
    heading = re.sub(r"^\d+(?:\.\d+)*\.?\s+", "", heading)
    heading = re.sub(
        r"^(?:[ivxlcdm]+|[a-z])(?:[.)]|\s*[-:])\s+",
        "",
        heading,
    )
    return heading
