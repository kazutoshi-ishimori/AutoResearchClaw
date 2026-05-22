"""Paper-writing evidence contract.

The paper contract is a deterministic handoff from experiment artifacts to
Stages 16-22.  It tells the writer exactly which numbers, conditions, and
figures are allowed so weaker local/cloud LLMs do not fill gaps with plausible
but unsupported results.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from researchclaw.pipeline._helpers import _utcnow_iso
from researchclaw.pipeline.verified_registry import VerifiedRegistry

_STRICT_MARKDOWN_HEADINGS = {
    "results",
    "main results",
    "experiments",
    "experimental setup",
    "experimental results",
    "evaluation",
    "ablation",
    "ablation study",
    "quantitative",
}

_NUMBER_RE = re.compile(
    r"(?<![a-zA-Z_\\])"
    r"(-?\d+\.?\d*(?:[eE][+-]?\d+)?)"
    r"(?![a-zA-Z_])"
)


def build_paper_contract(
    run_dir: Path,
    *,
    metric_direction: str = "maximize",
) -> dict[str, Any]:
    """Build a JSON-serializable contract from verified experiment artifacts."""
    registry = VerifiedRegistry.from_run_dir(
        run_dir,
        metric_direction=metric_direction,
        best_only=True,
    )
    allowed_numbers = sorted(
        {
            round(float(value), 8)
            for value in registry.values
            if math.isfinite(float(value))
        }
    )
    available_figures = _collect_available_figures(run_dir)
    contract: dict[str, Any] = {
        "version": 1,
        "generated": _utcnow_iso(),
        "source": "VerifiedRegistry.from_run_dir(best_only=True)",
        "metric_direction": metric_direction,
        "primary_metric": registry.primary_metric,
        "allowed_conditions": sorted(registry.condition_names),
        "allowed_numbers": allowed_numbers,
        "available_figures": available_figures,
        "claim_ledger": build_claim_ledger(run_dir),
        "rules": {
            "numbers": "only_allowed_numbers",
            "conditions": "only_allowed_conditions",
            "figures": "only_available_figures",
            "missing_results": "state_not_evaluated",
            "unsupported_statistics": "omit_or_mark_not_computed",
        },
    }
    return contract


def write_paper_contract(
    stage_dir: Path,
    run_dir: Path,
    *,
    metric_direction: str = "maximize",
) -> dict[str, Any]:
    """Build and persist ``paper_contract.json`` in *stage_dir*."""
    contract = build_paper_contract(run_dir, metric_direction=metric_direction)
    (stage_dir / "paper_contract.json").write_text(
        json.dumps(contract, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return contract


def load_paper_contract(run_dir: Path) -> dict[str, Any]:
    """Load the newest paper contract in *run_dir*, or return an empty dict."""
    candidates = sorted(run_dir.glob("stage-*/paper_contract.json"), reverse=True)
    for path in candidates:
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(loaded, dict):
            if "claim_ledger" not in loaded:
                loaded["claim_ledger"] = build_claim_ledger(run_dir)
            return loaded
    return {}


def render_paper_contract_instruction(contract: dict[str, Any]) -> str:
    """Render the contract as a compact hard-constraint prompt block."""
    if not contract:
        return ""
    conditions = contract.get("allowed_conditions") or []
    numbers = contract.get("allowed_numbers") or []
    figures = contract.get("available_figures") or []
    number_preview = ", ".join(_format_number(n) for n in numbers[:120])
    if len(numbers) > 120:
        number_preview += f", ... ({len(numbers)} total)"
    condition_preview = ", ".join(str(c) for c in conditions) or "(none)"
    figure_preview = ", ".join(str(f) for f in figures) or "(none)"
    return (
        "\n\n## PAPER CONTRACT (HARD CONSTRAINT)\n"
        "This contract is the authoritative boundary for the paper. "
        "ONLY use these numeric values for experimental claims; do not invent, "
        "interpolate, extrapolate, round beyond recognition, or add plausible "
        "baseline/statistical values.\n"
        f"- Metric direction: {contract.get('metric_direction', 'unknown')}\n"
        f"- Verified conditions: {condition_preview}\n"
        f"- Allowed figures: {figure_preview}\n"
        f"- Allowed numeric values: {number_preview}\n"
        "- If a desired dataset, method, metric, p-value, confidence interval, "
        "or ablation is not in this contract, write that it was not evaluated "
        "or omit the claim.\n"
        "- Tables must use only verified conditions and allowed numeric values. "
        "Use `--` for unsupported cells.\n"
        + _render_claim_ledger_instruction(contract.get("claim_ledger"))
    )


def build_claim_ledger(run_dir: Path) -> dict[str, Any]:
    """Build metric-specific evidence rules for paper claims.

    ``allowed_numbers`` is intentionally broad because the same value can be
    valid for one metric and impossible for another.  The claim ledger keeps
    the metric context so the writer can avoid plausible but invalid claims.
    """
    summary = _load_experiment_summary(run_dir)
    metrics_summary = summary.get("metrics_summary", {})
    condition_summaries = summary.get("condition_summaries", {})
    raw_claims: dict[tuple[str, str], dict[str, Any]] = {}

    if isinstance(metrics_summary, dict):
        for key, payload in metrics_summary.items():
            parsed = _parse_metric_path(str(key))
            if parsed is None or not isinstance(payload, dict):
                continue
            condition, metric = parsed
            value = _to_float(payload.get("mean"))
            if value is None:
                value = _to_float(payload.get("value"))
            if value is None:
                continue
            raw_claims[(condition, metric)] = {
                "condition": condition,
                "metric": metric,
                "value": round(value, 4),
                "count": int(payload.get("count") or 1),
            }

    if isinstance(condition_summaries, dict):
        for condition, payload in condition_summaries.items():
            if not isinstance(payload, dict):
                continue
            metrics = payload.get("metrics")
            if not isinstance(metrics, dict):
                continue
            count = int(payload.get("n_seeds") or payload.get("n_seed_metrics") or 1)
            for metric, value_raw in metrics.items():
                value = _to_float(value_raw)
                if value is None:
                    continue
                raw_claims.setdefault(
                    (str(condition), str(metric)),
                    {
                        "condition": str(condition),
                        "metric": str(metric),
                        "value": round(value, 4),
                        "count": count,
                    },
                )

    valid_claims: list[dict[str, Any]] = []
    invalid_claims: list[dict[str, Any]] = []
    for claim in sorted(
        raw_claims.values(),
        key=lambda item: (str(item["condition"]), str(item["metric"])),
    ):
        reason = _invalid_metric_reason(str(claim["metric"]), float(claim["value"]))
        if reason:
            invalid_claims.append({**claim, "reason": reason})
        else:
            valid_claims.append(claim)

    conditions = sorted({str(claim["condition"]) for claim in raw_claims.values()})
    baseline_conditions = [
        condition for condition in conditions if _looks_like_baseline(condition)
    ]
    comparative_policy = (
        "allow_verified_baseline_comparisons"
        if baseline_conditions
        else "forbid_percentage_improvement_claims"
    )
    return {
        "version": 1,
        "generated": _utcnow_iso(),
        "valid_metric_claims": valid_claims,
        "invalid_metric_claims": invalid_claims,
        "baseline_conditions": baseline_conditions,
        "comparative_claim_policy": comparative_policy,
        "rules": [
            "Use only valid_metric_claims in results tables.",
            "Do not report invalid_metric_claims as empirical findings.",
            "Do not claim percentage improvements without a verified baseline condition.",
            "Condition names are experimental regimes, not method names, unless labelled as baselines.",
        ],
    }


def metric_claim_invalid_reason(metric: str, value: float) -> str | None:
    """Return a deterministic reason when a metric-value pair is not publishable."""
    return _invalid_metric_reason(metric, value)


def _render_claim_ledger_instruction(ledger: Any) -> str:
    if not isinstance(ledger, dict):
        return ""
    valid = ledger.get("valid_metric_claims") or []
    invalid = ledger.get("invalid_metric_claims") or []
    baseline_conditions = ledger.get("baseline_conditions") or []
    policy = str(ledger.get("comparative_claim_policy") or "")

    lines = [
        "\n## CLAIM LEDGER (METRIC-SPECIFIC HARD CONSTRAINT)",
        "Use this ledger for tables and result claims; it is stricter than the raw number list.",
    ]
    if policy == "forbid_percentage_improvement_claims":
        lines.append(
            "- Do NOT claim percentage improvements, reductions, or superiority over a baseline; no verified baseline condition exists."
        )
    else:
        lines.append(
            "- Comparative claims are allowed only against verified baseline conditions: "
            + ", ".join(str(c) for c in baseline_conditions)
        )
    if valid:
        lines.append("- Valid metric claims for results tables:")
        for claim in valid[:60]:
            lines.append(
                "  - "
                f"{claim.get('condition')}/{claim.get('metric')}="
                f"{_format_number(claim.get('value'))}"
                f" (n={claim.get('count', 1)})"
            )
        if len(valid) > 60:
            lines.append(f"  - ... ({len(valid)} total valid metric claims)")
    if invalid:
        lines.append("- Invalid metric claims to omit or mark not evaluated:")
        for claim in invalid[:30]:
            reason = str(claim.get("reason") or "invalid_metric_claim")
            lines.append(
                "  - "
                f"{claim.get('condition')}/{claim.get('metric')}="
                f"{_format_number(claim.get('value'))}: "
                f"{_human_invalid_reason(reason)}"
            )
        if len(invalid) > 30:
            lines.append(f"  - ... ({len(invalid)} total invalid metric claims)")
    lines.append(
        "- Do not use experimental regime names as method names in tables; label the column as Condition or Regime."
    )
    return "\n".join(lines) + "\n"


def find_paper_contract_violations(
    paper: str,
    contract: dict[str, Any],
    *,
    tolerance: float = 0.01,
) -> list[str]:
    """Return deterministic contract violations in strict markdown sections."""
    allowed = {
        float(value)
        for value in contract.get("allowed_numbers", [])
        if isinstance(value, (int, float))
    }
    if not allowed:
        return []

    violations: list[str] = []
    in_strict = False
    for line_no, line in enumerate(paper.splitlines(), start=1):
        heading = _markdown_heading(line)
        if heading is not None:
            in_strict = heading in _STRICT_MARKDOWN_HEADINGS
            continue
        if not in_strict:
            continue
        check_line = _strip_citations_and_links(line)
        violations.extend(
            _claim_ledger_line_violations(
                check_line,
                contract,
                line_no=line_no,
                tolerance=tolerance,
            )
        )
        for match in _NUMBER_RE.finditer(check_line):
            num_str = match.group(1)
            try:
                value = float(num_str)
            except ValueError:
                continue
            if _is_allowed_number(value, allowed, tolerance):
                continue
            violations.append(
                f"line {line_no}: numeric value {num_str} is not allowed by paper_contract.json"
            )
    return violations


def sanitize_paper_contract_violations(
    paper: str,
    contract: dict[str, Any],
    *,
    tolerance: float = 0.01,
) -> tuple[str, dict[str, Any]]:
    """Replace unsupported numeric claims in strict sections with ``--``.

    This is a deterministic final fallback for local/cloud models that cannot
    fully obey the contract repair prompt.  It only touches strict empirical
    sections and leaves prose outside those sections unchanged.
    """
    allowed = {
        float(value)
        for value in contract.get("allowed_numbers", [])
        if isinstance(value, (int, float))
    }
    report: dict[str, Any] = {
        "replacement_count": 0,
        "replacements": [],
        "generated": _utcnow_iso(),
    }
    if not allowed:
        return paper, report

    sanitized_lines: list[str] = []
    in_strict = False
    for line_no, line in enumerate(paper.splitlines(), start=1):
        heading = _markdown_heading(line)
        if heading is not None:
            in_strict = heading in _STRICT_MARKDOWN_HEADINGS
            sanitized_lines.append(line)
            continue
        if not in_strict:
            sanitized_lines.append(line)
            continue

        replacements: list[tuple[tuple[int, int], str]] = []
        replacements.extend(
            _claim_ledger_line_replacements(
                line,
                contract,
                tolerance=tolerance,
            )
        )
        for match in _NUMBER_RE.finditer(line):
            num_str = match.group(1)
            try:
                value = float(num_str)
            except ValueError:
                continue
            if _is_allowed_number(value, allowed, tolerance):
                continue
            if not any(match.span(1) == span for span, _ in replacements):
                replacements.append((match.span(1), num_str))

        if not replacements:
            sanitized_lines.append(line)
            continue

        updated = line
        for (start, end), num_str in reversed(replacements):
            updated = updated[:start] + "--" + updated[end:]
            report["replacements"].append(
                {"line": line_no, "value": num_str, "replacement": "--"}
            )
        report["replacement_count"] += len(replacements)
        sanitized_lines.append(updated)

    return "\n".join(sanitized_lines), report


def repair_paper_contract_violations(
    paper: str,
    contract: dict[str, Any],
    *,
    llm: Any,
    stage_label: str,
    max_tokens: int = 12000,
) -> tuple[str, dict[str, Any]]:
    """Ask an LLM to repair contract violations, accepting only clean output.

    The original paper is returned unless the repaired draft has zero contract
    violations.  This keeps the retry from making the artifact worse when a
    local/cloud model ignores the constraint.
    """
    initial_violations = find_paper_contract_violations(paper, contract)
    report: dict[str, Any] = {
        "stage_label": stage_label,
        "repaired": False,
        "accepted": False,
        "initial_violation_count": len(initial_violations),
        "initial_violations": initial_violations[:50],
        "final_violation_count": len(initial_violations),
        "final_violations": initial_violations[:50],
        "generated": _utcnow_iso(),
    }
    if not initial_violations:
        report["accepted"] = True
        return paper, report

    instruction = render_paper_contract_instruction(contract)
    prompt = (
        f"{stage_label}: Repair the paper below so it satisfies paper_contract.json.\n\n"
        "You MUST preserve the paper's section structure and citations.\n"
        "You MUST remove, replace with `--`, or rewrite every unsupported numeric claim.\n"
        "Do NOT add new datasets, methods, baselines, p-values, confidence intervals, "
        "or hyperparameters unless they are explicitly allowed by the contract.\n"
        "Return ONLY the corrected markdown paper. Do not explain your changes.\n\n"
        f"{instruction}\n\n"
        "CONTRACT VIOLATIONS TO FIX:\n"
        + "\n".join(f"- {v}" for v in initial_violations[:50])
        + "\n\nPAPER TO REPAIR:\n"
        "```markdown\n"
        + paper
        + "\n```"
    )
    try:
        response = llm.chat(
            [{"role": "user", "content": prompt}],
            system=(
                "You are a strict scientific editor. You repair papers by "
                "removing unsupported numeric claims and preserving only "
                "contract-verified evidence."
            ),
            max_tokens=max_tokens,
            strip_thinking=True,
        )
        candidate = _strip_outer_markdown_fence(response.content).strip()
    except Exception as exc:  # noqa: BLE001
        report["error"] = str(exc)
        return paper, report

    final_violations = find_paper_contract_violations(candidate, contract)
    report.update(
        {
            "repaired": True,
            "final_violation_count": len(final_violations),
            "final_violations": final_violations[:50],
        }
    )
    if candidate and not final_violations:
        report["accepted"] = True
        return candidate, report
    return paper, report


def _collect_available_figures(run_dir: Path) -> list[str]:
    figures: list[str] = []
    seen: set[str] = set()
    for chart_dir in sorted(run_dir.glob("stage-14*/charts"), reverse=True):
        if not chart_dir.is_dir():
            continue
        for path in sorted(chart_dir.glob("*.png")):
            rel = f"charts/{path.name}"
            if rel not in seen:
                figures.append(rel)
                seen.add(rel)
        if figures:
            break
    return figures


def _load_experiment_summary(run_dir: Path) -> dict[str, Any]:
    candidates = [run_dir / "experiment_summary_best.json"]
    candidates.extend(sorted(run_dir.glob("stage-14*/experiment_summary.json"), reverse=True))
    for path in candidates:
        if not path.is_file():
            continue
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(loaded, dict):
            return loaded
    return {}


def _parse_metric_path(path: str) -> tuple[str, str] | None:
    parts = [part for part in path.split("/") if part]
    if len(parts) < 2:
        return None
    metric = parts[-1]
    condition_parts = parts[:-1]
    if condition_parts and condition_parts[-1].isdigit():
        # Skip raw per-seed claims; paper tables should use aggregate rows.
        return None
    condition = "/".join(condition_parts)
    if not condition:
        return None
    return condition, metric


def _to_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _invalid_metric_reason(metric: str, value: float) -> str | None:
    normalized = metric.lower().replace("-", "_")
    if normalized in {
        "average_prediction_set_size",
        "avg_prediction_set_size",
        "average_set_size",
        "avg_set_size",
        "set_size",
    } and value < 1.0:
        return "prediction_set_size_below_one"
    if normalized in {"coverage", "coverage_rate", "conditional_coverage"} and not (
        0.0 <= value <= 1.0
    ):
        return "coverage_out_of_unit_interval"
    if normalized in {"coverage_gap", "absolute_coverage_gap"} and not (
        0.0 <= value <= 1.0
    ):
        return "coverage_gap_out_of_unit_interval"
    return None


def _human_invalid_reason(reason: str) -> str:
    if reason == "prediction_set_size_below_one":
        return "prediction set size below 1.0 is invalid for classification prediction sets"
    if reason == "coverage_out_of_unit_interval":
        return "coverage must be between 0 and 1"
    if reason == "coverage_gap_out_of_unit_interval":
        return "coverage gap must be between 0 and 1"
    return reason.replace("_", " ")


def _looks_like_baseline(condition: str) -> bool:
    lowered = condition.lower()
    return any(
        token in lowered
        for token in ("baseline", "control", "split", "vanilla", "standard")
    )


def _claim_ledger_line_violations(
    line: str,
    contract: dict[str, Any],
    *,
    line_no: int,
    tolerance: float,
) -> list[str]:
    violations: list[str] = []
    for span, value, claim in _claim_ledger_line_matches(
        line,
        contract,
        tolerance=tolerance,
    ):
        _ = span
        metric = claim.get("metric", "metric")
        condition = claim.get("condition", "condition")
        reason = _human_invalid_reason(str(claim.get("reason", "invalid_metric_claim")))
        violations.append(
            f"line {line_no}: invalid metric claim {condition}/{metric}={value} ({reason})"
        )
    return violations


def _claim_ledger_line_replacements(
    line: str,
    contract: dict[str, Any],
    *,
    tolerance: float,
) -> list[tuple[tuple[int, int], str]]:
    return [
        (span, value)
        for span, value, _claim in _claim_ledger_line_matches(
            line,
            contract,
            tolerance=tolerance,
        )
    ]


def _claim_ledger_line_matches(
    line: str,
    contract: dict[str, Any],
    *,
    tolerance: float,
) -> list[tuple[tuple[int, int], str, dict[str, Any]]]:
    ledger = contract.get("claim_ledger")
    if not isinstance(ledger, dict):
        return []
    invalid = ledger.get("invalid_metric_claims") or []
    if not invalid:
        return []
    lowered = line.lower().replace("_", " ")
    matches: list[tuple[tuple[int, int], str, dict[str, Any]]] = []
    for claim in invalid:
        if not isinstance(claim, dict):
            continue
        metric = str(claim.get("metric") or "")
        if not any(phrase in lowered for phrase in _metric_text_phrases(metric)):
            continue
        claim_value = _to_float(claim.get("value"))
        if claim_value is None:
            continue
        for match in _NUMBER_RE.finditer(line):
            num_str = match.group(1)
            try:
                value = float(num_str)
            except ValueError:
                continue
            if _is_same_claim_value(value, claim_value, tolerance):
                matches.append((match.span(1), num_str, claim))
    return matches


def _metric_text_phrases(metric: str) -> set[str]:
    normalized = metric.lower().replace("_", " ").replace("-", " ").strip()
    phrases = {normalized}
    compact = normalized.replace(" ", "_")
    phrases.add(compact)
    if "set size" in normalized:
        phrases.update(
            {
                "set size",
                "prediction set size",
                "average set size",
                "average prediction set size",
            }
        )
    return {phrase for phrase in phrases if phrase}


def _is_same_claim_value(value: float, claim_value: float, tolerance: float) -> bool:
    if not math.isfinite(value) or not math.isfinite(claim_value):
        return False
    if claim_value == 0.0:
        return abs(value) < 1e-9
    return abs(value - claim_value) / max(abs(claim_value), 1e-9) <= tolerance


def _format_number(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.8g}"


def _markdown_heading(line: str) -> str | None:
    match = re.match(r"^\s*#{1,4}\s+(.+?)\s*$", line)
    if not match:
        return None
    heading = re.sub(r"[*_`]", "", match.group(1)).strip().lower()
    heading = re.sub(r"^\d+(?:\.\d+)*\.?\s+", "", heading)
    return heading


def _strip_citations_and_links(line: str) -> str:
    """Remove citation/link spans while preserving other numeric claims."""
    line = re.sub(r"\[[^\]]+\]\([^)]+\)", "", line)
    line = re.sub(
        r"\[[a-zA-Z][a-zA-Z0-9_-]*\d{4}[a-zA-Z0-9_-]*(?:\s*[,;]\s*[a-zA-Z][a-zA-Z0-9_-]*\d{4}[a-zA-Z0-9_-]*)*\]",
        "",
        line,
    )
    return line


def _strip_outer_markdown_fence(text: str) -> str:
    match = re.match(
        r"^\s*```(?:markdown|md|text)?\s*\n(.*?)\n```\s*$",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if match:
        return match.group(1)
    return text


def _is_allowed_number(
    value: float,
    allowed: set[float],
    tolerance: float,
) -> bool:
    if not math.isfinite(value):
        return False
    if 1900 <= value <= 2100 and value == int(value):
        return True
    if value == int(value) and abs(value) <= 5:
        return True
    for allowed_value in allowed:
        if allowed_value == 0.0:
            if abs(value) < 1e-9:
                return True
        elif abs(value - allowed_value) / max(abs(allowed_value), 1e-9) <= tolerance:
            return True
    return False
