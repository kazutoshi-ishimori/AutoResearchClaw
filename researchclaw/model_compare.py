"""Build offline comparison reports across ARC model runs."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml


_SCORE_WEIGHTS = {
    "quality_score": 0.35,
    "pdf_score": 0.25,
    "citation_integrity": 0.20,
    "completion_ratio": 0.15,
    "latex_success": 0.05,
}


def collect_run_summary(run_dir: Path) -> dict[str, Any]:
    """Collect comparable metrics from a single ARC run directory."""
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    pipeline = _read_json(run_dir / "pipeline_summary.json")
    if not pipeline:
        raise ValueError(f"No pipeline_summary.json found in {run_dir}")

    config = _read_run_config(run_dir)
    llm_config = config.get("llm") if isinstance(config.get("llm"), dict) else {}
    research_config = (
        config.get("research") if isinstance(config.get("research"), dict) else {}
    )
    model = (
        llm_config.get("primary_model")
        or pipeline.get("model")
        or _infer_model_from_name(run_dir.name)
        or "unknown"
    )
    provider = llm_config.get("provider") or "unknown"

    stages_done = _as_number(pipeline.get("stages_done"))
    stages_executed = _as_number(pipeline.get("stages_executed"))
    completion_ratio = None
    if stages_done is not None and stages_executed and stages_executed > 0:
        completion_ratio = max(0.0, min(1.0, stages_done / stages_executed))

    quality = _read_json(_find_stage_file(run_dir, "stage-20", "quality_report.json"))
    pdf_review = _read_json(_find_stage_file(run_dir, "stage-22", "pdf_review.json"))
    citation = _read_json(
        _find_stage_file(run_dir, "stage-23", "verification_report.json")
    )
    compilation = _read_json(
        _find_stage_file(run_dir, "stage-22", "compilation_quality.json")
    )
    integrity = _read_json(
        _find_stage_file(run_dir, "stage-22", "paper_integrity_report.json")
    )
    latex = _read_json(_find_stage_file(run_dir, "stage-22", "latex_compile_report.json"))
    experiment = _read_json(run_dir / "experiment_summary_best.json") or _read_json(
        _find_stage_file(run_dir, "stage-14", "experiment_summary.json")
    )

    quality_score = _first_number(
        quality,
        ("score_1_to_10", "score", "quality_score", "overall_score"),
    )
    pdf_score = _first_number(pdf_review, ("overall_score", "mean_score", "score"))
    citation_summary = citation.get("summary") if isinstance(citation, dict) else {}
    citation_integrity = _first_number(
        citation_summary if isinstance(citation_summary, dict) else citation,
        ("integrity_score",),
    )
    citation_total = _first_number(
        citation_summary if isinstance(citation_summary, dict) else citation,
        ("total", "total_references"),
    )
    citation_verified = _first_number(
        citation_summary if isinstance(citation_summary, dict) else citation,
        ("verified", "verified_count"),
    )
    page_count = _first_number(compilation, ("page_count",))
    latex_success = _latex_success(run_dir, latex)
    primary_metric = _primary_metric(experiment)

    row: dict[str, Any] = {
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "run_id": pipeline.get("run_id", run_dir.name),
        "model": str(model),
        "provider": str(provider),
        "topic": research_config.get("topic") or pipeline.get("topic") or "",
        "generated": pipeline.get("generated"),
        "final_status": pipeline.get("final_status", "unknown"),
        "from_stage": pipeline.get("from_stage"),
        "final_stage": pipeline.get("final_stage"),
        "stages_done": stages_done,
        "stages_executed": stages_executed,
        "completion_ratio": completion_ratio,
        "quality_score": quality_score,
        "quality_verdict": quality.get("verdict"),
        "pdf_score": pdf_score,
        "pdf_decision": pdf_review.get("decision"),
        "citation_integrity": citation_integrity,
        "citation_total": citation_total,
        "citation_verified": citation_verified,
        "page_count": page_count,
        "latex_success": latex_success,
        "primary_metric": primary_metric,
        "paper_integrity_warnings": integrity.get("warnings", []),
        "removed_citation_keys": integrity.get("removed_citation_keys", []),
        "skipped_figure_paths": integrity.get("skipped_figure_paths", []),
    }
    score, coverage = _rank_score(row)
    row["rank_score"] = score
    row["score_coverage"] = coverage
    row["flags"] = _flags(row)
    return row


def build_model_comparison(run_dirs: list[Path]) -> dict[str, Any]:
    """Build a model-comparison payload from multiple run directories."""
    if not run_dirs:
        raise ValueError("At least one run directory is required")
    rows = [collect_run_summary(Path(p)) for p in run_dirs]
    rows.sort(
        key=lambda r: (
            float(r.get("rank_score") or 0.0),
            float(r.get("score_coverage") or 0.0),
            str(r.get("run_name") or ""),
        ),
        reverse=True,
    )
    for idx, row in enumerate(rows, start=1):
        row["rank"] = idx

    best = rows[0]
    return {
        "version": 1,
        "run_count": len(rows),
        "runs": rows,
        "recommendation": {
            "best_model": best.get("model"),
            "best_run": best.get("run_name"),
            "rank_score": best.get("rank_score"),
            "score_coverage": best.get("score_coverage"),
            "note": _recommendation_note(best),
        },
    }


def render_model_comparison_markdown(comparison: dict[str, Any]) -> str:
    """Render a comparison payload as Markdown."""
    rows = comparison.get("runs") if isinstance(comparison.get("runs"), list) else []
    rec = comparison.get("recommendation")
    recommendation = rec if isinstance(rec, dict) else {}
    lines = [
        "# ARC Model Comparison Report",
        "",
        "## Recommendation",
        "",
        (
            f"- Best model: **{recommendation.get('best_model', 'unknown')}** "
            f"on `{recommendation.get('best_run', 'unknown')}`"
        ),
        f"- Rank score: {_fmt(recommendation.get('rank_score'))}/10",
        f"- Score coverage: {_fmt_pct(recommendation.get('score_coverage'))}",
        f"- Note: {recommendation.get('note', '')}",
        "",
        "## Runs",
        "",
        "| Rank | Model | Run | Status | Stages | Quality | PDF | Citations | Pages | Score | Coverage | Flags |",
        "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        stages = _stage_cell(row)
        citations = _citation_cell(row)
        flags = ", ".join(row.get("flags") or []) or "-"
        lines.append(
            "| {rank} | {model} | `{run}` | {status} | {stages} | {quality} | "
            "{pdf} | {citations} | {pages} | {score} | {coverage} | {flags} |".format(
                rank=row.get("rank", ""),
                model=row.get("model", "unknown"),
                run=row.get("run_name", "unknown"),
                status=row.get("final_status", "unknown"),
                stages=stages,
                quality=_fmt(row.get("quality_score")),
                pdf=_fmt(row.get("pdf_score")),
                citations=citations,
                pages=_fmt_int(row.get("page_count")),
                score=_fmt(row.get("rank_score")),
                coverage=_fmt_pct(row.get("score_coverage")),
                flags=flags,
            )
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Rank score is a weighted summary of available quality, PDF, citation, completion, and LaTeX signals.",
            "- Score coverage shows how much of that scoring evidence was present. Compare low-coverage runs carefully.",
            "- Stage-only runs remain visible, but paper-generation metrics are absent until later stages complete.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_model_comparison(
    run_dirs: list[Path],
    output_dir: Path,
) -> dict[str, Path]:
    """Write ``model_comparison.json`` and ``model_comparison.md``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison = build_model_comparison(run_dirs)
    json_path = output_dir / "model_comparison.json"
    md_path = output_dir / "model_comparison.md"
    json_path.write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    md_path.write_text(
        render_model_comparison_markdown(comparison),
        encoding="utf-8",
    )
    return {"json": json_path, "markdown": md_path}


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _read_run_config(run_dir: Path) -> dict[str, Any]:
    candidates = sorted(run_dir.glob("config*.yaml")) + sorted(
        run_dir.glob("config*.yml")
    )
    for path in candidates:
        try:
            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            continue
        if isinstance(loaded, dict):
            return loaded
    return {}


def _find_stage_file(run_dir: Path, stage_name: str, filename: str) -> Path | None:
    direct = run_dir / stage_name / filename
    if direct.is_file():
        return direct
    for candidate in sorted(run_dir.glob(f"{stage_name}*"), reverse=True):
        path = candidate / filename
        if path.is_file():
            return path
    deliverable = run_dir / "deliverables" / filename
    if deliverable.is_file():
        return deliverable
    return None


def _infer_model_from_name(name: str) -> str | None:
    lower = name.lower()
    if "gpt-oss" in lower:
        return "gpt-oss:120b-cloud" if "cloud" in lower else "openai/gpt-oss-120b"
    if "gemma4" in lower:
        return "gemma4:31b-cloud" if "cloud" in lower else "google/gemma-4"
    if "qwen35" in lower or "qwen3.5" in lower:
        return "qwen3.5:397b-cloud" if "397" in lower else "qwen3.5:cloud"
    if "qwen36" in lower or "qwen3.6" in lower:
        return "qwen/qwen3.6-27b"
    if "mistral" in lower:
        return "mistral-large-3:675b-cloud"
    if "kimi" in lower:
        return "kimi-k2.6"
    if "nemotron" in lower:
        return "nemotron-3-super"
    return None


def _first_number(data: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    if not isinstance(data, dict):
        return None
    for key in keys:
        value = data.get(key)
        number = _as_number(value)
        if number is not None:
            return number
    return None


def _as_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _primary_metric(data: dict[str, Any]) -> float | None:
    metrics = data.get("metrics_summary") if isinstance(data, dict) else None
    if not isinstance(metrics, dict):
        return None
    primary = metrics.get("primary_metric")
    if isinstance(primary, dict):
        return _first_number(primary, ("mean", "min", "max"))
    if isinstance(primary, (int, float)):
        return float(primary)
    for value in metrics.values():
        if isinstance(value, dict):
            number = _first_number(value, ("mean", "min", "max"))
            if number is not None:
                return number
    return None


def _latex_success(run_dir: Path, latex_report: dict[str, Any]) -> bool | None:
    if "success" in latex_report:
        return bool(latex_report.get("success"))
    if (run_dir / "stage-22" / "paper.pdf").is_file():
        return True
    if (run_dir / "deliverables" / "paper.pdf").is_file():
        return True
    if (run_dir / "stage-22").is_dir():
        return False
    return None


def _rank_score(row: dict[str, Any]) -> tuple[float, float]:
    weighted = 0.0
    coverage = 0.0
    components = {
        "quality_score": _normalize_10(row.get("quality_score")),
        "pdf_score": _normalize_10(row.get("pdf_score")),
        "citation_integrity": _normalize_1(row.get("citation_integrity")),
        "completion_ratio": _normalize_1(row.get("completion_ratio")),
        "latex_success": _normalize_bool(row.get("latex_success")),
    }
    for key, weight in _SCORE_WEIGHTS.items():
        value = components.get(key)
        if value is None:
            continue
        weighted += value * weight
        coverage += weight
    if coverage <= 0:
        return 0.0, 0.0
    score = (weighted / coverage) * 10.0

    page_count = _as_number(row.get("page_count"))
    if page_count and page_count > 10:
        score -= min(1.0, (page_count - 10) * 0.15)
    if row.get("final_status") not in ("done", "completed", "success"):
        score -= 1.0
    score = max(0.0, min(10.0, score))
    return round(score, 2), round(coverage, 3)


def _normalize_10(value: Any) -> float | None:
    number = _as_number(value)
    if number is None:
        return None
    return max(0.0, min(1.0, number / 10.0))


def _normalize_1(value: Any) -> float | None:
    number = _as_number(value)
    if number is None:
        return None
    return max(0.0, min(1.0, number))


def _normalize_bool(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    return None


def _flags(row: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    if (row.get("score_coverage") or 0) < 0.6:
        flags.append("low-evidence")
    if row.get("final_status") not in ("done", "completed", "success"):
        flags.append("not-complete")
    if row.get("latex_success") is False:
        flags.append("latex-failed")
    page_count = _as_number(row.get("page_count"))
    if page_count and page_count > 10:
        flags.append("over-page-limit")
    if row.get("removed_citation_keys"):
        flags.append("citation-drift")
    if row.get("paper_integrity_warnings"):
        flags.append("integrity-warnings")
    return flags


def _recommendation_note(best: dict[str, Any]) -> str:
    coverage = best.get("score_coverage") or 0
    if coverage < 0.6:
        return "Best among supplied runs, but evidence coverage is low."
    if best.get("flags"):
        return "Best score among supplied runs, with flags to review."
    return "Best score among supplied runs with sufficient comparison evidence."


def _stage_cell(row: dict[str, Any]) -> str:
    done = _fmt_int(row.get("stages_done"))
    total = _fmt_int(row.get("stages_executed"))
    return f"{done}/{total}" if total != "-" else done


def _citation_cell(row: dict[str, Any]) -> str:
    integrity = row.get("citation_integrity")
    total = _fmt_int(row.get("citation_total"))
    verified = _fmt_int(row.get("citation_verified"))
    if integrity is None:
        return "-"
    if total != "-" and verified != "-":
        return f"{_fmt_pct(integrity)} ({verified}/{total})"
    return _fmt_pct(integrity)


def _fmt(value: Any) -> str:
    number = _as_number(value)
    if number is None:
        return "-"
    return f"{number:.2f}".rstrip("0").rstrip(".")


def _fmt_int(value: Any) -> str:
    number = _as_number(value)
    if number is None:
        return "-"
    return str(int(number))


def _fmt_pct(value: Any) -> str:
    number = _as_number(value)
    if number is None:
        return "-"
    return f"{number * 100:.0f}%"
