from __future__ import annotations

import json
from pathlib import Path

from researchclaw.model_compare import (
    build_model_comparison,
    collect_run_summary,
    render_model_comparison_markdown,
    write_model_comparison,
)


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _make_run(
    root: Path,
    name: str,
    *,
    model: str,
    final_status: str = "done",
    stages_done: int = 23,
    quality_score: float | None = None,
    pdf_score: float | None = None,
    citation_integrity: float | None = None,
    page_count: int | None = None,
) -> Path:
    run_dir = root / name
    run_dir.mkdir(parents=True)
    (run_dir / f"config.{name}.yaml").write_text(
        "research:\n"
        "  topic: Small tabular calibration\n"
        "llm:\n"
        f"  primary_model: {model!r}\n",
        encoding="utf-8",
    )
    _write_json(
        run_dir / "pipeline_summary.json",
        {
            "run_id": name,
            "stages_executed": 23,
            "stages_done": stages_done,
            "stages_failed": 0 if final_status == "done" else 1,
            "final_status": final_status,
            "generated": "2026-05-20T00:00:00+00:00",
        },
    )
    _write_json(
        run_dir / "experiment_summary_best.json",
        {
            "metrics_summary": {
                "primary_metric": {"mean": 0.81, "count": 3},
                "accuracy": {"mean": 0.81, "count": 3},
            },
            "total_runs": 3,
        },
    )
    if quality_score is not None:
        _write_json(
            run_dir / "stage-20" / "quality_report.json",
            {"score_1_to_10": quality_score, "verdict": "accept"},
        )
    if pdf_score is not None:
        _write_json(
            run_dir / "stage-22" / "pdf_review.json",
            {"overall_score": pdf_score, "decision": "reject"},
        )
    if page_count is not None:
        _write_json(
            run_dir / "stage-22" / "compilation_quality.json",
            {"page_count": page_count, "warnings": []},
        )
    if citation_integrity is not None:
        _write_json(
            run_dir / "stage-23" / "verification_report.json",
            {
                "summary": {
                    "total": 10,
                    "verified": int(10 * citation_integrity),
                    "suspicious": 0,
                    "hallucinated": 0,
                    "integrity_score": citation_integrity,
                }
            },
        )
    return run_dir


def test_collect_run_summary_reads_model_and_quality_artifacts(tmp_path: Path) -> None:
    run_dir = _make_run(
        tmp_path,
        "run-gpt",
        model="gpt-oss:120b-cloud",
        quality_score=7.5,
        pdf_score=4.0,
        citation_integrity=1.0,
        page_count=12,
    )

    summary = collect_run_summary(run_dir)

    assert summary["run_name"] == "run-gpt"
    assert summary["model"] == "gpt-oss:120b-cloud"
    assert summary["quality_score"] == 7.5
    assert summary["pdf_score"] == 4.0
    assert summary["citation_integrity"] == 1.0
    assert summary["page_count"] == 12
    assert summary["primary_metric"] == 0.81
    assert summary["score_coverage"] > 0.7


def test_build_model_comparison_ranks_runs_and_preserves_missing_coverage(tmp_path: Path) -> None:
    weaker = _make_run(
        tmp_path,
        "run-weaker",
        model="gemma4:31b-cloud",
        stages_done=23,
        quality_score=5.0,
        pdf_score=3.0,
        citation_integrity=0.8,
    )
    stronger = _make_run(
        tmp_path,
        "run-stronger",
        model="gpt-oss:120b-cloud",
        stages_done=23,
        quality_score=8.0,
        pdf_score=7.0,
        citation_integrity=1.0,
    )
    stage9_only = _make_run(
        tmp_path,
        "run-stage9",
        model="qwen/qwen3.6-27b",
        stages_done=9,
    )

    comparison = build_model_comparison([weaker, stronger, stage9_only])

    rows = comparison["runs"]
    assert rows[0]["model"] == "gpt-oss:120b-cloud"
    assert rows[0]["rank"] == 1
    assert rows[2]["model"] == "qwen/qwen3.6-27b"
    assert rows[2]["score_coverage"] < rows[0]["score_coverage"]
    assert comparison["recommendation"]["best_model"] == "gpt-oss:120b-cloud"


def test_render_model_comparison_markdown_includes_table_and_warnings(tmp_path: Path) -> None:
    complete = _make_run(
        tmp_path,
        "run-complete",
        model="gpt-oss:120b-cloud",
        quality_score=8.0,
        pdf_score=7.0,
        citation_integrity=1.0,
    )
    partial = _make_run(
        tmp_path,
        "run-partial",
        model="qwen/qwen3.6-27b",
        stages_done=9,
    )

    markdown = render_model_comparison_markdown(
        build_model_comparison([partial, complete])
    )

    assert "# ARC Model Comparison Report" in markdown
    assert "| Rank | Model | Run | Status |" in markdown
    assert "gpt-oss:120b-cloud" in markdown
    assert "qwen/qwen3.6-27b" in markdown
    assert "score coverage" in markdown.lower()


def test_write_model_comparison_writes_json_and_markdown(tmp_path: Path) -> None:
    run_dir = _make_run(
        tmp_path,
        "run-gpt",
        model="gpt-oss:120b-cloud",
        quality_score=8.0,
        pdf_score=7.0,
        citation_integrity=1.0,
    )
    output_dir = tmp_path / "comparison"

    written = write_model_comparison([run_dir], output_dir)

    assert written["json"].name == "model_comparison.json"
    assert written["markdown"].name == "model_comparison.md"
    assert "gpt-oss:120b-cloud" in written["markdown"].read_text(encoding="utf-8")
    payload = json.loads(written["json"].read_text(encoding="utf-8"))
    assert payload["runs"][0]["model"] == "gpt-oss:120b-cloud"


def test_cli_model_compare_writes_report(tmp_path: Path, capsys) -> None:
    from researchclaw.cli import main

    run_dir = _make_run(
        tmp_path,
        "run-gpt",
        model="gpt-oss:120b-cloud",
        quality_score=8.0,
        pdf_score=7.0,
        citation_integrity=1.0,
    )
    output_dir = tmp_path / "comparison"

    code = main(
        [
            "model-compare",
            "--runs",
            str(run_dir),
            "--output",
            str(output_dir),
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "model_comparison.md" in captured.out
    assert (output_dir / "model_comparison.json").exists()
    assert (output_dir / "model_comparison.md").exists()
