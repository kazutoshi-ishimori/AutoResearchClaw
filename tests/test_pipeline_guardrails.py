from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch


def _make_config(tmp_path: Path):
    from researchclaw.config import RCConfig

    data = {
        "project": {"name": "guardrail-test", "mode": "docs-first"},
        "research": {
            "topic": "guardrail regression",
            "quality_threshold": 4.0,
            "graceful_degradation": False,
        },
        "runtime": {"timezone": "UTC"},
        "notifications": {"channel": "local"},
        "knowledge_base": {"backend": "markdown", "root": str(tmp_path / "kb")},
        "openclaw_bridge": {},
        "llm": {
            "provider": "openai-compatible",
            "base_url": "http://localhost:1234/v1",
            "api_key_env": "RC_TEST_KEY",
            "api_key": "inline-test-key",
        },
        "experiment": {
            "mode": "simulated",
            "metric_key": "primary_metric",
            "metric_direction": "maximize",
            "figure_agent": {"enabled": False},
            "repair": {"enabled": False},
        },
        "export": {"target_conference": "neurips_2025", "bib_file": "references"},
    }
    return RCConfig.from_dict(data, project_root=tmp_path, check_paths=False)


def test_pipeline_blocks_stage_16_when_experiment_metrics_are_empty(tmp_path: Path) -> None:
    from researchclaw.adapters import AdapterBundle
    from researchclaw.pipeline import runner as rc_runner
    from researchclaw.pipeline.executor import StageResult
    from researchclaw.pipeline.stages import Stage, StageStatus

    run_dir = tmp_path / "run"
    (run_dir / "stage-14").mkdir(parents=True)
    (run_dir / "stage-14" / "experiment_summary.json").write_text(
        json.dumps({"metrics_summary": {}, "total_runs": 1, "best_run": None}),
        encoding="utf-8",
    )
    config = _make_config(tmp_path)
    executed: list[Stage] = []

    def mock_execute_stage(stage: Stage, **kwargs) -> StageResult:
        _ = kwargs
        executed.append(stage)
        return StageResult(stage=stage, status=StageStatus.DONE, artifacts=("ok.md",))

    with patch.object(rc_runner, "execute_stage", side_effect=mock_execute_stage):
        results = rc_runner.execute_pipeline(
            run_dir=run_dir,
            run_id="empty-metrics",
            config=config,
            adapters=AdapterBundle(),
            from_stage=Stage.RESEARCH_DECISION,
            skip_noncritical=True,
        )

    assert Stage.PAPER_OUTLINE not in executed
    assert results[-1].stage is Stage.PAPER_OUTLINE
    assert results[-1].status is StageStatus.FAILED
    assert "empty" in (results[-1].error or "").lower()


def test_pipeline_blocks_stage_16_when_conditions_are_identical(tmp_path: Path) -> None:
    from researchclaw.adapters import AdapterBundle
    from researchclaw.pipeline import runner as rc_runner
    from researchclaw.pipeline.executor import StageResult
    from researchclaw.pipeline.stages import Stage, StageStatus

    run_dir = tmp_path / "run"
    (run_dir / "stage-14").mkdir(parents=True)
    (run_dir / "stage-14" / "experiment_summary.json").write_text(
        json.dumps(
            {
                "metrics_summary": {
                    "Baseline/primary_metric": {"mean": 1.0, "min": 1.0, "max": 1.0, "count": 1},
                    "Ablated/primary_metric": {"mean": 1.0, "min": 1.0, "max": 1.0, "count": 1},
                },
                "condition_summaries": {
                    "Baseline": {"metrics": {"primary_metric": 1.0}},
                    "Ablated": {"metrics": {"primary_metric": 1.0}},
                },
                "ablation_warnings": [
                    "ABLATION FAILURE: Conditions 'Baseline' and 'Ablated' produce identical outputs",
                    "ZERO VARIANCE: All 2 conditions have identical primary_metric",
                ],
            }
        ),
        encoding="utf-8",
    )
    config = _make_config(tmp_path)
    executed: list[Stage] = []

    def mock_execute_stage(stage: Stage, **kwargs) -> StageResult:
        _ = kwargs
        executed.append(stage)
        return StageResult(stage=stage, status=StageStatus.DONE, artifacts=("ok.md",))

    with patch.object(rc_runner, "execute_stage", side_effect=mock_execute_stage):
        results = rc_runner.execute_pipeline(
            run_dir=run_dir,
            run_id="identical-metrics",
            config=config,
            adapters=AdapterBundle(),
            from_stage=Stage.RESEARCH_DECISION,
            skip_noncritical=True,
        )

    assert Stage.PAPER_OUTLINE not in executed
    assert results[-1].stage is Stage.PAPER_OUTLINE
    assert results[-1].status is StageStatus.FAILED
    assert "identical" in (results[-1].error or "").lower()


def test_export_publish_fails_when_latex_compile_fails(tmp_path: Path) -> None:
    from researchclaw.adapters import AdapterBundle
    from researchclaw.pipeline.stage_impls._review_publish import _execute_export_publish
    from researchclaw.pipeline.stages import StageStatus
    from researchclaw.templates.compiler import CompileResult

    run_dir = tmp_path / "run"
    stage_dir = run_dir / "stage-22"
    stage_dir.mkdir(parents=True)
    (run_dir / "stage-19").mkdir(parents=True)
    (run_dir / "stage-19" / "paper_revised.md").write_text(
        "# Test Paper\n\n## Abstract\n\nA small paper.\n",
        encoding="utf-8",
    )
    config = _make_config(tmp_path)

    with patch(
        "researchclaw.templates.compiler.compile_latex",
        return_value=CompileResult(success=False, errors=["File ended while scanning use of \\underbrace"]),
    ):
        result = _execute_export_publish(
            stage_dir,
            run_dir,
            config,
            AdapterBundle(),
            llm=None,
            prompts=None,
        )

    assert result.status is StageStatus.FAILED
    assert "latex" in (result.error or "").lower()
    assert "paper.pdf" not in result.artifacts


def test_citation_verify_fails_when_references_bib_is_empty(tmp_path: Path) -> None:
    from researchclaw.adapters import AdapterBundle
    from researchclaw.pipeline.stage_impls._review_publish import _execute_citation_verify
    from researchclaw.pipeline.stages import StageStatus

    run_dir = tmp_path / "run"
    stage_dir = run_dir / "stage-23"
    stage_dir.mkdir(parents=True)
    (run_dir / "stage-22").mkdir(parents=True)
    (run_dir / "stage-22" / "paper_final.md").write_text("# Paper\n", encoding="utf-8")
    (run_dir / "stage-22" / "references.bib").write_text("   \n", encoding="utf-8")
    config = _make_config(tmp_path)

    result = _execute_citation_verify(
        stage_dir,
        run_dir,
        config,
        AdapterBundle(),
        llm=None,
        prompts=None,
    )

    report = json.loads((stage_dir / "verification_report.json").read_text(encoding="utf-8"))
    assert result.status is StageStatus.FAILED
    assert report["summary"]["integrity_score"] == 0.0
    assert "empty" in (result.error or "").lower()
