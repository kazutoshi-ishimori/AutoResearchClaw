from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

from researchclaw.llm.client import LLMResponse


class _FakeLLM:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMResponse:
        self.calls.append({"messages": messages, "kwargs": kwargs})
        idx = min(len(self.calls) - 1, len(self.responses) - 1)
        return LLMResponse(content=self.responses[idx], model="fake-model")


def _make_config(tmp_path: Path, export: dict[str, Any] | None = None):
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
        "export": export
        or {"target_conference": "neurips_2025", "bib_file": "references"},
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


def test_paper_outline_writes_paper_contract(tmp_path: Path) -> None:
    from researchclaw.adapters import AdapterBundle
    from researchclaw.pipeline.stage_impls._paper_writing import _execute_paper_outline

    run_dir = tmp_path / "run"
    stage14 = run_dir / "stage-14"
    stage16 = run_dir / "stage-16"
    stage16.mkdir(parents=True)
    stage14.mkdir(parents=True)
    (stage14 / "experiment_summary.json").write_text(
        json.dumps(
            {
                "metrics_summary": {
                    "Baseline/metric": {"mean": 0.72, "min": 0.71, "max": 0.73, "count": 3},
                    "Proposed/metric": {"mean": 0.81, "min": 0.80, "max": 0.82, "count": 3},
                },
                "condition_summaries": {
                    "Baseline": {"metrics": {"metric": 0.72}},
                    "Proposed": {"metrics": {"metric": 0.81}},
                },
                "best_run": {
                    "metrics": {
                        "Baseline/0/metric": 0.71,
                        "Baseline/1/metric": 0.72,
                        "Baseline/2/metric": 0.73,
                        "Proposed/0/metric": 0.80,
                        "Proposed/1/metric": 0.81,
                        "Proposed/2/metric": 0.82,
                        "primary_metric": 0.81,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    result = _execute_paper_outline(
        stage16,
        run_dir,
        _make_config(tmp_path),
        AdapterBundle(),
        llm=None,
        prompts=None,
    )

    contract = json.loads((stage16 / "paper_contract.json").read_text(encoding="utf-8"))
    assert result.status is not None
    assert "paper_contract.json" in result.artifacts
    assert contract["allowed_conditions"] == ["Baseline", "Proposed"]
    assert 0.81 in contract["allowed_numbers"]


def test_export_publish_sanitizes_paper_contract_violations_before_export_checks(
    tmp_path: Path,
) -> None:
    from researchclaw.adapters import AdapterBundle
    from researchclaw.pipeline.paper_contract import build_paper_contract
    from researchclaw.pipeline.stage_impls._review_publish import _execute_export_publish
    from researchclaw.templates.compiler import CompileResult

    run_dir = tmp_path / "run"
    stage14 = run_dir / "stage-14"
    stage16 = run_dir / "stage-16"
    stage19 = run_dir / "stage-19"
    stage22 = run_dir / "stage-22"
    for path in (stage14, stage16, stage19, stage22):
        path.mkdir(parents=True)
    summary = {
        "metrics_summary": {
            "Baseline/metric": {"mean": 0.72, "min": 0.71, "max": 0.73, "count": 3},
            "Proposed/metric": {"mean": 0.81, "min": 0.80, "max": 0.82, "count": 3},
        },
        "condition_summaries": {
            "Baseline": {"metrics": {"metric": 0.72}},
            "Proposed": {"metrics": {"metric": 0.81}},
        },
        "best_run": {"metrics": {"Baseline/0/metric": 0.72, "Proposed/0/metric": 0.81}},
    }
    (stage14 / "experiment_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (stage16 / "paper_contract.json").write_text(
        json.dumps(build_paper_contract(run_dir, metric_direction="maximize")),
        encoding="utf-8",
    )
    (stage19 / "paper_revised.md").write_text(
        "# Paper\n\n## Abstract\n\nShort.\n\n## Results\n\nThe proposed method reaches 0.95 accuracy.\n",
        encoding="utf-8",
    )
    (stage22 / "paper_contract_violations.json").write_text(
        '{"passed": false}',
        encoding="utf-8",
    )
    config = _make_config(tmp_path)

    with patch(
        "researchclaw.templates.compiler.compile_latex",
        return_value=CompileResult(success=True),
    ):
        result = _execute_export_publish(
            stage22,
            run_dir,
            config,
            AdapterBundle(),
            llm=None,
            prompts=None,
        )

    assert not (stage22 / "paper_contract_violations.json").exists()
    report = json.loads(
        (stage22 / "paper_contract_sanitization.json").read_text(encoding="utf-8")
    )
    assert report["passed"] is True
    assert report["replacement_count"] == 1
    assert "0.95" not in (stage22 / "paper_final.md").read_text(encoding="utf-8")
    assert "paper contract" not in (result.error or "").lower()


def test_export_publish_ignores_stale_empty_stage22_references_on_rerun(
    tmp_path: Path,
) -> None:
    from researchclaw.adapters import AdapterBundle
    from researchclaw.pipeline.stage_impls._review_publish import _execute_export_publish
    from researchclaw.templates.compiler import CompileResult

    run_dir = tmp_path / "run"
    stage04 = run_dir / "stage-04"
    stage19 = run_dir / "stage-19"
    stage22 = run_dir / "stage-22"
    for path in (stage04, stage19, stage22):
        path.mkdir(parents=True)
    (stage04 / "references.bib").write_text(
        "@article{smith2024test,\n"
        "  title={Test Paper},\n"
        "  author={Smith, Jane},\n"
        "  year={2024}\n"
        "}\n",
        encoding="utf-8",
    )
    (stage22 / "references.bib").write_text("", encoding="utf-8")
    (stage19 / "paper_revised.md").write_text(
        "# Paper\n\n"
        "## Abstract\n\nShort.\n\n"
        "## Introduction\n\nPrior work matters [cite_smith2024test].\n\n"
        "## Results\n\nNo numeric claims.\n",
        encoding="utf-8",
    )
    config = _make_config(tmp_path)

    with patch(
        "researchclaw.templates.compiler.compile_latex",
        return_value=CompileResult(success=True),
    ):
        _execute_export_publish(
            stage22,
            run_dir,
            config,
            AdapterBundle(),
            llm=None,
            prompts=None,
        )

    written_bib = (stage22 / "references.bib").read_text(encoding="utf-8")
    latex_md = (stage22 / "paper_final_latex.md").read_text(encoding="utf-8")
    assert "@article{smith2024test" in written_bib
    assert "\\cite{smith2024test}" in latex_md


def test_paper_draft_repairs_contract_violations_before_writing(tmp_path: Path) -> None:
    from researchclaw.adapters import AdapterBundle
    from researchclaw.pipeline.paper_contract import build_paper_contract
    from researchclaw.pipeline.stage_impls._paper_writing import _execute_paper_draft

    run_dir = tmp_path / "run"
    stage12_runs = run_dir / "stage-12" / "runs"
    stage14 = run_dir / "stage-14"
    stage16 = run_dir / "stage-16"
    stage17 = run_dir / "stage-17"
    for path in (stage12_runs, stage14, stage16, stage17):
        path.mkdir(parents=True)
    (stage12_runs / "run.json").write_text(
        json.dumps({"status": "completed", "metrics": {"primary_metric": 0.81}}),
        encoding="utf-8",
    )
    summary = {
        "metrics_summary": {
            "Baseline/metric": {"mean": 0.72, "min": 0.71, "max": 0.73, "count": 3},
            "Proposed/metric": {"mean": 0.81, "min": 0.80, "max": 0.82, "count": 3},
        },
        "condition_summaries": {
            "Baseline": {"metrics": {"metric": 0.72}},
            "Proposed": {"metrics": {"metric": 0.81}},
        },
        "best_run": {"metrics": {"Baseline/0/metric": 0.72, "Proposed/0/metric": 0.81}},
    }
    (stage14 / "experiment_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (stage16 / "paper_contract.json").write_text(
        json.dumps(build_paper_contract(run_dir, metric_direction="maximize")),
        encoding="utf-8",
    )
    (stage16 / "outline.md").write_text("# Outline\n", encoding="utf-8")
    bad_draft = "# Paper\n\n## Results\n\nThe proposed method reaches 0.95 accuracy.\n"
    fixed_draft = "# Paper\n\n## Results\n\nThe proposed method reaches 0.81 accuracy.\n"
    llm = _FakeLLM([fixed_draft])

    with patch(
        "researchclaw.pipeline.stage_impls._paper_writing._write_paper_sections",
        return_value=bad_draft,
    ):
        _execute_paper_draft(
            stage17,
            run_dir,
            _make_config(tmp_path),
            AdapterBundle(),
            llm=llm,  # type: ignore[arg-type]
            prompts=None,
        )

    written = (stage17 / "paper_draft.md").read_text(encoding="utf-8")
    report = json.loads((stage17 / "draft_contract_repair.json").read_text(encoding="utf-8"))
    assert "0.81" in written
    assert "0.95" not in written
    assert report["accepted"] is True


def test_paper_revision_repairs_contract_violations_before_writing(tmp_path: Path) -> None:
    from researchclaw.adapters import AdapterBundle
    from researchclaw.pipeline.paper_contract import build_paper_contract
    from researchclaw.pipeline.stage_impls._review_publish import _execute_paper_revision

    run_dir = tmp_path / "run"
    stage14 = run_dir / "stage-14"
    stage16 = run_dir / "stage-16"
    stage17 = run_dir / "stage-17"
    stage18 = run_dir / "stage-18"
    stage19 = run_dir / "stage-19"
    for path in (stage14, stage16, stage17, stage18, stage19):
        path.mkdir(parents=True)
    summary = {
        "metrics_summary": {
            "Baseline/metric": {"mean": 0.72, "min": 0.71, "max": 0.73, "count": 3},
            "Proposed/metric": {"mean": 0.81, "min": 0.80, "max": 0.82, "count": 3},
        },
        "condition_summaries": {
            "Baseline": {"metrics": {"metric": 0.72}},
            "Proposed": {"metrics": {"metric": 0.81}},
        },
        "best_run": {"metrics": {"Baseline/0/metric": 0.72, "Proposed/0/metric": 0.81}},
    }
    (stage14 / "experiment_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (stage16 / "paper_contract.json").write_text(
        json.dumps(build_paper_contract(run_dir, metric_direction="maximize")),
        encoding="utf-8",
    )
    (stage17 / "paper_draft.md").write_text(
        "# Paper\n\n## Results\n\nThe proposed method reaches 0.81 accuracy.\n",
        encoding="utf-8",
    )
    (stage18 / "reviews.md").write_text("Please revise.", encoding="utf-8")
    bad_revision = "# Paper\n\n## Results\n\nThe proposed method reaches 0.95 accuracy.\n"
    fixed_revision = "# Paper\n\n## Results\n\nThe proposed method reaches 0.81 accuracy.\n"
    llm = _FakeLLM([bad_revision, fixed_revision])

    _execute_paper_revision(
        stage19,
        run_dir,
        _make_config(tmp_path),
        AdapterBundle(),
        llm=llm,  # type: ignore[arg-type]
        prompts=None,
    )

    written = (stage19 / "paper_revised.md").read_text(encoding="utf-8")
    report = json.loads((stage19 / "revision_contract_repair.json").read_text(encoding="utf-8"))
    assert "0.81" in written
    assert "0.95" not in written
    assert report["accepted"] is True


def test_paper_revision_strips_new_citations_not_in_references(tmp_path: Path) -> None:
    from researchclaw.adapters import AdapterBundle
    from researchclaw.pipeline.stage_impls._review_publish import _execute_paper_revision

    run_dir = tmp_path / "run"
    stage04 = run_dir / "stage-04"
    stage16 = run_dir / "stage-16"
    stage17 = run_dir / "stage-17"
    stage18 = run_dir / "stage-18"
    stage19 = run_dir / "stage-19"
    for path in (stage04, stage16, stage17, stage18, stage19):
        path.mkdir(parents=True)
    (stage04 / "references.bib").write_text(
        "@article{smith2024test,\n"
        "  title={Test Paper},\n"
        "  author={Smith, Jane},\n"
        "  year={2024}\n"
        "}\n",
        encoding="utf-8",
    )
    (stage16 / "paper_contract.json").write_text(
        json.dumps({"allowed_numbers": [], "available_figures": []}),
        encoding="utf-8",
    )
    (stage17 / "paper_draft.md").write_text(
        "# Paper\n\n"
        "## Abstract\n\nA short study.\n\n"
        "## Introduction\n\nPrior work matters [smith2024test].\n\n"
        "## Results\n\nNo new numeric claims.\n",
        encoding="utf-8",
    )
    (stage18 / "reviews.md").write_text("Tighten citations.", encoding="utf-8")
    llm = _FakeLLM(
        [
            "# Paper\n\n"
            "## Abstract\n\nA short study.\n\n"
            "## Introduction\n\nPrior work matters [smith2024test, ghost2025fake].\n\n"
            "## Results\n\nNo new numeric claims.\n"
        ]
    )

    _execute_paper_revision(
        stage19,
        run_dir,
        _make_config(tmp_path),
        AdapterBundle(),
        llm=llm,  # type: ignore[arg-type]
        prompts=None,
    )

    written = (stage19 / "paper_revised.md").read_text(encoding="utf-8")
    report = json.loads((stage19 / "revision_integrity_report.json").read_text(encoding="utf-8"))
    assert "[smith2024test]" in written
    assert "ghost2025fake" not in written
    assert report["removed_citation_keys"] == ["ghost2025fake"]


def test_paper_revision_falls_back_when_required_sections_disappear(tmp_path: Path) -> None:
    from researchclaw.adapters import AdapterBundle
    from researchclaw.pipeline.stage_impls._review_publish import _execute_paper_revision

    run_dir = tmp_path / "run"
    stage16 = run_dir / "stage-16"
    stage17 = run_dir / "stage-17"
    stage18 = run_dir / "stage-18"
    stage19 = run_dir / "stage-19"
    for path in (stage16, stage17, stage18, stage19):
        path.mkdir(parents=True)
    (stage16 / "paper_contract.json").write_text(
        json.dumps({"allowed_numbers": [], "available_figures": []}),
        encoding="utf-8",
    )
    draft = (
        "# Paper\n\n"
        "## Abstract\n\nA short study.\n\n"
        "## Introduction\n\nContext.\n\n"
        "## Methods\n\nThe procedure is described.\n\n"
        "## Results\n\nThe evaluation is described without unsupported numbers.\n\n"
        "## Conclusion\n\nThe paper concludes.\n"
    )
    (stage17 / "paper_draft.md").write_text(draft, encoding="utf-8")
    (stage18 / "reviews.md").write_text("Improve flow.", encoding="utf-8")
    llm = _FakeLLM(
        [
            "# Paper\n\n"
            "## Abstract\n\nA short study.\n\n"
            "## Introduction\n\nContext improved.\n\n"
            "## Conclusion\n\nThe paper concludes.\n"
        ]
    )

    _execute_paper_revision(
        stage19,
        run_dir,
        _make_config(tmp_path),
        AdapterBundle(),
        llm=llm,  # type: ignore[arg-type]
        prompts=None,
    )

    written = (stage19 / "paper_revised.md").read_text(encoding="utf-8")
    report = json.loads((stage19 / "revision_integrity_report.json").read_text(encoding="utf-8"))
    assert "## Methods" in written
    assert "## Results" in written
    assert report["fallback_to_draft"] is True
    assert report["missing_sections"] == ["methods", "results"]


def test_paper_revision_allows_compact_revision_within_configured_target(
    tmp_path: Path,
) -> None:
    from researchclaw.adapters import AdapterBundle
    from researchclaw.pipeline.stage_impls._review_publish import _execute_paper_revision

    run_dir = tmp_path / "run"
    stage16 = run_dir / "stage-16"
    stage17 = run_dir / "stage-17"
    stage18 = run_dir / "stage-18"
    stage19 = run_dir / "stage-19"
    for path in (stage16, stage17, stage18, stage19):
        path.mkdir(parents=True)
    (stage16 / "paper_contract.json").write_text(
        json.dumps({"allowed_numbers": [], "available_figures": []}),
        encoding="utf-8",
    )
    draft = (
        "# Paper\n\n"
        "## Abstract\n\n" + " ".join(["draft"] * 120) + "\n\n"
        "## Introduction\n\n" + " ".join(["draft"] * 260) + "\n\n"
        "## Method\n\n" + " ".join(["draft"] * 260) + "\n\n"
        "## Results\n\n" + " ".join(["draft"] * 260) + "\n\n"
        "## Conclusion\n\n" + " ".join(["draft"] * 120) + "\n"
    )
    revised = (
        "# Paper\n\n"
        "## Abstract\n\n" + " ".join(["revised"] * 100) + "\n\n"
        "## Introduction\n\n" + " ".join(["revised"] * 180) + "\n\n"
        "## Method\n\n" + " ".join(["revised"] * 180) + "\n\n"
        "## Results\n\n" + " ".join(["revised"] * 180) + "\n\n"
        "## Conclusion\n\n" + " ".join(["revised"] * 100) + "\n"
    )
    (stage17 / "paper_draft.md").write_text(draft, encoding="utf-8")
    (stage18 / "reviews.md").write_text("Compress for page limit.", encoding="utf-8")
    llm = _FakeLLM([revised])

    _execute_paper_revision(
        stage19,
        run_dir,
        _make_config(
            tmp_path,
            export={
                "target_conference": "neurips_2025",
                "bib_file": "references",
                "page_limit": 8,
                "main_body_word_min": 600,
                "main_body_word_max": 900,
            },
        ),
        AdapterBundle(),
        llm=llm,  # type: ignore[arg-type]
        prompts=None,
    )

    assert len(llm.calls) == 1
    written = (stage19 / "paper_revised.md").read_text(encoding="utf-8")
    assert "revised" in written


def test_section_integrity_accepts_numbered_conference_headings() -> None:
    from researchclaw.pipeline.paper_integrity import section_integrity_warnings

    paper = (
        "# Paper\n\n"
        "## Abstract\n\nA short study.\n\n"
        "## I. Introduction\n\nContext.\n\n"
        "## III. Method\n\nMethod text.\n\n"
        "## V. Results\n\nResult text.\n\n"
        "## VII. Limitations and Conclusion\n\nConclusion text.\n"
    )

    assert section_integrity_warnings(paper) == []


def test_export_publish_strips_new_citations_without_resolving_when_contract_exists(
    tmp_path: Path,
) -> None:
    from researchclaw.adapters import AdapterBundle
    from researchclaw.pipeline.stage_impls._review_publish import _execute_export_publish
    from researchclaw.templates.compiler import CompileResult

    run_dir = tmp_path / "run"
    stage04 = run_dir / "stage-04"
    stage16 = run_dir / "stage-16"
    stage19 = run_dir / "stage-19"
    stage22 = run_dir / "stage-22"
    for path in (stage04, stage16, stage19, stage22):
        path.mkdir(parents=True)
    (stage04 / "references.bib").write_text(
        "@article{smith2024test,\n"
        "  title={Test Paper},\n"
        "  author={Smith, Jane},\n"
        "  year={2024}\n"
        "}\n",
        encoding="utf-8",
    )
    (stage16 / "paper_contract.json").write_text(
        json.dumps({"allowed_numbers": [], "available_figures": []}),
        encoding="utf-8",
    )
    (stage19 / "paper_revised.md").write_text(
        "# Paper\n\n"
        "## Abstract\n\nA short study.\n\n"
        "## Introduction\n\nKnown work [smith2024test, ghost2025fake].\n\n"
        "## Results\n\nNo new numeric claims.\n",
        encoding="utf-8",
    )

    with (
        patch("researchclaw.templates.compiler.compile_latex", return_value=CompileResult(success=True)),
        patch(
            "researchclaw.pipeline.stage_impls._review_publish._resolve_missing_citations",
            side_effect=AssertionError("Stage 22 must not synthesize new bibliography entries"),
        ),
    ):
        _execute_export_publish(
            stage22,
            run_dir,
            _make_config(tmp_path),
            AdapterBundle(),
            llm=None,
            prompts=None,
        )

    written = (stage22 / "paper_final.md").read_text(encoding="utf-8")
    report = json.loads((stage22 / "paper_integrity_report.json").read_text(encoding="utf-8"))
    assert "ghost2025fake" not in written
    assert "\\cite{smith2024test}" in (stage22 / "paper_final_latex.md").read_text(encoding="utf-8")
    assert report["removed_citation_keys"] == ["ghost2025fake"]


def test_export_publish_injects_only_contract_available_figures(tmp_path: Path) -> None:
    from researchclaw.adapters import AdapterBundle
    from researchclaw.pipeline.stage_impls._review_publish import _execute_export_publish
    from researchclaw.templates.compiler import CompileResult

    run_dir = tmp_path / "run"
    stage14_charts = run_dir / "stage-14" / "charts"
    stage16 = run_dir / "stage-16"
    stage19 = run_dir / "stage-19"
    stage22 = run_dir / "stage-22"
    for path in (stage14_charts, stage16, stage19, stage22):
        path.mkdir(parents=True)
    (stage14_charts / "allowed_result.png").write_bytes(b"png")
    (stage14_charts / "disallowed_result.png").write_bytes(b"png")
    (stage16 / "paper_contract.json").write_text(
        json.dumps(
            {
                "allowed_numbers": [],
                "available_figures": ["charts/allowed_result.png"],
            }
        ),
        encoding="utf-8",
    )
    (stage19 / "paper_revised.md").write_text(
        "# Paper\n\n"
        "## Abstract\n\nA short study.\n\n"
        "## Methods\n\nMethod text.\n\n"
        "## Results\n\nResult text.\n\n"
        "## Conclusion\n\nConclusion text.\n",
        encoding="utf-8",
    )

    with patch(
        "researchclaw.templates.compiler.compile_latex",
        return_value=CompileResult(success=True),
    ):
        _execute_export_publish(
            stage22,
            run_dir,
            _make_config(tmp_path),
            AdapterBundle(),
            llm=None,
            prompts=None,
        )

    written = (stage22 / "paper_final.md").read_text(encoding="utf-8")
    report = json.loads((stage22 / "paper_integrity_report.json").read_text(encoding="utf-8"))
    assert "charts/allowed_result.png" in written
    assert "charts/disallowed_result.png" not in written
    assert report["skipped_figure_paths"] == ["charts/disallowed_result.png"]


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
