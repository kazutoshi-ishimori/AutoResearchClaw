"""Phase 2 pipeline-wiring B: Stage 12 ``mode=agentic`` dispatch.

Before this wave, ``_execute_experiment_run`` had no branch for
``mode == "agentic"`` — pipeline runs would fall through every mode arm
and return ``DONE`` without writing ``run-1.json`` or invoking any agent.
These tests pin the new branch: the factory is consulted with the
BiomniConfig triplet, ``run_agent_session`` is invoked with the experiment
prompt, and an ``AgenticResult`` is materialised into ``runs/run-1.json``.

The factory and the AgenticSandbox are mocked — this is a wiring test, not
a Docker test.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from researchclaw.config import (
    AgenticConfig,
    BiomniConfig,
    ExperimentConfig,
    RCConfig,
)
from researchclaw.pipeline.stages import Stage, StageStatus


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    rd = tmp_path / "runs" / "run-x"
    rd.mkdir(parents=True, exist_ok=True)
    for d in ("stage-12",):
        (rd / d).mkdir(exist_ok=True)
    return rd


@pytest.fixture
def stage_dir(run_dir: Path) -> Path:
    sd = run_dir / "stage-12"
    sd.mkdir(parents=True, exist_ok=True)
    return sd


def _make_agentic_config(
    *,
    server_cmd: str = "",
    tool_allowlist: tuple[str, ...] = (),
    bridge_url: str = "",
) -> MagicMock:
    """Build a config MagicMock with ``experiment.mode='agentic'`` wired."""
    cfg = MagicMock(spec=RCConfig)
    exp = MagicMock(spec=ExperimentConfig)
    exp.mode = "agentic"
    exp.time_budget_sec = 1800
    exp.metric_key = "primary_metric"
    exp.agentic = AgenticConfig(
        image="rc-agentic:test",
        agent_install_cmd="",
        timeout_sec=1800,
    )
    exp.biomni = BiomniConfig(
        server_cmd=server_cmd,
        tool_allowlist=tool_allowlist,
        bridge_url=bridge_url,
        provenance_path="provenance.jsonl",
    )
    cfg.experiment = exp
    return cfg


def _make_agentic_result(
    *,
    returncode: int = 0,
    metrics: dict | None = None,
    elapsed_sec: float = 10.0,
    stdout: str = "{}",
    stderr: str = "",
) -> MagicMock:
    res = MagicMock()
    res.returncode = returncode
    # ``metrics is None`` → use the success default; an explicit ``{}`` from the
    # caller must NOT be coerced to non-empty (failed-result tests rely on it).
    res.metrics = {"auroc": 0.87} if metrics is None else metrics
    res.elapsed_sec = elapsed_sec
    res.stdout = stdout
    res.stderr = stderr
    res.output_files = []
    res.output_dirs = []
    res.agent_log = stdout
    res.steps_completed = 3
    return res


def _invoke(stage_dir: Path, run_dir: Path, config, *, agentic_result) -> object:
    """Invoke _execute_experiment_run with the AgenticSandbox factory mocked."""
    from researchclaw.pipeline.stage_impls._execution import (
        _execute_experiment_run,
    )

    sandbox = MagicMock()
    sandbox.run_agent_session.return_value = agentic_result

    with patch(
        "researchclaw.experiment.factory.create_agentic_sandbox",
        return_value=sandbox,
    ) as factory_spy, patch(
        "researchclaw.pipeline.stage_impls._execution._read_prior_artifact",
        return_value="",
    ), patch(
        "researchclaw.pipeline.stage_impls._execution._utcnow_iso",
        return_value="2026-06-27T10:00:00Z",
    ):
        sr = _execute_experiment_run(
            stage_dir, run_dir, config, MagicMock(),
            llm=None, prompts=None,
        )
        return sr, factory_spy, sandbox


def test_agentic_branch_invokes_factory_with_biomni_triplet(
    stage_dir: Path, run_dir: Path
) -> None:
    """The factory must be called with biomni_cfg + repo_root + ledger_dir."""
    cfg = _make_agentic_config(
        server_cmd="/opt/biomni/.venv/bin/python runner.py",
        tool_allowlist=("database.query_uniprot",),
    )
    result = _make_agentic_result()
    sr, factory_spy, sandbox = _invoke(
        stage_dir, run_dir, cfg, agentic_result=result
    )

    factory_spy.assert_called_once()
    kwargs = factory_spy.call_args.kwargs
    assert kwargs["biomni_cfg"] is cfg.experiment.biomni
    assert isinstance(kwargs["repo_root"], Path)
    assert kwargs["ledger_dir"] == run_dir
    sandbox.run_agent_session.assert_called_once()
    assert sr.status == StageStatus.DONE


def test_agentic_branch_writes_run1_json(
    stage_dir: Path, run_dir: Path
) -> None:
    """AgenticResult must be materialised into stage-12/runs/run-1.json."""
    cfg = _make_agentic_config()
    result = _make_agentic_result(metrics={"auroc": 0.91}, elapsed_sec=42.0)
    sr, _factory, _sandbox = _invoke(
        stage_dir, run_dir, cfg, agentic_result=result
    )

    run1 = stage_dir / "runs" / "run-1.json"
    assert run1.is_file(), "run-1.json must be written by the agentic branch"
    payload = json.loads(run1.read_text(encoding="utf-8"))
    assert payload["run_id"] == "run-1"
    assert payload["status"] == "completed"
    assert payload["metrics"] == {"auroc": 0.91}
    assert payload["elapsed_sec"] == 42.0
    assert sr.status == StageStatus.DONE


def test_agentic_branch_failed_result_marks_failed_status(
    stage_dir: Path, run_dir: Path
) -> None:
    """A non-zero returncode with no metrics must yield status='failed'."""
    cfg = _make_agentic_config()
    result = _make_agentic_result(
        returncode=-1, metrics={}, elapsed_sec=2.0, stderr="agent crashed"
    )
    sr, _f, _s = _invoke(stage_dir, run_dir, cfg, agentic_result=result)

    run1 = stage_dir / "runs" / "run-1.json"
    payload = json.loads(run1.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    # Stage 12 still returns a stage result (downstream decides what to do).
    assert sr.stage == Stage.EXPERIMENT_RUN


def test_agentic_branch_passes_workspace_and_timeout(
    stage_dir: Path, run_dir: Path
) -> None:
    """run_agent_session must receive a per-stage workspace and the timeout."""
    cfg = _make_agentic_config()
    result = _make_agentic_result()
    _sr, _f, sandbox = _invoke(
        stage_dir, run_dir, cfg, agentic_result=result
    )

    call = sandbox.run_agent_session.call_args
    workspace = call.kwargs.get("workspace") or (
        call.args[1] if len(call.args) > 1 else None
    )
    assert workspace is not None
    assert "agentic_workspace" in str(workspace)
    timeout = call.kwargs.get("timeout_sec")
    assert timeout == cfg.experiment.agentic.timeout_sec


# -- Phase 2 pipeline-wiring C: fabrication-guard for the agentic branch ----


def test_agentic_zero_metrics_fast_completion_returns_failed(
    stage_dir: Path, run_dir: Path
) -> None:
    """returncode=0 + empty metrics + elapsed < 30s → FAILED (misclassified)."""
    cfg = _make_agentic_config()
    result = _make_agentic_result(
        returncode=0, metrics={}, elapsed_sec=2.5, stdout="done"
    )
    sr, _f, _s = _invoke(stage_dir, run_dir, cfg, agentic_result=result)
    assert sr.status == StageStatus.FAILED
    assert "metric" in (sr.error or "").lower() or "fabricat" in (sr.error or "").lower()


def test_agentic_failed_with_zero_metrics_returns_failed(
    stage_dir: Path, run_dir: Path
) -> None:
    """A crashed agent with no metrics must NOT be marked DONE."""
    cfg = _make_agentic_config()
    result = _make_agentic_result(
        returncode=-1, metrics={}, elapsed_sec=5.0, stderr="agent crashed"
    )
    sr, _f, _s = _invoke(stage_dir, run_dir, cfg, agentic_result=result)
    assert sr.status == StageStatus.FAILED


def test_agentic_long_elapsed_with_zero_metrics_returns_failed(
    stage_dir: Path, run_dir: Path
) -> None:
    """Long runs without real metrics are still fabrication: FAILED."""
    cfg = _make_agentic_config()
    result = _make_agentic_result(
        returncode=0, metrics={}, elapsed_sec=900.0, stdout="done"
    )
    sr, _f, _s = _invoke(stage_dir, run_dir, cfg, agentic_result=result)
    assert sr.status == StageStatus.FAILED


def test_agentic_with_real_metrics_passes_guard(
    stage_dir: Path, run_dir: Path
) -> None:
    """Real numeric metrics with a normal completion must remain DONE."""
    cfg = _make_agentic_config()
    result = _make_agentic_result(
        returncode=0, metrics={"auroc": 0.91}, elapsed_sec=42.0
    )
    sr, _f, _s = _invoke(stage_dir, run_dir, cfg, agentic_result=result)
    assert sr.status == StageStatus.DONE


def test_agentic_nan_only_metrics_fail_guard(
    stage_dir: Path, run_dir: Path
) -> None:
    """A metric that is NaN/Inf only is not a real metric."""
    import math

    cfg = _make_agentic_config()
    result = _make_agentic_result(
        returncode=0, metrics={"auroc": math.nan}, elapsed_sec=60.0
    )
    sr, _f, _s = _invoke(stage_dir, run_dir, cfg, agentic_result=result)
    assert sr.status == StageStatus.FAILED


def test_non_agentic_modes_are_unaffected(stage_dir: Path, run_dir: Path) -> None:
    """Adding the agentic branch must not change ``mode='simulated'`` behaviour."""
    from researchclaw.pipeline.stage_impls._execution import (
        _execute_experiment_run,
    )

    cfg = MagicMock()
    cfg.experiment.mode = "simulated"
    cfg.experiment.time_budget_sec = 60
    cfg.experiment.metric_key = "primary_metric"

    with patch(
        "researchclaw.pipeline.stage_impls._execution._read_prior_artifact",
        return_value="",
    ), patch(
        "researchclaw.pipeline.stage_impls._execution._safe_json_loads",
        return_value={"tasks": []},
    ), patch(
        "researchclaw.pipeline.stage_impls._execution._utcnow_iso",
        return_value="2026-06-27T10:00:00Z",
    ):
        sr = _execute_experiment_run(
            stage_dir, run_dir, cfg, MagicMock(), llm=None, prompts=None
        )
    assert sr.status == StageStatus.DONE
