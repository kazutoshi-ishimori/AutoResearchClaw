"""Robustness: a wedged Docker engine must not hang the pipeline forever.

Regression for the 2026-07-02 incident where ``docker run -d`` on a
container-start-wedged Docker Desktop (WSL2 backend) blocked
``_start_container`` for 32 minutes: ``subprocess.run`` was called without a
``timeout``, so a non-returning engine turned into an unbounded hang instead
of a loud failure. Stage 12 never surfaced the problem — it just stalled.

This seam bounds the ``docker run`` invocation and converts a non-returning
engine into a prompt, actionable error (fail-loud is a feature).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from researchclaw.config import AgenticConfig
from researchclaw.experiment.agentic_sandbox import AgenticSandbox


def _make_sandbox(tmp_path: Path) -> AgenticSandbox:
    cfg = AgenticConfig(
        image="rc-agentic:test",
        agent_cli="claude",
        agent_install_cmd="",
        timeout_sec=60,
        memory_limit_mb=512,
        network_policy="full",
        mount_skills=False,
    )
    return AgenticSandbox(cfg, workdir=tmp_path / "wd", skills_dir=None)


def test_start_container_bounds_docker_run_with_timeout(tmp_path, monkeypatch) -> None:
    """The ``docker run -d`` subprocess call must carry a positive timeout."""
    seen: dict = {}

    def fake_run(argv, **kwargs):
        seen["kwargs"] = kwargs
        return subprocess.CompletedProcess(
            args=argv, returncode=0, stdout="", stderr=""
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    sb = _make_sandbox(tmp_path)
    sb._start_container("rc-agentic-test", tmp_path / "ws")

    timeout = seen.get("kwargs", {}).get("timeout")
    assert isinstance(timeout, (int, float)) and timeout > 0, (
        f"docker run must be bounded by a positive timeout, got {timeout!r}"
    )


def test_start_container_raises_clear_error_when_docker_run_hangs(
    tmp_path, monkeypatch
) -> None:
    """A non-returning ``docker run`` (TimeoutExpired) becomes a clear RuntimeError."""

    def fake_run(argv, **kwargs):
        raise subprocess.TimeoutExpired(cmd=argv, timeout=kwargs.get("timeout", 0))

    monkeypatch.setattr(subprocess, "run", fake_run)

    sb = _make_sandbox(tmp_path)
    with pytest.raises(RuntimeError) as ei:
        sb._start_container("rc-agentic-test", tmp_path / "ws")

    msg = str(ei.value).lower()
    assert "docker" in msg and (
        "timed out" in msg or "timeout" in msg or "unresponsive" in msg
    ), f"error should be actionable about the docker hang, got: {ei.value!r}"
