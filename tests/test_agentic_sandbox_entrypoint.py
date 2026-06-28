"""Phase 2 pipeline-wiring H: idle-loop container bypasses image ENTRYPOINT.

The idle-loop trick ``docker run -d ... <image> tail -f /dev/null`` assumes the
image has no ENTRYPOINT (or one that exec's its argv). Our domain images
inherit ``ENTRYPOINT ["/usr/local/bin/rc-entrypoint.sh"]`` from
``researchclaw/docker/entrypoint.sh`` which expects the first arg to be a
Python script under ``/workspace``. Without an override the container dies in
~1 second with ``python3: can't open file '/workspace/tail'``.

This seam forces ``--entrypoint tail`` so the keep-alive command runs verbatim,
independent of the image's stock entrypoint.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from researchclaw.config import AgenticConfig
from researchclaw.experiment.agentic_sandbox import AgenticSandbox


def _stub_run(capture: list):
    def fake_run(argv, **kwargs):
        capture.append(list(argv))
        return subprocess.CompletedProcess(
            args=argv, returncode=0, stdout="{}", stderr=""
        )

    return fake_run


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


def _docker_run_argv(calls: list) -> list[str]:
    runs = [c for c in calls if c[:2] == ["docker", "run"]]
    assert runs, "expected docker run to be invoked"
    return runs[0]


def test_start_container_overrides_entrypoint_to_tail(tmp_path, monkeypatch) -> None:
    """docker run argv contains ``--entrypoint tail`` before the image."""
    calls: list = []
    monkeypatch.setattr(subprocess, "run", _stub_run(calls))

    sb = _make_sandbox(tmp_path)
    sb.run_agent_session("hi", workspace=tmp_path / "ws", timeout_sec=10)

    argv = _docker_run_argv(calls)
    assert "--entrypoint" in argv, f"missing --entrypoint in {argv!r}"
    idx = argv.index("--entrypoint")
    assert argv[idx + 1] == "tail"


def test_entrypoint_override_appears_before_image(tmp_path, monkeypatch) -> None:
    """The ``--entrypoint`` flag must come before the image positional arg."""
    calls: list = []
    monkeypatch.setattr(subprocess, "run", _stub_run(calls))

    sb = _make_sandbox(tmp_path)
    sb.run_agent_session("hi", workspace=tmp_path / "ws", timeout_sec=10)

    argv = _docker_run_argv(calls)
    ep_idx = argv.index("--entrypoint")
    img_idx = argv.index("rc-agentic:test")
    assert ep_idx < img_idx, (
        f"--entrypoint must precede image: ep_idx={ep_idx} img_idx={img_idx}"
    )


def test_idle_loop_args_follow_image(tmp_path, monkeypatch) -> None:
    """After the image, the keep-alive args ``-f /dev/null`` must follow."""
    calls: list = []
    monkeypatch.setattr(subprocess, "run", _stub_run(calls))

    sb = _make_sandbox(tmp_path)
    sb.run_agent_session("hi", workspace=tmp_path / "ws", timeout_sec=10)

    argv = _docker_run_argv(calls)
    img_idx = argv.index("rc-agentic:test")
    tail_args = argv[img_idx + 1 :]
    assert tail_args == ["-f", "/dev/null"], f"got tail_args={tail_args!r}"
