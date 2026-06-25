"""Phase 2 ②-b: AgenticSandbox ↔ host-bridge lifecycle integration.

The agentic experiment flow must:

* enter the bridge context manager *before* the container starts;
* surface the bridge URL into the container via env (rewriting 127.0.0.1 to
  ``host.docker.internal`` so the container can reach the host);
* exit the lifecycle once the session ends (success or failure).

When no ``bridge_lifecycle`` is supplied the legacy behaviour must be a
zero-side-effect no-op.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from researchclaw.config import AgenticConfig
from researchclaw.experiment.agentic_sandbox import AgenticSandbox


class _FakeBridge:
    """Context manager standing in for :class:`BiomniBridgeProcess`."""

    def __init__(self, url: str = "http://127.0.0.1:54321") -> None:
        self.url = url
        self.entered = False
        self.exited = False

    def __enter__(self) -> str:
        self.entered = True
        return self.url

    def __exit__(self, exc_type, exc, tb) -> None:
        self.exited = True


def _stub_run(capture: list):
    """Return a stand-in for ``subprocess.run`` that records argv."""

    def fake_run(argv, **kwargs):
        capture.append((list(argv), kwargs))
        return subprocess.CompletedProcess(
            args=argv, returncode=0, stdout="{}", stderr=""
        )

    return fake_run


def _make_sandbox(
    tmp_path: Path,
    bridge=None,
    *,
    biomni_tools: tuple[str, ...] = (),
    biomni_client_path: Path | None = None,
) -> AgenticSandbox:
    cfg = AgenticConfig(
        image="rc-agentic:test",
        agent_cli="claude",
        agent_install_cmd="",  # skip the install step in unit tests
        timeout_sec=60,
        memory_limit_mb=512,
        network_policy="full",
        mount_skills=False,
    )
    return AgenticSandbox(
        cfg,
        workdir=tmp_path / "wd",
        skills_dir=None,
        bridge_lifecycle=bridge,
        biomni_tools=biomni_tools,
        biomni_client_path=biomni_client_path,
    )


def test_no_bridge_keeps_legacy_behaviour(tmp_path, monkeypatch) -> None:
    """Without a bridge, no BIOMNI_BRIDGE_URL leaks into the container."""
    calls: list = []
    monkeypatch.setattr(subprocess, "run", _stub_run(calls))

    sb = _make_sandbox(tmp_path)
    sb.run_agent_session("hello", workspace=tmp_path / "ws", timeout_sec=10)

    docker_run_calls = [c for c in calls if c[0][:2] == ["docker", "run"]]
    assert docker_run_calls, "expected docker run to be invoked"
    argv = docker_run_calls[0][0]
    assert not any("BIOMNI_BRIDGE_URL" in a for a in argv)
    # And no --add-host plumbing without a bridge.
    assert "--add-host" not in argv


def test_bridge_lifecycle_entered_before_container_and_url_passed(
    tmp_path, monkeypatch
) -> None:
    """The bridge must be live *before* docker run and its URL must reach env."""
    calls: list = []
    bridge = _FakeBridge(url="http://127.0.0.1:60123")

    seen_order: list[str] = []

    def fake_run(argv, **kwargs):
        if argv[:2] == ["docker", "run"]:
            seen_order.append("docker_run")
            # The bridge must already be entered by the time the container starts.
            assert bridge.entered is True
            assert bridge.exited is False
        calls.append((list(argv), kwargs))
        return subprocess.CompletedProcess(
            args=argv, returncode=0, stdout="{}", stderr=""
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    sb = _make_sandbox(tmp_path, bridge=bridge)
    sb.run_agent_session("hi", workspace=tmp_path / "ws", timeout_sec=10)

    assert bridge.entered is True
    assert bridge.exited is True, "bridge must be torn down after the session"
    assert seen_order == ["docker_run"]

    docker_run_argv = [c[0] for c in calls if c[0][:2] == ["docker", "run"]][0]
    # 127.0.0.1 → host.docker.internal rewrite for the container.
    env_flags = [
        docker_run_argv[i + 1]
        for i, a in enumerate(docker_run_argv)
        if a == "-e" and i + 1 < len(docker_run_argv)
    ]
    assert any(
        e == "BIOMNI_BRIDGE_URL=http://host.docker.internal:60123" for e in env_flags
    ), f"expected BIOMNI_BRIDGE_URL env, got env_flags={env_flags!r}"
    # Linux Docker needs the explicit host-gateway mapping for that hostname.
    assert "--add-host" in docker_run_argv
    add_host_val = docker_run_argv[docker_run_argv.index("--add-host") + 1]
    assert add_host_val == "host.docker.internal:host-gateway"


def test_bridge_exits_even_when_agent_raises(tmp_path, monkeypatch) -> None:
    """A crash inside the agent must still trigger bridge teardown."""
    bridge = _FakeBridge()

    def fake_run(argv, **kwargs):
        if argv[:2] == ["docker", "exec"]:
            raise subprocess.TimeoutExpired(cmd=argv, timeout=1)
        return subprocess.CompletedProcess(
            args=argv, returncode=0, stdout="", stderr=""
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    sb = _make_sandbox(tmp_path, bridge=bridge)
    result = sb.run_agent_session("boom", workspace=tmp_path / "ws", timeout_sec=1)

    # The session returns a timed-out AgenticResult and the bridge is gone.
    assert bridge.entered is True
    assert bridge.exited is True
    assert result.returncode == -1


def test_bridge_passes_non_loopback_url_unchanged(tmp_path, monkeypatch) -> None:
    """If the bridge already exposes a routable hostname, don't rewrite it."""
    bridge = _FakeBridge(url="http://10.0.0.4:8765")
    calls: list = []
    monkeypatch.setattr(subprocess, "run", _stub_run(calls))

    sb = _make_sandbox(tmp_path, bridge=bridge)
    sb.run_agent_session("hi", workspace=tmp_path / "ws", timeout_sec=10)

    docker_run_argv = [c[0] for c in calls if c[0][:2] == ["docker", "run"]][0]
    env_flags = [
        docker_run_argv[i + 1]
        for i, a in enumerate(docker_run_argv)
        if a == "-e" and i + 1 < len(docker_run_argv)
    ]
    assert "BIOMNI_BRIDGE_URL=http://10.0.0.4:8765" in env_flags


# -- Phase 2 ②-c: in-container client mount + prompt guidance ---------------


def test_biomni_client_path_is_mounted_read_only(tmp_path, monkeypatch) -> None:
    """``biomni_client.py`` must reach the container at the published path."""
    bridge = _FakeBridge()
    calls: list = []
    monkeypatch.setattr(subprocess, "run", _stub_run(calls))

    client = tmp_path / "biomni_client.py"
    client.write_text("# placeholder\n")
    sb = _make_sandbox(tmp_path, bridge=bridge, biomni_client_path=client)
    sb.run_agent_session("hi", workspace=tmp_path / "ws", timeout_sec=10)

    docker_run_argv = [c[0] for c in calls if c[0][:2] == ["docker", "run"]][0]
    mounts = [
        docker_run_argv[i + 1]
        for i, a in enumerate(docker_run_argv)
        if a == "-v" and i + 1 < len(docker_run_argv)
    ]
    expected = f"{client.resolve()}:/usr/local/bin/biomni_client.py:ro"
    assert expected in mounts, f"expected client mount in {mounts!r}"


def test_biomni_tools_inject_guidance_into_agent_prompt(
    tmp_path, monkeypatch
) -> None:
    """Allowlisted tools must appear in the prompt the agent actually sees."""
    bridge = _FakeBridge()
    calls: list = []
    monkeypatch.setattr(subprocess, "run", _stub_run(calls))

    sb = _make_sandbox(
        tmp_path,
        bridge=bridge,
        biomni_tools=("database.query_uniprot", "database.query_kegg"),
    )
    sb.run_agent_session(
        "Run the experiment.", workspace=tmp_path / "ws", timeout_sec=10
    )

    docker_exec_calls = [c[0] for c in calls if c[0][:2] == ["docker", "exec"]]
    assert docker_exec_calls, "expected docker exec to be invoked"
    # The agent CLI command is the last argv element (bash -c "<cmd>").
    bash_cmd = docker_exec_calls[0][-1]
    assert "biomni_client.py" in bash_cmd
    assert "$BIOMNI_BRIDGE_URL" in bash_cmd
    assert "database.query_uniprot" in bash_cmd
    assert "database.query_kegg" in bash_cmd
    # The original task must still be present after the guidance preface.
    assert "Run the experiment." in bash_cmd


def test_biomni_features_are_noop_without_bridge(tmp_path, monkeypatch) -> None:
    """Without a bridge, neither the mount nor the guidance must appear."""
    calls: list = []
    monkeypatch.setattr(subprocess, "run", _stub_run(calls))

    client = tmp_path / "biomni_client.py"
    client.write_text("# placeholder\n")
    sb = _make_sandbox(
        tmp_path,
        bridge=None,  # no lifecycle → biomni machinery must stay dormant
        biomni_tools=("database.query_uniprot",),
        biomni_client_path=client,
    )
    sb.run_agent_session("hi", workspace=tmp_path / "ws", timeout_sec=10)

    docker_run_argv = [c[0] for c in calls if c[0][:2] == ["docker", "run"]][0]
    assert not any("biomni_client.py" in a for a in docker_run_argv)

    docker_exec_calls = [c[0] for c in calls if c[0][:2] == ["docker", "exec"]]
    bash_cmd = docker_exec_calls[0][-1] if docker_exec_calls else ""
    assert "biomni_client.py" not in bash_cmd
    assert "database.query_uniprot" not in bash_cmd
