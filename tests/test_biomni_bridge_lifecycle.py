"""Tests for the Phase 2 ②-b BiomniBridgeProcess lifecycle helper.

The agentic experiment flow spins up the host-side bridge daemon
(:mod:`external/biomni_bridge/biomni_bridge_daemon`) before launching the
container, then tears it down afterwards. This module covers that wrapper
in isolation: subprocess invocation is injected so no Biomni venv (or even
a network socket) is required for the unit tests.
"""

from __future__ import annotations

import pytest

from researchclaw.experiment.verify.bridge_lifecycle import (
    BiomniBridgeProcess,
    BridgeProcessConfig,
)


class _FakeProc:
    """Stand-in for ``subprocess.Popen`` returned by the spawn callable."""

    def __init__(self, announce_url: str = "http://127.0.0.1:54321") -> None:
        # Mimic a process whose first stdout line is the daemon's announcement.
        self._announce = f"biomni bridge listening on {announce_url}\n"
        self.stdout = self
        self.terminated = False
        self.killed = False
        self.waited = False

    # -- file-object surface used by the lifecycle helper -------------------

    def readline(self) -> str:
        line, self._announce = self._announce, ""
        return line

    # -- Popen surface ------------------------------------------------------

    def terminate(self) -> None:
        self.terminated = True

    def wait(self, timeout: float | None = None) -> int:
        self.waited = True
        return 0

    def kill(self) -> None:
        self.killed = True


def _stub_health(*, always_ok: bool = True) -> tuple[list[str], callable]:
    calls: list[str] = []

    def check(url: str) -> bool:
        calls.append(url)
        return always_ok

    return calls, check


def test_start_spawns_daemon_with_expected_argv(tmp_path) -> None:
    """start() must invoke the daemon CLI with the configured flags."""
    captured: dict = {}

    def spawn(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return _FakeProc()

    _, health = _stub_health()
    cfg = BridgeProcessConfig(
        daemon_script=tmp_path / "biomni_bridge_daemon.py",
        python_executable="/usr/bin/python3",
        ledger_path=tmp_path / "ledger.jsonl",
        allowlist=("database.query_uniprot", "database.query_kegg"),
        timeout=42.5,
        spawn=spawn,
        health_check=health,
    )
    bp = BiomniBridgeProcess(cfg)
    url = bp.start()
    bp.stop()

    assert url.startswith("http://127.0.0.1:")
    argv = captured["argv"]
    assert argv[0] == "/usr/bin/python3"
    assert argv[1].endswith("biomni_bridge_daemon.py")
    assert "--ledger" in argv
    assert str(cfg.ledger_path) in argv
    assert "--allowlist" in argv
    # Allowlist must be comma-joined for the daemon's CLI parser.
    allow_idx = argv.index("--allowlist") + 1
    assert argv[allow_idx] == "database.query_uniprot,database.query_kegg"
    assert "--port" in argv  # 0 (OS picks) is the default
    assert argv[argv.index("--port") + 1] == "0"
    assert "--timeout" in argv
    assert argv[argv.index("--timeout") + 1] == "42.5"


def test_start_returns_announced_url_and_polls_health(tmp_path) -> None:
    """The URL on stdout is published as the bridge URL after /health passes."""
    fake = _FakeProc(announce_url="http://127.0.0.1:55555")
    calls, health = _stub_health()

    cfg = BridgeProcessConfig(
        daemon_script=tmp_path / "daemon.py",
        python_executable="python3",
        ledger_path=tmp_path / "l.jsonl",
        allowlist=("database.query_uniprot",),
        spawn=lambda *a, **k: fake,
        health_check=health,
        health_timeout=1.0,
        health_interval=0.0,
    )
    bp = BiomniBridgeProcess(cfg)
    url = bp.start()
    bp.stop()

    assert url == "http://127.0.0.1:55555"
    assert calls and calls[0] == url


def test_start_raises_when_health_never_succeeds(tmp_path) -> None:
    """If /health stays down within ``health_timeout`` we surface a clear error."""
    fake = _FakeProc()
    _, health = _stub_health(always_ok=False)

    cfg = BridgeProcessConfig(
        daemon_script=tmp_path / "daemon.py",
        python_executable="python3",
        ledger_path=tmp_path / "l.jsonl",
        allowlist=("database.query_uniprot",),
        spawn=lambda *a, **k: fake,
        health_check=health,
        health_timeout=0.05,
        health_interval=0.0,
    )
    bp = BiomniBridgeProcess(cfg)
    with pytest.raises(RuntimeError, match="/health"):
        bp.start()
    # And on health failure we must have torn the process down already.
    assert fake.terminated is True


def test_context_manager_terminates_on_exit(tmp_path) -> None:
    """`with BiomniBridgeProcess(...) as url:` must terminate the daemon."""
    fake = _FakeProc(announce_url="http://127.0.0.1:60000")
    _, health = _stub_health()

    cfg = BridgeProcessConfig(
        daemon_script=tmp_path / "daemon.py",
        python_executable="python3",
        ledger_path=tmp_path / "l.jsonl",
        allowlist=("database.query_uniprot",),
        spawn=lambda *a, **k: fake,
        health_check=health,
        health_timeout=1.0,
        health_interval=0.0,
    )
    with BiomniBridgeProcess(cfg) as url:
        assert url == "http://127.0.0.1:60000"
        assert fake.terminated is False
    assert fake.terminated is True
    assert fake.waited is True


def test_start_announce_mismatch_is_fatal(tmp_path) -> None:
    """Garbage on stdout (e.g. crash) raises so the caller can fail loud."""

    class _BadProc(_FakeProc):
        def readline(self) -> str:
            return "Traceback (most recent call last):\n"

    bad = _BadProc()
    _, health = _stub_health()
    cfg = BridgeProcessConfig(
        daemon_script=tmp_path / "daemon.py",
        python_executable="python3",
        ledger_path=tmp_path / "l.jsonl",
        allowlist=("x",),
        spawn=lambda *a, **k: bad,
        health_check=health,
    )
    with pytest.raises(RuntimeError, match="announce"):
        BiomniBridgeProcess(cfg).start()


def test_server_cmd_override_is_passed_through(tmp_path) -> None:
    """A non-empty ``server_cmd`` must be forwarded to the daemon's CLI."""
    captured: dict = {}

    def spawn(argv, **kwargs):
        captured["argv"] = argv
        return _FakeProc()

    _, health = _stub_health()
    cfg = BridgeProcessConfig(
        daemon_script=tmp_path / "daemon.py",
        python_executable="python3",
        ledger_path=tmp_path / "l.jsonl",
        allowlist=("x",),
        server_cmd="/opt/biomni/.venv/bin/python runner.py",
        spawn=spawn,
        health_check=health,
    )
    BiomniBridgeProcess(cfg).start()
    argv = captured["argv"]
    assert "--server-cmd" in argv
    assert argv[argv.index("--server-cmd") + 1] == "/opt/biomni/.venv/bin/python runner.py"
