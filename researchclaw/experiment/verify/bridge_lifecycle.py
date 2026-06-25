"""Lifecycle manager for the host-side Biomni HTTP bridge (Phase 2, ②-b).

The agentic experiment flow runs Claude Code inside a Docker container while
the Biomni venv and provenance ledger live on the host. This module spins up
``external/biomni_bridge/biomni_bridge_daemon.py`` as a subprocess, waits for
``/health`` to respond, and surfaces the ephemeral URL so the container can
reach it (typically via ``host.docker.internal`` rewriting). On context exit
the daemon is terminated and reaped, keeping the executor's observer firmly
on the host side per the B+(1-5) design invariant.

The implementation is deliberately decoupled from real subprocesses and
sockets: both ``spawn`` and ``health_check`` are injectable, so the unit
tests in ``tests/test_biomni_bridge_lifecycle.py`` need no Biomni venv and
no listening port.
"""

from __future__ import annotations

import json
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


_ANNOUNCE_PREFIX = "biomni bridge listening on "


def _default_health_check(url: str) -> bool:
    """Default ``/health`` poll — returns True on HTTP 200 + ``{"ok": true}``."""
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=1.0) as resp:
            if resp.status != 200:
                return False
            payload = json.loads(resp.read())
            return bool(payload.get("ok"))
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        return False


@dataclass
class BridgeProcessConfig:
    """Knobs for :class:`BiomniBridgeProcess`.

    ``spawn`` and ``health_check`` are injected so the lifecycle helper can
    be unit-tested without a live daemon.
    """

    daemon_script: Path
    python_executable: str
    ledger_path: Path
    allowlist: tuple[str, ...]
    server_cmd: str = ""
    timeout: float = 120.0
    host: str = "127.0.0.1"
    port: int = 0  # 0 = let the OS pick an ephemeral port
    health_timeout: float = 5.0
    health_interval: float = 0.05
    spawn: Callable[..., Any] = field(default=subprocess.Popen)
    health_check: Callable[[str], bool] = field(default=_default_health_check)


class BiomniBridgeProcess:
    """Context manager that owns one host-side bridge daemon process.

    Typical use::

        with BiomniBridgeProcess(cfg) as url:
            run_agent(env={"BIOMNI_BRIDGE_URL": url})
    """

    def __init__(self, cfg: BridgeProcessConfig) -> None:
        self.cfg = cfg
        self._proc: Any | None = None
        self._url: str | None = None

    # -- public ----------------------------------------------------------

    @property
    def url(self) -> str:
        if self._url is None:
            raise RuntimeError("bridge has not been started yet")
        return self._url

    def start(self) -> str:
        if self._proc is not None:
            raise RuntimeError("bridge already running")
        argv = [
            self.cfg.python_executable,
            str(self.cfg.daemon_script),
            "--host", self.cfg.host,
            "--port", str(self.cfg.port),
            "--ledger", str(self.cfg.ledger_path),
            "--allowlist", ",".join(self.cfg.allowlist),
            "--timeout", str(self.cfg.timeout),
        ]
        if self.cfg.server_cmd:
            argv.extend(["--server-cmd", self.cfg.server_cmd])

        proc = self.cfg.spawn(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self._proc = proc

        # The daemon prints the bound URL on its first stdout line.
        announcement = (proc.stdout.readline() or "").strip()
        if not announcement.startswith(_ANNOUNCE_PREFIX):
            self._terminate_proc()
            raise RuntimeError(
                f"bridge daemon failed to announce its URL (got: {announcement!r})"
            )
        self._url = announcement[len(_ANNOUNCE_PREFIX):].strip()

        # Block until /health passes (or give up loudly).
        deadline = time.monotonic() + max(self.cfg.health_timeout, 0.0)
        while True:
            if self.cfg.health_check(self._url):
                return self._url
            if time.monotonic() >= deadline:
                break
            time.sleep(max(self.cfg.health_interval, 0.0))
        self._terminate_proc()
        url = self._url
        self._url = None
        raise RuntimeError(f"bridge /health did not respond at {url}")

    def stop(self) -> None:
        self._terminate_proc()
        self._url = None

    # -- context-manager API --------------------------------------------

    def __enter__(self) -> str:
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    # -- internals -------------------------------------------------------

    def _terminate_proc(self) -> None:
        proc = self._proc
        if proc is None:
            return
        try:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        finally:
            self._proc = None
