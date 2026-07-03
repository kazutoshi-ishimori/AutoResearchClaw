"""The agentic workspace must be writable by the in-container agent user.

The container runs as a non-root user (``researcher``, UID 999) baked into the
image, but the host workspace dir is created by the pipeline as the host user
(UID 1000, mode 0o755). Bind-mounted at ``/workspace``, that dir is then
un-writable to UID 999 — so the agent silently falls back to ``$HOME`` and the
pipeline collects nothing from ``/workspace`` ("zero real metrics"). The sandbox
owns the mount contract, so it must make the mounted dir writable regardless of
the container UID before the container starts.
"""

from __future__ import annotations

import stat
from pathlib import Path

from researchclaw.experiment.agentic_sandbox import _ensure_workspace_writable


def test_ensure_workspace_writable_creates_and_opens_permissions(tmp_path: Path) -> None:
    ws = tmp_path / "agentic_workspace"

    _ensure_workspace_writable(ws)

    assert ws.is_dir()
    mode = stat.S_IMODE(ws.stat().st_mode)
    # World-writable so any container UID (e.g. 999) can create results.json.
    assert mode & 0o002, f"workspace not world-writable: {oct(mode)}"


def test_ensure_workspace_writable_opens_existing_restrictive_dir(tmp_path: Path) -> None:
    ws = tmp_path / "agentic_workspace"
    ws.mkdir(mode=0o755)
    assert not (stat.S_IMODE(ws.stat().st_mode) & 0o002)

    _ensure_workspace_writable(ws)

    assert stat.S_IMODE(ws.stat().st_mode) & 0o002
