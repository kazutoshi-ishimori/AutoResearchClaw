"""Host-side adapter to the isolated Biomni venv (Phase 1, ③ activation).

The deterministic-replay gate (layer ③) needs to *re-execute* a recorded tool
call. Biomni lives in its own venv (``~/workspace/Biomni/.venv``) with a 15 GB
data lake, kept out of ARC's environment; the bridge therefore runs the tool by
shelling out to the standalone runner
(``external/biomni_bridge/biomni_tool_runner.py``) via ``server_cmd``.

:class:`BiomniSubprocessRunner` exposes the ``(tool, args) -> raw_return``
callable :func:`researchclaw.experiment.verify.replay.replay_ledger` expects.
The subprocess call is injected (``run``) so command construction and stdout
parsing stay unit-testable without a live Biomni environment.

Note: replaying re-runs real tools (network/API for the deterministic query
tools), so the pipeline gates it behind ``BiomniConfig.replay_enabled``.
"""

from __future__ import annotations

import json
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# Canonical isolated-venv interpreter + standalone runner. Pinned to the
# dedicated Biomni venv (conda was never available; see project memory).
_BIOMNI_VENV_PYTHON = Path.home() / "workspace" / "Biomni" / ".venv" / "bin" / "python"
_RUNNER_REL = "external/biomni_bridge/biomni_tool_runner.py"


def default_server_cmd() -> str:
    """The canonical ``server_cmd``: Biomni-venv python + the standalone runner."""
    return f"{_BIOMNI_VENV_PYTHON} {_RUNNER_REL}"


def tool_modules_from_allowlist(allowlist: tuple[str, ...]) -> dict[str, str]:
    """Map bare tool names to their ``biomni.tool`` submodule.

    The allowlist holds qualified ``module.tool`` entries (e.g.
    ``database.query_uniprot``); replay records only the bare tool name, so
    this recovers the module needed to re-invoke it.
    """
    mapping: dict[str, str] = {}
    for qualified in allowlist:
        if "." in qualified:
            module, _, tool = qualified.partition(".")
            mapping[tool] = module
    return mapping


@dataclass
class BiomniSubprocessRunner:
    """Invoke one allowlisted Biomni tool in the isolated venv, return its JSON.

    Parameters
    ----------
    server_cmd:
        Base command (``python runner.py``) — split with :func:`shlex.split`.
    tool_modules:
        ``{tool: module}`` map (see :func:`tool_modules_from_allowlist`).
    ledger_path:
        Throwaway ledger the runner appends to; replay only needs the stdout.
    timeout:
        Per-call subprocess timeout in seconds.
    run:
        Injected subprocess runner (defaults to :func:`subprocess.run`).
    """

    server_cmd: str
    tool_modules: dict[str, str]
    ledger_path: Path
    timeout: float = 120.0
    run: Callable[..., Any] = field(default=subprocess.run)

    def __call__(self, tool: str, args: dict) -> Any:
        if tool not in self.tool_modules:
            raise ValueError(f"no module mapping for tool '{tool}' (check tool_allowlist)")
        argv = shlex.split(self.server_cmd) + [
            "--module", self.tool_modules[tool],
            "--tool", tool,
            "--args-json", json.dumps(args),
            "--ledger", str(self.ledger_path),
        ]
        proc = self.run(
            argv,
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )
        if getattr(proc, "returncode", 0) != 0:
            raise RuntimeError(
                f"Biomni runner failed for '{tool}' (rc={proc.returncode}): {proc.stderr.strip()}"
            )
        return json.loads(proc.stdout)
