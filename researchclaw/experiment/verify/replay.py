"""Deterministic replay gate (Phase 1, layer ③).

Re-executes recorded tool calls and confirms the return still anchors to the
ledger's sha256. A deterministic API tool (``query_uniprot``, ``query_kegg``,
…) returns byte-identical JSON on replay, so its canonical hash matches
exactly; a tool whose floats jitter in the last bits passes via the numeric
tolerance path as long as its structure is unchanged. Anything else is *drift*
— non-reproducibility or a tampered ledger — and is reported for fail-loud
handling.

The tool runner ``(tool, args) -> raw_return`` is injected, so the gate is
fully unit-testable. The pipeline supplies a runner that shells out to the
isolated Biomni venv (see ``external/biomni_bridge/biomni_tool_runner.py``).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from researchclaw.experiment.biomni_bridge import canonical_hash


@dataclass
class ReplayReport:
    """Result of replaying recorded tool calls against the ledger."""

    matched: list[str] = field(default_factory=list)
    drifted: list[tuple[str, str, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.drifted

    def summary(self) -> str:
        if self.ok:
            return f"replay: {len(self.matched)} tool call(s) reproduced."
        tools = ", ".join(t for t, _, _ in self.drifted)
        return f"replay: {len(self.drifted)} non-reproducible call(s): {tools}"


def _close(a: Any, b: Any, tol: float) -> bool:
    """Structural equality with a numeric tolerance on float/int leaves."""
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_close(a[k], b[k], tol) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_close(x, y, tol) for x, y in zip(a, b))
    return a == b


def replay_ledger(
    ledger_path: Path,
    runner: Callable[[str, dict], Any],
    *,
    tol: float = 1e-9,
    tools: set[str] | None = None,
) -> ReplayReport:
    """Re-run each recorded call and confirm its return still anchors clean.

    ``tools``, when given, restricts replay to those tool names so expensive
    or non-deterministic calls can be excluded.
    """
    report = ReplayReport()
    for line in Path(ledger_path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        entry = json.loads(line)
        tool = entry.get("tool", "")
        if tools is not None and tool not in tools:
            continue
        recorded_sha = entry.get("sha256", "")
        replayed = runner(tool, entry.get("args", {}))
        replay_sha = canonical_hash(replayed)
        if replay_sha == recorded_sha or _close(replayed, entry.get("raw_return"), tol):
            report.matched.append(tool)
        else:
            report.drifted.append((tool, recorded_sha, replay_sha))
    return report
