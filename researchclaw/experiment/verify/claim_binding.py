"""Claim->ledger binding gate (Phase 0, layer ⑤).

Trust-minimal check: a numeric metric is *backed* only if its value appears
among the numbers found in some recorded raw tool return. Values that appear
nowhere in the provenance ledger are flagged as unbacked (fabricated).

Known Phase 0 limitation: a metric *derived* by arithmetic from several tool
returns (e.g. a ratio) will not value-match any single raw number and would be
flagged. Such derived metrics need explicit provenance annotation; that is a
Phase 1 refinement. For Phase 0 the gate targets the worst case — a headline
number that was never produced by any tool.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ClaimBindingReport:
    """Result of binding claimed metrics to the provenance ledger."""

    backed: list[tuple[str, float]] = field(default_factory=list)
    unbacked: list[tuple[str, float]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.unbacked

    def summary(self) -> str:
        if self.ok:
            return f"claim_binding: all {len(self.backed)} metric(s) backed by ledger."
        keys = ", ".join(k for k, _ in self.unbacked)
        return f"claim_binding: {len(self.unbacked)} unbacked metric(s): {keys}"


def _collect_numbers(obj: Any, out: set[float]) -> None:
    """Recursively gather every numeric leaf in *obj*."""
    if isinstance(obj, bool):
        return  # bools are ints in Python; not scientific numbers
    if isinstance(obj, (int, float)):
        out.add(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            _collect_numbers(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _collect_numbers(v, out)


def _ledger_numbers(ledger_path: Path) -> set[float]:
    numbers: set[float] = set()
    for line in Path(ledger_path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        entry = json.loads(line)
        _collect_numbers(entry.get("raw_return"), numbers)
    return numbers


def bind_claims(
    metrics: dict[str, Any],
    ledger_path: Path,
    *,
    tol: float = 1e-9,
) -> ClaimBindingReport:
    """Check that every numeric metric traces to a recorded tool return."""
    ledger = _ledger_numbers(ledger_path)
    report = ClaimBindingReport()
    for key, value in metrics.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue  # only quantitative scientific claims are bound
        fval = float(value)
        if any(math.isclose(fval, lv, rel_tol=tol, abs_tol=tol) for lv in ledger):
            report.backed.append((key, fval))
        else:
            report.unbacked.append((key, fval))
    return report
