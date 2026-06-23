"""Independent recompute oracle (Phase 1, layer ④).

Layers ① + ⑤ prove every reported number was *produced by some tool*. They do
not prove the agent *combined* those numbers correctly into a headline metric.
Layer ④ closes that gap: for each metric with a registered oracle, an
independent re-derivation is compared against the claimed value.

Design stance — fire only on contradiction:
  * agree within tolerance        -> verified
  * disagree beyond tolerance     -> mismatched (critical)
  * no oracle / non-numeric claim -> skipped (silence is not failure)

Oracles are zero-argument callables returning a float; the caller builds them
as closures over whatever raw data / ledger context the recomputation needs.
This keeps the gate fully unit-testable without a live Biomni environment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class RecomputeReport:
    """Result of independently recomputing claimed headline metrics."""

    verified: list[tuple[str, float, float]] = field(default_factory=list)
    mismatched: list[tuple[str, float, float]] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.mismatched

    def summary(self) -> str:
        if self.ok:
            return (
                f"recompute: {len(self.verified)} metric(s) reproduced, "
                f"{len(self.skipped)} unchecked."
            )
        parts = [f"{k} (claimed {c} vs recomputed {r})" for k, c, r in self.mismatched]
        return "recompute: MISMATCH -> " + "; ".join(parts)


def recompute_metrics(
    claimed: dict[str, object],
    oracles: dict[str, Callable[[], float]],
    *,
    tol: float = 1e-9,
) -> RecomputeReport:
    """Compare each claimed numeric metric against its independent oracle."""
    report = RecomputeReport()
    for name, value in claimed.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            report.skipped.append(name)
            continue
        oracle = oracles.get(name)
        if oracle is None:
            report.skipped.append(name)
            continue
        recomputed = float(oracle())
        claimed_val = float(value)
        if math.isclose(claimed_val, recomputed, rel_tol=tol, abs_tol=tol):
            report.verified.append((name, claimed_val, recomputed))
        else:
            report.mismatched.append((name, claimed_val, recomputed))
    return report
