"""Phase 2 pipeline-wiring D: COVID oracle registry for layer ④.

Layer ④ (independent recompute) is wired generically by
:func:`researchclaw.experiment.verify.recompute.recompute_metrics`, which takes
a dict ``{metric_name -> zero-arg callable}``. The remaining gap is the bridge
between :class:`BiomniConfig.recompute_headline_metrics` (a tuple of *names*)
and the actual *closures* that recompute those metrics from run artefacts.

These tests pin a small registry: :func:`build_covid_oracles` accepts the
metric names that the operator declared in config plus a
:class:`RecomputeContext` (ranking + positives) and returns the oracle dict.
Unknown metric names must be SILENTLY DROPPED — this matches the existing
"silence is not failure" stance of layer ④. Numeric work is a pure-python
Mann-Whitney AUROC; no sklearn.
"""

from __future__ import annotations

import math

import pytest

from researchclaw.experiment.verify.oracle_registry import (
    RecomputeContext,
    build_covid_oracles,
)
from researchclaw.experiment.verify.recompute import recompute_metrics


def _toy_context() -> RecomputeContext:
    # Drugs A, B (positives) score above C, D, E (negatives) — perfect AUROC.
    return RecomputeContext(
        ranking=(("A", 0.9), ("B", 0.8), ("C", 0.4), ("D", 0.3), ("E", 0.1)),
        positives=frozenset({"A", "B"}),
    )


def test_unknown_metric_name_is_silently_dropped() -> None:
    """Unknown metrics must not raise — silence preserves layer ④ stance."""
    ctx = _toy_context()
    oracles = build_covid_oracles(("totally_unknown_metric",), ctx)
    assert oracles == {}


def test_known_metric_is_wired_as_closure() -> None:
    """A known name must produce a zero-arg callable, not the raw number."""
    ctx = _toy_context()
    oracles = build_covid_oracles(("auroc_phase2plus",), ctx)
    assert "auroc_phase2plus" in oracles
    assert callable(oracles["auroc_phase2plus"])
    # The closure must close over the context, so calling it now yields a float.
    value = oracles["auroc_phase2plus"]()
    assert isinstance(value, float)


def test_auroc_phase2plus_perfect_ranking_returns_one() -> None:
    """All positives ranked above all negatives ⇒ AUROC = 1.0."""
    ctx = _toy_context()
    oracle = build_covid_oracles(("auroc_phase2plus",), ctx)["auroc_phase2plus"]
    assert math.isclose(oracle(), 1.0)


def test_auroc_phase2plus_inverted_ranking_returns_zero() -> None:
    """All positives ranked below all negatives ⇒ AUROC = 0.0."""
    ctx = RecomputeContext(
        ranking=(("A", 0.1), ("B", 0.2), ("C", 0.7), ("D", 0.8), ("E", 0.9)),
        positives=frozenset({"A", "B"}),
    )
    oracle = build_covid_oracles(("auroc_phase2plus",), ctx)["auroc_phase2plus"]
    assert math.isclose(oracle(), 0.0)


def test_auroc_phase2plus_handles_ties_with_half_credit() -> None:
    """Ties between a positive and a negative count as 0.5 (Mann-Whitney)."""
    ctx = RecomputeContext(
        ranking=(("A", 0.5), ("B", 0.5), ("C", 0.5), ("D", 0.1)),
        positives=frozenset({"A"}),
    )
    oracle = build_covid_oracles(("auroc_phase2plus",), ctx)["auroc_phase2plus"]
    # A vs B: tie (0.5). A vs C: tie (0.5). A vs D: win (1.0). -> (0.5+0.5+1)/3
    assert math.isclose(oracle(), (0.5 + 0.5 + 1.0) / 3.0)


def test_registry_plugs_into_recompute_metrics() -> None:
    """End-to-end: build oracles from names, hand them to recompute_metrics."""
    ctx = _toy_context()
    oracles = build_covid_oracles(("auroc_phase2plus",), ctx)
    # Claimed matches what an honest agent would have produced.
    report = recompute_metrics({"auroc_phase2plus": 1.0}, oracles)
    assert report.ok
    assert ("auroc_phase2plus", 1.0, 1.0) in report.verified


def test_registry_with_no_positives_yields_nan_and_is_skipped() -> None:
    """No positives ⇒ AUROC undefined; recompute reports mismatch vs nan -> skipped/verified rule.

    With nan the oracle returns NaN and math.isclose treats NaN ≠ anything; so
    a claimed numeric value vs NaN recomputed will be flagged as mismatched.
    We document this current behaviour so future changes are deliberate.
    """
    ctx = RecomputeContext(
        ranking=(("A", 0.1), ("B", 0.2)),
        positives=frozenset(),
    )
    oracle = build_covid_oracles(("auroc_phase2plus",), ctx)["auroc_phase2plus"]
    assert math.isnan(oracle())
