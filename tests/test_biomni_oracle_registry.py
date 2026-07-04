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


def test_mean_rank_known_metric_is_wired_as_closure() -> None:
    """``mean_rank_of_positives`` must resolve to a zero-arg float closure."""
    ctx = _toy_context()
    oracles = build_covid_oracles(("mean_rank_of_positives",), ctx)
    assert "mean_rank_of_positives" in oracles
    assert callable(oracles["mean_rank_of_positives"])
    assert isinstance(oracles["mean_rank_of_positives"](), float)


def test_mean_rank_positives_at_top_is_near_one() -> None:
    """Positives A (0.9) and B (0.8) hold descending ranks 1 and 2 ⇒ mean 1.5."""
    ctx = _toy_context()
    oracle = build_covid_oracles(("mean_rank_of_positives",), ctx)[
        "mean_rank_of_positives"
    ]
    assert math.isclose(oracle(), 1.5)


def test_mean_rank_uses_midrank_for_ties() -> None:
    """A three-way tie at the top gives the tied positive the midrank (2.0)."""
    ctx = RecomputeContext(
        ranking=(("A", 0.5), ("B", 0.5), ("C", 0.5), ("D", 0.1)),
        positives=frozenset({"A"}),
    )
    oracle = build_covid_oracles(("mean_rank_of_positives",), ctx)[
        "mean_rank_of_positives"
    ]
    # ranks of the three tied 0.5s average to (1+2+3)/3 = 2.0; D is rank 4.
    assert math.isclose(oracle(), 2.0)


def test_mean_rank_is_order_independent() -> None:
    """Midrank depends on the score multiset, not the emitted ranking order."""
    shuffled = RecomputeContext(
        ranking=(("D", 0.3), ("A", 0.9), ("E", 0.1), ("C", 0.4), ("B", 0.8)),
        positives=frozenset({"A", "B"}),
    )
    oracle = build_covid_oracles(("mean_rank_of_positives",), shuffled)[
        "mean_rank_of_positives"
    ]
    assert math.isclose(oracle(), 1.5)


def test_mean_rank_no_positives_yields_nan() -> None:
    ctx = RecomputeContext(ranking=(("A", 0.1), ("B", 0.2)), positives=frozenset())
    oracle = build_covid_oracles(("mean_rank_of_positives",), ctx)[
        "mean_rank_of_positives"
    ]
    assert math.isnan(oracle())


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


def test_ppi_enrichment_fold_oracle_recomputes_ratio() -> None:
    ctx = RecomputeContext(
        ranking=(),
        positives=frozenset(),
        network={"number_of_edges": 44.0, "expected_number_of_edges": 1.0},
    )
    oracles = build_covid_oracles(("ppi_enrichment_fold",), ctx)
    assert set(oracles) == {"ppi_enrichment_fold"}
    assert oracles["ppi_enrichment_fold"]() == 44.0


def test_ppi_enrichment_fold_zero_expected_is_nan() -> None:
    ctx = RecomputeContext(
        ranking=(),
        positives=frozenset(),
        network={"number_of_edges": 44.0, "expected_number_of_edges": 0.0},
    )
    oracles = build_covid_oracles(("ppi_enrichment_fold",), ctx)
    assert math.isnan(oracles["ppi_enrichment_fold"]())


def test_ppi_enrichment_fold_dropped_without_network() -> None:
    ctx = RecomputeContext(ranking=(), positives=frozenset())  # network defaults None
    oracles = build_covid_oracles(("ppi_enrichment_fold",), ctx)
    assert oracles == {}


def test_network_default_keeps_ranking_oracles_working() -> None:
    ctx = RecomputeContext(
        ranking=(("D1", 2.0), ("D2", 1.0)),
        positives=frozenset({"D1"}),
    )
    oracles = build_covid_oracles(("auroc_phase2plus",), ctx)
    assert "auroc_phase2plus" in oracles
    assert oracles["auroc_phase2plus"]() == 1.0
