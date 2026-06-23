"""Independent recompute oracle (Phase 1, layer ④).

The strongest anti-fabrication layer: even when every tool call is real and
every number traces to the ledger (layers ① + ⑤), the agent can still combine
them into a *wrong* headline metric. The recompute oracle re-derives a metric
independently and flags any active disagreement. A metric with no oracle is
*skipped*, not failed — layer ④ only fires on contradiction, never on silence.
"""

from __future__ import annotations

from researchclaw.experiment.verify.recompute import RecomputeReport, recompute_metrics


def test_agreeing_oracle_verifies() -> None:
    report = recompute_metrics({"auroc": 0.78}, {"auroc": lambda: 0.78})
    assert report.ok
    assert report.verified == [("auroc", 0.78, 0.78)]
    assert report.mismatched == []


def test_disagreeing_oracle_is_mismatch() -> None:
    report = recompute_metrics({"auroc": 0.78}, {"auroc": lambda: 0.55})
    assert not report.ok
    assert report.mismatched == [("auroc", 0.78, 0.55)]
    assert report.verified == []


def test_metric_without_oracle_is_skipped_not_failed() -> None:
    report = recompute_metrics({"auroc": 0.78, "proximity": -2.1}, {"auroc": lambda: 0.78})
    assert report.ok  # no oracle for 'proximity' -> silence, not failure
    assert report.skipped == ["proximity"]
    assert report.verified == [("auroc", 0.78, 0.78)]


def test_tolerance_respected() -> None:
    # within tol -> verified
    r1 = recompute_metrics({"x": 1.0}, {"x": lambda: 1.0 + 1e-12}, tol=1e-9)
    assert r1.ok
    # beyond tol -> mismatch
    r2 = recompute_metrics({"x": 1.0}, {"x": lambda: 1.01}, tol=1e-9)
    assert not r2.ok


def test_non_numeric_claim_is_skipped() -> None:
    report = recompute_metrics({"label": "approved", "x": 1.0}, {"x": lambda: 1.0})
    assert report.ok
    assert "label" in report.skipped


def test_empty_is_ok() -> None:
    report = recompute_metrics({}, {})
    assert report.ok
    assert isinstance(report, RecomputeReport)
    assert "recompute" in report.summary()
