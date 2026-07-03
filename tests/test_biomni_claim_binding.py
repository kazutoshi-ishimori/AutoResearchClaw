"""Tests for the claim->ledger binding gate (Phase 0, layer ⑤).

A numeric metric in results.json is "backed" only if its value can be traced
to a raw tool return recorded in the provenance ledger. A claimed number with
no ledger backing is treated as fabricated (the executor invented it rather
than computing it from a real tool call).
"""

from __future__ import annotations

import json
from pathlib import Path

from researchclaw.experiment.verify.claim_binding import bind_claims


def _write_ledger(path: Path, entries: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(e) for e in entries) + "\n")


def test_all_metrics_backed_by_ledger(tmp_path: Path) -> None:
    ledger = tmp_path / "provenance.jsonl"
    _write_ledger(
        ledger,
        [
            {"tool": "network_proximity", "raw_return": {"proximity": 0.42}},
            {"tool": "query_gene", "raw_return": {"id": "ACE2", "score": 1.5}},
        ],
    )
    report = bind_claims({"proximity": 0.42, "gene_score": 1.5}, ledger)
    assert report.ok
    assert report.unbacked == []


def test_unbacked_metric_is_flagged(tmp_path: Path) -> None:
    ledger = tmp_path / "provenance.jsonl"
    _write_ledger(
        ledger,
        [{"tool": "network_proximity", "raw_return": {"proximity": 0.42}}],
    )
    report = bind_claims({"proximity": 0.42, "made_up": 9.99}, ledger)
    assert not report.ok
    assert report.unbacked == [("made_up", 9.99)]


def test_backed_within_floating_point_tolerance(tmp_path: Path) -> None:
    ledger = tmp_path / "provenance.jsonl"
    _write_ledger(
        ledger,
        [{"tool": "t", "raw_return": {"v": 0.42}}],
    )
    report = bind_claims({"proximity": 0.4200000001}, ledger, tol=1e-6)
    assert report.ok


def test_numbers_nested_in_lists_are_collected(tmp_path: Path) -> None:
    ledger = tmp_path / "provenance.jsonl"
    _write_ledger(
        ledger,
        [{"tool": "ppi", "raw_return": {"neighbors": [{"deg": 3}, {"deg": 7}]}}],
    )
    report = bind_claims({"max_degree": 7}, ledger)
    assert report.ok


# -----------------------------------------------------------------------------
# ④↔⑤ reconciliation: a derived metric independently reproduced by a layer-④
# oracle has provenance (the trusted recomputation), so it must NOT be flagged
# as fabricated even though its value never appears as a raw tool return.
# -----------------------------------------------------------------------------

def test_recompute_verified_metric_is_backed_without_ledger_value(tmp_path: Path) -> None:
    """A headline AUROC is *derived* from the ranking, not returned by any tool,
    so it never appears in the ledger. When layer ④ has independently verified
    it, ⑤ must treat it as backed — otherwise a legitimately-derived metric is
    wrongly flagged FABRICATED_METRIC (the Phase-0 limitation claim_binding.py
    documents)."""
    ledger = tmp_path / "provenance.jsonl"
    _write_ledger(
        ledger,
        [{"tool": "query_uniprot", "raw_return": {"length": 1273}}],
    )
    report = bind_claims(
        {"auroc_phase2plus": 0.40444444444444444},
        ledger,
        verified_metrics=frozenset({"auroc_phase2plus"}),
    )
    assert report.ok
    assert report.unbacked == []
    assert ("auroc_phase2plus", 0.40444444444444444) in report.backed


def test_unverified_metric_still_flagged_despite_reconciliation(tmp_path: Path) -> None:
    """Reconciliation must not become a blanket bypass: a metric with neither a
    ledger value nor a layer-④ verification is still fabricated."""
    ledger = tmp_path / "provenance.jsonl"
    _write_ledger(
        ledger,
        [{"tool": "query_uniprot", "raw_return": {"length": 1273}}],
    )
    report = bind_claims(
        {"auroc_phase2plus": 0.404, "made_up": 9.99},
        ledger,
        verified_metrics=frozenset({"auroc_phase2plus"}),
    )
    assert not report.ok
    assert report.unbacked == [("made_up", 9.99)]
