"""Stage-14 wiring for the Biomni verification gates (Phase 0, ①+②).

These tests cover the *pure* seam that the pipeline runner glues into
``_run_experiment_diagnosis``: given a parsed ``experiment_summary.json``, an
optional provenance ledger, and an optional entity gate, run layers ⑤ and ②
and report what is unbacked / unknown — without touching the pipeline itself.
The runner then folds the reports into the diagnosis via the already-tested
``add_biomni_verification_deficiencies``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from researchclaw.config import BiomniConfig
from researchclaw.experiment.verify.entity_gate import EntityGate
from researchclaw.experiment.verify.stage_hook import (
    build_entity_gate,
    collect_claimed_entities,
    collect_claimed_metrics,
    collect_recompute_context,
    resolve_ledger_path,
    run_biomni_gates,
    run_recompute_gate,
)
from researchclaw.pipeline.experiment_diagnosis import (
    DeficiencyType,
    ExperimentDiagnosis,
    add_biomni_verification_deficiencies,
)


def _write_ledger(path: Path, raw_returns: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rr in raw_returns:
            fh.write(json.dumps({"tool": "t", "args": {}, "raw_return": rr,
                                 "sha256": "x", "ts": "now"}) + "\n")


# --- metric / entity extraction --------------------------------------------

def test_collect_metrics_prefers_best_run() -> None:
    summary = {
        "best_run": {"metrics": {"proximity_z": -2.3, "auroc": 0.78}},
        "metrics": {"ignored": 1.0},
    }
    assert collect_claimed_metrics(summary) == {"proximity_z": -2.3, "auroc": 0.78}


def test_collect_metrics_falls_back_to_top_level() -> None:
    assert collect_claimed_metrics({"metrics": {"x": 1.0}}) == {"x": 1.0}


def test_collect_entities_reads_entities_block() -> None:
    summary = {"entities": {"gene": ["ACE2"], "drug": ["DB00001"]}}
    assert collect_claimed_entities(summary) == {"gene": ["ACE2"], "drug": ["DB00001"]}


def test_collect_entities_absent_returns_empty() -> None:
    assert collect_claimed_entities({"best_run": {}}) == {}


# --- run_biomni_gates -------------------------------------------------------

def test_gates_flag_fabricated_metric(tmp_path: Path) -> None:
    ledger = tmp_path / "provenance.jsonl"
    _write_ledger(ledger, [{"length": 805}])
    summary = {"best_run": {"metrics": {"seq_length": 805, "made_up": 123456}}}

    cb, er = run_biomni_gates(summary, ledger_path=ledger, gate=None)

    assert er is None
    assert cb is not None
    assert cb.backed == [("seq_length", 805.0)]
    assert cb.unbacked == [("made_up", 123456.0)]


def test_gates_exempt_recompute_verified_metric(tmp_path: Path) -> None:
    """A derived headline metric verified by layer ④ is not in the ledger, but
    must not be flagged by ⑤ — run_biomni_gates threads the verified names
    through to bind_claims so the two gates agree."""
    ledger = tmp_path / "provenance.jsonl"
    _write_ledger(ledger, [{"length": 1273}])
    summary = {"best_run": {"metrics": {"auroc_phase2plus": 0.40444444444444444}}}

    cb, er = run_biomni_gates(
        summary,
        ledger_path=ledger,
        gate=None,
        verified_metrics=frozenset({"auroc_phase2plus"}),
    )

    assert cb is not None
    assert cb.ok
    assert cb.unbacked == []
    assert ("auroc_phase2plus", 0.40444444444444444) in cb.backed


def test_gates_flag_unknown_drug(tmp_path: Path) -> None:
    known = hashlib.sha256(b"DBGOOD").hexdigest()
    gate = EntityGate(drug_hashes={known})
    summary = {"entities": {"drug": ["DBGOOD", "DBBAD"]}}

    cb, er = run_biomni_gates(summary, ledger_path=None, gate=gate)

    assert cb is None
    assert er is not None
    assert er.unknown == {"drug": ["DBBAD"]}


def test_gates_noop_when_nothing_configured() -> None:
    cb, er = run_biomni_gates({"best_run": {"metrics": {"x": 1.0}}},
                              ledger_path=None, gate=None)
    assert cb is None and er is None


def test_gates_skip_entity_when_no_entities_claimed() -> None:
    gate = EntityGate(drug_hashes={"abc"})
    cb, er = run_biomni_gates({"best_run": {"metrics": {}}}, ledger_path=None, gate=gate)
    assert cb is None and er is None


def test_gates_missing_ledger_file_is_noop(tmp_path: Path) -> None:
    cb, er = run_biomni_gates({"best_run": {"metrics": {"x": 1.0}}},
                              ledger_path=tmp_path / "nope.jsonl", gate=None)
    assert cb is None and er is None


# --- end-to-end fold into the diagnosis ------------------------------------

def test_gate_failure_becomes_critical_diagnosis(tmp_path: Path) -> None:
    ledger = tmp_path / "provenance.jsonl"
    _write_ledger(ledger, [{"length": 805}])
    summary = {"best_run": {"metrics": {"seq_length": 805, "made_up": 999.0}}}

    cb, er = run_biomni_gates(summary, ledger_path=ledger, gate=None)
    diag = ExperimentDiagnosis()
    add_biomni_verification_deficiencies(diag, claim_binding=cb, entity=er)

    assert diag.has_critical()
    assert DeficiencyType.FABRICATED_METRIC in [d.type for d in diag.deficiencies]


# --- config-driven resolution ----------------------------------------------

def test_resolve_ledger_finds_nested_ledger(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    nested = run_dir / "stage-12-experiment_run" / "ws"
    nested.mkdir(parents=True)
    ledger = nested / "provenance.jsonl"
    ledger.write_text("{}\n", encoding="utf-8")

    cfg = BiomniConfig(provenance_path="provenance.jsonl")
    assert resolve_ledger_path(cfg, run_dir) == ledger


def test_resolve_ledger_absent_returns_none(tmp_path: Path) -> None:
    cfg = BiomniConfig(provenance_path="provenance.jsonl")
    assert resolve_ledger_path(cfg, tmp_path) is None


def test_build_entity_gate_none_when_unconfigured(tmp_path: Path) -> None:
    assert build_entity_gate(BiomniConfig(), tmp_path) is None


# --- layer ④ recompute hook (Phase 2 pipeline-wiring E) --------------------


def _summary_with_ranking() -> dict:
    return {
        "best_run": {"metrics": {"auroc_phase2plus": 1.0}},
        "ranking": [["A", 0.9], ["B", 0.8], ["C", 0.4], ["D", 0.1]],
        "positives": ["A", "B"],
    }


def test_collect_recompute_context_extracts_from_summary() -> None:
    """Summary with ranking + positives → frozen RecomputeContext."""
    ctx = collect_recompute_context(_summary_with_ranking())
    assert ctx is not None
    assert ctx.ranking == (("A", 0.9), ("B", 0.8), ("C", 0.4), ("D", 0.1))
    assert ctx.positives == frozenset({"A", "B"})


def test_collect_recompute_context_missing_ranking_returns_none() -> None:
    """No ranking ⇒ None (silence is not failure)."""
    assert collect_recompute_context({"positives": ["A"]}) is None


def test_collect_recompute_context_missing_positives_returns_none() -> None:
    assert collect_recompute_context({"ranking": [["A", 0.9]]}) is None


def test_collect_recompute_context_malformed_entries_returns_none() -> None:
    """A non-numeric score in the ranking must not raise — return None instead."""
    bad = {"ranking": [["A", "not-a-number"]], "positives": ["A"]}
    assert collect_recompute_context(bad) is None


def test_run_recompute_gate_no_metric_names_is_noop() -> None:
    """recompute_headline_metrics empty ⇒ None (layer ④ disabled by config)."""
    cfg = BiomniConfig()  # default: recompute_headline_metrics=()
    assert run_recompute_gate(_summary_with_ranking(), cfg) is None


def test_run_recompute_gate_no_ranking_in_summary_is_noop() -> None:
    """No ranking in summary ⇒ None even when names are configured."""
    cfg = BiomniConfig(recompute_headline_metrics=("auroc_phase2plus",))
    summary = {"best_run": {"metrics": {"auroc_phase2plus": 1.0}}}
    assert run_recompute_gate(summary, cfg) is None


def test_run_recompute_gate_perfect_agreement_is_ok() -> None:
    """Configured name + ranking + matching claim ⇒ verified report."""
    cfg = BiomniConfig(recompute_headline_metrics=("auroc_phase2plus",))
    report = run_recompute_gate(_summary_with_ranking(), cfg)
    assert report is not None
    assert report.ok
    assert ("auroc_phase2plus", 1.0, 1.0) in report.verified


def test_run_recompute_gate_disagreement_is_mismatch() -> None:
    """Claim of 0.5 with recomputed 1.0 ⇒ mismatch flagged."""
    cfg = BiomniConfig(recompute_headline_metrics=("auroc_phase2plus",))
    summary = _summary_with_ranking()
    summary["best_run"]["metrics"]["auroc_phase2plus"] = 0.5
    report = run_recompute_gate(summary, cfg)
    assert report is not None
    assert not report.ok
    assert any(name == "auroc_phase2plus" for name, *_ in report.mismatched)


def test_run_recompute_gate_unknown_name_is_silently_dropped() -> None:
    """A configured name with no builder ⇒ no oracle ⇒ None (no gate ran)."""
    cfg = BiomniConfig(recompute_headline_metrics=("never_registered",))
    assert run_recompute_gate(_summary_with_ranking(), cfg) is None


def test_build_entity_gate_loads_existing_db(tmp_path: Path) -> None:
    drug_db = tmp_path / "drugbank_ids.sha256"
    h = hashlib.sha256(b"DB00001").hexdigest()
    drug_db.write_text(h + "\n", encoding="utf-8")

    cfg = BiomniConfig(drug_hashes_db="drugbank_ids.sha256")
    gate = build_entity_gate(cfg, tmp_path)

    assert gate is not None
    assert gate.drug_hashes == {h}
    assert gate.check({"drug": ["DB00001"]}).ok
    assert not gate.check({"drug": ["DBBAD"]}).ok


def _stringdb_ledger(tmp_path: Path, edges: int, expected: int) -> Path:
    led = tmp_path / "provenance.jsonl"
    entry = {
        "tool": "query_stringdb",
        "raw_return": [{"number_of_edges": edges, "expected_number_of_edges": expected}],
    }
    led.write_text(json.dumps(entry) + "\n", encoding="utf-8")
    return led


def test_context_populates_network_from_ledger(tmp_path: Path) -> None:
    led = _stringdb_ledger(tmp_path, 44, 1)
    ctx = collect_recompute_context({}, ledger_path=led)
    assert ctx is not None
    assert ctx.network == {"number_of_edges": 44.0, "expected_number_of_edges": 1.0}


def test_context_none_without_ranking_or_network() -> None:
    assert collect_recompute_context({}, ledger_path=None) is None


def test_recompute_gate_verifies_fold_from_ledger(tmp_path: Path) -> None:
    led = _stringdb_ledger(tmp_path, 44, 1)
    cfg = BiomniConfig(recompute_headline_metrics=("ppi_enrichment_fold",))
    summary = {"metrics": {"ppi_enrichment_fold": 44.0}}
    report = run_recompute_gate(summary, cfg, ledger_path=led)
    assert report is not None
    assert report.ok
    assert any(name == "ppi_enrichment_fold" for name, _c, _r in report.verified)


def test_recompute_gate_mismatch_when_claim_tampered(tmp_path: Path) -> None:
    led = _stringdb_ledger(tmp_path, 44, 1)
    cfg = BiomniConfig(recompute_headline_metrics=("ppi_enrichment_fold",))
    summary = {"metrics": {"ppi_enrichment_fold": 99.0}}
    report = run_recompute_gate(summary, cfg, ledger_path=led)
    assert report is not None
    assert not report.ok


def _network_ledger(tmp_path: Path, scores: list[float]) -> Path:
    led = tmp_path / "provenance.jsonl"
    entry = {
        "tool": "query_stringdb",
        "raw_return": {
            "success": True,
            "result": [{"preferredName_A": "ACE2", "score": s} for s in scores],
        },
    }
    led.write_text(json.dumps(entry) + "\n", encoding="utf-8")
    return led


def test_context_populates_interactions_from_ledger(tmp_path: Path) -> None:
    # A-③: the stage hook must also surface the network edge scores into the
    # context so the mean_interaction_score oracle can fire.
    led = _network_ledger(tmp_path, [0.9, 0.6, 0.3])
    ctx = collect_recompute_context({}, ledger_path=led)
    assert ctx is not None
    assert ctx.interactions == (0.9, 0.6, 0.3)


def test_context_none_when_ledger_has_no_network_or_ranking(tmp_path: Path) -> None:
    led = tmp_path / "provenance.jsonl"
    led.write_text(json.dumps({"tool": "query_uniprot", "raw_return": {"length": 805}}) + "\n")
    assert collect_recompute_context({}, ledger_path=led) is None


def test_recompute_gate_verifies_mean_interaction_score_from_ledger(tmp_path: Path) -> None:
    led = _network_ledger(tmp_path, [0.9, 0.6, 0.3])  # mean 0.6
    cfg = BiomniConfig(recompute_headline_metrics=("mean_interaction_score",))
    summary = {"metrics": {"mean_interaction_score": 0.6}}
    report = run_recompute_gate(summary, cfg, ledger_path=led)
    assert report is not None
    assert report.ok
    assert any(name == "mean_interaction_score" for name, _c, _r in report.verified)


def test_recompute_gate_mismatch_when_mean_interaction_tampered(tmp_path: Path) -> None:
    led = _network_ledger(tmp_path, [0.9, 0.6, 0.3])
    cfg = BiomniConfig(recompute_headline_metrics=("mean_interaction_score",))
    summary = {"metrics": {"mean_interaction_score": 0.99}}
    report = run_recompute_gate(summary, cfg, ledger_path=led)
    assert report is not None
    assert not report.ok


def _cross_tool_ledger(tmp_path: Path, edges: int = 44, length: int = 805) -> Path:
    """A-④: a ledger with TWO tools' returns — STRING ppi_enrichment (edge count)
    AND query_uniprot ACE2 (residue length) — the two provenance anchors the
    cross-tool oracle composes into one metric.
    """
    led = tmp_path / "provenance.jsonl"
    stringdb = {
        "tool": "query_stringdb",
        "raw_return": [{"number_of_edges": edges, "expected_number_of_edges": 1}],
    }
    uniprot = {
        "tool": "query_uniprot",
        "raw_return": {"success": True, "result": {"sequence": {"length": length}}},
    }
    with led.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(stringdb) + "\n")
        fh.write(json.dumps(uniprot) + "\n")
    return led


def test_context_populates_protein_length_from_ledger(tmp_path: Path) -> None:
    # A-④: the stage hook must surface the UniProt residue length so the
    # cross-tool oracle can bind it to the STRING edge count.
    led = _cross_tool_ledger(tmp_path, edges=44, length=805)
    ctx = collect_recompute_context({}, ledger_path=led)
    assert ctx is not None
    assert ctx.network == {"number_of_edges": 44.0, "expected_number_of_edges": 1.0}
    assert ctx.protein_length == 805.0


def test_recompute_gate_verifies_edges_per_ace2_residue_from_ledger(tmp_path: Path) -> None:
    led = _cross_tool_ledger(tmp_path, edges=44, length=805)
    cfg = BiomniConfig(recompute_headline_metrics=("edges_per_ace2_residue",))
    summary = {"metrics": {"edges_per_ace2_residue": 44.0 / 805.0}}
    report = run_recompute_gate(summary, cfg, ledger_path=led)
    assert report is not None
    assert report.ok
    assert any(name == "edges_per_ace2_residue" for name, _c, _r in report.verified)


def test_recompute_gate_mismatch_when_edges_per_residue_tampered(tmp_path: Path) -> None:
    led = _cross_tool_ledger(tmp_path, edges=44, length=805)
    cfg = BiomniConfig(recompute_headline_metrics=("edges_per_ace2_residue",))
    summary = {"metrics": {"edges_per_ace2_residue": 0.999}}
    report = run_recompute_gate(summary, cfg, ledger_path=led)
    assert report is not None
    assert not report.ok
