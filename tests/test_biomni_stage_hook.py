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
    resolve_ledger_path,
    run_biomni_gates,
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
