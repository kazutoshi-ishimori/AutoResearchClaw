"""Regression test: COVID biomni-mode config wiring stays consistent.

The actual workspace config lives at
``workspaces/covid-repurposing-paper/config_covid_biomni.yaml`` but
``workspaces/`` is gitignored (workspaces are local-only). This test pins the
wiring through a fixture under ``tests/fixtures/`` so CI catches drift between
``BiomniConfig`` and the YAML schema even though the workspace copy isn't
tracked. Keep the fixture in sync when changing biomni keys.
"""

from __future__ import annotations

from pathlib import Path

from researchclaw.config import load_config

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "covid_biomni_minimal.yaml"


def test_covid_biomni_fixture_uses_agentic_mode() -> None:
    cfg = load_config(FIXTURE, check_paths=False)
    assert cfg.experiment.mode == "agentic"


def test_covid_biomni_fixture_wires_biomni_block() -> None:
    cfg = load_config(FIXTURE, check_paths=False)
    b = cfg.experiment.biomni
    # server_cmd routes to the isolated Biomni venv + standalone runner.
    assert "biomni_tool_runner.py" in b.server_cmd
    assert ".venv/bin/python" in b.server_cmd
    # Allowlist: at least the deterministic database tools we plan to replay.
    assert "database.query_uniprot" in b.tool_allowlist
    # Entity-DB pointers (paths only — gitignored files, regenerated locally).
    assert b.gene_db.endswith("genes_human.tsv")
    assert b.drug_hashes_db.endswith("drugbank_ids.sha256")
    assert b.pathway_db.endswith("pathways_kegg2021.tsv")
    # Provenance ledger (relative path — runner.py resolves it under run_dir).
    assert b.provenance_path
    # Replay gate opted in for COVID (deterministic DB queries are safe to replay).
    assert b.replay_enabled is True
    # Recompute headline metric opted in (oracle is registered by experiment code).
    assert "auroc_phase2plus" in b.recompute_headline_metrics
