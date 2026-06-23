"""Tests for the experiment.biomni config block (Phase 0, C2).

The Biomni bridge runs under experiment mode ``agentic`` (ARC's own agent
orchestrates Biomni tools over MCP), so no new EXPERIMENT_MODES entry is
needed — only this optional sub-config carrying the bridge command, the tool
allowlist, the provenance ledger path, and the entity-DB / verification knobs.
"""

from __future__ import annotations

from researchclaw.config import BiomniConfig, _parse_experiment_config


def test_defaults_when_no_biomni_block() -> None:
    exp = _parse_experiment_config({})
    assert isinstance(exp.biomni, BiomniConfig)
    assert exp.biomni.provenance_path == "provenance.jsonl"
    assert exp.biomni.tool_allowlist == ()
    assert exp.biomni.replay_enabled is False


def test_parses_biomni_block() -> None:
    exp = _parse_experiment_config(
        {
            "biomni": {
                "server_cmd": "~/workspace/Biomni/.venv/bin/python external/biomni_bridge/biomni_tool_runner.py",
                "tool_allowlist": ["database.query_uniprot", "database.query_kegg"],
                "provenance_path": "provenance.jsonl",
                "gene_db": "data/refs/genes_human.tsv",
                "drug_hashes_db": "data/refs/drugbank_ids.sha256",
                "pathway_db": "data/refs/pathways_kegg2021.tsv",
                "replay_enabled": True,
                "replay_tolerance": 1e-6,
                "recompute_headline_metrics": ["proximity_z"],
            }
        }
    )
    b = exp.biomni
    assert "biomni_tool_runner.py" in b.server_cmd
    assert b.tool_allowlist == ("database.query_uniprot", "database.query_kegg")
    assert b.gene_db == "data/refs/genes_human.tsv"
    assert b.drug_hashes_db.endswith(".sha256")
    assert b.replay_enabled is True
    assert b.replay_tolerance == 1e-6
    assert b.recompute_headline_metrics == ("proximity_z",)


def test_tool_allowlist_accepts_single_string() -> None:
    exp = _parse_experiment_config({"biomni": {"tool_allowlist": "query_gene"}})
    assert exp.biomni.tool_allowlist == ("query_gene",)
