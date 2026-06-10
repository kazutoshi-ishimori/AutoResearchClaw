"""Tests for the entity-ID gate (Phase 0, layer ②).

Gene and pathway IDs are checked against plaintext authoritative reference
sets. Drug IDs are checked by sha256 *hash membership only*, so the licensed
DrugBank identifier space never has to be stored in plaintext — honouring the
project's hash-only DrugBank policy.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from researchclaw.experiment.verify.entity_gate import EntityGate


def _h(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def _gate() -> EntityGate:
    return EntityGate(
        gene={"ACE2", "TMPRSS2"},
        drug_hashes={_h("DB00945")},
        pathway={"R-HSA-168256"},
    )


def test_known_ids_pass() -> None:
    report = _gate().check(
        {"gene": ["ACE2", "TMPRSS2"], "drug": ["DB00945"], "pathway": ["R-HSA-168256"]}
    )
    assert report.ok
    assert report.unknown == {}


def test_unknown_gene_is_flagged() -> None:
    report = _gate().check({"gene": ["ACE2", "NOTAGENE"]})
    assert not report.ok
    assert report.unknown == {"gene": ["NOTAGENE"]}


def test_drug_checked_by_hash_only() -> None:
    # The known drug passes via hash membership; an invented ID is flagged.
    report = _gate().check({"drug": ["DB00945", "DB99999"]})
    assert not report.ok
    assert report.unknown == {"drug": ["DB99999"]}


def test_loads_reference_sets_from_files(tmp_path: Path) -> None:
    gene_f = tmp_path / "hgnc.tsv"
    gene_f.write_text("ACE2\nTMPRSS2\n# comment\n\n")
    drug_f = tmp_path / "drugbank_ids.sha256"
    drug_f.write_text(_h("DB00945") + "\n")
    path_f = tmp_path / "reactome.tsv"
    path_f.write_text("R-HSA-168256\n")

    gate = EntityGate.from_files(gene=gene_f, drug_hashes=drug_f, pathway=path_f)
    report = gate.check({"gene": ["ACE2"], "drug": ["DB00945"], "pathway": ["R-HSA-168256"]})
    assert report.ok
