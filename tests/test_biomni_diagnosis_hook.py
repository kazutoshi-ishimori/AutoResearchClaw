"""Tests for wiring the Biomni verification gates into experiment_diagnosis (C5).

A failed claim-binding (⑤) or entity-ID (②) check must surface as a *critical*
deficiency so the existing fail-loud machinery (has_critical) halts the
pipeline rather than writing a paper on unverified numbers.
"""

from __future__ import annotations

from researchclaw.experiment.verify.claim_binding import ClaimBindingReport
from researchclaw.experiment.verify.entity_gate import EntityReport
from researchclaw.experiment.verify.recompute import RecomputeReport
from researchclaw.experiment.verify.replay import ReplayReport
from researchclaw.pipeline.experiment_diagnosis import (
    DeficiencyType,
    ExperimentDiagnosis,
    add_biomni_verification_deficiencies,
)


def test_unbacked_claim_becomes_critical_deficiency() -> None:
    diag = ExperimentDiagnosis()
    claim = ClaimBindingReport(backed=[("proximity", 0.42)], unbacked=[("made_up", 9.99)])
    add_biomni_verification_deficiencies(diag, claim_binding=claim)

    assert diag.has_critical()
    types = [d.type for d in diag.deficiencies]
    assert DeficiencyType.FABRICATED_METRIC in types
    assert any("made_up" in d.description for d in diag.deficiencies)


def test_unknown_entity_becomes_critical_deficiency() -> None:
    diag = ExperimentDiagnosis()
    entity = EntityReport(unknown={"drug": ["DB99999"]})
    add_biomni_verification_deficiencies(diag, entity=entity)

    assert diag.has_critical()
    types = [d.type for d in diag.deficiencies]
    assert DeficiencyType.UNKNOWN_ENTITY in types
    assert any("DB99999" in d.description for d in diag.deficiencies)


def test_nonreproducible_tool_becomes_critical_deficiency() -> None:
    diag = ExperimentDiagnosis()
    replay = ReplayReport(drifted=[("query_kegg", "sha_a", "sha_b")])
    add_biomni_verification_deficiencies(diag, replay=replay)

    assert diag.has_critical()
    types = [d.type for d in diag.deficiencies]
    assert DeficiencyType.NONREPRODUCIBLE_TOOL in types
    assert any("query_kegg" in d.description for d in diag.deficiencies)


def test_recompute_mismatch_becomes_critical_deficiency() -> None:
    diag = ExperimentDiagnosis()
    recompute = RecomputeReport(mismatched=[("auroc", 0.78, 0.55)])
    add_biomni_verification_deficiencies(diag, recompute=recompute)

    assert diag.has_critical()
    types = [d.type for d in diag.deficiencies]
    assert DeficiencyType.RECOMPUTE_MISMATCH in types
    assert any("auroc" in d.description for d in diag.deficiencies)


def test_clean_reports_add_no_deficiencies() -> None:
    diag = ExperimentDiagnosis()
    add_biomni_verification_deficiencies(
        diag,
        claim_binding=ClaimBindingReport(backed=[("proximity", 0.42)]),
        entity=EntityReport(),
        replay=ReplayReport(matched=["query_uniprot"]),
        recompute=RecomputeReport(verified=[("auroc", 0.78, 0.78)]),
    )
    assert not diag.has_critical()
    assert diag.deficiencies == []
