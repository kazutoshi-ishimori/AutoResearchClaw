"""Renderers for the paper's Figure 2 (coverage grid) and Table 1 (evidence).

The renderers are *pure functions of ``run_suite`` output*: given the outcome
rows for each dataset column, they emit LaTeX. The whole point of generating the
figure/table mechanically is that a future regression (a row that stops being
caught) surfaces as a ``\\miss`` in the artifact instead of being silently
transcribed away — so these tests hand-build outcome lists that deliberately
include a by-design gap (△) and an undetected regression (✗), and assert the
render reflects them faithfully.
"""

from __future__ import annotations

from researchclaw.experiment.verify.fabrication_attacks import AttackOutcome


def _caught(attack_id: str, evidence=()) -> AttackOutcome:
    return AttackOutcome(
        attack_id=attack_id,
        name="caught attack",
        layer="④ recompute",
        honest_passes=True,
        detected=True,
        evidence=evidence,
    )


def _gap(attack_id: str = "T3′") -> AttackOutcome:
    return AttackOutcome(
        attack_id=attack_id,
        name="by-design gap",
        layer="④/⑤ (insufficient)",
        honest_passes=True,
        detected=False,
        by_design_gap=True,
        detail="routing contract → study_findings",
        evidence=(("auroc_phase2plus", 1.0, 1.0),),
    )


def _regression(attack_id: str) -> AttackOutcome:
    """A row that SHOULD be caught but isn't — a genuine regression, not by design."""
    return AttackOutcome(
        attack_id=attack_id,
        name="silently broken defense",
        layer="④ recompute",
        honest_passes=True,
        detected=False,
        by_design_gap=False,
    )


def _false_positive(attack_id: str) -> AttackOutcome:
    """Catches the attack BUT also fires on the honest control — a false positive.

    This is the DUAL regression to a miss: the harness promises true-positives
    *and* the absence of false-positives, so a defense that flags honest input
    must render distinctly, never as a plain \\caught.
    """
    return AttackOutcome(
        attack_id=attack_id,
        name="trigger-happy defense",
        layer="④ recompute",
        honest_passes=False,
        detected=True,
        by_design_gap=False,
    )


def _fp_cols():
    from researchclaw.experiment.verify.coverage_report import Column

    col = Column(
        label="IPF",
        ledger_path="x/provenance.jsonl",
        ledger_sha256="aaaa1111",
        outcomes=(_caught("T1"), _false_positive("T2")),
    )
    return [col]


def _cols():
    from researchclaw.experiment.verify.coverage_report import Column

    ipf = Column(
        label="IPF",
        ledger_path="workspaces/ipf-repurposing/.../provenance.jsonl",
        ledger_sha256="aaaa1111",
        outcomes=(
            _caught("T1", evidence=(("ppi_enrichment_fold", 1008.0, 9.0),)),
            _gap("T3′"),
            _regression("T6"),
        ),
    )
    covid = Column(
        label="COVID",
        ledger_path="workspaces/covid-repurposing-paper/.../provenance.jsonl",
        ledger_sha256="bbbb2222",
        outcomes=(
            _caught("T1", evidence=(("ppi_enrichment_fold", 1043.0, 44.0),)),
            _gap("T3′"),
            _regression("T6"),
        ),
    )
    return [ipf, covid]


# --- Figure 2: the compact attack × dataset grid ----------------------------

def test_figure2_renders_one_symbol_per_attack_per_dataset():
    from researchclaw.experiment.verify.coverage_report import render_figure2

    tex = render_figure2(_cols(), harness_commit="deadbee")

    assert "IPF" in tex and "COVID" in tex          # both columns labelled
    assert r"T1 & \caught & \caught" in tex          # caught in both datasets
    assert r"\gap" in tex                            # the by-design row is a gap, not ✅
    # A regression MUST show as a miss, never silently render green.
    assert r"\miss" in tex


def test_figure2_marks_undetected_nonbydesign_as_regression():
    from researchclaw.experiment.verify.coverage_report import render_figure2

    tex = render_figure2(_cols(), harness_commit="deadbee")
    # T6 is detected=False & by_design_gap=False in both columns → \miss, not \gap.
    assert r"T6 & \miss & \miss" in tex


def test_figure2_false_positive_renders_distinct_from_caught():
    """A defense that fires on the honest control is a false positive — the DUAL
    of a miss. It must render distinctly (\\falsepos), never as a plain \\caught,
    or the harness's 'absence of false-positives' claim is invisible."""
    from researchclaw.experiment.verify.coverage_report import render_figure2

    tex = render_figure2(_fp_cols(), harness_commit="deadbee")
    assert r"\falsepos" in tex
    # T2 is honest_passes=False → must NOT be the plain caught symbol.
    assert r"T2 & \caught" not in tex


# --- Table 1: per-row evidence (claimed vs recomputed) ----------------------

def test_table1_renders_claimed_vs_recomputed_numbers():
    from researchclaw.experiment.verify.coverage_report import render_table1

    tex = render_table1(_cols(), harness_commit="deadbee")

    # The numbers that earn the second column must appear verbatim — this is the
    # provenance the paper's thesis demands, not a hand-copied constant.
    assert "1008" in tex and "9" in tex        # IPF T1 claimed 1008 vs recomputed 9
    assert "1043" in tex and "44" in tex       # COVID T1 claimed 1043 vs recomputed 44
    # The per-attack note prose now lives in the caption (render_evidence_notes),
    # NOT in the table body — so the body stays legible at a fixed font size
    # without a \resizebox. The mitigation prose must be absent here.
    assert "findings" not in tex
    assert "Note" not in tex                   # no Note column header either


def test_render_evidence_notes_emits_per_attack_caption_prose():
    """The per-attack notes (dataset-independent: by-design gap / conditional /
    detail) are emitted as a caption-ready fragment, keyed by attack id, so the
    table body can drop its wide Note column. Same provenance header as the rest."""
    from researchclaw.experiment.verify.coverage_report import render_evidence_notes

    tex = render_evidence_notes(_cols(), harness_commit="deadbee")

    # the by-design-gap row's mitigation note is keyed by its attack id
    assert "T3" in tex
    assert "by-design gap" in tex
    assert "study" in tex and "findings" in tex   # routing-contract mitigation prose
    # provenance header binds the fragment to the ledgers + harness commit
    assert "deadbee" in tex


def test_table1_false_positive_renders_distinct_from_caught():
    """Table 1 must surface the honest dimension too: a false-positive row shows
    \\falsepos in its status cell, not the plain caught symbol."""
    from researchclaw.experiment.verify.coverage_report import render_table1

    tex = render_table1(_fp_cols(), harness_commit="deadbee")
    assert r"\falsepos" in tex
    # The T2 row must not open its status cell with the plain caught symbol.
    assert r"T2 & " in tex
    t2_line = next(ln for ln in tex.splitlines() if ln.startswith(r"T2 & "))
    assert r"\falsepos" in t2_line and r"\caught" not in t2_line


# --- provenance header (thesis applied to the artifact itself) --------------

def test_render_binds_ledger_sha_and_harness_commit():
    from researchclaw.experiment.verify.coverage_report import (
        render_figure2,
        render_table1,
    )

    for render in (render_figure2, render_table1):
        tex = render(_cols(), harness_commit="deadbee")
        assert "aaaa1111" in tex and "bbbb2222" in tex   # each ledger's sha
        assert "deadbee" in tex                          # harness commit
