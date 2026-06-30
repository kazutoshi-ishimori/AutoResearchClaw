"""Producer side of the layer-④ recompute contract (Phase 2 pipeline-wiring).

``stage_hook.collect_recompute_context`` reads ``ranking`` + ``positives`` from
the *top level* of a parsed ``experiment_summary.json``. The canonical
``results.json`` the agent writes lands in ``exp_data['structured_results']``
during stage 14, but nothing promotes those two fields to the top of the
summary — so layer ④ can never see them. This seam closes that gap with a pure,
generic passthrough: any experiment whose results.json carries a well-formed
ranking/positives gets the recompute oracle for free, with no COVID specifics
baked into the generic pipeline.

Silence-on-absence parity with the rest of the verification stack: a missing or
malformed field is simply not surfaced (never raised), so the recompute gate
stays quiet exactly as it does for a missing oracle.
"""

from __future__ import annotations

from researchclaw.pipeline.stage_impls._analysis import _recompute_contract_fields


def test_surfaces_ranking_and_positives_from_structured_results() -> None:
    """A well-formed results.json yields both contract fields verbatim."""
    structured = {
        "ranking": [["DB001", 0.91], ["DB002", 0.80], ["DB003", 0.12]],
        "positives": ["DB001", "DB002"],
        "metrics": {"auroc_phase2plus": 0.83},
    }
    out = _recompute_contract_fields(structured)
    assert out == {
        "ranking": [["DB001", 0.91], ["DB002", 0.80], ["DB003", 0.12]],
        "positives": ["DB001", "DB002"],
    }


def test_missing_fields_yield_empty_dict() -> None:
    """No ranking/positives ⇒ nothing surfaced (silence is not failure)."""
    assert _recompute_contract_fields({"metrics": {"auroc_phase2plus": 0.5}}) == {}


def test_non_dict_structured_results_yields_empty_dict() -> None:
    """A list / None / scalar results.json must not raise."""
    assert _recompute_contract_fields(None) == {}
    assert _recompute_contract_fields([1, 2, 3]) == {}
    assert _recompute_contract_fields("nope") == {}


def test_empty_lists_are_not_surfaced() -> None:
    """An empty ranking or positives carries no signal; drop it."""
    out = _recompute_contract_fields({"ranking": [], "positives": []})
    assert out == {}


def test_fields_are_surfaced_independently() -> None:
    """Only one field present ⇒ surface just that one (gate stays silent)."""
    assert _recompute_contract_fields({"ranking": [["A", 0.5]]}) == {
        "ranking": [["A", 0.5]]
    }
    assert _recompute_contract_fields({"positives": ["A"]}) == {"positives": ["A"]}


def test_round_trips_into_collect_recompute_context() -> None:
    """The surfaced fields feed the consumer contract without translation."""
    from researchclaw.experiment.verify.stage_hook import collect_recompute_context

    structured = {
        "ranking": [["A", 0.9], ["B", 0.8], ["C", 0.4]],
        "positives": ["A", "B"],
    }
    summary = {"best_run": {"metrics": {}}}
    summary.update(_recompute_contract_fields(structured))

    ctx = collect_recompute_context(summary)
    assert ctx is not None
    assert ctx.ranking == (("A", 0.9), ("B", 0.8), ("C", 0.4))
    assert ctx.positives == frozenset({"A", "B"})
