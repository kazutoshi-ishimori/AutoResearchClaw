"""Stage 12 (agentic): the results-emission contract appended to the prompt.

The free-form agentic agent is handed the Stage-9 ``exp_plan.yaml`` as its
prompt. Nothing there tells it the machine-readable shape the downstream
Biomni harness needs: a flat ``metrics`` object whose keys match the
``recompute_headline_metrics`` oracle names, so layer ④ can recompute each
from the ledger and layer ⑤ can bind it. Without that contract the agent
writes free-form output, ``_parse_result_metrics`` finds nothing, and Stage 12
fails with zero metrics.

These tests pin ``_build_agentic_results_contract`` (a pure, config-driven
helper) and the wiring that appends it to the agent prompt.

Honest-scope invariant: the contract specifies FORMULAS and FORMAT only. It
never coaches a target numeric value, and it routes ground-truth-dependent
quantities (AUROC, precision@k) OUT of the ledger-bound ``metrics`` object.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from researchclaw.config import (
    AgenticConfig,
    BiomniConfig,
    ExperimentConfig,
    RCConfig,
)
from researchclaw.pipeline.stages import StageStatus

_NETWORK_METRICS = (
    "ppi_enrichment_fold",
    "avg_node_degree",
    "mean_interaction_score",
)


def _make_config(recompute: tuple[str, ...]) -> MagicMock:
    cfg = MagicMock(spec=RCConfig)
    exp = MagicMock(spec=ExperimentConfig)
    exp.mode = "agentic"
    exp.time_budget_sec = 1800
    exp.metric_key = "auroc_ipf"
    exp.metric_direction = "maximize"
    exp.max_iterations = 1
    exp.agentic = AgenticConfig(
        image="rc-agentic:test",
        agent_install_cmd="",
        timeout_sec=1800,
    )
    exp.biomni = BiomniConfig(
        server_cmd="/opt/biomni/.venv/bin/python runner.py",
        tool_allowlist=("database.query_stringdb", "database.query_uniprot"),
        provenance_path="provenance.jsonl",
        recompute_headline_metrics=recompute,
    )
    cfg.experiment = exp
    return cfg


# ── unit: the pure contract builder ─────────────────────────────────────────


def test_contract_names_every_configured_network_metric() -> None:
    from researchclaw.pipeline.stage_impls._execution import (
        _build_agentic_results_contract,
    )

    contract = _build_agentic_results_contract(_make_config(_NETWORK_METRICS))

    for key in _NETWORK_METRICS:
        assert key in contract, f"contract must name the oracle key {key!r}"


def test_contract_states_each_recompute_formula() -> None:
    from researchclaw.pipeline.stage_impls._execution import (
        _build_agentic_results_contract,
    )

    contract = _build_agentic_results_contract(_make_config(_NETWORK_METRICS))

    # Formulas are stated in terms of the ledger's raw field names — never a
    # precomputed numeric target.
    assert "number_of_edges" in contract
    assert "expected_number_of_edges" in contract
    assert "number_of_nodes" in contract
    assert "score" in contract


def test_contract_pins_results_json_metrics_shape() -> None:
    from researchclaw.pipeline.stage_impls._execution import (
        _build_agentic_results_contract,
    )

    contract = _build_agentic_results_contract(_make_config(_NETWORK_METRICS))

    assert "results.json" in contract
    assert "metrics" in contract
    # AUROC / ground-truth-dependent results are reported but kept OUT of the
    # ledger-bound metrics object.
    assert "study_findings" in contract


def test_contract_routes_auroc_out_of_metrics_object() -> None:
    """AUROC must be named as a NON-ledger-bound quantity, not a metrics key."""
    from researchclaw.pipeline.stage_impls._execution import (
        _build_agentic_results_contract,
    )

    contract = _build_agentic_results_contract(_make_config(_NETWORK_METRICS))
    # The exact primary metric key must be discussed as excluded-from-metrics,
    # so the agent does not put it where ⑤ would flag it as fabricated.
    assert "auroc_ipf" in contract


def test_contract_is_config_driven_single_metric() -> None:
    """Only the configured subset appears — not the whole known registry."""
    from researchclaw.pipeline.stage_impls._execution import (
        _build_agentic_results_contract,
    )

    contract = _build_agentic_results_contract(
        _make_config(("mean_interaction_score",))
    )
    assert "mean_interaction_score" in contract
    assert "ppi_enrichment_fold" not in contract
    assert "avg_node_degree" not in contract


def test_contract_empty_when_no_recompute_metrics() -> None:
    """No configured oracle names → no contract (layer ④ disabled)."""
    from researchclaw.pipeline.stage_impls._execution import (
        _build_agentic_results_contract,
    )

    assert _build_agentic_results_contract(_make_config(())) == ""


def test_contract_omits_unknown_names() -> None:
    """A configured name with no known STRING/UniProt oracle is not coached."""
    from researchclaw.pipeline.stage_impls._execution import (
        _build_agentic_results_contract,
    )

    contract = _build_agentic_results_contract(
        _make_config(("some_unmapped_metric",))
    )
    # No oracle mapping → nothing to instruct → empty contract.
    assert contract == ""


def test_contract_states_no_numeric_target() -> None:
    """Honest-scope: the contract carries formulas, not a preconceived value.

    A blunt guard: none of the metric lines may pin an '= <number>' target.
    """
    import re

    from researchclaw.pipeline.stage_impls._execution import (
        _build_agentic_results_contract,
    )

    contract = _build_agentic_results_contract(_make_config(_NETWORK_METRICS))
    # e.g. "ppi_enrichment_fold = 44.0" would be coaching. The only digit we
    # permit in a formula is the literal factor 2 in 2*edges/nodes.
    for m in re.finditer(r"=\s*([0-9]+\.[0-9]+)", contract):
        pytest.fail(f"contract coaches a numeric target: {m.group(0)!r}")


# ── wiring: the contract reaches the agent prompt ───────────────────────────


def test_composed_prompt_reaches_run_agent_session(tmp_path: Path) -> None:
    from researchclaw.pipeline.stage_impls._execution import (
        _execute_experiment_run,
    )

    run_dir = tmp_path / "runs" / "run-x"
    (run_dir / "stage-12").mkdir(parents=True, exist_ok=True)
    stage_dir = run_dir / "stage-12"

    cfg = _make_config(_NETWORK_METRICS)

    result = MagicMock()
    result.returncode = 0
    result.metrics = {"ppi_enrichment_fold": 44.0}
    result.elapsed_sec = 120.0
    result.stdout = "{}"
    result.stderr = ""

    sandbox = MagicMock()
    sandbox.run_agent_session.return_value = result

    with patch(
        "researchclaw.experiment.factory.create_agentic_sandbox",
        return_value=sandbox,
    ), patch(
        "researchclaw.pipeline.stage_impls._execution._read_prior_artifact",
        return_value="# Base experiment plan\n",
    ), patch(
        "researchclaw.pipeline.stage_impls._execution._utcnow_iso",
        return_value="2026-07-11T10:00:00Z",
    ):
        sr = _execute_experiment_run(
            stage_dir, run_dir, cfg, MagicMock(), llm=None, prompts=None
        )

    assert sr.status == StageStatus.DONE
    prompt_arg = sandbox.run_agent_session.call_args.args[0]
    assert "# Base experiment plan" in prompt_arg
    assert "ppi_enrichment_fold" in prompt_arg
    assert "study_findings" in prompt_arg
