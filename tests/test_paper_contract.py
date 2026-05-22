from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from researchclaw.llm.client import LLMResponse


class _FakeLLM:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMResponse:
        self.calls.append({"messages": messages, "kwargs": kwargs})
        return LLMResponse(content=self.response, model="fake-model")


def _write_summary(run_dir: Path) -> None:
    stage14 = run_dir / "stage-14"
    stage14.mkdir(parents=True)
    (stage14 / "charts").mkdir()
    (stage14 / "charts" / "accuracy.png").write_bytes(b"png")
    summary = {
        "metrics_summary": {
            "Baseline/metric": {"mean": 0.72, "min": 0.71, "max": 0.73, "count": 3},
            "Proposed/metric": {"mean": 0.81, "min": 0.80, "max": 0.82, "count": 3},
        },
        "condition_summaries": {
            "Baseline": {"metrics": {"metric": 0.72}},
            "Proposed": {"metrics": {"metric": 0.81}},
        },
        "best_run": {
            "metrics": {
                "Baseline/0/metric": 0.71,
                "Baseline/1/metric": 0.72,
                "Baseline/2/metric": 0.73,
                "Proposed/0/metric": 0.80,
                "Proposed/1/metric": 0.81,
                "Proposed/2/metric": 0.82,
                "primary_metric": 0.81,
            }
        },
    }
    (stage14 / "experiment_summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )


def _write_conformal_summary_without_baseline(run_dir: Path) -> None:
    stage14 = run_dir / "stage-14"
    stage14.mkdir(parents=True)
    summary = {
        "metrics_summary": {
            "marginal_covariate_0.5/coverage_rate": {
                "mean": 0.9061,
                "min": 0.9061,
                "max": 0.9061,
                "count": 3,
            },
            "marginal_covariate_0.5/coverage_gap": {
                "mean": 0.0061,
                "min": 0.0039,
                "max": 0.0191,
                "count": 3,
            },
            "marginal_covariate_0.5/average_prediction_set_size": {
                "mean": 0.9061,
                "min": 0.8961,
                "max": 0.9191,
                "count": 3,
            },
            "conditional_covariate_0.5/average_prediction_set_size": {
                "mean": 1.7672,
                "min": 1.7003,
                "max": 1.9331,
                "count": 3,
            },
            "marginal_covariate_0.7/coverage_rate": {
                "mean": 0.8941,
                "min": 0.8941,
                "max": 0.8941,
                "count": 3,
            },
        },
        "condition_summaries": {
            "marginal_covariate_0.5": {
                "metrics": {
                    "coverage_rate": 0.9061,
                    "coverage_gap": 0.0061,
                    "average_prediction_set_size": 0.9061,
                },
                "n_seeds": 3,
            },
            "conditional_covariate_0.5": {
                "metrics": {
                    "coverage_rate": 0.9081,
                    "coverage_gap": 0.0081,
                    "average_prediction_set_size": 1.7672,
                },
                "n_seeds": 3,
            },
        },
    }
    (stage14 / "experiment_summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )


def test_build_paper_contract_records_allowed_numbers_conditions_and_figures(tmp_path: Path) -> None:
    from researchclaw.pipeline.paper_contract import build_paper_contract

    run_dir = tmp_path / "run"
    _write_summary(run_dir)

    contract = build_paper_contract(run_dir, metric_direction="maximize")

    assert contract["allowed_conditions"] == ["Baseline", "Proposed"]
    assert 0.81 in contract["allowed_numbers"]
    assert 81.0 in contract["allowed_numbers"]
    assert contract["available_figures"] == ["charts/accuracy.png"]
    assert contract["rules"]["missing_results"] == "state_not_evaluated"


def test_build_paper_contract_excludes_invalid_conformal_set_size_claims(
    tmp_path: Path,
) -> None:
    from researchclaw.pipeline.paper_contract import build_paper_contract

    run_dir = tmp_path / "run"
    _write_conformal_summary_without_baseline(run_dir)

    contract = build_paper_contract(run_dir, metric_direction="minimize")

    ledger = contract["claim_ledger"]
    invalid = ledger["invalid_metric_claims"]
    assert any(
        item["metric"] == "average_prediction_set_size"
        and item["condition"] == "marginal_covariate_0.5"
        and item["reason"] == "prediction_set_size_below_one"
        for item in invalid
    )
    assert {
        "condition": "conditional_covariate_0.5",
        "metric": "average_prediction_set_size",
        "value": 1.7672,
        "count": 3,
    } in ledger["valid_metric_claims"]
    assert 0.5 in contract["allowed_numbers"]


def test_render_paper_contract_instruction_forbids_comparative_claims_without_baseline(
    tmp_path: Path,
) -> None:
    from researchclaw.pipeline.paper_contract import (
        build_paper_contract,
        render_paper_contract_instruction,
    )

    run_dir = tmp_path / "run"
    _write_conformal_summary_without_baseline(run_dir)

    instruction = render_paper_contract_instruction(
        build_paper_contract(run_dir, metric_direction="minimize")
    )

    assert "CLAIM LEDGER" in instruction
    assert "Do NOT claim percentage improvements" in instruction
    assert "average_prediction_set_size" in instruction
    assert "prediction set size below 1.0" in instruction


def test_load_paper_contract_backfills_missing_claim_ledger(tmp_path: Path) -> None:
    from researchclaw.pipeline.paper_contract import load_paper_contract

    run_dir = tmp_path / "run"
    _write_conformal_summary_without_baseline(run_dir)
    stage16 = run_dir / "stage-16"
    stage16.mkdir()
    (stage16 / "paper_contract.json").write_text(
        json.dumps(
            {
                "version": 1,
                "allowed_conditions": ["marginal_covariate_0.5"],
                "allowed_numbers": [0.9061],
                "available_figures": [],
            }
        ),
        encoding="utf-8",
    )

    contract = load_paper_contract(run_dir)

    assert "claim_ledger" in contract
    assert contract["claim_ledger"]["invalid_metric_claims"]
    assert 0.7 in contract["allowed_numbers"]


def test_render_paper_contract_instruction_is_strict_about_evidence(tmp_path: Path) -> None:
    from researchclaw.pipeline.paper_contract import (
        build_paper_contract,
        render_paper_contract_instruction,
    )

    run_dir = tmp_path / "run"
    _write_summary(run_dir)

    instruction = render_paper_contract_instruction(
        build_paper_contract(run_dir, metric_direction="maximize")
    )

    assert "PAPER CONTRACT" in instruction
    assert "ONLY use these numeric values" in instruction
    assert "Baseline" in instruction
    assert "Proposed" in instruction
    assert "not evaluated" in instruction


def test_contract_violation_detection_rejects_unverified_result_numbers(tmp_path: Path) -> None:
    from researchclaw.pipeline.paper_contract import (
        build_paper_contract,
        find_paper_contract_violations,
    )

    run_dir = tmp_path / "run"
    _write_summary(run_dir)
    contract = build_paper_contract(run_dir, metric_direction="maximize")
    paper = """
## Results

The proposed method reaches 0.95 accuracy, while the baseline reaches 0.72.

## Discussion

Smith et al. reported 95.0 in 2025.
"""

    violations = find_paper_contract_violations(paper, contract)

    assert any("0.95" in v for v in violations)
    assert not any("0.72" in v for v in violations)
    assert not any("2025" in v for v in violations)


def test_contract_violation_detection_rejects_invalid_metric_context(
    tmp_path: Path,
) -> None:
    from researchclaw.pipeline.paper_contract import (
        build_paper_contract,
        find_paper_contract_violations,
        sanitize_paper_contract_violations,
    )

    run_dir = tmp_path / "run"
    _write_conformal_summary_without_baseline(run_dir)
    contract = build_paper_contract(run_dir, metric_direction="minimize")
    paper = """
## Results

The marginal regime achieved an average prediction set size of 0.9061.
"""

    violations = find_paper_contract_violations(paper, contract)
    sanitized, report = sanitize_paper_contract_violations(paper, contract)

    assert any("invalid metric claim" in v for v in violations)
    assert "average prediction set size of --" in sanitized
    assert report["replacement_count"] == 1


def test_contract_violation_detection_does_not_treat_small_integers_as_invalid_metric_values(
    tmp_path: Path,
) -> None:
    from researchclaw.pipeline.paper_contract import (
        build_paper_contract,
        find_paper_contract_violations,
    )

    run_dir = tmp_path / "run"
    _write_conformal_summary_without_baseline(run_dir)
    contract = build_paper_contract(run_dir, metric_direction="minimize")
    paper = """
## Results

The average set size analysis is reported in Table -1 and Figure -5.
"""

    violations = find_paper_contract_violations(paper, contract)

    assert violations == []


def test_contract_violation_detection_scopes_invalid_metric_to_nearby_value(
    tmp_path: Path,
) -> None:
    from researchclaw.pipeline.paper_contract import (
        build_paper_contract,
        find_paper_contract_violations,
    )

    run_dir = tmp_path / "run"
    _write_conformal_summary_without_baseline(run_dir)
    contract = build_paper_contract(run_dir, metric_direction="minimize")
    paper = """
## Results

The coverage rate was 0.9061 and the average set size was 1.7672.
"""

    violations = find_paper_contract_violations(paper, contract)

    assert violations == []


def test_contract_violation_detection_stops_scope_at_other_metric_phrase(
    tmp_path: Path,
) -> None:
    from researchclaw.pipeline.paper_contract import (
        build_paper_contract,
        find_paper_contract_violations,
    )

    run_dir = tmp_path / "run"
    _write_conformal_summary_without_baseline(run_dir)
    contract = build_paper_contract(run_dir, metric_direction="minimize")
    paper = """
## Results

S-CP-C0.5 has an average set size of 1.7672, while S-CP-M0.7 has a coverage rate of 0.8941 and a set size of 1.7672.
"""

    violations = find_paper_contract_violations(paper, contract)

    assert violations == []


def test_contract_violation_detection_ignores_setup_and_figure_reference_numbers(
    tmp_path: Path,
) -> None:
    from researchclaw.pipeline.paper_contract import (
        build_paper_contract,
        find_paper_contract_violations,
    )

    run_dir = tmp_path / "run"
    _write_conformal_summary_without_baseline(run_dir)
    contract = build_paper_contract(run_dir, metric_direction="minimize")
    paper = """
## Experiments

The synthetic benchmark has 20 features and uses a window of 100 samples.

## Results

As shown in Figure 6 and Table 3, the severe condition has lambda 0.7.
"""

    violations = find_paper_contract_violations(paper, contract)

    assert violations == []


def test_contract_violation_detection_still_checks_lines_with_citations(tmp_path: Path) -> None:
    from researchclaw.pipeline.paper_contract import (
        build_paper_contract,
        find_paper_contract_violations,
    )

    run_dir = tmp_path / "run"
    _write_summary(run_dir)
    contract = build_paper_contract(run_dir, metric_direction="maximize")
    paper = """
## Results

The proposed method reaches 0.95 accuracy [smith2025baseline].
"""

    violations = find_paper_contract_violations(paper, contract)

    assert any("0.95" in v for v in violations)


def test_contract_violation_detection_handles_numbered_result_headings(tmp_path: Path) -> None:
    from researchclaw.pipeline.paper_contract import (
        build_paper_contract,
        find_paper_contract_violations,
    )

    run_dir = tmp_path / "run"
    _write_summary(run_dir)
    contract = build_paper_contract(run_dir, metric_direction="maximize")
    paper = """
## 7. Results

The proposed method reaches 0.95 accuracy.
"""

    violations = find_paper_contract_violations(paper, contract)

    assert any("0.95" in v for v in violations)


def test_contract_violation_detection_ignores_heading_numbers(tmp_path: Path) -> None:
    from researchclaw.pipeline.paper_contract import (
        build_paper_contract,
        find_paper_contract_violations,
    )

    run_dir = tmp_path / "run"
    _write_summary(run_dir)
    contract = build_paper_contract(run_dir, metric_direction="maximize")
    paper = """
## 5.4 Ablation Study

The proposed method reaches 0.81 accuracy.
"""

    violations = find_paper_contract_violations(paper, contract)

    assert violations == []


def test_sanitize_paper_contract_violations_replaces_only_unsupported_numbers(
    tmp_path: Path,
) -> None:
    from researchclaw.pipeline.paper_contract import (
        build_paper_contract,
        find_paper_contract_violations,
        sanitize_paper_contract_violations,
    )

    run_dir = tmp_path / "run"
    _write_summary(run_dir)
    contract = build_paper_contract(run_dir, metric_direction="maximize")
    paper = """
## Results

| Method | Accuracy |
| --- | --- |
| Proposed | 0.81 |
| Unsupported | 0.95 |

## Discussion

Smith et al. reported 95.0 in 2025.
"""

    sanitized, report = sanitize_paper_contract_violations(paper, contract)

    assert "| Proposed | 0.81 |" in sanitized
    assert "| Unsupported | -- |" in sanitized
    assert "95.0 in 2025" in sanitized
    assert report["replacement_count"] == 1
    assert find_paper_contract_violations(sanitized, contract) == []


def test_repair_paper_contract_violations_accepts_clean_llm_revision(tmp_path: Path) -> None:
    from researchclaw.pipeline.paper_contract import (
        build_paper_contract,
        repair_paper_contract_violations,
    )

    run_dir = tmp_path / "run"
    _write_summary(run_dir)
    contract = build_paper_contract(run_dir, metric_direction="maximize")
    paper = """
## Results

The proposed method reaches 0.95 accuracy.
"""
    llm = _FakeLLM(
        """
## Results

The proposed method reaches 0.81 accuracy.
"""
    )

    repaired, report = repair_paper_contract_violations(
        paper,
        contract,
        llm=llm,
        stage_label="Stage 17",
    )

    assert "0.81" in repaired
    assert "0.95" not in repaired
    assert report["repaired"] is True
    assert report["accepted"] is True
    assert report["initial_violation_count"] == 1
    assert report["final_violation_count"] == 0
    assert len(llm.calls) == 1


def test_repair_paper_contract_violations_keeps_original_when_revision_still_violates(
    tmp_path: Path,
) -> None:
    from researchclaw.pipeline.paper_contract import (
        build_paper_contract,
        repair_paper_contract_violations,
    )

    run_dir = tmp_path / "run"
    _write_summary(run_dir)
    contract = build_paper_contract(run_dir, metric_direction="maximize")
    paper = """
## Results

The proposed method reaches 0.95 accuracy.
"""
    llm = _FakeLLM(
        """
## Results

The proposed method reaches 0.94 accuracy.
"""
    )

    repaired, report = repair_paper_contract_violations(
        paper,
        contract,
        llm=llm,
        stage_label="Stage 19",
    )

    assert repaired == paper
    assert report["repaired"] is True
    assert report["accepted"] is False
    assert report["initial_violation_count"] == 1
    assert report["final_violation_count"] == 1
