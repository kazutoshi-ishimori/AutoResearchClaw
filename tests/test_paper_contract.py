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
