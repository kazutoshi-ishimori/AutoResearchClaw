from __future__ import annotations

from types import SimpleNamespace

from researchclaw.pipeline.stage_impls._experiment_design import (
    _apply_single_cell_sandbox_execution_constraints,
    _apply_tabular_cpu_budget_constraints,
    _flatten_guardrail_text,
    _stage9_single_cell_sandbox_guidance,
    _stage9_tabular_cpu_budget_guidance,
    _validate_domain_benchmark_guardrail,
)


def _profile(domain_id: str, display_name: str = "") -> SimpleNamespace:
    return SimpleNamespace(domain_id=domain_id, display_name=display_name)


def test_single_cell_rejects_generic_image_benchmark_dataset() -> None:
    result = _validate_domain_benchmark_guardrail(
        topic="Fly Cell Atlas h5ad scRNA-seq analysis of Drosophila ecdysone response",
        domain_profile=_profile("biology_singlecell", "Single-Cell Biology"),
        plan={"datasets": ["CIFAR-10"], "baselines": ["ResNet"]},
    )

    assert result["ok"] is False
    assert result["single_cell_domain"] is True
    assert result["violations"][0]["family"] == "image_classification"


def test_single_cell_rejects_generic_citation_graph_benchmark_plan() -> None:
    result = _validate_domain_benchmark_guardrail(
        topic="Fly Cell Atlas h5ad scRNA-seq analysis of Drosophila ecdysone response",
        domain_profile=_profile("biology_singlecell", "Single-Cell Biology"),
        plan={"datasets": ["Fly Cell Atlas h5ad"]},
        benchmark_plan={
            "selected_benchmarks": [
                {"name": "PubMed", "api": "torch_geometric.datasets.Planetoid"}
            ]
        },
    )

    assert result["ok"] is False
    assert result["violations"][0]["family"] == "citation_graph"


def test_single_cell_accepts_fca_h5ad_plan() -> None:
    result = _validate_domain_benchmark_guardrail(
        topic="Fly Cell Atlas h5ad scRNA-seq analysis of Drosophila ecdysone response",
        domain_profile=_profile("biology_singlecell", "Single-Cell Biology"),
        plan={
            "datasets": ["Fly Cell Atlas AnnData h5ad"],
            "baselines": ["scanpy PCA", "scVI"],
            "metrics": ["silhouette score"],
        },
    )

    assert result["ok"] is True
    assert result["violations"] == []


def test_vision_topic_allows_cifar_benchmark() -> None:
    result = _validate_domain_benchmark_guardrail(
        topic="Image classification with efficient convolutional networks",
        domain_profile=_profile("ml_vision", "Computer Vision"),
        plan={"datasets": ["CIFAR-10"], "baselines": ["ResNet"]},
    )

    assert result["ok"] is True


def test_graph_topic_allows_pubmed_planetoid_benchmark() -> None:
    result = _validate_domain_benchmark_guardrail(
        topic="Graph neural network node classification on citation graphs",
        domain_profile=_profile("ml_graph", "Graph Machine Learning"),
        plan={"datasets": ["PubMed Planetoid"], "baselines": ["GCN"]},
    )

    assert result["ok"] is True


def test_single_cell_ignores_survey_only_benchmark_mentions() -> None:
    result = _validate_domain_benchmark_guardrail(
        topic="Fly Cell Atlas h5ad scRNA-seq analysis of Drosophila ecdysone response",
        domain_profile=_profile("biology_singlecell", "Single-Cell Biology"),
        plan={"datasets": ["Fly Cell Atlas h5ad"]},
        benchmark_plan={
            "survey_results": [{"name": "CIFAR-10"}, {"name": "PubMed"}],
            "selected_benchmarks": [],
        },
    )

    assert result["ok"] is True


def test_single_cell_sandbox_rewrites_heavy_tool_requirements() -> None:
    plan = {
        "datasets": ["Fly Cell Atlas AnnData"],
        "proposed_methods": [
            {
                "name": "hdWGCNA_fingerprint_mapper",
                "implementation_spec": {
                    "algorithm_steps": (
                        "Run hdWGCNA, validate with MAGIC and ALRA, then run "
                        "CellChat for ligand-receptor inference."
                    ),
                    "key_hyperparameters": {"score_method": "UCell"},
                },
            }
        ],
        "baselines": ["scanpy PCA baseline"],
    }

    constrained, report = _apply_single_cell_sandbox_execution_constraints(
        plan,
        topic="Fly Cell Atlas h5ad scRNA-seq analysis",
        domain_profile=_profile("biology_singlecell", "Single-Cell Biology"),
        experiment_mode="sandbox",
    )

    text = _flatten_guardrail_text({
        "datasets": constrained.get("datasets"),
        "proposed_methods": constrained.get("proposed_methods"),
        "baselines": constrained.get("baselines"),
        "sandbox_execution": constrained.get("sandbox_execution"),
    }).lower()
    for forbidden in ("hdwgcna", "magic", "alra", "cellchat", "scanpy", "ucell"):
        assert forbidden not in text
    for required in ("numpy", "pandas", "sklearn", "synthetic"):
        assert required in text
    assert report["ok"] is True
    assert report["rewritten"] is True


def test_single_cell_sandbox_constraints_do_not_touch_other_domains() -> None:
    plan = {"proposed_methods": ["CellChat-style message passing benchmark"]}

    constrained, report = _apply_single_cell_sandbox_execution_constraints(
        plan,
        topic="Graph neural network node classification",
        domain_profile=_profile("ml_graph", "Graph Machine Learning"),
        experiment_mode="sandbox",
    )

    assert constrained == plan
    assert report["rewritten"] is False


def test_stage9_single_cell_sandbox_guidance_mentions_lightweight_mapping() -> None:
    guidance = _stage9_single_cell_sandbox_guidance(
        topic="Fly Cell Atlas single-cell RNA-seq AnnData analysis",
        domain_profile=_profile("biology_singlecell", "Single-Cell Biology"),
        experiment_mode="sandbox",
    )

    assert "hdWGCNA" in guidance
    assert "CellChat" in guidance
    assert "numpy/pandas/sklearn" in guidance


def test_tabular_cpu_budget_constraints_clamp_heavy_synthetic_plan() -> None:
    plan = {
        "datasets": [
            {
                "name": "synthetic_covariate_shift",
                "generation_protocol": "Generate N=20000 tabular samples",
            }
        ],
        "proposed_methods": [
            {
                "name": "stability_aware_conformal",
                "key_hyperparameters": {
                    "n_samples": 20000,
                    "n_estimators": 200,
                },
            }
        ],
        "baselines": ["standard split conformal"],
        "regime_factors": {
            "shift_type": ["continuous", "categorical", "missingness"],
            "shift_magnitude": [0.2, 0.5, 0.8],
        },
        "compute_budget": {"n_samples": 20000, "n_estimators": 200},
        "seeds": [11, 22, 33, 44, 55],
    }

    constrained, report = _apply_tabular_cpu_budget_constraints(
        plan,
        topic=(
            "Stability-aware conformal prediction for reliable tabular "
            "classification under covariate shift using controlled synthetic "
            "benchmark data"
        ),
        domain_profile=_profile("ml_tabular", "Tabular Machine Learning"),
        experiment_mode="sandbox",
    )

    cpu_budget = constrained["sandbox_execution"]["cpu_budget"]
    assert cpu_budget["max_samples"] == 5000
    assert cpu_budget["max_estimators"] == 50
    assert cpu_budget["max_shift_regimes"] == 4
    assert cpu_budget["seed_count"] == 3
    assert constrained["regime_factors"]["shift_type"] == ["continuous", "categorical"]
    assert constrained["regime_factors"]["shift_magnitude"] == [0.2, 0.5]
    assert constrained["compute_budget"]["n_samples"] == 5000
    assert constrained["compute_budget"]["n_estimators"] == 50
    assert constrained["seeds"] == [11, 22, 33]
    text = _flatten_guardrail_text(constrained)
    assert "N<=5000" in text
    assert report["ok"] is True
    assert report["rewritten"] is True


def test_tabular_cpu_budget_constraints_do_not_touch_other_domains() -> None:
    plan = {
        "datasets": ["CIFAR-10"],
        "compute_budget": {"n_samples": 50000, "n_estimators": 200},
    }

    constrained, report = _apply_tabular_cpu_budget_constraints(
        plan,
        topic="Image classification with convolutional networks",
        domain_profile=_profile("ml_vision", "Computer Vision"),
        experiment_mode="sandbox",
    )

    assert constrained == plan
    assert report["rewritten"] is False


def test_stage9_tabular_cpu_budget_guidance_mentions_exact_limits() -> None:
    guidance = _stage9_tabular_cpu_budget_guidance(
        topic="Conformal prediction for tabular classification under covariate shift",
        domain_profile=_profile("ml_tabular", "Tabular Machine Learning"),
        experiment_mode="sandbox",
    )

    assert "n_samples <= 5000" in guidance
    assert "n_estimators <= 50" in guidance
    assert "shift regimes <= 4" in guidance
    assert "seeds = 3" in guidance
