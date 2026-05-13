from __future__ import annotations

from types import SimpleNamespace

from researchclaw.pipeline.stage_impls._experiment_design import (
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
