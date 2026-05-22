"""Stage 9: Experiment design."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import yaml

from researchclaw.adapters import AdapterBundle
from researchclaw.config import RCConfig
from researchclaw.llm.client import LLMClient
from researchclaw.pipeline._helpers import (
    StageResult,
    _build_context_preamble,
    _chat_with_prompt,
    _extract_yaml_block,
    _get_evolution_overlay,
    _load_hardware_profile,
    _read_prior_artifact,
    _safe_json_loads,
    _utcnow_iso,
)
from researchclaw.pipeline.stages import Stage, StageStatus
from researchclaw.prompts import PromptManager

logger = logging.getLogger(__name__)


def _normalize_plan_field(value: Any) -> list:
    """Normalize a plan field (baselines, proposed_methods, ablations, datasets)
    from any shape the LLM might produce into a flat list of items.

    Handles: list[str], list[dict], dict[str, Any], str, None.
    When the input is a dict, we preserve the full structure by converting each
    key-value pair into a dict item (with at least a 'name' key), rather than
    discarding either keys or values.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, dict):
        result = []
        for k, v in value.items():
            if isinstance(v, dict):
                # e.g. {"baseline_1": {"params": ...}} -> {"name": "baseline_1", "params": ...}
                item = dict(v)
                item.setdefault("name", str(k))
                result.append(item)
            else:
                # e.g. {"baseline_1": "description"} -> {"name": "baseline_1", "description": str(v)}
                result.append({"name": str(k), "description": str(v) if v else ""})
        return result
    if isinstance(value, list):
        return list(value)
    return [value]


def _plan_field_names(items: list) -> list[str]:
    """Extract string names from a normalized plan field for display/dedup."""
    result = []
    for item in items:
        if isinstance(item, dict):
            result.append(item.get("name", str(item)))
        else:
            result.append(str(item))
    return result


_SINGLE_CELL_TOPIC_TERMS = (
    "single cell",
    "single-cell",
    "scrna",
    "scrna-seq",
    "h5ad",
    "anndata",
    "scanpy",
    "scvi",
    "fly cell atlas",
    "cell atlas",
    "drosophila ecdysone",
)

_GENERIC_IMAGE_BENCHMARK_TERMS = (
    "cifar",
    "mnist",
    "fashionmnist",
    "fashion mnist",
    "stl-10",
    "stl10",
    "svhn",
    "celeba",
    "imagenet",
    "torchvision.datasets",
)

_GENERIC_CITATION_GRAPH_BENCHMARK_TERMS = (
    "planetoid",
    "pubmed",
    "cora",
    "citeseer",
    "torch_geometric.datasets.planetoid",
)

_SINGLE_CELL_SANDBOX_HEAVY_TOOL_REPLACEMENTS = (
    ("hdWGCNA", r"(?i)(?<![a-z0-9])hdwgcna(?=$|[^a-z0-9])", "lightweight_coexpression"),
    ("WGCNA", r"(?i)(?<![a-z0-9])wgcna(?=$|[^a-z0-9])", "lightweight_coexpression"),
    ("MAGIC", r"(?i)(?<![a-z0-9])magic(?=$|[^a-z0-9])", "dropout_noise_sensitivity"),
    ("ALRA", r"(?i)(?<![a-z0-9])alra(?=$|[^a-z0-9])", "dropout_noise_sensitivity"),
    ("CellChat", r"(?i)(?<![a-z0-9])cellchat(?=$|[^a-z0-9])", "ligand_expression_coupling"),
    ("UCell", r"(?i)(?<![a-z0-9])ucell(?=$|[^a-z0-9])", "rank_mean_module_score"),
    ("AddCure", r"(?i)(?<![a-z0-9])addcure(?=$|[^a-z0-9])", "rank_mean_module_score"),
    ("AddModuleScore", r"(?i)(?<![a-z0-9])addmodulescore(?=$|[^a-z0-9])", "rank_mean_module_score"),
    ("scanpy", r"(?i)(?<![a-z0-9])scanpy(?=$|[^a-z0-9])", "numpy_pandas_sklearn_preprocessing"),
    ("scVI", r"(?i)(?<![a-z0-9])scvi(?=$|[^a-z0-9])", "sklearn_pca_clustering"),
    ("AnnData", r"(?i)(?<![a-z0-9])anndata(?=$|[^a-z0-9])", "fca_like_synthetic_matrix"),
    ("h5ad", r"(?i)(?<![a-z0-9])h5ad(?=$|[^a-z0-9])", "self_contained_synthetic_matrix"),
)

_TABULAR_CPU_TOPIC_TERMS = (
    "tabular",
    "structured data",
    "conformal prediction",
    "covariate shift",
    "reliable machine learning",
    "uncertainty quantification",
    "synthetic benchmark",
)

_TABULAR_CPU_MAX_SAMPLES = 5000
_TABULAR_CPU_MAX_ESTIMATORS = 50
_TABULAR_CPU_MAX_SHIFT_REGIMES = 4
_TABULAR_CPU_SEED_COUNT = 3
_TABULAR_CONFORMAL_REQUIRED_BASELINE = {
    "name": "split_cp_baseline",
    "description": (
        "standard Split Conformal Prediction using the same base classifier, "
        "calibration split, seeds, and shift regimes as the proposed method"
    ),
    "role": "reference",
    "required_metrics": [
        "coverage_rate",
        "coverage_gap",
        "average_prediction_set_size",
    ],
}


def _profile_id(domain_profile: Any | None) -> str:
    return str(getattr(domain_profile, "domain_id", "") or "")


def _profile_display(domain_profile: Any | None) -> str:
    return str(getattr(domain_profile, "display_name", "") or "")


def _flatten_guardrail_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(
            f"{k} {_flatten_guardrail_text(v)}" for k, v in value.items()
        )
    if isinstance(value, (list, tuple, set)):
        return " ".join(_flatten_guardrail_text(item) for item in value)
    return str(value)


def _contains_guardrail_term(text: str, term: str) -> bool:
    haystack = text.lower()
    needle = term.lower()
    if "." in needle or "_" in needle:
        return needle in haystack
    return re.search(
        rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])",
        haystack,
    ) is not None


def _is_single_cell_topic(
    topic: str,
    domain_profile: Any | None,
    plan: dict[str, Any] | None = None,
) -> bool:
    domain_id = _profile_id(domain_profile)
    if domain_id == "biology_singlecell":
        return True
    context = " ".join(
        (
            topic,
            domain_id,
            _profile_display(domain_profile),
            _flatten_guardrail_text(plan) if plan else "",
        )
    )
    return any(
        _contains_guardrail_term(context, term)
        for term in _SINGLE_CELL_TOPIC_TERMS
    )


def _is_tabular_cpu_budget_topic(
    topic: str,
    domain_profile: Any | None,
    plan: dict[str, Any] | None = None,
) -> bool:
    """Detect tabular synthetic/shift topics that need local CPU budget caps."""
    domain_id = _profile_id(domain_profile)
    context = " ".join(
        (
            topic,
            domain_id,
            _profile_display(domain_profile),
            _flatten_guardrail_text(plan) if plan else "",
        )
    )
    if domain_id == "ml_tabular":
        return True
    has_tabular_signal = any(
        _contains_guardrail_term(context, term)
        for term in ("tabular", "structured data")
    )
    has_budgeted_ml_signal = any(
        _contains_guardrail_term(context, term)
        for term in (
            "conformal prediction",
            "covariate shift",
            "reliable machine learning",
            "uncertainty quantification",
            "synthetic benchmark",
        )
    )
    return has_tabular_signal and has_budgeted_ml_signal


def _stage9_tabular_cpu_budget_guidance(
    *,
    topic: str,
    domain_profile: Any | None,
    experiment_mode: str,
) -> str:
    """Prompt guidance for runnable tabular CPU experiments in sandbox mode."""
    if experiment_mode != "sandbox":
        return ""
    if not _is_tabular_cpu_budget_topic(topic, domain_profile):
        return ""
    return (
        "\n\n## TABULAR CPU BUDGET CONSTRAINT (MANDATORY)\n"
        "- The Stage 10 implementation will run on local CPU in sandbox mode.\n"
        "- Use self-contained synthetic tabular data only unless a local dataset "
        "is already generated by the project code.\n"
        "- Set n_samples <= 5000 for every generated dataset split combined.\n"
        "- Set tree/ensemble n_estimators <= 50 for every model.\n"
        "- Evaluate shift regimes <= 4 total, for example 2 shift types x 2 "
        "magnitudes.\n"
        "- Use seeds = 3 exactly. Report mean and standard deviation across "
        "those seeds.\n"
        "- Include a required baseline named `split_cp_baseline`: standard "
        "Split Conformal Prediction using the same base classifier, "
        "calibration split, seeds, and shift regimes as the proposed method.\n"
        "- The plan MUST report coverage_rate, coverage_gap, and "
        "average_prediction_set_size for both `split_cp_baseline` and the "
        "proposed method in every shift regime.\n"
        "- Prefer numpy/pandas/sklearn/scipy implementations and avoid any "
        "network download or GPU requirement.\n"
    )


def _stage9_single_cell_sandbox_guidance(
    *,
    topic: str,
    domain_profile: Any | None,
    experiment_mode: str,
) -> str:
    """Prompt guidance for runnable single-cell plans in local sandbox mode."""
    if experiment_mode != "sandbox":
        return ""
    if not _is_single_cell_topic(topic, domain_profile):
        return ""
    return (
        "\n\n## SINGLE-CELL SANDBOX EXECUTION CONSTRAINT (MANDATORY)\n"
        "- The Stage 10 implementation will run in local sandbox mode with no "
        "scanpy/anndata requirement and no real FCA h5ad file.\n"
        "- Do NOT make hdWGCNA, MAGIC, ALRA, CellChat, UCell, scanpy, scVI, "
        "or AnnData required implementation dependencies.\n"
        "- If those tools are scientifically relevant, describe them only as "
        "conceptual inspirations and map them to a runnable "
        "numpy/pandas/sklearn approximation.\n"
        "- The experiment plan MUST include self-contained synthetic "
        "single-cell-like data with tissue, cell_type, gene metadata, two "
        "baselines, one proposed method, and non-empty metrics across seeds.\n"
    )


def _rewrite_single_cell_sandbox_text(value: str) -> tuple[str, list[str]]:
    rewritten = value
    hits: list[str] = []
    for label, pattern, replacement in _SINGLE_CELL_SANDBOX_HEAVY_TOOL_REPLACEMENTS:
        if re.search(pattern, rewritten):
            hits.append(label)
            rewritten = re.sub(pattern, replacement, rewritten)
    return rewritten, hits


def _rewrite_single_cell_sandbox_value(value: Any) -> tuple[Any, list[str]]:
    if isinstance(value, str):
        return _rewrite_single_cell_sandbox_text(value)
    if isinstance(value, list):
        new_items = []
        hits: list[str] = []
        for item in value:
            new_item, item_hits = _rewrite_single_cell_sandbox_value(item)
            new_items.append(new_item)
            hits.extend(item_hits)
        return new_items, hits
    if isinstance(value, dict):
        new_dict: dict[str, Any] = {}
        hits = []
        for key, item in value.items():
            new_key, key_hits = _rewrite_single_cell_sandbox_text(str(key))
            new_item, item_hits = _rewrite_single_cell_sandbox_value(item)
            new_dict[new_key] = new_item
            hits.extend(key_hits)
            hits.extend(item_hits)
        return new_dict, hits
    return value, []


def _apply_single_cell_sandbox_execution_constraints(
    plan: dict[str, Any],
    *,
    topic: str,
    domain_profile: Any | None,
    experiment_mode: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Keep single-cell sandbox plans runnable with local lightweight tools."""
    single_cell_domain = _is_single_cell_topic(topic, domain_profile, plan)
    report: dict[str, Any] = {
        "ok": True,
        "single_cell_domain": single_cell_domain,
        "experiment_mode": experiment_mode,
        "rewritten": False,
        "rewritten_terms": [],
        "warnings": [],
    }
    if experiment_mode != "sandbox" or not single_cell_domain:
        return plan, report

    constrained, hits = _rewrite_single_cell_sandbox_value(plan)
    if not isinstance(constrained, dict):
        constrained = dict(plan)
    unique_hits = sorted(set(hits))
    report["rewritten_terms"] = unique_hits
    report["rewritten"] = bool(unique_hits)
    if unique_hits:
        report["warnings"].append(
            "Heavy single-cell tool requirements were rewritten to local "
            "numpy/pandas/sklearn approximations for sandbox execution."
        )

    constrained["sandbox_execution"] = {
        "mode": "lightweight_single_cell_approximation",
        "allowed_libraries": ["numpy", "pandas", "sklearn", "scipy"],
        "dependency_policy": (
            "Do not require specialized single-cell IO or analysis packages in "
            "the runnable sandbox path."
        ),
        "data_contract": (
            "Generate a self-contained synthetic single-cell-like expression "
            "matrix with tissue, cell_type, gene annotations, pathway genes, "
            "and seeded perturbations before analysis."
        ),
        "method_mapping": {
            "coexpression_modules": (
                "Pearson/Spearman correlation matrices, module scores, and "
                "bootstrap stability summaries"
            ),
            "dropout_robustness": (
                "seeded dropout/noise perturbation sensitivity instead of "
                "external imputation packages"
            ),
            "ligand_coupling": (
                "curated ligand/receptor gene expression correlation scores"
            ),
        },
        "minimum_outputs": (
            "non-empty metrics for at least two baselines and one proposed "
            "condition across three seeds"
        ),
    }
    return constrained, report


def _rewrite_tabular_cpu_budget_value(value: Any) -> tuple[Any, bool]:
    """Clamp common literal budget fields inside a plan-like structure."""
    rewritten = False
    if isinstance(value, str):
        new_value = re.sub(
            r"(?i)\bN\s*=\s*\d+",
            f"N<={_TABULAR_CPU_MAX_SAMPLES}",
            value,
        )
        new_value = re.sub(
            r"(?i)\bn_samples\s*=\s*\d+",
            f"n_samples<={_TABULAR_CPU_MAX_SAMPLES}",
            new_value,
        )
        new_value = re.sub(
            r"(?i)\bn_estimators\s*=\s*\d+",
            f"n_estimators<={_TABULAR_CPU_MAX_ESTIMATORS}",
            new_value,
        )
        return new_value, new_value != value
    if isinstance(value, list):
        new_items = []
        for item in value:
            new_item, item_rewritten = _rewrite_tabular_cpu_budget_value(item)
            new_items.append(new_item)
            rewritten = rewritten or item_rewritten
        return new_items, rewritten
    if isinstance(value, dict):
        new_dict: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key).lower()
            new_item, item_rewritten = _rewrite_tabular_cpu_budget_value(item)
            if (
                key_text in {"n_samples", "num_samples", "sample_count", "n_train"}
                and isinstance(new_item, int)
                and new_item > _TABULAR_CPU_MAX_SAMPLES
            ):
                new_item = _TABULAR_CPU_MAX_SAMPLES
                item_rewritten = True
            if (
                key_text in {"n_estimators", "num_estimators"}
                and isinstance(new_item, int)
                and new_item > _TABULAR_CPU_MAX_ESTIMATORS
            ):
                new_item = _TABULAR_CPU_MAX_ESTIMATORS
                item_rewritten = True
            new_dict[key] = new_item
            rewritten = rewritten or item_rewritten
        return new_dict, rewritten
    return value, False


def _has_split_cp_baseline(items: Any) -> bool:
    text = _flatten_guardrail_text(items).lower()
    return any(
        marker in text
        for marker in (
            "split_cp_baseline",
            "split cp",
            "split-cp",
            "split conformal",
            "standard conformal",
        )
    )


def _apply_tabular_cpu_budget_constraints(
    plan: dict[str, Any],
    *,
    topic: str,
    domain_profile: Any | None,
    experiment_mode: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Keep tabular synthetic sandbox plans within local CPU limits."""
    tabular_domain = _is_tabular_cpu_budget_topic(topic, domain_profile, plan)
    report: dict[str, Any] = {
        "ok": True,
        "tabular_cpu_budget_domain": tabular_domain,
        "experiment_mode": experiment_mode,
        "rewritten": False,
        "warnings": [],
        "limits": {
            "max_samples": _TABULAR_CPU_MAX_SAMPLES,
            "max_estimators": _TABULAR_CPU_MAX_ESTIMATORS,
            "max_shift_regimes": _TABULAR_CPU_MAX_SHIFT_REGIMES,
            "seed_count": _TABULAR_CPU_SEED_COUNT,
        },
    }
    if experiment_mode != "sandbox" or not tabular_domain:
        return plan, report

    constrained, rewritten = _rewrite_tabular_cpu_budget_value(plan)
    if not isinstance(constrained, dict):
        constrained = dict(plan)

    baselines = _normalize_plan_field(constrained.get("baselines"))
    required_baselines: list[str] = []
    if not _has_split_cp_baseline(baselines):
        baselines.insert(0, dict(_TABULAR_CONFORMAL_REQUIRED_BASELINE))
        constrained["baselines"] = baselines
        required_baselines.append("split_cp_baseline")
        rewritten = True

    regime_factors = constrained.get("regime_factors")
    if isinstance(regime_factors, dict):
        shift_types = regime_factors.get("shift_type") or regime_factors.get("shift_types")
        shift_magnitudes = (
            regime_factors.get("shift_magnitude")
            or regime_factors.get("shift_magnitudes")
        )
        if isinstance(shift_types, list) and isinstance(shift_magnitudes, list):
            max_types = min(len(shift_types), 2)
            max_mags = min(
                len(shift_magnitudes),
                max(1, _TABULAR_CPU_MAX_SHIFT_REGIMES // max(1, max_types)),
            )
            if len(shift_types) > max_types:
                regime_factors["shift_type"] = shift_types[:max_types]
                if "shift_types" in regime_factors:
                    regime_factors["shift_types"] = shift_types[:max_types]
                rewritten = True
            if len(shift_magnitudes) > max_mags:
                regime_factors["shift_magnitude"] = shift_magnitudes[:max_mags]
                if "shift_magnitudes" in regime_factors:
                    regime_factors["shift_magnitudes"] = shift_magnitudes[:max_mags]
                rewritten = True

    seeds = constrained.get("seeds")
    if isinstance(seeds, list) and len(seeds) > _TABULAR_CPU_SEED_COUNT:
        constrained["seeds"] = seeds[:_TABULAR_CPU_SEED_COUNT]
        rewritten = True
    elif not isinstance(seeds, list):
        constrained["seeds"] = [42, 123, 456]
        rewritten = True

    compute_budget = constrained.get("compute_budget")
    if not isinstance(compute_budget, dict):
        compute_budget = {}
        constrained["compute_budget"] = compute_budget
        rewritten = True
    for key, limit in (
        ("n_samples", _TABULAR_CPU_MAX_SAMPLES),
        ("max_samples", _TABULAR_CPU_MAX_SAMPLES),
        ("n_estimators", _TABULAR_CPU_MAX_ESTIMATORS),
        ("max_estimators", _TABULAR_CPU_MAX_ESTIMATORS),
    ):
        current = compute_budget.get(key)
        if current is None or (isinstance(current, int) and current > limit):
            compute_budget[key] = limit
            rewritten = True
    compute_budget["max_shift_regimes"] = _TABULAR_CPU_MAX_SHIFT_REGIMES
    compute_budget["seed_count"] = _TABULAR_CPU_SEED_COUNT

    sandbox_execution = constrained.get("sandbox_execution")
    if not isinstance(sandbox_execution, dict):
        sandbox_execution = {}
        constrained["sandbox_execution"] = sandbox_execution
        rewritten = True
    sandbox_execution["mode"] = "cpu_bounded_tabular_synthetic"
    sandbox_execution["cpu_budget"] = {
        "max_samples": _TABULAR_CPU_MAX_SAMPLES,
        "max_estimators": _TABULAR_CPU_MAX_ESTIMATORS,
        "max_shift_regimes": _TABULAR_CPU_MAX_SHIFT_REGIMES,
        "seed_count": _TABULAR_CPU_SEED_COUNT,
    }
    sandbox_execution["data_contract"] = (
        "Use self-contained synthetic tabular data generated by the project "
        f"code with N<={_TABULAR_CPU_MAX_SAMPLES}; do not download external data."
    )
    sandbox_execution["allowed_libraries"] = ["numpy", "pandas", "sklearn", "scipy"]
    sandbox_execution["required_baselines"] = [
        {
            "name": "split_cp_baseline",
            "metrics": [
                "coverage_rate",
                "coverage_gap",
                "average_prediction_set_size",
            ],
        }
    ]

    if not constrained.get("metrics"):
        constrained["metrics"] = [
            {"name": "coverage_gap", "direction": "minimize"},
            {"name": "coverage_rate", "direction": "target_0.90"},
            {"name": "average_prediction_set_size", "direction": "minimize"},
            {"name": "runtime_seconds", "direction": "minimize"},
        ]
        rewritten = True
    report["rewritten"] = rewritten
    report["required_baselines"] = required_baselines or ["split_cp_baseline"]
    if rewritten:
        report["warnings"].append(
            "Tabular synthetic sandbox plan was constrained to local CPU "
            "limits: n_samples<=5000, n_estimators<=50, shift regimes<=4, "
            "seeds=3."
        )
    return constrained, report


def _collect_guardrail_texts(value: Any, location: str) -> list[tuple[str, str]]:
    if value is None:
        return []
    if isinstance(value, dict):
        texts: list[tuple[str, str]] = []
        for key, item in value.items():
            texts.extend(_collect_guardrail_texts(item, f"{location}.{key}"))
        return texts
    if isinstance(value, (list, tuple, set)):
        texts = []
        for index, item in enumerate(value):
            texts.extend(_collect_guardrail_texts(item, f"{location}[{index}]"))
        return texts
    return [(location, str(value))]


def _collect_plan_benchmark_texts(plan: dict[str, Any]) -> list[tuple[str, str]]:
    texts: list[tuple[str, str]] = []
    for key in ("dataset", "datasets", "benchmark", "benchmarks"):
        if key in plan:
            texts.extend(_collect_guardrail_texts(plan.get(key), f"exp_plan.{key}"))
    return texts


def _collect_benchmark_plan_texts(
    benchmark_plan: dict[str, Any] | None,
) -> list[tuple[str, str]]:
    if not isinstance(benchmark_plan, dict):
        return []
    texts: list[tuple[str, str]] = []
    selected_keys = (
        "selected_benchmarks",
        "selected_datasets",
        "final_benchmarks",
        "final_datasets",
    )
    for key in selected_keys:
        if key in benchmark_plan:
            texts.extend(
                _collect_guardrail_texts(
                    benchmark_plan.get(key),
                    f"benchmark_plan.{key}",
                )
            )
    return texts


def _validate_domain_benchmark_guardrail(
    *,
    topic: str,
    domain_profile: Any | None,
    plan: dict[str, Any],
    benchmark_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Reject generic benchmark choices only when they conflict with the topic.

    Single-cell/FCA/h5ad work should not silently drift into generic vision or
    citation-graph benchmarks. Vision and graph ML domains can still use those
    benchmarks normally.
    """
    domain_id = _profile_id(domain_profile) or "unknown"
    single_cell_domain = _is_single_cell_topic(topic, domain_profile, plan)
    result: dict[str, Any] = {
        "ok": True,
        "domain_id": domain_id,
        "single_cell_domain": single_cell_domain,
        "violations": [],
        "warnings": [],
    }
    if not single_cell_domain:
        return result

    targets = _collect_plan_benchmark_texts(plan)
    targets.extend(_collect_benchmark_plan_texts(benchmark_plan))

    families = (
        ("image_classification", _GENERIC_IMAGE_BENCHMARK_TERMS),
        ("citation_graph", _GENERIC_CITATION_GRAPH_BENCHMARK_TERMS),
    )
    seen: set[tuple[str, str, str]] = set()
    for location, text in targets:
        for family, terms in families:
            for term in terms:
                if not _contains_guardrail_term(text, term):
                    continue
                marker = (family, term, location)
                if marker in seen:
                    continue
                seen.add(marker)
                result["violations"].append({
                    "family": family,
                    "term": term,
                    "location": location,
                    "message": (
                        f"Single-cell/FCA/h5ad experiment selected generic "
                        f"{family} benchmark '{term}' at {location}."
                    ),
                })

    if result["violations"]:
        result["ok"] = False
    return result


def _execute_experiment_design(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    hypotheses = _read_prior_artifact(run_dir, "hypotheses.md") or ""
    preamble = _build_context_preamble(
        config, run_dir, include_goal=True, include_hypotheses=True
    )
    plan: dict[str, Any] | None = None

    # ── Domain detection ──────────────────────────────────────────────────
    # Detect the research domain early so we can adapt experiment design
    # and code generation. For ML domains, existing behavior is unchanged.
    _domain_profile = None
    try:
        from researchclaw.domains.detector import detect_domain as _detect_domain_adv
        _domain_profile = _detect_domain_adv(
            topic=config.research.topic,
            hypotheses=hypotheses,
        )
        logger.info(
            "Domain detected: %s (%s)",
            _domain_profile.display_name,
            _domain_profile.domain_id,
        )
        # Persist domain profile for Stage 10
        import json as _json_dd
        (stage_dir / "domain_profile.json").write_text(
            _json_dd.dumps({
                "domain_id": _domain_profile.domain_id,
                "display_name": _domain_profile.display_name,
                "experiment_paradigm": _domain_profile.experiment_paradigm,
                "core_libraries": _domain_profile.core_libraries,
                "gpu_required": _domain_profile.gpu_required,
            }, indent=2),
            encoding="utf-8",
        )
    except Exception:  # noqa: BLE001
        logger.debug("Domain detection unavailable", exc_info=True)
    if llm is not None:
        _pm = prompts or PromptManager()
        # Pass dataset_guidance block for experiment design
        try:
            _dg_block = _pm.block("dataset_guidance")
        except (KeyError, Exception):  # noqa: BLE001
            _dg_block = ""
        # I-08: Inject RL step guidance for RL topics
        _rl_kws = ("reinforcement learning", "ppo", "sac", "td3", "ddpg",
                    "dqn", "mujoco", "continuous control", "actor-critic",
                    "policy gradient", "exploration bonus")
        _is_rl_topic = any(kw in config.research.topic.lower() for kw in _rl_kws)
        if _is_rl_topic:
            try:
                _dg_block += _pm.block("rl_step_guidance")
            except Exception:  # noqa: BLE001
                pass
            # Improvement G: For RL with short budget, constrain to classic control
            if config.experiment.time_budget_sec <= 3600:
                _dg_block += (
                    "\n\n## RL TIME CONSTRAINT (MANDATORY):\n"
                    f"Your time budget is {config.experiment.time_budget_sec}s (≤ 3600s).\n"
                    "You MUST use ONLY classic control environments: "
                    "CartPole-v1, Pendulum-v1, MountainCar-v0, Acrobot-v1, LunarLander-v3.\n"
                    "Do NOT use MuJoCo (HalfCheetah, Hopper, Walker2d, Ant, Humanoid) — "
                    "they require >5000s for meaningful training.\n"
                )
            if config.experiment.time_budget_sec <= 1800:
                _dg_block += (
                    "Time budget ≤ 1800s: use ONLY CartPole-v1 or Pendulum-v1 "
                    "(the simplest environments).\n"
                )
        # F-01: Inject framework docs for experiment design
        try:
            from researchclaw.data import detect_frameworks, load_framework_docs
            _fw_ids = detect_frameworks(config.research.topic, hypotheses)
            if _fw_ids:
                _fw_docs = load_framework_docs(_fw_ids, max_chars=4000)
                if _fw_docs:
                    _dg_block += _fw_docs
        except Exception:  # noqa: BLE001
            pass
        _dg_block += _stage9_single_cell_sandbox_guidance(
            topic=config.research.topic,
            domain_profile=_domain_profile,
            experiment_mode=config.experiment.mode,
        )
        _dg_block += _stage9_tabular_cpu_budget_guidance(
            topic=config.research.topic,
            domain_profile=_domain_profile,
            experiment_mode=config.experiment.mode,
        )
        # Improvement A: Compute hardware profile + per-condition budget
        _hw_profile_str = (
            "- GPU: NVIDIA RTX 6000 Ada (49140 MB VRAM)\n"
            "- GPU count: 1\n"
            "- CPU: shared server"
        )
        _per_condition_sec = int(config.experiment.time_budget_sec * 0.7 / 6)
        _tier1 = "CIFAR-10, CIFAR-100, MNIST, FashionMNIST, STL-10, SVHN"

        _overlay = _get_evolution_overlay(run_dir, "experiment_design")
        sp = _pm.for_stage(
            "experiment_design",
            evolution_overlay=_overlay,
            preamble=preamble,
            hypotheses=hypotheses,
            dataset_guidance=_dg_block,
            time_budget_sec=config.experiment.time_budget_sec,
            metric_key=config.experiment.metric_key,
            metric_direction=config.experiment.metric_direction,
            hardware_profile=_hw_profile_str,
            per_condition_budget_sec=_per_condition_sec,
            available_tier1_datasets=_tier1,
        )
        resp = _chat_with_prompt(
            llm,
            sp.system,
            sp.user,
            json_mode=sp.json_mode,
            max_tokens=sp.max_tokens,
        )
        raw_yaml = _extract_yaml_block(resp.content)
        try:
            parsed = yaml.safe_load(raw_yaml)
        except yaml.YAMLError:
            parsed = None
        # Fallback: reasoning models sometimes emit the YAML without fences
        # or wrapped in prose. Try parsing the whole response as YAML.
        if not isinstance(parsed, dict):
            try:
                parsed = yaml.safe_load(resp.content)
            except yaml.YAMLError:
                pass
        # Last fallback: try to find any YAML-like dict in the response
        if not isinstance(parsed, dict):
            import re as _re_yaml

            # Look for lines starting with known keys
            _yaml_lines = []
            _capturing = False
            for line in resp.content.splitlines():
                if _re_yaml.match(
                    r"^(baselines|proposed_methods|ablations|datasets|"
                    r"metrics|objectives|risks|compute_budget)\s*:",
                    line,
                ):
                    _capturing = True
                if _capturing:
                    if line.strip() == "" or line.startswith("```"):
                        continue
                    if line.startswith("#") or line.startswith("**"):
                        continue
                    _yaml_lines.append(line)
            if _yaml_lines:
                try:
                    parsed = yaml.safe_load("\n".join(_yaml_lines))
                except yaml.YAMLError:
                    pass
        if isinstance(parsed, dict):
            plan = parsed
        else:
            logger.warning(
                "Stage 09: LLM response could not be parsed as YAML "
                "(len=%d, first 200 chars: %s). Content extraction method "
                "returned: %s",
                len(resp.content),
                resp.content[:200],
                raw_yaml[:200] if raw_yaml else "<empty>",
            )
            # BUG-12: Retry with a stricter, shorter prompt
            if llm is not None:
                logger.info("Stage 09: Retrying with strict YAML-only prompt...")
                _retry_prompt = (
                    "Output ONLY valid YAML. No prose, no markdown fences, no explanation.\n"
                    f"Topic: {config.research.topic}\n"
                    "Required keys: baselines, proposed_methods, ablations, "
                    "datasets, metrics, objectives, risks, compute_budget.\n"
                    "Each key maps to a list of strings."
                )
                _retry_resp = _chat_with_prompt(
                    llm,
                    "You output ONLY valid YAML. Nothing else.",
                    _retry_prompt,
                    max_tokens=4096,
                )
                try:
                    _retry_parsed = yaml.safe_load(_retry_resp.content)
                    if isinstance(_retry_parsed, dict):
                        plan = _retry_parsed
                        logger.info("Stage 09: Strict YAML retry succeeded.")
                except yaml.YAMLError:
                    pass

    # BUG-12: Fallback 4 — extract method/baseline names from Stage 8 hypotheses
    if plan is None:
        _hyp_text = _read_prior_artifact(run_dir, "hypotheses.md") or ""
        if _hyp_text:
            import re as _re_hyp
            # Extract method-like names from hypothesis text
            _method_candidates = _re_hyp.findall(
                r"(?:proposed|our|novel|new)\s+(?:method|approach|algorithm|framework|model)[:\s]+[\"']?([A-Za-z][\w-]+)",
                _hyp_text, _re_hyp.IGNORECASE,
            )
            _baseline_candidates = _re_hyp.findall(
                r"(?:baseline|compare|existing|standard|traditional)\s+(?:method|approach|model)?[:\s]+[\"']?([A-Za-z][\w-]+)",
                _hyp_text, _re_hyp.IGNORECASE,
            )
            if _method_candidates or _baseline_candidates:
                logger.info(
                    "Stage 09: Extracted names from hypotheses: methods=%s, baselines=%s",
                    _method_candidates[:3], _baseline_candidates[:3],
                )
                plan = {
                    "topic": config.research.topic,
                    "generated": _utcnow_iso(),
                    "objectives": ["Evaluate hypotheses with controlled experiments"],
                    "datasets": ["primary_dataset"],
                    "baselines": _baseline_candidates[:3] or ["baseline_1", "baseline_2"],
                    "proposed_methods": _method_candidates[:3] or ["proposed_method"],
                    "ablations": ["without_key_component", "simplified_version"],
                    "metrics": [config.experiment.metric_key, "secondary_metric"],
                    "risks": ["validity threats", "confounding variables"],
                    "compute_budget": {"max_gpu": 1, "max_hours": 4},
                }

    if plan is None:
        # BUG-12: Use domain-aware names instead of fully generic placeholders
        _topic_prefix = config.research.topic.split()[0] if config.research.topic else "method"
        logger.warning(
            "Stage 09: LLM failed to produce valid experiment plan YAML. "
            "Using topic-derived fallback."
        )
        plan = {
            "topic": config.research.topic,
            "generated": _utcnow_iso(),
            "objectives": ["Evaluate hypotheses with controlled experiments"],
            "datasets": ["primary_dataset", "secondary_dataset"],
            "baselines": [f"{_topic_prefix}_baseline_1", f"{_topic_prefix}_baseline_2"],
            "proposed_methods": [f"{_topic_prefix}_proposed", f"{_topic_prefix}_variant"],
            "ablations": ["without_key_component", "simplified_version"],
            "metrics": [config.experiment.metric_key, "secondary_metric"],
            "risks": ["validity threats", "confounding variables"],
            "compute_budget": {"max_gpu": 1, "max_hours": 4},
        }
    # ── BA: BenchmarkAgent — intelligent dataset/baseline selection ──────
    _benchmark_plan = None
    # BUG-40: Skip BenchmarkAgent for non-ML domains — it has no relevant
    # benchmarks for physics/chemistry/mathematics/etc. and would inject
    # wrong datasets (e.g., CIFAR-10 for PDE topics).
    _ba_domain_profile = _domain_profile
    if _ba_domain_profile is None:
        try:
            from researchclaw.domains.detector import detect_domain as _detect_domain_adv
            _ba_domain_profile = _detect_domain_adv(
                topic=config.research.topic,
                hypotheses=hypotheses,
            )
        except Exception:  # noqa: BLE001
            logger.debug("BenchmarkAgent domain detection unavailable", exc_info=True)
    _ba_domain_id = (
        _ba_domain_profile.domain_id
        if _ba_domain_profile is not None
        else "generic"
    )
    _ba_domain_ok = _ba_domain_id.startswith("ml_")
    if not _ba_domain_ok:
        logger.info(
            "BenchmarkAgent skipped: domain profile '%s' is not an ML profile (topic: %s)",
            _ba_domain_id, config.research.topic[:80],
        )
    if (
        _ba_domain_ok
        and config.experiment.benchmark_agent.enabled
        and config.experiment.mode in ("sandbox", "docker")
        and llm is not None
    ):
        try:
            from researchclaw.agents.benchmark_agent import BenchmarkOrchestrator
            from researchclaw.agents.benchmark_agent.orchestrator import (
                BenchmarkAgentConfig as _BACfg,
            )

            _ba_cfg_raw = config.experiment.benchmark_agent
            _ba_cfg = _BACfg(
                enabled=_ba_cfg_raw.enabled,
                enable_hf_search=_ba_cfg_raw.enable_hf_search,
                max_hf_results=_ba_cfg_raw.max_hf_results,
                enable_web_search=_ba_cfg_raw.enable_web_search,
                max_web_results=_ba_cfg_raw.max_web_results,
                web_search_min_local=_ba_cfg_raw.web_search_min_local,
                tier_limit=_ba_cfg_raw.tier_limit,
                min_benchmarks=_ba_cfg_raw.min_benchmarks,
                min_baselines=_ba_cfg_raw.min_baselines,
                prefer_cached=_ba_cfg_raw.prefer_cached,
                max_iterations=_ba_cfg_raw.max_iterations,
            )

            _hw = _load_hardware_profile(run_dir)
            _ba = BenchmarkOrchestrator(
                llm,
                config=_ba_cfg,
                gpu_memory_mb=(
                    _hw.get("gpu_memory_mb", 49000) if _hw else 49000
                ),
                time_budget_sec=config.experiment.time_budget_sec,
                network_policy=(
                    config.experiment.docker.network_policy
                    if config.experiment.mode == "docker"
                    else "full"
                ),
                stage_dir=stage_dir / "benchmark_agent",
            )
            _benchmark_plan = _ba.orchestrate({
                "topic": config.research.topic,
                "hypothesis": hypotheses,
                "experiment_plan": plan.get("objectives", "") if isinstance(plan, dict) else "",
            })

            # Inject BenchmarkAgent selections into experiment plan
            if isinstance(plan, dict) and _benchmark_plan.selected_benchmarks:
                plan["datasets"] = [
                    b.get("name", "Unknown") for b in _benchmark_plan.selected_benchmarks
                ]
                # Normalize existing baselines — LLM may emit dict, list of
                # dicts, or list of strings.
                _baselines_from_plan = _plan_field_names(
                    _normalize_plan_field(plan.get("baselines", []))
                )
                plan["baselines"] = [
                    bl.get("name", "Unknown") for bl in _benchmark_plan.selected_baselines
                ] + _baselines_from_plan
                # Deduplicate baselines
                plan["baselines"] = list(dict.fromkeys(plan["baselines"]))

            logger.info(
                "BenchmarkAgent: %d benchmarks, %d baselines selected (%d LLM calls, %.1fs)",
                len(_benchmark_plan.selected_benchmarks),
                len(_benchmark_plan.selected_baselines),
                _benchmark_plan.total_llm_calls,
                _benchmark_plan.elapsed_sec,
            )
        except Exception as _ba_exc:
            logger.warning("BenchmarkAgent failed (non-fatal): %s", _ba_exc)

    # Save benchmark plan for code_generation stage
    if _benchmark_plan is not None:
        try:
            (stage_dir / "benchmark_plan.json").write_text(
                json.dumps(_benchmark_plan.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:  # noqa: BLE001
            pass

    plan.setdefault("topic", config.research.topic)
    plan, _sandbox_execution_report = (
        _apply_single_cell_sandbox_execution_constraints(
            plan,
            topic=config.research.topic,
            domain_profile=_domain_profile or _ba_domain_profile,
            experiment_mode=config.experiment.mode,
        )
    )
    plan, _tabular_cpu_budget_report = _apply_tabular_cpu_budget_constraints(
        plan,
        topic=config.research.topic,
        domain_profile=_domain_profile or _ba_domain_profile,
        experiment_mode=config.experiment.mode,
    )
    (stage_dir / "single_cell_sandbox_execution_guardrail.json").write_text(
        json.dumps(_sandbox_execution_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (stage_dir / "tabular_cpu_budget_guardrail.json").write_text(
        json.dumps(_tabular_cpu_budget_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # BUG-R41-09: Enforce condition count limit based on time budget.
    # Too many conditions (30+) guarantee timeouts and wasted compute.
    _time_budget = getattr(
        getattr(config, "experiment", None), "time_budget_sec", 3600
    )
    _max_conditions = 8  # default for budgets ≤ 3600s
    if _time_budget > 3600:
        _max_conditions = 12
    if _time_budget > 7200:
        _max_conditions = 20

    _baselines = _normalize_plan_field(plan.get("baselines", []))
    _proposed = _normalize_plan_field(plan.get("proposed_methods", []))
    _ablations = _normalize_plan_field(plan.get("ablations", []))
    _total = len(_baselines) + len(_proposed) + len(_ablations)

    if _total > _max_conditions:
        logger.warning(
            "Stage 9: Plan has %d conditions (limit %d for %ds budget). "
            "Trimming to fit.",
            _total, _max_conditions, _time_budget,
        )
        # Keep all proposed methods (up to max), trim baselines and ablations
        _proposed_count = min(len(_proposed), max(1, _max_conditions - 4))
        _remaining = max(0, _max_conditions - _proposed_count)
        _baseline_budget = max(1, _remaining // 2)
        _ablation_budget = max(0, _remaining - _baseline_budget)
        if len(_proposed) > _proposed_count:
            plan["proposed_methods"] = _proposed[:_proposed_count]
            logger.info(
                "Stage 9: Trimmed proposed methods %d → %d",
                len(_proposed), _proposed_count,
            )

        if len(_baselines) > _baseline_budget:
            plan["baselines"] = _baselines[:_baseline_budget]
            logger.info(
                "Stage 9: Trimmed baselines %d → %d",
                len(_baselines), _baseline_budget,
            )
        if len(_ablations) > _ablation_budget:
            plan["ablations"] = _ablations[:_ablation_budget]
            logger.info(
                "Stage 9: Trimmed ablations %d → %d",
                len(_ablations), _ablation_budget,
            )

    # --- HITL: Read human guidance if available ---
    guidance_file = stage_dir / "hitl_guidance.md"
    if guidance_file.exists():
        try:
            guidance = guidance_file.read_text(encoding="utf-8").strip()
            if guidance and llm is not None and isinstance(plan, dict):
                logger.info("Applying HITL guidance to experiment design")
                resp = llm.chat(
                    [{"role": "user", "content": (
                        f"The human researcher provided this guidance for "
                        f"the experiment design:\n\n{guidance}\n\n"
                        f"Current experiment plan:\n"
                        f"```yaml\n{yaml.dump(plan, default_flow_style=False)}\n```\n\n"
                        f"Update the YAML plan to incorporate the guidance. "
                        f"Return ONLY the updated YAML."
                    )}],
                    max_tokens=4096,
                )
                updated = _extract_yaml_block(resp.content)
                try:
                    parsed_update = yaml.safe_load(updated)
                    if isinstance(parsed_update, dict):
                        plan = parsed_update
                except yaml.YAMLError:
                    pass
        except Exception:
            logger.debug("HITL guidance application failed (non-blocking)")

    # --- HITL: Baseline Navigator data persistence ---
    try:
        from researchclaw.hitl.workshops.baseline import BaselineNavigator, BaselineCandidate

        nav = BaselineNavigator(run_dir, llm_client=llm)
        if isinstance(plan, dict):
            baselines = plan.get("baselines", [])
            if isinstance(baselines, list):
                for b in baselines:
                    if isinstance(b, dict):
                        nav.baselines.append(BaselineCandidate(
                            name=b.get("name", str(b)),
                            description=b.get("description", ""),
                        ))
                    elif isinstance(b, str):
                        nav.baselines.append(BaselineCandidate(name=b))
            metrics = plan.get("metrics", [])
            if isinstance(metrics, list):
                nav.metrics = [str(m) for m in metrics]
        nav.save()
    except Exception:
        pass

    (stage_dir / "exp_plan.yaml").write_text(
        yaml.dump(plan, default_flow_style=False, allow_unicode=True),
        encoding="utf-8",
    )
    _benchmark_plan_dict = None
    if _benchmark_plan is not None:
        try:
            _benchmark_plan_dict = _benchmark_plan.to_dict()
        except Exception:  # noqa: BLE001
            _benchmark_plan_dict = None
    _guardrail_report = _validate_domain_benchmark_guardrail(
        topic=config.research.topic,
        domain_profile=_domain_profile or _ba_domain_profile,
        plan=plan,
        benchmark_plan=_benchmark_plan_dict,
    )
    (stage_dir / "domain_benchmark_guardrail.json").write_text(
        json.dumps(_guardrail_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if not _guardrail_report["ok"]:
        _messages = [
            str(v.get("message", v))
            for v in _guardrail_report.get("violations", [])
        ]
        _error = (
            "Stage 9 domain benchmark guardrail rejected incompatible benchmarks: "
            + "; ".join(_messages)
        )
        logger.error(_error)
        return StageResult(
            stage=Stage.EXPERIMENT_DESIGN,
            status=StageStatus.FAILED,
            artifacts=("exp_plan.yaml", "domain_benchmark_guardrail.json"),
            error=_error,
            evidence_refs=(
                "stage-09/exp_plan.yaml",
                "stage-09/domain_benchmark_guardrail.json",
            ),
        )
    return StageResult(
        stage=Stage.EXPERIMENT_DESIGN,
        status=StageStatus.DONE,
        artifacts=(
            "exp_plan.yaml",
            "domain_benchmark_guardrail.json",
            "single_cell_sandbox_execution_guardrail.json",
            "tabular_cpu_budget_guardrail.json",
        ),
        evidence_refs=(
            "stage-09/exp_plan.yaml",
            "stage-09/domain_benchmark_guardrail.json",
            "stage-09/single_cell_sandbox_execution_guardrail.json",
            "stage-09/tabular_cpu_budget_guardrail.json",
        ),
    )
