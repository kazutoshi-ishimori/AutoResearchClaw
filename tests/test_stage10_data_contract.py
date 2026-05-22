from __future__ import annotations

from pathlib import Path
import time
from typing import Any

import pytest

from researchclaw.adapters import AdapterBundle
from researchclaw.config import RCConfig
from researchclaw.llm.client import LLMResponse
from researchclaw.pipeline.stages import StageStatus
from researchclaw.pipeline.stage_impls._code_generation import (
    _Stage10TimeoutError,
    _execute_code_generation,
    _run_with_stage10_timeout,
    _stage10_data_contract_guidance,
    _stage10_runnable_metric_guidance,
    _stage10_tabular_cpu_budget_guidance,
    _validate_stage10_data_contract,
    _validate_stage10_runnable_metric_contract,
    _validate_tabular_cpu_budget_contract,
)


class FakeLLM:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[list[dict[str, str]]] = []

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMResponse:
        _ = kwargs
        self.calls.append(messages)
        return LLMResponse(content=self.response, model="fake-model")


def _single_cell_config(tmp_path: Path) -> RCConfig:
    return RCConfig.from_dict(
        {
            "project": {"name": "rc-test", "mode": "full-auto"},
            "research": {
                "topic": (
                    "Mapping cell-type-specific expression of ecdysone genes "
                    "using Fly Cell Atlas single-cell RNA-seq AnnData data"
                ),
                "domains": ["single-cell-genomics"],
                "daily_paper_count": 2,
                "quality_threshold": 4.0,
            },
            "runtime": {"timezone": "UTC"},
            "notifications": {"channel": "console"},
            "knowledge_base": {"backend": "markdown", "root": str(tmp_path / "kb")},
            "openclaw_bridge": {},
            "llm": {
                "provider": "openai-compatible",
                "base_url": "http://localhost:1234/v1",
                "api_key_env": "RC_TEST_KEY",
                "api_key": "test",
                "primary_model": "fake-model",
                "fallback_models": [],
            },
            "security": {"hitl_required_stages": []},
            "experiment": {
                "mode": "sandbox",
                "time_budget_sec": 7200,
                "metric_key": "primary_metric",
                "metric_direction": "maximize",
                "opencode": {"enabled": False},
                "code_agent": {"enabled": False},
            },
        },
        project_root=tmp_path,
        check_paths=False,
    )


def test_stage10_contract_rejects_missing_generated_npz_for_single_cell() -> None:
    files = {
        "main.py": (
            "import numpy as np\n"
            "data = np.load('fly_cell_atlas_synthetic.npz')\n"
            "print('primary_metric: 1.0')\n"
        )
    }

    violations = _validate_stage10_data_contract(
        files,
        topic="Fly Cell Atlas single-cell RNA-seq AnnData analysis",
        experiment_mode="sandbox",
        network_policy="none",
    )

    assert any(v["category"] == "missing_generated_data_file" for v in violations)


def test_stage10_contract_allows_npz_generated_by_project_code() -> None:
    files = {
        "main.py": (
            "import numpy as np\n"
            "import pandas as pd\n"
            "from sklearn.metrics import silhouette_score\n"
            "import generate_data\n"
            "generate_data.main()\n"
            "data = np.load('fly_cell_atlas_synthetic.npz')\n"
            "metadata = pd.DataFrame({'cell_type': ['a', 'b', 'c']})\n"
            "_score = silhouette_score(data['X'], [0, 1, 1])\n"
            "print('primary_metric: 1.0')\n"
        ),
        "generate_data.py": (
            "import numpy as np\n"
            "def main():\n"
            "    np.savez('fly_cell_atlas_synthetic.npz', X=np.ones((3, 2)))\n"
        ),
    }

    violations = _validate_stage10_data_contract(
        files,
        topic="Fly Cell Atlas single-cell RNA-seq AnnData analysis",
        experiment_mode="sandbox",
        network_policy="none",
    )

    assert not any(v["category"] == "missing_generated_data_file" for v in violations)


def test_stage10_contract_rejects_hard_scanpy_import_in_sandbox() -> None:
    files = {
        "main.py": (
            "import scanpy as sc\n"
            "print('primary_metric: 1.0')\n"
        )
    }

    violations = _validate_stage10_data_contract(
        files,
        topic="Fly Cell Atlas single-cell RNA-seq AnnData analysis",
        experiment_mode="sandbox",
        network_policy="none",
    )

    assert any(v["category"] == "forbidden_sandbox_dependency" for v in violations)


def test_stage10_contract_rejects_optional_scanpy_import_in_sandbox() -> None:
    files = {
        "main.py": (
            "import numpy as np\n"
            "import pandas as pd\n"
            "from sklearn.metrics import silhouette_score\n"
            "try:\n"
            "    import scanpy as sc\n"
            "except ImportError:\n"
            "    sc = None\n"
            "print('primary_metric: 1.0')\n"
        )
    }

    violations = _validate_stage10_data_contract(
        files,
        topic="Fly Cell Atlas single-cell RNA-seq AnnData analysis",
        experiment_mode="sandbox",
        network_policy="none",
    )

    assert any(v["category"] == "forbidden_sandbox_dependency" for v in violations)


def test_stage10_contract_requires_numpy_pandas_sklearn_fallback() -> None:
    files = {
        "main.py": (
            "import numpy as np\n"
            "print('primary_metric: 1.0')\n"
        )
    }

    violations = _validate_stage10_data_contract(
        files,
        topic="Fly Cell Atlas single-cell RNA-seq AnnData analysis",
        experiment_mode="sandbox",
        network_policy="none",
    )

    assert any(
        v["category"] == "missing_numpy_pandas_sklearn_fallback"
        for v in violations
    )


def test_stage10_contract_rejects_heavy_single_cell_tool_imports() -> None:
    files = {
        "main.py": (
            "import numpy as np\n"
            "import pandas as pd\n"
            "from sklearn.metrics import silhouette_score\n"
            "import magic\n"
            "print('primary_metric: 1.0')\n"
        )
    }

    violations = _validate_stage10_data_contract(
        files,
        topic="Fly Cell Atlas single-cell RNA-seq AnnData analysis",
        experiment_mode="sandbox",
        network_policy="none",
    )

    assert any(v["category"] == "forbidden_sandbox_dependency" for v in violations)


def test_stage10_guidance_mentions_self_contained_single_cell_data() -> None:
    guidance = _stage10_data_contract_guidance(
        topic="Fly Cell Atlas single-cell RNA-seq AnnData analysis",
        experiment_mode="sandbox",
        network_policy="none",
    )

    assert "self-contained" in guidance.lower()
    assert "synthetic" in guidance.lower()
    assert "scanpy" in guidance.lower()
    assert "anndata" in guidance.lower()
    assert "hdWGCNA" in guidance
    assert "CellChat" in guidance


def test_stage10_tabular_cpu_budget_contract_rejects_heavy_literals() -> None:
    files = {
        "config.py": (
            "N_SAMPLES = 20000\n"
            "SEEDS = [42, 123, 456, 789]\n"
            "SHIFT_TYPES = ['continuous', 'categorical', 'missingness']\n"
            "SHIFT_MAGNITUDES = [0.2, 0.5, 0.8]\n"
        ),
        "models.py": "model = GradientBoostingClassifier(n_estimators=200)\n",
    }

    violations = _validate_tabular_cpu_budget_contract(
        files,
        topic="Conformal prediction for tabular classification under covariate shift",
        experiment_mode="sandbox",
        network_policy="none",
    )

    categories = {v["category"] for v in violations}
    assert "tabular_cpu_budget_n_samples" in categories
    assert "tabular_cpu_budget_n_estimators" in categories
    assert "tabular_cpu_budget_seeds" in categories
    assert "tabular_cpu_budget_shift_regimes" in categories


def test_stage10_tabular_cpu_budget_contract_allows_small_literals() -> None:
    files = {
        "config.py": (
            "n_samples: int = 5000\n"
            "seeds = [42, 123, 456]\n"
            "shift_types = ['continuous', 'categorical']\n"
            "shift_magnitudes = [0.2, 0.5]\n"
        ),
        "models.py": "model = GradientBoostingClassifier(n_estimators=50)\n",
        "baseline.py": "def split_cp_baseline():\n    return {'coverage_rate': 0.9}\n",
    }

    violations = _validate_tabular_cpu_budget_contract(
        files,
        topic="Conformal prediction for tabular classification under covariate shift",
        experiment_mode="sandbox",
        network_policy="none",
    )

    assert violations == []


def test_stage10_tabular_cpu_budget_guidance_mentions_exact_limits() -> None:
    guidance = _stage10_tabular_cpu_budget_guidance(
        topic="Conformal prediction for tabular classification under covariate shift",
        experiment_mode="sandbox",
        network_policy="none",
    )

    assert "n_samples <= 5000" in guidance
    assert "n_estimators <= 50" in guidance
    assert "shift regimes <= 4" in guidance
    assert "seeds = 3" in guidance
    assert "split_cp_baseline" in guidance
    assert "average_prediction_set_size" in guidance
    assert "Do not create alias baselines" in guidance
    assert "condition=<name>" in guidance


def test_stage10_tabular_cpu_budget_contract_requires_split_cp_baseline() -> None:
    files = {
        "main.py": (
            "def main():\n"
            "    print('coverage_gap: 0.1')\n"
            "    print('stability_aware_conformal/coverage_rate: 0.9')\n"
        )
    }

    violations = _validate_tabular_cpu_budget_contract(
        files,
        topic="Conformal prediction for tabular classification under covariate shift",
        experiment_mode="sandbox",
        network_policy="none",
    )

    assert any(v["category"] == "tabular_split_cp_baseline" for v in violations)


def test_stage10_runnable_metric_contract_rejects_class_only_main() -> None:
    files = {
        "main.py": (
            "class ExperimentConfig:\n"
            "    def __init__(self):\n"
            "        self.n_samples = 2000\n"
        )
    }

    violations = _validate_stage10_runnable_metric_contract(
        files,
        metric="coverage_gap",
    )

    categories = {v["category"] for v in violations}
    assert "missing_executable_entrypoint" in categories
    assert "missing_primary_metric_output" in categories


def test_stage10_runnable_metric_contract_allows_metric_print() -> None:
    files = {
        "main.py": (
            "def main():\n"
            "    print('coverage_gap: 0.1')\n"
            "if __name__ == '__main__':\n"
            "    main()\n"
        )
    }

    violations = _validate_stage10_runnable_metric_contract(
        files,
        metric="coverage_gap",
    )

    assert violations == []


def test_stage10_runnable_metric_guidance_mentions_primary_metric() -> None:
    guidance = _stage10_runnable_metric_guidance(metric="coverage_gap")

    assert "coverage_gap: <float>" in guidance
    assert "results.json" in guidance
    assert "if __name__" in guidance


def test_stage10_timeout_helper_raises_explicit_error() -> None:
    def slow_call() -> None:
        time.sleep(0.2)

    with pytest.raises(_Stage10TimeoutError) as excinfo:
        _run_with_stage10_timeout(
            slow_call,
            timeout_sec=0.05,
            label="CodeAgent",
        )

    assert "CodeAgent timed out" in str(excinfo.value)


def test_execute_code_generation_fails_on_stage10_contract_violation(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    stage09 = run_dir / "stage-09"
    stage09.mkdir()
    (stage09 / "exp_plan.yaml").write_text(
        "datasets:\n- Fly Cell Atlas AnnData\nmetrics:\n- primary_metric\n",
        encoding="utf-8",
    )
    stage_dir = run_dir / "stage-10"
    stage_dir.mkdir()
    llm = FakeLLM(
        "```filename:main.py\n"
        "import numpy as np\n"
        "data = np.load('fly_cell_atlas_synthetic.npz')\n"
        "print('primary_metric: 1.0')\n"
        "```\n"
    )

    result = _execute_code_generation(
        stage_dir,
        run_dir,
        _single_cell_config(tmp_path),
        AdapterBundle(),
        llm=llm,  # type: ignore[arg-type]
    )

    assert result.status is StageStatus.FAILED
    report = stage_dir / "stage10_data_contract.json"
    assert report.exists()
    assert "fly_cell_atlas_synthetic.npz" in report.read_text(encoding="utf-8")
