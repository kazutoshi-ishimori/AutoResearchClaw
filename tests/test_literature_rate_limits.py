from pathlib import Path

from researchclaw.config import RCConfig
from researchclaw.pipeline.stage_impls._literature import (
    _cap_search_queries,
    _literature_limit_per_query,
)


def _config_with_daily_count(count: int, tmp_path: Path) -> RCConfig:
    data = {
        "project": {"name": "demo", "mode": "full-auto"},
        "research": {
            "topic": "Bootstrap optimizer comparison",
            "domains": ["machine-learning"],
            "daily_paper_count": count,
        },
        "runtime": {"timezone": "Asia/Tokyo"},
        "notifications": {"channel": "console"},
        "knowledge_base": {"backend": "markdown", "root": "docs/kb"},
        "openclaw_bridge": {},
        "llm": {
            "provider": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key_env": "OPENROUTER_API_KEY",
            "primary_model": "openai/gpt-oss-120b",
        },
        "experiment": {"mode": "sandbox"},
    }
    return RCConfig.from_dict(data, project_root=tmp_path, check_paths=False)


def test_literature_limit_per_query_uses_daily_paper_count(tmp_path: Path):
    assert _literature_limit_per_query(_config_with_daily_count(10, tmp_path)) == 10
    assert _literature_limit_per_query(_config_with_daily_count(2, tmp_path)) == 5
    assert _literature_limit_per_query(_config_with_daily_count(50, tmp_path)) == 20


def test_cap_search_queries_deduplicates_and_limits_to_four():
    queries = ["alpha", "beta", "alpha", "gamma", "delta", "epsilon"]

    assert _cap_search_queries(queries) == ["alpha", "beta", "gamma", "delta"]
