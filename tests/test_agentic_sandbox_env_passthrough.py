"""Phase 2 pipeline-wiring G: env passthrough into the agentic container.

The host-bridge wiring (E/F) only covers tool-call provenance. The *agent*
itself — Claude Code, Codex, etc. — still has to authenticate with its model
provider, which means env vars like ``ANTHROPIC_API_KEY`` must reach the
container. ``_start_container`` already injects ``BIOMNI_BRIDGE_URL`` for the
host bridge; this seam adds a generic ``env_passthrough`` list so the agent
can be authenticated without baking secrets into the image.

Silence is preserved: a configured name that is *unset* on the host is
skipped rather than passed as ``KEY=`` (which would override the container's
own default to an empty string and is rarely what the operator wants).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from researchclaw.config import AgenticConfig
from researchclaw.experiment.agentic_sandbox import AgenticSandbox


def _stub_run(capture: list):
    def fake_run(argv, **kwargs):
        capture.append(list(argv))
        return subprocess.CompletedProcess(
            args=argv, returncode=0, stdout="{}", stderr=""
        )

    return fake_run


def _make_sandbox(tmp_path: Path, env_passthrough: tuple[str, ...]) -> AgenticSandbox:
    cfg = AgenticConfig(
        image="rc-agentic:test",
        agent_cli="claude",
        agent_install_cmd="",
        timeout_sec=60,
        memory_limit_mb=512,
        network_policy="full",
        mount_skills=False,
        env_passthrough=env_passthrough,
    )
    return AgenticSandbox(cfg, workdir=tmp_path / "wd", skills_dir=None)


def _docker_run_argv(calls: list) -> list[str]:
    runs = [c for c in calls if c[:2] == ["docker", "run"]]
    assert runs, "expected docker run to be invoked"
    return runs[0]


def _env_flags(argv: list[str]) -> list[str]:
    return [argv[i + 1] for i, a in enumerate(argv) if a == "-e" and i + 1 < len(argv)]


def test_env_passthrough_default_is_empty(tmp_path, monkeypatch) -> None:
    """Default AgenticConfig() has empty env_passthrough — no -e leaks."""
    cfg = AgenticConfig()
    assert cfg.env_passthrough == ()


def test_env_passthrough_propagates_set_env(tmp_path, monkeypatch) -> None:
    """A configured + set env var lands in docker run as ``-e NAME=VALUE``."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-123")
    calls: list = []
    monkeypatch.setattr(subprocess, "run", _stub_run(calls))

    sb = _make_sandbox(tmp_path, env_passthrough=("ANTHROPIC_API_KEY",))
    sb.run_agent_session("hi", workspace=tmp_path / "ws", timeout_sec=10)

    argv = _docker_run_argv(calls)
    assert "ANTHROPIC_API_KEY=sk-ant-test-123" in _env_flags(argv)


def test_env_passthrough_skips_unset_env_silently(tmp_path, monkeypatch) -> None:
    """A configured but unset env var is dropped (not passed as KEY=)."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    calls: list = []
    monkeypatch.setattr(subprocess, "run", _stub_run(calls))

    sb = _make_sandbox(tmp_path, env_passthrough=("ANTHROPIC_API_KEY",))
    sb.run_agent_session("hi", workspace=tmp_path / "ws", timeout_sec=10)

    argv = _docker_run_argv(calls)
    flags = _env_flags(argv)
    assert not any(f.startswith("ANTHROPIC_API_KEY") for f in flags)


def test_env_passthrough_multiple_names(tmp_path, monkeypatch) -> None:
    """Multiple configured names all appear; unset ones are skipped."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-oa-2")
    monkeypatch.delenv("THIRD_SECRET", raising=False)
    calls: list = []
    monkeypatch.setattr(subprocess, "run", _stub_run(calls))

    sb = _make_sandbox(
        tmp_path,
        env_passthrough=("ANTHROPIC_API_KEY", "THIRD_SECRET", "OPENAI_API_KEY"),
    )
    sb.run_agent_session("hi", workspace=tmp_path / "ws", timeout_sec=10)

    argv = _docker_run_argv(calls)
    flags = _env_flags(argv)
    assert "ANTHROPIC_API_KEY=sk-ant-1" in flags
    assert "OPENAI_API_KEY=sk-oa-2" in flags
    assert not any(f.startswith("THIRD_SECRET") for f in flags)


def test_env_passthrough_empty_default_no_env_flags(tmp_path, monkeypatch) -> None:
    """Without env_passthrough, no agent-auth -e flags leak in."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "should-not-leak")
    calls: list = []
    monkeypatch.setattr(subprocess, "run", _stub_run(calls))

    sb = _make_sandbox(tmp_path, env_passthrough=())
    sb.run_agent_session("hi", workspace=tmp_path / "ws", timeout_sec=10)

    argv = _docker_run_argv(calls)
    flags = _env_flags(argv)
    assert not any(f.startswith("ANTHROPIC_API_KEY") for f in flags)


def test_env_passthrough_yaml_parses_into_config(tmp_path) -> None:
    """YAML loader parses experiment.agentic.env_passthrough → tuple[str,...]."""
    from researchclaw.config import load_config

    cfg_text = """
project:
  name: x
  mode: full-auto
research:
  topic: t
runtime: {timezone: Asia/Tokyo}
notifications: {channel: console}
knowledge_base: {backend: markdown, root: docs/kb}
llm:
  provider: openai-compatible
  base_url: https://example.com
  api_key_env: OPENAI_API_KEY
  primary_model: gpt-5.4-mini
experiment:
  mode: agentic
  agentic:
    image: rc-agentic:test
    env_passthrough:
      - ANTHROPIC_API_KEY
      - OPENAI_API_KEY
"""
    (tmp_path / "docs" / "kb").mkdir(parents=True)
    p = tmp_path / "c.yaml"
    p.write_text(cfg_text, encoding="utf-8")
    cfg = load_config(p)
    assert cfg.experiment.agentic.env_passthrough == ("ANTHROPIC_API_KEY", "OPENAI_API_KEY")
