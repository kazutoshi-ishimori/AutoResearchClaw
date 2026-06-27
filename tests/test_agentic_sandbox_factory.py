"""Phase 2 pipeline-wiring A: ``create_agentic_sandbox`` ↔ BiomniConfig.

The factory used to ignore Biomni entirely. With this wave the factory
takes an optional :class:`BiomniConfig` plus a ``repo_root`` and forwards
the resulting ``bridge_lifecycle`` / ``biomni_tools`` / ``biomni_client_path``
triplet into :class:`AgenticSandbox`. We do not exercise Docker here — the
availability probe is monkeypatched so the factory's contract can be
inspected in isolation.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from researchclaw.config import (
    AgenticConfig,
    BiomniConfig,
    ExperimentConfig,
)
from researchclaw.experiment import factory as factory_mod
from researchclaw.experiment.agentic_sandbox import AgenticSandbox
from researchclaw.experiment.verify.bridge_lifecycle import (
    BiomniBridgeProcess,
    StaticBridgeUrl,
)


@pytest.fixture(autouse=True)
def _force_docker_available(monkeypatch):
    """Tests must not depend on a live Docker daemon."""
    monkeypatch.setattr(
        AgenticSandbox, "check_docker_available", staticmethod(lambda: True)
    )


def _repo_root_with_bridge(tmp_path: Path) -> Path:
    bridge_dir = tmp_path / "external" / "biomni_bridge"
    bridge_dir.mkdir(parents=True, exist_ok=True)
    (bridge_dir / "biomni_client.py").write_text("# stub\n", encoding="utf-8")
    (bridge_dir / "biomni_bridge_daemon.py").write_text("# stub\n", encoding="utf-8")
    return tmp_path


def _make_experiment_config(biomni: BiomniConfig | None = None) -> ExperimentConfig:
    return ExperimentConfig(
        mode="agentic",
        agentic=AgenticConfig(image="rc-agentic:test", agent_install_cmd=""),
        biomni=biomni or BiomniConfig(),
    )


def test_factory_without_biomni_keeps_legacy_signature(tmp_path) -> None:
    """No biomni_cfg ⇒ AgenticSandbox stays Biomni-free (back-compat)."""
    cfg = _make_experiment_config()
    sb = factory_mod.create_agentic_sandbox(cfg, workdir=tmp_path / "wd")
    assert isinstance(sb, AgenticSandbox)
    # Internals: defaults must keep the sandbox dormant w.r.t. Biomni.
    assert sb._bridge_lifecycle is None
    assert sb._biomni_tools == ()
    assert sb._biomni_client_path is None


def test_factory_self_hosted_biomni_wires_full_triplet(tmp_path) -> None:
    """server_cmd set ⇒ self-hosted lifecycle + allowlist + bundled client."""
    repo_root = _repo_root_with_bridge(tmp_path)
    biomni = BiomniConfig(
        server_cmd="/opt/biomni/.venv/bin/python runner.py",
        tool_allowlist=("database.query_uniprot", "database.query_kegg"),
        provenance_path="provenance.jsonl",
    )
    cfg = _make_experiment_config(biomni)
    ledger_dir = tmp_path / "run-001"

    sb = factory_mod.create_agentic_sandbox(
        cfg,
        workdir=tmp_path / "wd",
        biomni_cfg=biomni,
        repo_root=repo_root,
        ledger_dir=ledger_dir,
    )

    assert isinstance(sb._bridge_lifecycle, BiomniBridgeProcess)
    assert sb._biomni_tools == (
        "database.query_uniprot",
        "database.query_kegg",
    )
    assert sb._biomni_client_path == (
        repo_root / "external" / "biomni_bridge" / "biomni_client.py"
    )
    # The relative provenance_path must have been re-rooted at ledger_dir.
    assert sb._bridge_lifecycle.cfg.ledger_path == ledger_dir / "provenance.jsonl"


def test_factory_external_bridge_url_yields_static_lifecycle(tmp_path) -> None:
    """bridge_url-only config ⇒ StaticBridgeUrl is forwarded."""
    repo_root = _repo_root_with_bridge(tmp_path)
    biomni = BiomniConfig(
        bridge_url="http://host.docker.internal:8765",
        tool_allowlist=("database.query_uniprot",),
    )
    cfg = _make_experiment_config(biomni)

    sb = factory_mod.create_agentic_sandbox(
        cfg,
        workdir=tmp_path / "wd",
        biomni_cfg=biomni,
        repo_root=repo_root,
        ledger_dir=tmp_path / "run",
    )

    assert isinstance(sb._bridge_lifecycle, StaticBridgeUrl)
    assert sb._biomni_tools == ("database.query_uniprot",)
    assert sb._biomni_client_path is not None
    with sb._bridge_lifecycle as url:
        assert url == "http://host.docker.internal:8765"


def test_factory_empty_biomni_config_is_noop(tmp_path) -> None:
    """Passing a Biomni config with everything empty stays a no-op."""
    repo_root = _repo_root_with_bridge(tmp_path)
    biomni = BiomniConfig()  # all defaults
    cfg = _make_experiment_config(biomni)

    sb = factory_mod.create_agentic_sandbox(
        cfg,
        workdir=tmp_path / "wd",
        biomni_cfg=biomni,
        repo_root=repo_root,
        ledger_dir=tmp_path / "run",
    )
    assert sb._bridge_lifecycle is None
    assert sb._biomni_tools == ()
    assert sb._biomni_client_path is None
