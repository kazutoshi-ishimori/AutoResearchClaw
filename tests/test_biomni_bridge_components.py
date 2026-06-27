"""Phase 2 pipeline-wiring A: build_biomni_bridge_components helper.

The factory that constructs an :class:`AgenticSandbox` for ``mode=agentic``
needs a single seam to turn a :class:`BiomniConfig` into the three pieces
the sandbox already accepts:

* ``bridge_lifecycle`` — a context manager that yields the bridge URL;
* ``biomni_tools``     — the allowlist that the prompt guidance enumerates;
* ``biomni_client_path`` — the host path to the bundled CLI client.

This module unit-tests that translation. Subprocesses are NOT spawned: the
returned :class:`BiomniBridgeProcess` carries its own spawn injection, so we
only need to confirm what the helper *builds*, not what it runs.
"""

from __future__ import annotations

from pathlib import Path

from researchclaw.config import BiomniConfig
from researchclaw.experiment.verify.bridge_lifecycle import (
    BiomniBridgeProcess,
    build_biomni_bridge_components,
)


def _repo_root(tmp_path: Path) -> Path:
    """Pretend tmp_path is the repo root and lay out the bundled client."""
    client_dir = tmp_path / "external" / "biomni_bridge"
    client_dir.mkdir(parents=True, exist_ok=True)
    (client_dir / "biomni_client.py").write_text("# stub\n", encoding="utf-8")
    daemon = client_dir / "biomni_bridge_daemon.py"
    daemon.write_text("# stub\n", encoding="utf-8")
    return tmp_path


def test_no_bridge_when_both_endpoints_unset(tmp_path) -> None:
    """Empty server_cmd AND bridge_url ⇒ strict no-op triplet."""
    cfg = BiomniConfig()  # all defaults: server_cmd="", bridge_url=""
    repo_root = _repo_root(tmp_path)
    lifecycle, tools, client = build_biomni_bridge_components(
        cfg, ledger_dir=tmp_path / "run", repo_root=repo_root
    )
    assert lifecycle is None
    assert tools == ()
    assert client is None


def test_server_cmd_builds_self_hosted_daemon_lifecycle(tmp_path) -> None:
    """server_cmd set ⇒ BiomniBridgeProcess wired with allowlist + ledger."""
    repo_root = _repo_root(tmp_path)
    cfg = BiomniConfig(
        server_cmd="/opt/biomni/.venv/bin/python runner.py",
        tool_allowlist=("database.query_uniprot", "database.query_kegg"),
        provenance_path="provenance.jsonl",
    )
    ledger_dir = tmp_path / "run-001"
    lifecycle, tools, client = build_biomni_bridge_components(
        cfg, ledger_dir=ledger_dir, repo_root=repo_root
    )

    assert isinstance(lifecycle, BiomniBridgeProcess)
    bcfg = lifecycle.cfg
    assert bcfg.server_cmd == "/opt/biomni/.venv/bin/python runner.py"
    assert bcfg.allowlist == ("database.query_uniprot", "database.query_kegg")
    # Relative provenance_path is rooted at the per-run ledger directory.
    assert bcfg.ledger_path == ledger_dir / "provenance.jsonl"
    # The daemon script must come from the repo's external/ bundle.
    assert bcfg.daemon_script == repo_root / "external" / "biomni_bridge" / "biomni_bridge_daemon.py"

    assert tools == ("database.query_uniprot", "database.query_kegg")
    assert client == repo_root / "external" / "biomni_bridge" / "biomni_client.py"


def test_absolute_provenance_path_is_respected(tmp_path) -> None:
    """If provenance_path is absolute, use it verbatim (don't re-root it)."""
    repo_root = _repo_root(tmp_path)
    abs_ledger = tmp_path / "elsewhere" / "p.jsonl"
    cfg = BiomniConfig(
        server_cmd="/opt/biomni/.venv/bin/python runner.py",
        tool_allowlist=("database.query_uniprot",),
        provenance_path=str(abs_ledger),
    )
    lifecycle, _, _ = build_biomni_bridge_components(
        cfg, ledger_dir=tmp_path / "run", repo_root=repo_root
    )
    assert lifecycle is not None
    assert lifecycle.cfg.ledger_path == abs_ledger


def test_external_bridge_url_yields_static_context_manager(tmp_path) -> None:
    """bridge_url set + server_cmd empty ⇒ trust the external daemon."""
    repo_root = _repo_root(tmp_path)
    cfg = BiomniConfig(
        bridge_url="http://host.docker.internal:8765",
        tool_allowlist=("database.query_uniprot",),
    )
    lifecycle, tools, client = build_biomni_bridge_components(
        cfg, ledger_dir=tmp_path / "run", repo_root=repo_root
    )
    assert lifecycle is not None
    # The context-manager must yield the configured URL verbatim — the
    # bridge is owned by someone else; we just point at it.
    with lifecycle as url:
        assert url == "http://host.docker.internal:8765"
    assert tools == ("database.query_uniprot",)
    assert client == repo_root / "external" / "biomni_bridge" / "biomni_client.py"


def test_self_hosted_takes_precedence_over_static_url(tmp_path) -> None:
    """If both server_cmd and bridge_url are set, the self-hosted path wins.

    Rationale: the ARC daemon owns the ledger; deferring to an arbitrary
    external URL while we also have credentials to spawn our own daemon
    would split provenance ownership and violate the design invariant.
    """
    repo_root = _repo_root(tmp_path)
    cfg = BiomniConfig(
        server_cmd="/opt/biomni/.venv/bin/python runner.py",
        bridge_url="http://elsewhere.example:9999",
        tool_allowlist=("x",),
    )
    lifecycle, _, _ = build_biomni_bridge_components(
        cfg, ledger_dir=tmp_path / "run", repo_root=repo_root
    )
    assert isinstance(lifecycle, BiomniBridgeProcess)
