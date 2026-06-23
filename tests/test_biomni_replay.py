"""Deterministic replay gate (Phase 1, layer ③).

Re-executes recorded tool calls and confirms the return still anchors to the
ledger's sha256 — catching non-determinism, environment drift, or a tampered
ledger. The tool runner is injected, so the gate is unit-testable without a
live Biomni environment; the pipeline supplies a runner that shells out to the
isolated Biomni venv.

Exact canonical-hash equality is the pass condition; a numeric tolerance path
covers tools whose floats jitter in the last bits while their structure is
unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

from researchclaw.experiment.biomni_bridge import canonical_hash
from researchclaw.experiment.verify.replay import ReplayReport, replay_ledger


def _write_ledger(path: Path, entries: list[tuple[str, dict, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for tool, args, raw in entries:
            fh.write(json.dumps({
                "tool": tool, "args": args, "raw_return": raw,
                "sha256": canonical_hash(raw), "ts": "now",
            }) + "\n")


def test_deterministic_tool_replays_clean(tmp_path: Path) -> None:
    ledger = tmp_path / "p.jsonl"
    _write_ledger(ledger, [("query_uniprot", {"id": "Q9BYF1"}, {"length": 805})])

    def runner(tool: str, args: dict) -> object:
        return {"length": 805}  # same value

    report = replay_ledger(ledger, runner)
    assert report.ok
    assert report.matched == ["query_uniprot"]
    assert report.drifted == []


def test_changed_return_drifts(tmp_path: Path) -> None:
    ledger = tmp_path / "p.jsonl"
    _write_ledger(ledger, [("query_uniprot", {"id": "Q9BYF1"}, {"length": 805})])

    def runner(tool: str, args: dict) -> object:
        return {"length": 999}  # tampered / non-reproducible

    report = replay_ledger(ledger, runner)
    assert not report.ok
    assert len(report.drifted) == 1
    tool, recorded_sha, replay_sha = report.drifted[0]
    assert tool == "query_uniprot"
    assert recorded_sha == canonical_hash({"length": 805})
    assert replay_sha == canonical_hash({"length": 999})


def test_numeric_jitter_within_tolerance_matches(tmp_path: Path) -> None:
    ledger = tmp_path / "p.jsonl"
    _write_ledger(ledger, [("score", {}, {"v": 1.0})])

    def runner(tool: str, args: dict) -> object:
        return {"v": 1.0 + 1e-12}

    report = replay_ledger(ledger, runner, tol=1e-9)
    assert report.ok
    assert report.matched == ["score"]


def test_numeric_drift_beyond_tolerance_drifts(tmp_path: Path) -> None:
    ledger = tmp_path / "p.jsonl"
    _write_ledger(ledger, [("score", {}, {"v": 1.0})])

    def runner(tool: str, args: dict) -> object:
        return {"v": 1.5}

    report = replay_ledger(ledger, runner, tol=1e-9)
    assert not report.ok


def test_tools_filter_limits_replay(tmp_path: Path) -> None:
    ledger = tmp_path / "p.jsonl"
    _write_ledger(ledger, [
        ("cheap_api", {}, {"a": 1}),
        ("expensive_sim", {}, {"b": 2}),
    ])
    called: list[str] = []

    def runner(tool: str, args: dict) -> object:
        called.append(tool)
        return {"a": 1} if tool == "cheap_api" else {"b": 2}

    report = replay_ledger(ledger, runner, tools={"cheap_api"})
    assert called == ["cheap_api"]  # expensive_sim never re-run
    assert report.matched == ["cheap_api"]


def test_empty_ledger_is_ok(tmp_path: Path) -> None:
    ledger = tmp_path / "p.jsonl"
    ledger.write_text("", encoding="utf-8")
    report = replay_ledger(ledger, lambda t, a: None)
    assert report.ok
    assert isinstance(report, ReplayReport)
    assert "replay" in report.summary()
