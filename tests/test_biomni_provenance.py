"""Tests for the Biomni bridge provenance recorder (Phase 0, layer ①).

Design invariant: the provenance ledger is written by the *observer*
(the bridge that actually invokes the tool), never by the executor's
self-report. Every real tool invocation appends one JSON line carrying
the raw return value and its sha256 anchor, which the claim-binding gate
(layer ⑤) later checks results.json numbers against.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from researchclaw.experiment.biomni_bridge import ProvenanceRecorder, canonical_hash


def test_records_tool_call_with_hash_of_return(tmp_path: Path) -> None:
    ledger = tmp_path / "provenance.jsonl"
    recorder = ProvenanceRecorder(ledger)

    def fake_tool(*, seeds: list[str]) -> dict[str, float]:
        return {"proximity": 0.42}

    result = recorder.call("network_proximity", fake_tool, seeds=["TP53"])

    # The recorder is transparent: it returns the tool's real result.
    assert result == {"proximity": 0.42}

    lines = ledger.read_text().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["tool"] == "network_proximity"
    assert entry["args"] == {"seeds": ["TP53"]}
    assert entry["raw_return"] == {"proximity": 0.42}
    assert entry["sha256"] == canonical_hash({"proximity": 0.42})
    assert "ts" in entry and entry["ts"]


def test_appends_one_line_per_call(tmp_path: Path) -> None:
    ledger = tmp_path / "provenance.jsonl"
    recorder = ProvenanceRecorder(ledger)

    recorder.call("query_gene", lambda *, symbol: {"id": symbol}, symbol="ACE2")
    recorder.call("query_gene", lambda *, symbol: {"id": symbol}, symbol="TMPRSS2")

    lines = ledger.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["args"] == {"symbol": "ACE2"}
    assert json.loads(lines[1])["args"] == {"symbol": "TMPRSS2"}


def test_canonical_hash_is_order_independent_for_dicts(tmp_path: Path) -> None:
    # Same content, different key insertion order -> same anchor.
    a = canonical_hash({"x": 1, "y": 2})
    b = canonical_hash({"y": 2, "x": 1})
    assert a == b
    # And it really is sha256 over the canonical JSON.
    expected = hashlib.sha256(
        json.dumps({"x": 1, "y": 2}, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert a == expected
