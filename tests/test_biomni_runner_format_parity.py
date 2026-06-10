"""Format-parity guard between the two provenance writers.

The ledger is written by two independent implementations:
  * ARC venv  -> researchclaw.experiment.biomni_bridge.ProvenanceRecorder
  * Biomni venv -> external/biomni_bridge/biomni_tool_runner.py (standalone)

If their canonical hashing or entry schema drift apart, the claim-binding gate
silently stops matching real tool outputs. This test loads the standalone
runner by path (its top-level imports are stdlib-only, so no Biomni needed) and
asserts byte-for-byte parity plus claim_binding consumability.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from researchclaw.experiment.biomni_bridge import canonical_hash as arc_hash
from researchclaw.experiment.verify.claim_binding import bind_claims

_RUNNER = (
    Path(__file__).resolve().parent.parent
    / "external"
    / "biomni_bridge"
    / "biomni_tool_runner.py"
)


def _load_runner():
    spec = importlib.util.spec_from_file_location("biomni_tool_runner", _RUNNER)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_canonical_hash_matches_arc_recorder() -> None:
    runner = _load_runner()
    payload = {"y": 2, "x": 1, "nested": [3, {"z": 0.5}]}
    assert runner.canonical_hash(payload) == arc_hash(payload)


def test_recorded_entry_is_consumable_by_claim_binding(tmp_path: Path) -> None:
    runner = _load_runner()
    ledger = tmp_path / "provenance.jsonl"
    runner.record(ledger, "query_uniprot", {"endpoint": "x"}, {"length": 805})

    entry = json.loads(ledger.read_text().splitlines()[0])
    assert entry["tool"] == "query_uniprot"
    assert entry["sha256"] == arc_hash({"length": 805})

    report = bind_claims({"seq_length": 805, "made_up": 1.23}, ledger)
    assert report.backed == [("seq_length", 805.0)]
    assert report.unbacked == [("made_up", 1.23)]
