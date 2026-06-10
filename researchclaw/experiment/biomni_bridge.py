"""Biomni bridge — provenance-recording proxy for tool calls (Phase 0, layer ①).

This module is the *observer* seam of the B+(1-5) design: every Biomni tool
invocation is routed through :class:`ProvenanceRecorder`, which appends one
JSON line to the provenance ledger carrying the raw return value and its
sha256 anchor.  Downstream gates (claim binding, layer ⑤) check the numbers
claimed in ``results.json`` against these anchors, so a claimed value with no
ledger backing is provably fabricated rather than executed.

The tool callable is injected, so the recorder is fully testable without a
live Biomni environment.  The standalone MCP server
(``external/biomni_bridge/biomni_mcp_server.py``) wraps real Biomni tools in
this recorder once the ``biomni_env`` conda environment exists.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


def canonical_hash(value: Any) -> str:
    """Return the sha256 anchor of *value* over its canonical JSON form.

    Canonicalisation sorts dict keys and strips insignificant whitespace so
    that semantically identical returns hash identically regardless of key
    insertion order.
    """
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


class ProvenanceRecorder:
    """Append-only recorder that wraps tool invocations.

    Parameters
    ----------
    ledger_path:
        Path to the ``provenance.jsonl`` ledger.  Parent directories are
        created on first write.
    """

    def __init__(self, ledger_path: Path) -> None:
        self.ledger_path = Path(ledger_path)

    def call(self, tool: str, fn: Callable[..., Any], **args: Any) -> Any:
        """Invoke *fn(**args)*, record the call, and return its real result."""
        result = fn(**args)
        entry = {
            "tool": tool,
            "args": args,
            "raw_return": result,
            "sha256": canonical_hash(result),
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with self.ledger_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")
        return result
