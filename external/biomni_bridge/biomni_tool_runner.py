#!/usr/bin/env python3
"""Standalone Biomni tool runner (Phase 0, layer ① — the observer seam).

This script runs **inside the isolated Biomni venv** (``~/workspace/Biomni/.venv``),
not ARC's venv, so it deliberately imports nothing from ``researchclaw``: only
the standard library plus ``biomni``. It invokes one allowlisted Biomni tool
and appends a provenance entry to the ledger in exactly the format
:class:`researchclaw.experiment.biomni_bridge.ProvenanceRecorder` produces, so
the ARC-side claim-binding gate (layer ⑤) can verify results.json numbers
against a ledger written by the *executor's observer*, never by the model.

Usage
-----
    python biomni_tool_runner.py \
        --module database --tool query_uniprot \
        --args-json '{"endpoint": "https://rest.uniprot.org/uniprotkb/Q9BYF1"}' \
        --ledger provenance.jsonl

The tool's raw return is printed to stdout as JSON and recorded to the ledger.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def canonical_hash(value: Any) -> str:
    """sha256 over canonical JSON — must match the ARC-side recorder byte-for-byte."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def record(ledger: Path, tool: str, args: dict[str, Any], result: Any) -> None:
    entry = {
        "tool": tool,
        "args": args,
        "raw_return": result,
        "sha256": canonical_hash(result),
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, default=str) + "\n")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run one Biomni tool with provenance recording.")
    p.add_argument("--module", required=True, help="biomni.tool submodule, e.g. database")
    p.add_argument("--tool", required=True, help="function name within the submodule")
    p.add_argument("--args-json", default="{}", help="JSON object of keyword arguments")
    p.add_argument("--ledger", required=True, help="provenance.jsonl path")
    p.add_argument(
        "--allowlist",
        default="",
        help="comma-separated allowed tool names; empty disables the check",
    )
    ns = p.parse_args(argv)

    if ns.allowlist:
        allowed = {t.strip() for t in ns.allowlist.split(",") if t.strip()}
        if ns.tool not in allowed:
            print(f"ERROR: tool '{ns.tool}' not in allowlist", file=sys.stderr)
            return 2

    args: dict[str, Any] = json.loads(ns.args_json)
    mod = importlib.import_module(f"biomni.tool.{ns.module}")
    fn = getattr(mod, ns.tool)
    result = fn(**args)

    record(Path(ns.ledger), ns.tool, args, result)
    json.dump(result, sys.stdout, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
