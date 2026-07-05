"""Extract STRING ppi_enrichment fields from the provenance ledger (layer ④ input).

Layer ④'s ``ppi_enrichment_fold`` oracle recomputes a derived metric from the
*raw* STRING numbers the executor's observer recorded — never from anything the
agent wrote into results.json. Reading from the ledger is what binds the metric
to a real, provenance-anchored tool call. The ppi_enrichment endpoint returns a
one-element list of dicts; an error return is a dict without the numeric fields,
so we filter for entries that actually carry ``number_of_edges``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _iter_ledger(ledger_path: Path):
    with ledger_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _enrichment_row(raw_return: Any) -> dict | None:
    result = raw_return.get("result", raw_return) if isinstance(raw_return, dict) else raw_return
    if isinstance(result, list) and result and isinstance(result[0], dict):
        return result[0]
    if isinstance(result, dict):
        return result
    return None


def extract_ppi_enrichment(ledger_path: Path) -> dict[str, float] | None:
    """Most-recent query_stringdb ppi_enrichment fields, or ``None`` if absent."""
    latest: dict | None = None
    for entry in _iter_ledger(Path(ledger_path)):
        if entry.get("tool") != "query_stringdb":
            continue
        row = _enrichment_row(entry.get("raw_return"))
        if not isinstance(row, dict):
            continue
        if "number_of_edges" not in row or "expected_number_of_edges" not in row:
            continue
        latest = row
    if latest is None:
        return None
    try:
        out = {
            "number_of_edges": float(latest["number_of_edges"]),
            "expected_number_of_edges": float(latest["expected_number_of_edges"]),
        }
    except (TypeError, ValueError):
        return None
    # number_of_nodes is the layer ④ input for the avg_node_degree oracle
    # (2*edges/nodes). Surface it when present, but keep it optional — its
    # absence must not drop an otherwise-valid enrichment row.
    nodes = latest.get("number_of_nodes")
    if nodes is not None:
        try:
            out["number_of_nodes"] = float(nodes)
        except (TypeError, ValueError):
            pass
    return out
