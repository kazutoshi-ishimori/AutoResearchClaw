"""Extract STRING network numbers from the provenance ledger (layer ④ input).

Layer ④'s oracles recompute derived metrics from the *raw* STRING numbers the
executor's observer recorded — never from anything the agent wrote into
results.json. Reading from the ledger is what binds a metric to a real,
provenance-anchored tool call.

Two distinct query_stringdb payload shapes are surfaced here:

* the ``ppi_enrichment`` endpoint returns a one-element list of dicts carrying
  ``number_of_edges`` etc. → :func:`extract_ppi_enrichment` (fold, avg degree);
* the ``network`` endpoint returns a list of *edge* dicts each carrying a
  combined ``score`` (0-1) → :func:`extract_interaction_scores` (mean score).

Both arrive under the same tool name ``query_stringdb``, so each extractor
disambiguates by payload shape (presence of ``number_of_edges`` vs ``score``).
An error return is a dict without those fields and is filtered out.
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


def _edge_scores(raw_return: Any) -> list[float] | None:
    """Per-edge ``score`` list from a network return, or ``None`` if not one.

    A network return's ``result`` is a *list of edge dicts* each carrying a
    ``score``. A ppi_enrichment return's single row has no ``score``, and an
    error return has no ``result`` list — both yield ``None`` here.
    """
    result = raw_return.get("result", raw_return) if isinstance(raw_return, dict) else raw_return
    if not isinstance(result, list) or not result:
        return None
    scores: list[float] = []
    for row in result:
        if not isinstance(row, dict) or "score" not in row:
            return None
        try:
            scores.append(float(row["score"]))
        except (TypeError, ValueError):
            return None
    return scores


def extract_interaction_scores(ledger_path: Path) -> tuple[float, ...] | None:
    """Per-edge combined scores from the most-recent query_stringdb network call.

    Layer ④'s ``mean_interaction_score`` oracle aggregates these (Σscore/n) — a
    value the tool does not return directly. Returns ``None`` when no
    query_stringdb entry carries an edge list with scores (so the recompute gate
    stays silent on absence, and a ppi_enrichment-only run is not misread).
    """
    latest: list[float] | None = None
    for entry in _iter_ledger(Path(ledger_path)):
        if entry.get("tool") != "query_stringdb":
            continue
        scores = _edge_scores(entry.get("raw_return"))
        if scores is None:
            continue
        latest = scores
    return tuple(latest) if latest is not None else None
