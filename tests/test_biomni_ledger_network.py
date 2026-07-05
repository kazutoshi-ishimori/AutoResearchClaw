from __future__ import annotations

import json
from pathlib import Path

from researchclaw.experiment.verify.ledger_network import (
    extract_interaction_scores,
    extract_ppi_enrichment,
)


def _write_ledger(path: Path, entries: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for e in entries:
            fh.write(json.dumps(e) + "\n")


def _stringdb_entry() -> dict:
    return {
        "tool": "query_stringdb",
        "args": {"endpoint": "https://version-12-0.string-db.org/api/json/ppi_enrichment?identifiers=ACE2&species=9606"},
        "raw_return": [
            {
                "number_of_nodes": 10,
                "number_of_edges": 44,
                "average_node_degree": 8.8,
                "local_clustering_coefficient": 0.978,
                "expected_number_of_edges": 1,
                "p_value": 0.0,
            }
        ],
        "sha256": "deadbeef",
        "ts": "2026-07-04T00:00:00+00:00",
    }


def test_extracts_edges_and_expected(tmp_path: Path) -> None:
    led = tmp_path / "p.jsonl"
    _write_ledger(led, [_stringdb_entry()])
    got = extract_ppi_enrichment(led)
    assert got == {
        "number_of_nodes": 10.0,
        "number_of_edges": 44.0,
        "expected_number_of_edges": 1.0,
    }


def test_extracts_number_of_nodes_for_avg_degree_oracle(tmp_path: Path) -> None:
    """A-②: avg_node_degree = 2*edges/nodes needs number_of_nodes surfaced too."""
    led = tmp_path / "p.jsonl"
    _write_ledger(led, [_stringdb_entry()])
    got = extract_ppi_enrichment(led)
    assert got is not None
    assert got["number_of_nodes"] == 10.0


def test_omits_number_of_nodes_when_absent(tmp_path: Path) -> None:
    """number_of_nodes is optional — its absence must not drop the whole row."""
    led = tmp_path / "p.jsonl"
    entry = _stringdb_entry()
    del entry["raw_return"][0]["number_of_nodes"]
    _write_ledger(led, [entry])
    got = extract_ppi_enrichment(led)
    assert got == {"number_of_edges": 44.0, "expected_number_of_edges": 1.0}


def _network_entry(scores: list[float] | None = None) -> dict:
    """A query_stringdb *network* endpoint return: a list of edge dicts, each
    carrying a combined `score` (0-1). Distinct payload shape from ppi_enrichment.
    """
    if scores is None:
        scores = [0.9, 0.6, 0.3]
    return {
        "tool": "query_stringdb",
        "args": {"endpoint": "https://version-12-0.string-db.org/api/json/network?identifiers=ACE2&species=9606"},
        "raw_return": {
            "success": True,
            "query_info": {"endpoint": "network"},
            "result": [
                {
                    "stringId_A": f"A{i}",
                    "stringId_B": f"B{i}",
                    "preferredName_A": "ACE2",
                    "preferredName_B": "TMPRSS2",
                    "score": s,
                }
                for i, s in enumerate(scores)
            ],
        },
        "sha256": "cafef00d",
        "ts": "2026-07-05T01:00:00+00:00",
    }


def test_extract_interaction_scores_returns_per_edge_scores(tmp_path: Path) -> None:
    # A-③: mean_interaction_score = Σscore/n aggregates the per-edge scores.
    led = tmp_path / "p.jsonl"
    _write_ledger(led, [_network_entry([0.9, 0.6, 0.3])])
    assert extract_interaction_scores(led) == (0.9, 0.6, 0.3)


def test_interaction_scores_none_for_ppi_enrichment_only(tmp_path: Path) -> None:
    # Disambiguation: a ppi_enrichment row carries no `score`, so it must NOT be
    # mistaken for an edge list.
    led = tmp_path / "p.jsonl"
    _write_ledger(led, [_stringdb_entry()])
    assert extract_interaction_scores(led) is None


def test_ppi_enrichment_none_for_network_only(tmp_path: Path) -> None:
    # Symmetric disambiguation: a network edge list has no number_of_edges row.
    led = tmp_path / "p.jsonl"
    _write_ledger(led, [_network_entry()])
    assert extract_ppi_enrichment(led) is None


def test_latest_network_wins_for_interaction_scores(tmp_path: Path) -> None:
    led = tmp_path / "p.jsonl"
    _write_ledger(led, [_network_entry([0.1, 0.2]), _network_entry([0.5, 0.5, 0.5])])
    assert extract_interaction_scores(led) == (0.5, 0.5, 0.5)


def test_interaction_scores_none_without_stringdb(tmp_path: Path) -> None:
    led = tmp_path / "p.jsonl"
    _write_ledger(led, [{"tool": "query_uniprot", "raw_return": {"success": True}}])
    assert extract_interaction_scores(led) is None


def test_returns_none_without_stringdb(tmp_path: Path) -> None:
    led = tmp_path / "p.jsonl"
    _write_ledger(led, [{"tool": "query_uniprot", "raw_return": {"success": True}}])
    assert extract_ppi_enrichment(led) is None


def test_returns_none_for_stringdb_error(tmp_path: Path) -> None:
    led = tmp_path / "p.jsonl"
    _write_ledger(led, [{"tool": "query_stringdb", "raw_return": {"success": False, "error": "boom"}}])
    assert extract_ppi_enrichment(led) is None


def test_latest_stringdb_wins(tmp_path: Path) -> None:
    led = tmp_path / "p.jsonl"
    first = _stringdb_entry()
    second = _stringdb_entry()
    second["raw_return"][0]["number_of_edges"] = 50
    second["raw_return"][0]["expected_number_of_edges"] = 2
    _write_ledger(led, [first, second])
    assert extract_ppi_enrichment(led) == {
        "number_of_nodes": 10.0,
        "number_of_edges": 50.0,
        "expected_number_of_edges": 2.0,
    }
