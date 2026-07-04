from __future__ import annotations

import json
from pathlib import Path

from researchclaw.experiment.verify.ledger_network import extract_ppi_enrichment


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
    assert got == {"number_of_edges": 44.0, "expected_number_of_edges": 1.0}


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
    assert extract_ppi_enrichment(led) == {"number_of_edges": 50.0, "expected_number_of_edges": 2.0}
