"""Layer ④ oracle input: principled seed-module selection from the ledger.

Motivation (IPF autonomous run, honest-red 2026-07-11): the extractors picked
the *most-recent* matching query_stringdb entry. That heuristic held for the
strict two-step COVID scripts (latest == canonical), but a free-form agent
that probes STRING dozens of times leaves the LAST entry as a degenerate
exploratory subset (e.g. a 5-gene ablation) — so layer ④ recomputed the wrong
slice and MISMATCHED the agent's honest, canonical claim.

The principled fix is claim-INDEPENDENT: select the ledger entry whose queried
identifier set (carried in the endpoint URL) equals the study's a-priori seed
module, declared in config (``BiomniConfig.recompute_identifiers``). This anchors
the oracle to the *scientific question* (which genes define the disease module),
never to the claimed *value*. The arithmetic recompute is unchanged, so a
fabricated value still mismatches (see the negative-control tests).

Opt-in and additive: with ``select_identifiers=None`` (no config seed set) the
extractors keep their exact prior "latest wins" behaviour, so the proven COVID
A-①〜④ greens do not regress.
"""

from __future__ import annotations

import json
from pathlib import Path

from researchclaw.experiment.verify.ledger_network import (
    extract_interaction_scores,
    extract_ppi_enrichment,
)

# The IPF a-priori seed module (9 genes) — the study's pre-registered scientific
# object. Order/case here is deliberately scrambled to prove set-comparison.
_SEED = frozenset(
    {"tgfb1", "COL1A1", "sftpc", "MUC5B", "TERT", "terc", "TOLLIP", "DSP", "AKAP13"}
)


def _write_ledger(path: Path, entries: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for e in entries:
            fh.write(json.dumps(e) + "\n")


def _endpoint(kind: str, idents: list[str]) -> str:
    # Faithful to the live payload: identifiers are %0d(CR)-joined and
    # url-encoded inside the endpoint query string.
    joined = "%0d".join(idents)
    return (
        f"https://version-12-0.string-db.org/api/json/{kind}"
        f"?identifiers={joined}&species=9606"
    )


def _ppi_entry(idents: list[str], *, edges: int, expected: int, nodes: int) -> dict:
    return {
        "tool": "query_stringdb",
        "args": {"endpoint": _endpoint("ppi_enrichment", idents)},
        "raw_return": [
            {
                "number_of_nodes": nodes,
                "number_of_edges": edges,
                "expected_number_of_edges": expected,
            }
        ],
        "sha256": "d0",
        "ts": "2026-07-11T00:00:00+00:00",
    }


def _network_entry(idents: list[str], scores: list[float]) -> dict:
    return {
        "tool": "query_stringdb",
        "args": {"endpoint": _endpoint("network", idents)},
        "raw_return": {
            "success": True,
            "result": [
                {"stringId_A": f"A{i}", "stringId_B": f"B{i}", "score": s}
                for i, s in enumerate(scores)
            ],
        },
        "sha256": "d1",
        "ts": "2026-07-11T01:00:00+00:00",
    }


_SEED_GENES = ["MUC5B", "TERT", "TERC", "TOLLIP", "DSP", "AKAP13", "TGFB1", "COL1A1", "SFTPC"]


# ── ppi_enrichment: pick the seed module, not the latest ────────────────────


def test_ppi_selects_seed_module_over_latest_subset(tmp_path: Path) -> None:
    """Canonical 9-gene call first, degenerate 5-gene ablation LAST.

    ``select_identifiers`` must pull the 9-gene row (edges=9, exp=1, nodes=9),
    NOT the latest 5-gene subset (edges=1, exp=0, nodes=5).
    """
    led = tmp_path / "p.jsonl"
    _write_ledger(
        led,
        [
            _ppi_entry(_SEED_GENES, edges=9, expected=1, nodes=9),  # canonical
            _ppi_entry(["TERT", "TERC", "TOLLIP", "DSP", "AKAP13"], edges=1, expected=0, nodes=5),  # latest, degenerate
        ],
    )
    got = extract_ppi_enrichment(led, select_identifiers=_SEED)
    assert got == {
        "number_of_nodes": 9.0,
        "number_of_edges": 9.0,
        "expected_number_of_edges": 1.0,
    }


def test_ppi_none_when_no_entry_matches_seed(tmp_path: Path) -> None:
    """No canonical-seed call in the ledger → None (honest fail-loud).

    We must NOT fall back to 'latest' — that is exactly the bug. Absence of a
    seed-scoped call means the claim cannot be verified, so layer ④ stays
    silent and layer ⑤ flags the unbacked metric.
    """
    led = tmp_path / "p.jsonl"
    _write_ledger(
        led,
        [_ppi_entry(["TERT", "TERC", "TOLLIP"], edges=1, expected=0, nodes=3)],
    )
    assert extract_ppi_enrichment(led, select_identifiers=_SEED) is None


# ── network: pick the seed module, not the latest, not the biggest ──────────


def test_network_selects_seed_over_latest_and_biggest(tmp_path: Path) -> None:
    """The seed 9-gene network (mean 0.6053) must win over BOTH a bigger
    expanded 19-gene network (more edges) AND a later degenerate subset.

    This is the case naive 'max edges' gets wrong: the 19-gene expanded net has
    the most edges but is a different scientific object.
    """
    led = tmp_path / "p.jsonl"
    seed_scores = [0.9, 0.6, 0.3, 0.9, 0.6, 0.3, 0.9, 0.6, 0.3]  # mean 0.6
    expanded = _SEED_GENES + ["FGFR1", "FGFR2", "FGFR3", "PDGFRA"]
    _write_ledger(
        led,
        [
            _network_entry(_SEED_GENES, seed_scores),          # canonical seed
            _network_entry(expanded, [0.8] * 40),              # biggest (max edges)
            _network_entry(["TERT", "TERC"], [0.99]),          # latest, degenerate
        ],
    )
    got = extract_interaction_scores(led, select_identifiers=_SEED)
    assert got == tuple(seed_scores)


def test_network_exact_set_not_superset(tmp_path: Path) -> None:
    """A superset network (seed ∪ extra genes) must NOT satisfy a seed match.

    Set EQUALITY, not containment — otherwise the expanded net would be picked.
    """
    led = tmp_path / "p.jsonl"
    superset = _SEED_GENES + ["FGFR1"]
    _write_ledger(led, [_network_entry(superset, [0.5, 0.5])])
    assert extract_interaction_scores(led, select_identifiers=_SEED) is None


# ── regression: no seed set → exact prior 'latest' behaviour ────────────────


def test_none_select_preserves_latest_ppi(tmp_path: Path) -> None:
    """select_identifiers=None must reproduce the legacy 'latest wins' path."""
    led = tmp_path / "p.jsonl"
    _write_ledger(
        led,
        [
            _ppi_entry(_SEED_GENES, edges=9, expected=1, nodes=9),
            _ppi_entry(["TERT", "TERC"], edges=1, expected=0, nodes=2),
        ],
    )
    got = extract_ppi_enrichment(led)  # default None
    assert got == {
        "number_of_nodes": 2.0,
        "number_of_edges": 1.0,
        "expected_number_of_edges": 0.0,
    }


def test_none_select_preserves_latest_network(tmp_path: Path) -> None:
    led = tmp_path / "p.jsonl"
    _write_ledger(
        led,
        [
            _network_entry(_SEED_GENES, [0.6, 0.6]),
            _network_entry(["TERT", "TERC"], [0.99]),
        ],
    )
    assert extract_interaction_scores(led) == (0.99,)
