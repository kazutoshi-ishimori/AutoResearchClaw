#!/usr/bin/env python3
"""Build the Biomni entity-gate reference DBs (Phase 0, layer ②).

Emits three authoritative reference sets under ``data/refs/`` from local,
license-clean (or hash-only) sources:

  * ``genes_human.tsv``       — human gene symbols (STRING v12 preferred names)
  * ``pathways_kegg2021.tsv`` — KEGG 2021 Human pathway names
  * ``drugbank_ids.sha256``   — sha256 of every DrugBank ID (HASH ONLY)

The DrugBank source is licensed; only non-reversible sha256 digests are
written, never the plaintext IDs — consistent with the repo's hash-only
DrugBank policy.  Re-run after refreshing any source.

Usage:
    python scripts/build_entity_refs.py
"""

from __future__ import annotations

import gzip
import hashlib
import itertools
import json
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
REFS = REPO / "data" / "refs"
COVID = REPO / "workspaces" / "covid-repurposing-paper" / "data"

STRING_INFO = COVID / "string" / "9606.protein.info.v12.0.txt.gz"
KEGG_JSON = COVID / "kegg" / "KEGG_2021_Human.json"
DRUGS_TSV = COVID / "drugbank" / "extracted" / "drugs.tsv"


def _header(source: str, n: int) -> str:
    return (
        f"# Biomni entity-gate reference (Phase 0, layer ②)\n"
        f"# source: {source}\n"
        f"# built: {date.today().isoformat()}  count: {n}\n"
    )


def build_genes() -> int:
    symbols: set[str] = set()
    with gzip.open(STRING_INFO, "rt", encoding="utf-8") as fh:
        next(fh)  # header line
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2 and parts[1]:
                symbols.add(parts[1])
    out = REFS / "genes_human.tsv"
    out.write_text(
        _header("STRING v12.0 9606.protein.info preferred_name", len(symbols))
        + "\n".join(sorted(symbols)) + "\n",
        encoding="utf-8",
    )
    return len(symbols)


def build_pathways() -> int:
    data = json.loads(KEGG_JSON.read_text(encoding="utf-8"))
    names = sorted(data.keys())
    out = REFS / "pathways_kegg2021.tsv"
    out.write_text(
        _header("KEGG_2021_Human (Enrichr) pathway names", len(names))
        + "\n".join(names) + "\n",
        encoding="utf-8",
    )
    return len(names)


def build_drug_hashes() -> int:
    ids: set[str] = set()
    with DRUGS_TSV.open("r", encoding="utf-8") as fh:
        next(fh)  # header: drugbank_id\tname\ttype\tgroups
        for line in fh:
            did = line.split("\t", 1)[0].strip()
            if did:
                ids.add(did)
    digests = sorted(hashlib.sha256(i.encode()).hexdigest() for i in ids)
    out = REFS / "drugbank_ids.sha256"
    out.write_text(
        _header("DrugBank full database (LICENSED) — sha256 of drugbank_id ONLY", len(digests))
        + "\n".join(digests) + "\n",
        encoding="utf-8",
    )
    return len(digests)


def main() -> int:
    REFS.mkdir(parents=True, exist_ok=True)
    g = build_genes()
    p = build_pathways()
    d = build_drug_hashes()
    print(f"genes_human.tsv:        {g} symbols")
    print(f"pathways_kegg2021.tsv:  {p} pathways")
    print(f"drugbank_ids.sha256:    {d} hashes (hash-only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
