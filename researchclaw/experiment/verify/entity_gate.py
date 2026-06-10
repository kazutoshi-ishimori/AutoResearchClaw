"""Entity-ID gate (Phase 0, layer ②).

Validates biomedical identifiers claimed in results.json against authoritative
reference sets:

  * gene, pathway  — plaintext ID sets (e.g. HGNC symbols, Reactome IDs)
  * drug           — sha256 hash membership only, so the licensed DrugBank
                     identifier space is never stored in plaintext

An identifier absent from its reference is flagged. This is the gate that
catches identifier mix-ups (e.g. a model reporting "Tofacitinib" data under a
DrugBank ID that actually belongs to a different molecule).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EntityReport:
    """Result of checking claimed identifiers against the references."""

    unknown: dict[str, list[str]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.unknown

    def summary(self) -> str:
        if self.ok:
            return "entity_gate: all identifiers recognised."
        parts = [f"{kind}: {', '.join(ids)}" for kind, ids in self.unknown.items()]
        return "entity_gate: unknown identifiers -> " + "; ".join(parts)


def _load_lines(path: Path) -> set[str]:
    out: set[str] = set()
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.add(line)
    return out


@dataclass
class EntityGate:
    """Holds reference sets and checks claimed identifiers against them."""

    gene: set[str] = field(default_factory=set)
    drug_hashes: set[str] = field(default_factory=set)
    pathway: set[str] = field(default_factory=set)

    @classmethod
    def from_files(
        cls,
        *,
        gene: Path | None = None,
        drug_hashes: Path | None = None,
        pathway: Path | None = None,
    ) -> "EntityGate":
        return cls(
            gene=_load_lines(gene) if gene else set(),
            drug_hashes=_load_lines(drug_hashes) if drug_hashes else set(),
            pathway=_load_lines(pathway) if pathway else set(),
        )

    def check(self, entities: dict[str, list[str]]) -> EntityReport:
        report = EntityReport()
        for gid in entities.get("gene", []):
            if gid not in self.gene:
                report.unknown.setdefault("gene", []).append(gid)
        for pid in entities.get("pathway", []):
            if pid not in self.pathway:
                report.unknown.setdefault("pathway", []).append(pid)
        for did in entities.get("drug", []):
            if hashlib.sha256(did.encode()).hexdigest() not in self.drug_hashes:
                report.unknown.setdefault("drug", []).append(did)
        return report
