"""Stage-14 wiring for the Biomni verification gates (Phase 0, layers ⑤ + ②).

This is the *pure* seam between a parsed ``experiment_summary.json`` and the
two trust-minimal gates. The pipeline runner resolves the provenance ledger
and entity-reference DBs from :class:`~researchclaw.config.BiomniConfig`,
calls :func:`run_biomni_gates`, then folds the reports into the experiment
diagnosis via ``add_biomni_verification_deficiencies`` so any failure becomes a
*critical* deficiency and the existing fail-loud machinery halts before a paper
is written on unverified numbers.

Nothing here imports the pipeline, so it stays unit-testable in isolation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from researchclaw.experiment.verify.claim_binding import ClaimBindingReport, bind_claims
from researchclaw.experiment.verify.entity_gate import EntityGate, EntityReport

_ENTITY_KINDS = ("gene", "drug", "pathway")


def collect_claimed_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    """Headline metrics the experiment reported.

    Prefers ``best_run.metrics`` (the canonical results.json numbers) and falls
    back to a top-level ``metrics`` block. Returns an empty dict when neither
    is present.
    """
    best = summary.get("best_run")
    if isinstance(best, dict) and isinstance(best.get("metrics"), dict):
        return dict(best["metrics"])
    metrics = summary.get("metrics")
    return dict(metrics) if isinstance(metrics, dict) else {}


def collect_claimed_entities(summary: dict[str, Any]) -> dict[str, list[str]]:
    """Biomedical identifiers the experiment claims to have analysed.

    Reads ``summary['entities']`` — a dict with ``gene`` / ``drug`` / ``pathway``
    lists. Returns ``{}`` when absent so the caller skips the entity gate.
    """
    entities = summary.get("entities")
    if not isinstance(entities, dict):
        return {}
    out: dict[str, list[str]] = {}
    for kind in _ENTITY_KINDS:
        ids = entities.get(kind)
        if isinstance(ids, list) and ids:
            out[kind] = [str(i) for i in ids]
    return out


def run_biomni_gates(
    summary: dict[str, Any],
    *,
    ledger_path: Path | None = None,
    gate: EntityGate | None = None,
) -> tuple[ClaimBindingReport | None, EntityReport | None]:
    """Run layer ⑤ (claim binding) and layer ② (entity gate) over *summary*.

    Each report is ``None`` when its inputs are unavailable — no ledger file,
    no configured gate, or no claimed entities — so the caller injects a
    deficiency only when a gate actually ran.
    """
    claim_report: ClaimBindingReport | None = None
    if ledger_path is not None and Path(ledger_path).exists():
        claim_report = bind_claims(collect_claimed_metrics(summary), Path(ledger_path))

    entity_report: EntityReport | None = None
    if gate is not None:
        entities = collect_claimed_entities(summary)
        if entities:
            entity_report = gate.check(entities)

    return claim_report, entity_report


def _resolve_db(rel: str, base_dir: Path) -> Path | None:
    if not rel:
        return None
    p = Path(rel)
    return p if p.is_absolute() else base_dir / p


def build_entity_gate(biomni_cfg: Any, base_dir: Path) -> EntityGate | None:
    """Build an :class:`EntityGate` from configured reference DBs.

    DB paths are resolved relative to *base_dir* (unless absolute). Only paths
    that exist are loaded. Returns ``None`` when no DB is configured at all.
    """
    gene = _resolve_db(getattr(biomni_cfg, "gene_db", ""), base_dir)
    drug = _resolve_db(getattr(biomni_cfg, "drug_hashes_db", ""), base_dir)
    pathway = _resolve_db(getattr(biomni_cfg, "pathway_db", ""), base_dir)
    if not any((gene, drug, pathway)):
        return None
    return EntityGate.from_files(
        gene=gene if gene and gene.exists() else None,
        drug_hashes=drug if drug and drug.exists() else None,
        pathway=pathway if pathway and pathway.exists() else None,
    )


def resolve_ledger_path(biomni_cfg: Any, run_dir: Path) -> Path | None:
    """Locate the provenance ledger for a run, or ``None`` if it is absent.

    An absolute ``provenance_path`` is used as-is. A relative one is looked up
    first directly under *run_dir*, then by searching the run tree for its
    basename (the bridge writes it into the stage-12 workspace). The most
    recent match wins.
    """
    name = getattr(biomni_cfg, "provenance_path", "") or "provenance.jsonl"
    p = Path(name)
    if p.is_absolute():
        return p if p.exists() else None

    direct = run_dir / name
    if direct.exists():
        return direct

    matches = sorted(run_dir.glob(f"**/{p.name}"))
    return matches[-1] if matches else None
