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
from researchclaw.experiment.verify.ledger_network import (
    extract_interaction_scores,
    extract_ppi_enrichment,
    extract_uniprot_length,
)
from researchclaw.experiment.verify.oracle_registry import (
    RecomputeContext,
    build_covid_oracles,
)
from researchclaw.experiment.verify.recompute import (
    RecomputeReport,
    recompute_metrics,
)

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
    verified_metrics: frozenset[str] = frozenset(),
) -> tuple[ClaimBindingReport | None, EntityReport | None]:
    """Run layer ⑤ (claim binding) and layer ② (entity gate) over *summary*.

    Each report is ``None`` when its inputs are unavailable — no ledger file,
    no configured gate, or no claimed entities — so the caller injects a
    deficiency only when a gate actually ran.

    ``verified_metrics`` carries the names layer ④ has already reproduced from
    the (ranking, positives) contract; those are threaded into ⑤ so a
    legitimately-derived headline metric (not present as a raw tool return) is
    treated as backed instead of flagged fabricated. See :func:`bind_claims`.
    """
    claim_report: ClaimBindingReport | None = None
    if ledger_path is not None and Path(ledger_path).exists():
        claim_report = bind_claims(
            collect_claimed_metrics(summary),
            Path(ledger_path),
            verified_metrics=verified_metrics,
        )

    entity_report: EntityReport | None = None
    if gate is not None:
        entities = collect_claimed_entities(summary)
        if entities:
            entity_report = gate.check(entities)

    return claim_report, entity_report


def collect_recompute_context(
    summary: dict[str, Any], ledger_path: Path | None = None
) -> RecomputeContext | None:
    """Build a :class:`RecomputeContext` from the summary and (optionally) the ledger.

    Two independent input sources feed layer ④:

    * ``ranking`` + ``positives`` (top level of the summary) — the ranking oracles;
    * the ledger's most-recent query_stringdb ppi_enrichment return — the network
      oracles (fold, avg degree);
    * the ledger's most-recent query_stringdb ``network`` return — the
      mean_interaction_score oracle (per-edge scores);
    * the ledger's most-recent query_uniprot return — the UniProt residue length
      for the cross-tool edges_per_ace2_residue oracle (A-④).

    These raw numbers are read from the *ledger*, never the summary, so the
    metric binds to provenance-anchored tool calls — for the cross-tool oracle,
    to two different tools' calls at once.

    Returns ``None`` only when neither source yields anything, so the recompute
    gate stays silent on absence (layer ④ fires on contradiction, not absence).
    """
    ranking: tuple[tuple[str, float], ...] = ()
    positives: frozenset[str] = frozenset()
    ranking_raw = summary.get("ranking")
    positives_raw = summary.get("positives")
    if (
        isinstance(ranking_raw, list)
        and ranking_raw
        and isinstance(positives_raw, list)
        and positives_raw
    ):
        try:
            ranking = tuple((str(item[0]), float(item[1])) for item in ranking_raw)
            positives = frozenset(str(p) for p in positives_raw)
        except (TypeError, ValueError, IndexError):
            ranking = ()
            positives = frozenset()

    network = None
    interactions = None
    protein_length = None
    if ledger_path is not None and Path(ledger_path).exists():
        network = extract_ppi_enrichment(Path(ledger_path))
        interactions = extract_interaction_scores(Path(ledger_path))
        protein_length = extract_uniprot_length(Path(ledger_path))

    if (
        not ranking
        and network is None
        and interactions is None
        and protein_length is None
    ):
        return None
    return RecomputeContext(
        ranking=ranking,
        positives=positives,
        network=network,
        interactions=interactions,
        protein_length=protein_length,
    )


def run_recompute_gate(
    summary: dict[str, Any],
    biomni_cfg: Any,
    ledger_path: Path | None = None,
) -> RecomputeReport | None:
    """Run layer ④ over *summary* (+ ledger) if oracles and inputs align.

    Returns ``None`` — no report at all — when any prerequisite is missing:

    * ``recompute_headline_metrics`` is empty (layer ④ disabled by config);
    * neither a ranking/positives contract nor a ledger network return exists;
    * none of the configured names are registered in the oracle registry.

    Otherwise returns a :class:`RecomputeReport` that the runner folds into
    the experiment diagnosis. Silence on absence keeps the gate consistent
    with the other layers; mismatch on contradiction is loud.
    """
    metric_names = tuple(getattr(biomni_cfg, "recompute_headline_metrics", ()) or ())
    if not metric_names:
        return None
    ctx = collect_recompute_context(summary, ledger_path=ledger_path)
    if ctx is None:
        return None
    oracles = build_covid_oracles(metric_names, ctx)
    if not oracles:
        return None
    return recompute_metrics(collect_claimed_metrics(summary), oracles)


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
