"""COVID-domain oracle registry for layer ④ (Phase 2 pipeline-wiring D).

:func:`researchclaw.experiment.verify.recompute.recompute_metrics` already
implements the generic "claim vs independent recompute" gate. What it needs
from the caller is a dict ``{metric_name -> zero-arg callable}`` of oracles.
The :class:`BiomniConfig` exposes only the *names* the operator wants
recomputed; this module bridges the two.

Design stance — keep the registry narrow and silent on the unknown:

* A name with no registered builder is **dropped**, not flagged. Layer ④ only
  fires on contradiction; an unregistered metric is "no oracle" and so the
  recompute gate silently skips it, exactly like
  :func:`recompute_metrics` already does for missing oracles.
* Oracles close over a :class:`RecomputeContext` so the call site (Stage 14)
  can inject ranking + ground-truth without ever touching this module's
  internals. Tests stay pure-python.

First resident: ``auroc_phase2plus`` — Mann-Whitney–U–style AUROC over a
positive set drawn from the candidate ranking. No sklearn dependency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Mapping


@dataclass(frozen=True)
class RecomputeContext:
    """Inputs an oracle closure may need to recompute a headline metric.

    ``ranking`` is the agent-produced ordered list of candidates as
    ``(drug_id, score)`` tuples. ``positives`` is the held-out ground-truth
    positive set (Phase 2/3+ approved drugs in COVID's case).
    """

    ranking: tuple[tuple[str, float], ...]
    positives: frozenset[str]
    # Ledger-sourced raw numbers for network oracles (e.g. STRING ppi_enrichment).
    # None when the run has no such tool call. Never hashed by this module.
    network: Mapping[str, float] | None = None
    # Ledger-sourced per-edge scores from the STRING `network` endpoint, for the
    # mean_interaction_score oracle. Independent of ``network``; None when absent.
    interactions: tuple[float, ...] | None = None
    # Ledger-sourced UniProt residue count (a DIFFERENT tool, query_uniprot), for
    # the cross-tool edges_per_ace2_residue oracle. None when absent. The whole
    # point of A-④: one metric bound to two provenance anchors from two tools.
    protein_length: float | None = None


def _auroc(ranking: tuple[tuple[str, float], ...], positives: frozenset[str]) -> float:
    """Pure-python Mann-Whitney–U AUROC. Ties contribute 0.5 to the win count.

    Returns NaN if either class is empty (AUROC undefined).
    """
    pos_scores: list[float] = []
    neg_scores: list[float] = []
    for drug_id, score in ranking:
        if drug_id in positives:
            pos_scores.append(score)
        else:
            neg_scores.append(score)
    if not pos_scores or not neg_scores:
        return float("nan")
    wins = 0.0
    for p in pos_scores:
        for n in neg_scores:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(pos_scores) * len(neg_scores))


def _build_auroc_phase2plus(ctx: RecomputeContext) -> Callable[[], float]:
    return lambda: _auroc(ctx.ranking, ctx.positives)


def _mean_rank_of_positives(
    ranking: tuple[tuple[str, float], ...], positives: frozenset[str]
) -> float:
    """Mean 1-based descending rank of the positives (1 = best), midrank ties.

    Mirrors ``methods_v2.eval.metrics.mean_rank_of_positives`` — ``rankdata`` on
    ``-scores`` with average-tie handling — but in pure python over the
    (ranking, positives) context. Because midrank is a function of the score
    *multiset* (not any emitted ordering), this is safe to recompute from the
    contract regardless of how ties were broken when the ranking was serialised.
    Returns NaN when there are no scoreable positives (rank undefined).
    """
    scores = [s for _, s in ranking]
    pos_scores = [s for drug_id, s in ranking if drug_id in positives]
    if not pos_scores:
        return float("nan")
    total = 0.0
    for s in pos_scores:
        n_greater = sum(1 for other in scores if other > s)
        n_equal = sum(1 for other in scores if other == s)
        total += n_greater + (n_equal + 1) / 2.0  # midrank of this score
    return total / len(pos_scores)


def _build_mean_rank_of_positives(ctx: RecomputeContext) -> Callable[[], float]:
    return lambda: _mean_rank_of_positives(ctx.ranking, ctx.positives)


def _ppi_enrichment_fold(network: Mapping[str, float]) -> float:
    """Observed/expected STRING interaction-edge ratio (network enrichment fold).

    A *derived* metric: STRING returns the two raw counts, not their ratio, so
    layer ④ must recompute it. Returns NaN when expected is missing or zero
    (ratio undefined) — parity with the ranking oracles' empty-class NaN.
    """
    obs = network.get("number_of_edges")
    exp = network.get("expected_number_of_edges")
    if obs is None or exp is None or exp == 0:
        return float("nan")
    return float(obs) / float(exp)


def _build_ppi_enrichment_fold(ctx: RecomputeContext) -> Callable[[], float] | None:
    if ctx.network is None:
        return None
    net = dict(ctx.network)
    return lambda: _ppi_enrichment_fold(net)


def _avg_node_degree(network: Mapping[str, float]) -> float:
    """Undirected average node degree = 2*edges/nodes (derived from the ledger).

    A second *derived* metric over the SAME STRING return the fold oracle uses.
    STRING reports ``average_node_degree`` directly, but layer ④ recomputes it
    independently from the two raw counts so the number the agent reports stays
    checkable. Returns NaN when nodes is missing or zero (degree undefined) —
    parity with the fold oracle's NaN on undefined ratio. That the two oracles
    read the same network dict is exactly what lets a tamper on one metric
    MISMATCH in isolation while the other stays verified.
    """
    edges = network.get("number_of_edges")
    nodes = network.get("number_of_nodes")
    if edges is None or nodes is None or nodes == 0:
        return float("nan")
    return 2.0 * float(edges) / float(nodes)


def _build_avg_node_degree(ctx: RecomputeContext) -> Callable[[], float] | None:
    if ctx.network is None:
        return None
    net = dict(ctx.network)
    return lambda: _avg_node_degree(net)


def _mean_interaction_score(interactions: tuple[float, ...]) -> float:
    """Mean STRING combined-confidence score over the network's edges (Σscore/n).

    A *list-aggregation* derived metric: the STRING ``network`` endpoint returns
    a per-edge score list, not their mean, so layer ④ aggregates it from the
    ledger. The mean of a score *multiset* is order-independent, so it is safe to
    recompute regardless of the edge order STRING serialised. Returns NaN on an
    empty edge list (mean undefined) — parity with the other oracles' NaN.
    """
    if not interactions:
        return float("nan")
    return sum(interactions) / len(interactions)


def _build_mean_interaction_score(ctx: RecomputeContext) -> Callable[[], float] | None:
    if ctx.interactions is None:
        return None
    scores = tuple(ctx.interactions)
    return lambda: _mean_interaction_score(scores)


def _edges_per_residue(network: Mapping[str, float], protein_length: float) -> float:
    """STRING edge count per UniProt residue = number_of_edges / sequence length.

    A *cross-tool* derived metric: the edge count comes from ``query_stringdb``
    and the residue length from ``query_uniprot`` — two different tools, two
    different provenance anchors. Neither returns the ratio, so layer ④ composes
    it. Returns NaN when the edge count is missing or the length is zero (ratio
    undefined) — parity with the other oracles' NaN on an undefined quantity.
    """
    edges = network.get("number_of_edges")
    if edges is None or protein_length == 0:
        return float("nan")
    return float(edges) / float(protein_length)


def _build_edges_per_ace2_residue(ctx: RecomputeContext) -> Callable[[], float] | None:
    # Requires BOTH anchors — the STRING network dict AND the UniProt length.
    # Missing either drops the oracle, so a single-tool run stays silent.
    if ctx.network is None or ctx.protein_length is None:
        return None
    net = dict(ctx.network)
    length = float(ctx.protein_length)
    return lambda: _edges_per_residue(net, length)


# Only metrics that are (a) a deterministic function of (ranking, positives) and
# (b) independent of how ties were broken when the ranking was serialised belong
# here — layer ④ fires on contradiction, so a tie-break- or RNG-sensitive oracle
# would raise false MISMATCHes. Deliberately EXCLUDED for that reason:
#   * precision/recall/hit/enrichment @k — the top-k boundary is tie-break
#     dependent, so an oracle that re-ranks the contract can disagree with the
#     experiment's own index-tie-broken top-k at ties.
#   * perm_pvalue, auroc_ci_* — RNG-driven (label shuffles / bootstrap resamples)
#     and not reproducible across processes without the exact random stream.
#   * class_recall_at_k — needs per-drug class labels, which the context omits.
_BUILDERS: dict[str, Callable[[RecomputeContext], Callable[[], float] | None]] = {
    "auroc_phase2plus": _build_auroc_phase2plus,
    "mean_rank_of_positives": _build_mean_rank_of_positives,
    "ppi_enrichment_fold": _build_ppi_enrichment_fold,
    "avg_node_degree": _build_avg_node_degree,
    "mean_interaction_score": _build_mean_interaction_score,
    "edges_per_ace2_residue": _build_edges_per_ace2_residue,
}


def build_covid_oracles(
    metric_names: tuple[str, ...],
    context: RecomputeContext,
) -> dict[str, Callable[[], float]]:
    """Translate the configured metric names into a dict of oracle closures.

    Unknown names are silently dropped. A registered builder that returns
    ``None`` (its required inputs are absent from the context) is likewise
    dropped, so layer ④ stays silent on absence.
    """
    oracles: dict[str, Callable[[], float]] = {}
    for name in metric_names:
        builder = _BUILDERS.get(name)
        if builder is None:
            continue
        oracle = builder(context)
        if oracle is None:
            continue
        oracles[name] = oracle
    return oracles


__all__ = ["RecomputeContext", "build_covid_oracles"]
