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
from typing import Callable


@dataclass(frozen=True)
class RecomputeContext:
    """Inputs an oracle closure may need to recompute a headline metric.

    ``ranking`` is the agent-produced ordered list of candidates as
    ``(drug_id, score)`` tuples. ``positives`` is the held-out ground-truth
    positive set (Phase 2/3+ approved drugs in COVID's case).
    """

    ranking: tuple[tuple[str, float], ...]
    positives: frozenset[str]


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
_BUILDERS: dict[str, Callable[[RecomputeContext], Callable[[], float]]] = {
    "auroc_phase2plus": _build_auroc_phase2plus,
    "mean_rank_of_positives": _build_mean_rank_of_positives,
}


def build_covid_oracles(
    metric_names: tuple[str, ...],
    context: RecomputeContext,
) -> dict[str, Callable[[], float]]:
    """Translate the configured metric names into a dict of oracle closures.

    Unknown names are silently dropped — see module docstring for rationale.
    """
    oracles: dict[str, Callable[[], float]] = {}
    for name in metric_names:
        builder = _BUILDERS.get(name)
        if builder is None:
            continue
        oracles[name] = builder(context)
    return oracles


__all__ = ["RecomputeContext", "build_covid_oracles"]
