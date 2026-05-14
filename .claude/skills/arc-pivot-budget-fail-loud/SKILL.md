---
name: arc-pivot-budget-fail-loud
description: >
  When AutoResearchClaw exhausts max_pivots (default 2) with quality gate
  still failing, BLOCK paper writing and surface the underlying defect for
  human triage instead of continuing into a "degraded: true" final state.
  Set degraded_block_writing=true or check pipeline_summary.degraded
  manually before running stages 16-22. Use when configuring the pipeline,
  when pipeline_summary.json shows degraded:true, or when seeing
  "Max pivots reached" in quality_warning.txt. Triggers on:
  "max pivots reached", "degraded: true", "pipeline_summary.degraded",
  "quality_warning.txt", "Paper will be written but may have significant
  issues".
---

# ARC Pivot Budget: Fail Loud, Don't Ship Degraded

ARC's default policy when `max_pivots` is exhausted with the quality gate
still failing is to **continue into stages 16-22 and produce a paper
flagged `degraded: true`**. The paper compiles, the LaTeX builds, the
PDF lands in `deliverables/` — but the underlying experiment is broken
and the manuscript is unsubmittable. This is the worst possible outcome:
hours of compute spent producing a clean-looking PDF that nobody can use.

## Symptom signature

```bash
RUN_DIR=artifacts/<run-id>
# All three present together = this skill applies
test -f "$RUN_DIR/quality_warning.txt"          && echo "✓ warning file"
test -f "$RUN_DIR/deliverables/paper.pdf"       && echo "✓ paper exists"
grep '"degraded": true' "$RUN_DIR/pipeline_summary.json" && echo "✓ degraded flag"
grep -c "Max pivots" "$RUN_DIR/quality_warning.txt"
```

## Pre-flight policy — block writing on degraded

Two layers of defense:

### Layer 1 — config policy (proactive)

In `config.arc.yaml`, add an explicit guard:

```yaml
research:
  # When True, max_pivots exhaustion HALTS the pipeline at stage 15
  # instead of degraded-continuing into stages 16-22.
  block_writing_on_pivot_exhaustion: true   # default: false
```

If your ARC version doesn't yet expose this flag, post a hook into stage
15 (RESEARCH_DECISION) that returns `decision: block` instead of
`decision: refine` once `len(decision_history) >= max_pivots`.

### Layer 2 — pre-stage-16 bash guard (reactive, no code change)

Wrap the pipeline invocation:

```bash
RUN_DIR=artifacts/<run-id>
researchclaw run --to-stage 15 --output "$RUN_DIR"

if [ -f "$RUN_DIR/quality_warning.txt" ]; then
    echo "✗ Quality gate failed. Halting before stage 16."
    cat "$RUN_DIR/quality_warning.txt"
    exit 1
fi

researchclaw run --from-stage 16 --output "$RUN_DIR"
```

## What to do when degraded triggers

This is the triage flowchart for the human (or this Claude session):

```
quality_warning.txt exists
├── "Max pivots reached" + identical_conditions?
│     → invoke arc-ablation-identical-conditions-circuit-breaker
│       (fix the wiring bug, then resume from stage 13)
│
├── "Max pivots reached" + peer_review failed N times?
│     → invoke arc-peer-review-model-tier
│       (pin stage 18 to mid-tier, resume from stage 18)
│
├── degraded: true but no specific lesson?
│     → inspect stage-15/research_decision.json for the REFINE rationale
│       and decide whether to escalate (HITL) or accept the degraded paper
│
└── degraded: true with quality_score <= 3 in stage-20/quality_report.json
      → DO NOT EXPORT. Discard deliverables/, fix root cause, rerun.
```

## When a degraded paper IS acceptable

A `degraded: true` paper can be useful as a **first draft for human
review**, NOT as a submission. Acceptable cases:

- The ARC run was an exploratory pilot and you wanted to see what came out
- You are using ARC to generate scaffolding (figures, references, structure)
  that you will manually rewrite
- You are debugging the pipeline itself

In all these cases, mark the artifact directory as draft (e.g.,
`mv artifacts/rc-XXXX artifacts/draft-rc-XXXX`) so it is not confused
with a publishable run.

## Anti-patterns

- ❌ Bumping `max_pivots` from 2 to 5 — the pivots are LLM prompt
  mutations, not code fixes. If 2 didn't help, 5 won't either.
- ❌ Suppressing `quality_warning.txt` to make the pipeline "succeed" —
  hides the very signal you need.
- ❌ Submitting a `degraded: true` paper to a venue — peer review will
  reject (e.g., quality_score=2 verdict=reject in the SIR run that
  motivated this skill).

## Provenance

Distilled from
`workspaces/sir-failure-map-paper/artifacts/rc-20260418-071251-89d552/`:
- `quality_warning.txt`: "Max pivots (2) reached. Quality gate failed:
  4 ablation warnings... Paper will be written but may have significant
  issues."
- `pipeline_summary.json`: `degraded: true` but `final_status: done`
- `stage-20/quality_report.json`: `score: 2, verdict: reject`
- A 264 KB PDF was produced and nobody could submit it.
