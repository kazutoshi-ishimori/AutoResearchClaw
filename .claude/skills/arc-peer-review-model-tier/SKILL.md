---
name: arc-peer-review-model-tier
description: >
  Never assign nano-tier or other small fallback models to AutoResearchClaw
  stage 18 (PEER_REVIEW). Stage 18 prompts are long (full paper draft + 3
  reviewer personas) and consistently exceed nano-tier reliability budgets,
  causing "All models failed. Last error: LLM call failed after 3 retries"
  with no usable output. Use when configuring fallback_models, when stage 18
  fails repeatedly, or when seeing "Stage peer_review failed" in lessons.jsonl.
  Triggers on: "peer_review failed", "stage 18", "All models failed",
  "gpt-5.4-nano", "fallback_models", "PEER_REVIEW retry".
---

# ARC Stage 18: Reserve Primary-Tier Models for Peer Review

`stage-18/peer_review` constructs prompts that include the entire paper
draft (8-12k tokens) plus three reviewer-persona instructions and produces
3 structured reviews. Nano-tier models (`gpt-5.4-nano`, equivalent
`-mini`-of-mini variants) systematically fail this load with no graceful
degradation — they hit context, rate, or coherence limits and the stage
exits with `LLM call failed after 3 retries`.

## Diagnosis

```bash
RUN_DIR=artifacts/<run-id>
grep -c "Stage peer_review failed" "$RUN_DIR/evolution/lessons.jsonl"
# If > 0, this skill applies.
```

Confirm the fallback chain ordering:

```bash
grep -A 5 'fallback_models' config.arc.yaml
```

If `gpt-5.4-nano` (or any `*-nano`, `*-mini-mini`) appears in the chain
**before** stage 18 reaches a working model, the pipeline burns retries
on a model that cannot complete the task.

## Fix recipe — split model tiers by stage

Two equivalent options:

### Option A — global config (simplest)

In `config.arc.yaml`, ensure the fallback chain only contains models
that can handle 12k-token structured-output prompts:

```yaml
llm:
  primary_model: "gpt-5.4-mini"     # OK for stage 18
  fallback_models:
    - "gpt-5.4"                      # Heavier but reliable
    # - "gpt-5.4-nano"               # ← REMOVE, breaks stage 18
```

### Option B — per-stage model override (preferred when supported)

If your prompts file supports it, pin stage 18 to primary tier only:

```yaml
# prompts.default.yaml or custom prompts file
peer_review:
  model_override: "gpt-5.4-mini"
  no_fallback: true   # Don't degrade to nano on retry
```

## Stages where nano-tier IS acceptable

Nano models work for short, well-bounded calls:

| Stage | Task | Nano OK? |
|---|---|---|
| 02 LITERATURE_SEARCH | Query rewriting | ✓ |
| 04 BASELINE_NAVIGATION | Single-baseline summary | ✓ |
| 14 RESULT_INTERPRETATION | One-table interpretation | ✓ |
| **18 PEER_REVIEW** | **3 reviewers × full paper** | **✗** |
| **19 PAPER_REVISION** | **Full-paper rewrite** | **✗** |
| **20 QUALITY_ASSESSMENT** | **Cross-check tables vs summary** | **✗** |

The pattern: any stage that ingests the full paper or full experiment
artifact needs at least mid-tier capacity.

## Resume after the fix

```bash
cd ~/workspace/AutoResearchClaw
researchclaw run --from-stage 18 --output artifacts/<run-id>
```

## Anti-patterns

- ❌ Increasing retry count from 3 to 10 — wastes time on a model that
  fundamentally cannot complete the task.
- ❌ Removing the peer_review stage entirely — loses the strongest
  quality signal in the pipeline.
- ❌ Switching the entire pipeline to gpt-5.4 (heavy tier) just to fix
  stage 18 — burns budget on cheap stages that nano handles fine.

## Provenance

Distilled from
`workspaces/sir-failure-map-paper/artifacts/rc-20260418-071251-89d552/evolution/lessons.jsonl`:
5 consecutive `peer_review failed: LLM call failed after 3 retries for
model gpt-5.4-nano` errors across 5 separate `run_id`s before the
pipeline finally landed on a working model.
