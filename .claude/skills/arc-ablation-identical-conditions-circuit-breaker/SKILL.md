---
name: arc-ablation-identical-conditions-circuit-breaker
description: >
  When AutoResearchClaw's experiment_diagnosis flags "identical_conditions"
  (≥2 ablation conditions producing byte-identical outputs), STOP the
  pipeline and fix the experiment code rather than burning pivot budget.
  The differentiating parameter is not wired into the forward pass — no
  amount of stage-19 paper revision can fix that. Use during stage 13
  ITERATIVE_REFINE, when reviewing experiment_diagnosis.json, or when the
  REFINE decision repeats. Triggers on: "identical_conditions",
  "experiment_diagnosis.json", "ablation pair(s) produce identical outputs",
  "REFINE", "max pivots reached", "differentiating parameter is likely
  not wired".
---

# ARC Ablation: Identical-Conditions Circuit Breaker

`stage-13` produces `experiment_diagnosis.json`. When it contains:

```json
{
  "type": "identical_conditions",
  "severity": "major",
  "description": "N ablation pair(s) produce identical outputs."
}
```

…the experiment script **failed to wire the condition parameter through
the model**. The ablation knob is logged but never read in the forward
pass. The default ARC behavior is to PIVOT (rerun with prompt patches)
up to `max_pivots` (default 2), then proceed to writing with a degraded
flag. **This always wastes the pivot budget on a defect that is
mechanically detectable and code-fixable, never prompt-fixable.**

## Detection

```bash
RUN_DIR=artifacts/<run-id>
DIAG="$RUN_DIR/experiment_diagnosis.json"
# Count identical-condition deficiencies
python -c "
import json
d = json.load(open('$DIAG'))
defs = d.get('diagnosis', {}).get('deficiencies', [])
ic = [x for x in defs if x.get('type') == 'identical_conditions']
print(f'identical_conditions deficiencies: {len(ic)}')
for x in ic:
    print(f'  affected: {x.get(\"affected_conditions\", [])}')"
```

If count > 0, this skill applies. **Do not pivot. Open the experiment
script and inspect the conditional branches.**

## Fix recipe — find the unwired knob

```bash
SCRIPT="$RUN_DIR/stage-13/experiment_final.py"   # or stage-12/experiment.py

# 1. List all condition names from diagnosis
python -c "
import json
d = json.load(open('$RUN_DIR/experiment_diagnosis.json'))
ic = [x for x in d['diagnosis']['deficiencies'] if x['type']=='identical_conditions'][0]
for c in ic['affected_conditions']: print(c)"
# e.g. delay_aware_underreporting_poisson_sir
#      negative_binomial_overdispersion_sir

# 2. For each condition name, grep where it's BRANCHED on (not just logged)
for c in $(python -c "..."); do
    echo "=== $c ==="
    grep -n -E "(if|elif|case|condition).*['\"]$c['\"]" "$SCRIPT"
done
```

If the only matches are `condition_name = "..."` (assignment) or
`logging.info(f"Running {condition_name}")` (logging) but **no branching
on `condition_name`**, that is the bug. The script runs the same forward
pass for every condition.

## Three common root causes

1. **Loop variable not consumed** — outer loop iterates over conditions
   but the inner `model.fit(data)` ignores the variable.

   ```python
   # ✗ Bug
   for cond in conditions:
       model = SIRModel()        # ← always default config
       results[cond] = model.fit(data)

   # ✓ Fix
   for cond in conditions:
       model = SIRModel(observation_kind=cond)   # ← actually wire it
       results[cond] = model.fit(data)
   ```

2. **Config dict built but unread** — the dispatcher constructs a config
   per condition but the model class ignores unknown kwargs.

3. **String comparison typo** — `if cond == "delay_aware"` but the
   condition is named `"delay_aware_poisson_sir"`. The comparison
   silently falls through to default.

## Why pivoting cannot fix this

Stage 19 (`paper_revision`) and stage 13 PIVOT (`refine_sandbox_v2`) are
LLM-driven prompt mutations. They cannot detect that `if cond == X`
never matches `X` in the actual data. Pivoting will:

- Burn 30-60 minutes per pivot rerun
- Produce identical numeric outputs again
- Hit `max_pivots` and proceed to writing
- Generate a paper with `5/8 conditions identical` that is
  unsubmittable (peer_review will reject with score 2)

## Resume after the fix

After patching `experiment.py`:

```bash
cd ~/workspace/AutoResearchClaw
# Resume from stage 13, NOT from stage 1
researchclaw run --from-stage 13 --output artifacts/<run-id>
```

## Anti-patterns

- ❌ Letting the pipeline pivot 2× and continue with `degraded: true`
  — the resulting paper will fail peer review.
- ❌ Adding more ablation conditions to "average out" the identical ones
  — the bug stays.
- ❌ Manually editing the results JSON to perturb identical values — that
  is fabrication and the quality assessor (stage 20) will catch it.

## Provenance

Distilled from
`workspaces/sir-failure-map-paper/artifacts/rc-20260418-071251-89d552/`:
- `experiment_diagnosis.json`: 5 of 8 SIR conditions identical
- `decision_history.json`: 2 PIVOT attempts, both failed to fix
- `quality_warning.txt`: "Max pivots (2) reached"
- `stage-20/quality_report.json`: `verdict: reject, score: 2`

A code fix at the `condition_name → model_kwarg` plumbing would have
shipped a publishable paper in one rerun instead of a degraded one
after two failed pivots.
