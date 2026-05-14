# Mortality Suppression Paper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a focused AutoResearchClaw run workspace for the mortality suppression paper.

**Architecture:** Add workspace-level documentation, a dedicated ARC config, and prompt overrides that constrain the pipeline to a synthetic-truth public-health statistics benchmark. Validate config loading and prompt rendering before running ARC.

**Tech Stack:** AutoResearchClaw CLI, YAML prompt overrides, Python 3.11, numpy-only sandbox experiments.

---

### Task 1: Workspace Files

**Files:**
- Create: `workspaces/mortality-suppression-paper/README.md`
- Create: `workspaces/mortality-suppression-paper/RESEARCH_DESIGN.md`
- Create: `workspaces/mortality-suppression-paper/config_mortality_suppression.yaml`
- Create: `workspaces/mortality-suppression-paper/prompts_mortality_suppression.yaml`

- [x] **Step 1: Create workspace documentation**

Document the research question, ARC run commands, and privacy guardrails.

- [x] **Step 2: Create ARC config**

Use sandbox mode, `primary_metric`, `metric_direction: minimize`, HITL gates,
and a custom prompt file path.

- [x] **Step 3: Create prompt overrides**

Override search strategy, experiment design, code generation, result analysis,
and paper drafting prompts so ARC uses synthetic ground truth and does not infer
real suppressed cells.

### Task 2: Validation

**Files:**
- Read: `workspaces/mortality-suppression-paper/config_mortality_suppression.yaml`
- Read: `workspaces/mortality-suppression-paper/prompts_mortality_suppression.yaml`

- [x] **Step 1: Validate config**

Run:

```bash
.venv/bin/researchclaw validate --config workspaces/mortality-suppression-paper/config_mortality_suppression.yaml
```

Expected: validation succeeds or reports only non-blocking environment warnings.

- [x] **Step 2: Validate prompt rendering**

Run:

```bash
.venv/bin/python -c "from researchclaw.prompts import PromptManager; pm=PromptManager('workspaces/mortality-suppression-paper/prompts_mortality_suppression.yaml'); p=pm.for_stage('experiment_design', preamble='P', hypotheses='H', mortality_suppression_guardrails=pm.block('mortality_suppression_guardrails')); assert 'synthetic county-by-age mortality truth' in p.user; print('prompt ok')"
```

Expected: `prompt ok`.

### Task 3: First ARC Run

**Files:**
- Read: `workspaces/mortality-suppression-paper/config_mortality_suppression.yaml`

- [ ] **Step 1: Run to Stage 9**

Run:

```bash
.venv/bin/researchclaw run \
  --config workspaces/mortality-suppression-paper/config_mortality_suppression.yaml \
  --mode co-pilot \
  --to-stage EXPERIMENT_DESIGN
```

Expected: ARC pauses after producing the Stage 9 experiment plan for review.

- [ ] **Step 2: Review Stage 9**

Check that the plan includes synthetic truth, suppression of 0-9 deaths,
unreliable flags below 20 deaths, ranking metrics, and numpy-only feasibility.
