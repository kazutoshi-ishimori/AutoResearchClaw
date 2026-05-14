# SIR Inference Failure Map Run Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run AutoResearchClaw in an isolated workspace to produce a complete paper draft on SIR inference failure regions under observation noise and reporting delays, with outputs easy to extract from `deliverables/`.

**Architecture:** Use a dedicated git worktree plus a paper-specific local config copied from `config.arc.yaml`. Keep the first run conservative and full-auto, then only add a prompt override file if the generated draft drifts away from the approved spec. Treat `artifacts/<run-id>/deliverables/` as the export boundary for the paper bundle.

**Tech Stack:** Git worktree, Python virtualenv, AutoResearchClaw CLI, YAML config, optional prompt override YAML

---

## File Map

- Reuse: `config.arc.yaml`
  - Stable tracked baseline with the local environment already wired
- Reuse: `docs/superpowers/specs/2026-04-18-sir-inference-failure-map-design.md`
  - Approved paper spec
- Create: `config_sir_failure_map.yaml`
  - Worktree-local run config for this paper
- Create if needed: `prompts.sir_failure_map.yaml`
  - Worktree-local prompt overrides if the first run drifts from the approved topic
- Create if needed: `notes/sir_failure_map_run.md`
  - Short rerun notes and artifact handoff notes

## Task 1: Create The Isolated Workspace

**Files:**
- Reuse: `.gitignore`
- Reuse: `docs/superpowers/specs/2026-04-18-sir-inference-failure-map-design.md`

- [ ] **Step 1: Confirm the repo root and current branch**

Run: `git rev-parse --show-toplevel && git branch --show-current`
Expected: repo root path and `main`

- [ ] **Step 2: Create an isolated worktree under the ignored `workspaces/` directory**

Run: `git worktree add workspaces/sir-failure-map-paper -b sir-failure-map-paper`
Expected: a new worktree created at `workspaces/sir-failure-map-paper`

- [ ] **Step 3: Enter the new workspace and verify the spec is visible**

Run: `cd workspaces/sir-failure-map-paper && test -f docs/superpowers/specs/2026-04-18-sir-inference-failure-map-design.md && echo OK`
Expected: `OK`

- [ ] **Step 4: Activate the existing Python environment in the worktree**

Run: `source .venv/bin/activate`
Expected: shell prompt shows the venv is active

- [ ] **Step 5: Smoke-check the CLI before editing config**

Run: `researchclaw --help >/tmp/researchclaw-help.txt && tail -n 5 /tmp/researchclaw-help.txt`
Expected: CLI help prints without import errors

## Task 2: Prepare The Paper-Specific Config

**Files:**
- Reuse: `config.arc.yaml`
- Create: `config_sir_failure_map.yaml`

- [ ] **Step 1: Copy the tracked baseline config into a local paper config**

Run: `cp config.arc.yaml config_sir_failure_map.yaml`
Expected: `config_sir_failure_map.yaml` exists in the worktree root

- [ ] **Step 2: Edit the copied config to target the paper topic**

Update `config_sir_failure_map.yaml` so these fields match the approved spec:

```yaml
project:
  name: "sir-inference-failure-map"
  mode: "full-auto"

research:
  topic: "Map when SIR parameter inference and peak prediction break under synthetic observation noise, persistent under-reporting, and variable reporting delays, comparing least-squares and likelihood-based inference across mild, moderate, and rapid epidemic regimes."
  domains:
    - "biology"
    - "epidemiology"
    - "statistical-modeling"
```

- [ ] **Step 3: Keep the first run conservative**

Adjust `config_sir_failure_map.yaml` to reduce moving parts in the first run:

```yaml
experiment:
  mode: "sandbox"
  time_budget_sec: 300
  max_iterations: 10
  opencode:
    enabled: false
    auto: false
```

If a `cli_agent` block exists in the copied config, disable it for the first run.

- [ ] **Step 4: Keep outputs easy to cut out**

Ensure the config preserves:

```yaml
knowledge_base:
  backend: "markdown"
  root: "docs/kb"
```

Do not change the default artifacts root. The plan relies on `artifacts/<run-id>/deliverables/`.

- [ ] **Step 5: Verify the local config stays untracked**

Run: `git status --short`
Expected: `config_sir_failure_map.yaml` does not appear as a tracked-file edit

## Task 3: Sanity-Check The Run Inputs

**Files:**
- Reuse: `config_sir_failure_map.yaml`

- [ ] **Step 1: Re-read the final topic and project name from the config**

Run: `rg -n "name:|topic:|domains:" config_sir_failure_map.yaml`
Expected: the paper-specific project name and topic appear exactly once in the intended sections

- [ ] **Step 2: Run the built-in LLM preflight with the paper config**

Run: `researchclaw run --config config_sir_failure_map.yaml --topic "SIR inference failure map smoke check" --from-stage topic_init --auto-approve`
Expected: preflight succeeds and a run directory is created before the early-stage execution begins

- [ ] **Step 3: Inspect the newest artifact root after the smoke check**

Run: `ls -dt artifacts/rc-* | head -n 1`
Expected: the latest run directory path

- [ ] **Step 4: Remove the smoke-check artifact only if it is clearly disposable**

Run: `rm -rf <latest-smoke-check-artifact>`
Expected: the temporary artifact directory is gone

Only do this if the directory was created by the smoke-check topic and contains no useful outputs.

## Task 4: Execute The First Full Paper Run

**Files:**
- Reuse: `config_sir_failure_map.yaml`
- Reuse: `docs/superpowers/specs/2026-04-18-sir-inference-failure-map-design.md`

- [ ] **Step 1: Launch the full run with the paper config**

Run:

```bash
researchclaw run \
  --config config_sir_failure_map.yaml \
  --auto-approve
```

Expected: a new `artifacts/rc-*/` directory is created and the pipeline advances through the full paper workflow

- [ ] **Step 2: If the run stops unexpectedly, resume instead of starting over**

Run:

```bash
researchclaw run \
  --config config_sir_failure_map.yaml \
  --resume \
  --auto-approve
```

Expected: the pipeline resumes from the most recent matching artifact directory

- [ ] **Step 3: Record the final artifact directory**

Run: `ls -dt artifacts/rc-* | head -n 1`
Expected: the final run directory path used for all later inspection

## Task 5: Review The Generated Deliverables Against The Spec

**Files:**
- Reuse: `docs/superpowers/specs/2026-04-18-sir-inference-failure-map-design.md`
- Reuse: `artifacts/<run-id>/deliverables/`
- Create if needed: `notes/sir_failure_map_run.md`

- [ ] **Step 1: Verify the expected final bundle exists**

Run: `find artifacts/<run-id>/deliverables -maxdepth 2 -type f | sort`
Expected: paper draft, LaTeX, bibliography, charts, and review-related outputs are present

- [ ] **Step 2: Read the generated paper draft and compare it to the approved spec**

Check for these points:

- SIR only, not SEIR
- synthetic data only
- observation noise plus under-reporting plus variable reporting delay
- least-squares versus likelihood-based inference
- mild, moderate, rapid regimes
- failure-map-centered narrative

- [ ] **Step 3: Inspect the charts folder for at least one strong failure-map figure**

Run: `find artifacts/<run-id> -path '*charts*' -type f | sort`
Expected: at least one heatmap-style figure or equivalent reliability-region figure is present

- [ ] **Step 4: Write a short run note if anything drifted**

If the run misses the spec, create `notes/sir_failure_map_run.md` with:

- what matched the plan
- what drifted
- whether the drift is a prompt problem, config problem, or pipeline problem

## Task 6: Add Prompt Overrides Only If The First Run Drifted

**Files:**
- Create if needed: `prompts.sir_failure_map.yaml`
- Reuse: `prompts.default.yaml`
- Reuse: `config_sir_failure_map.yaml`

- [ ] **Step 1: Export or copy the default prompt file as a local override starting point**

Run: `cp prompts.default.yaml prompts.sir_failure_map.yaml`
Expected: a local prompt override file exists

- [ ] **Step 2: Narrow only the stages that drifted from the approved paper**

Edit `prompts.sir_failure_map.yaml` only for the stages that need stronger guidance, such as:

- topic framing
- experiment design
- paper outline
- paper drafting

Do not rewrite unrelated prompts.

- [ ] **Step 3: Wire the override file into the local paper config**

Update `config_sir_failure_map.yaml`:

```yaml
prompts:
  custom_file: "prompts.sir_failure_map.yaml"
```

- [ ] **Step 4: Re-run the paper with the same config and compare outputs**

Run:

```bash
researchclaw run \
  --config config_sir_failure_map.yaml \
  --auto-approve
```

Expected: a new artifact directory with outputs closer to the approved failure-map paper design

## Task 7: Package The Final Paper Bundle

**Files:**
- Reuse: `artifacts/<run-id>/deliverables/`
- Create if needed: `notes/sir_failure_map_run.md`

- [ ] **Step 1: Identify the final accepted run directory**

Run: `ls -dt artifacts/rc-* | head -n 3`
Expected: recent candidate run directories listed newest first

- [ ] **Step 2: Verify the deliverables folder is self-contained**

Run: `find artifacts/<accepted-run-id>/deliverables -maxdepth 2 -type f | sort`
Expected: all paper outputs needed for review are visible from this boundary

- [ ] **Step 3: Capture a short handoff note for future reruns**

If not already present, write `notes/sir_failure_map_run.md` with:

- accepted run id
- config file used
- whether prompt overrides were used
- any obvious caveats about the outputs

- [ ] **Step 4: Report the final extraction path**

Final handoff should point to:

```text
artifacts/<accepted-run-id>/deliverables/
```

This path is the cut-out unit for the paper bundle.
