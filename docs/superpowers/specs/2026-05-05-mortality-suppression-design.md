# Mortality Suppression Paper Design

## Goal

Prepare an AutoResearchClaw workspace for a paper that quantifies how public
mortality-data suppression and unreliable-rate rules distort small-area risk
estimation and county rankings under synthetic ground truth.

## Architecture

The workspace contains a focused ARC configuration, a prompt override file, and
a research design note. The ARC run should stop at Stage 9 first so the
experiment protocol can be reviewed before code generation and execution.

## Scope

In scope:

- Synthetic county-by-age mortality benchmark.
- WONDER-style suppression mechanism for 0-9 deaths.
- Unreliable-rate flag for counts below 20.
- Estimation and ranking metrics.
- Paper-writing guardrails that prevent claims about real suppressed cells.

Out of scope:

- Automated county-level CDC WONDER extraction.
- Complementary disclosure or reconstruction of suppressed real cells.
- Production data pipelines.

## Validation

Validate the config with:

```bash
.venv/bin/researchclaw validate --config workspaces/mortality-suppression-paper/config_mortality_suppression.yaml
```

Validate prompt rendering by loading the custom prompt file through
`researchclaw.prompts.PromptManager`.

