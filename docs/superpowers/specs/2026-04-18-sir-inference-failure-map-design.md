# SIR Inference Failure Map Design

- Date: 2026-04-18
- Status: Draft for review
- Topic: Observation noise and reporting delays in SIR inference

## Goal

Create a solid but memorable paper using AutoResearchClaw around a simple, simulation-first epidemiology question:

> When does SIR inference break under realistic observation corruption?

The paper should stay methodologically conservative while presenting the main findings through failure maps that show where inference becomes unreliable.

## Paper Direction

- Tone: rigorous, readable, slightly sharper than a textbook comparison paper
- Core contribution: map the reliability boundary of SIR parameter inference and peak prediction under observation noise and reporting delays
- Style of novelty: not a new epidemiological model, but a systematic comparison study with practical decision value
- Deliverable target: one complete AutoResearchClaw run that produces a coherent draft, figures, LaTeX, and review artifacts

## Final Scope

- Model family: SIR only
- Data source: synthetic data only
- Observation corruption:
  - day-to-day random observation noise
  - persistent under-reporting
  - stochastic reporting delays with variable lag
- Epidemic regimes:
  - mild spread
  - moderate spread
  - rapid spread
- Estimation methods:
  - least squares
  - likelihood-based inference
- Main outcomes:
  - parameter estimation error for beta and gamma
  - peak timing prediction error
  - peak magnitude prediction error
- Main presentation style:
  - failure-map-centered paper

## Research Questions

1. How quickly do beta and gamma estimates degrade as observation noise and reporting delay strength increase?
2. Do least squares and likelihood-based inference fail in the same regions of the corruption space?
3. Are some epidemic regimes more robust to corruption than others?
4. Does forecast usefulness collapse gradually or abruptly once corruption exceeds certain thresholds?

## Proposed Claim Shape

The paper should aim to support a claim of the following form:

> SIR inference remains reasonably stable only within a limited observation-quality region, and the failure boundary depends jointly on epidemic speed, under-reporting, and reporting-delay variability. Least squares and likelihood-based inference exhibit distinct breakdown patterns, especially for peak prediction.

This framing keeps the paper conservative while giving it a strong central message.

## Experimental Design

### 1. Ground-Truth Epidemic Generation

Generate clean SIR outbreaks under three epidemic regimes that differ primarily in transmission intensity.

- Mild regime: slower growth, flatter peak
- Moderate regime: balanced reference regime
- Rapid regime: faster growth, sharper peak

The simulation layer should remain separate from the observation layer so that the paper can cleanly distinguish epidemic dynamics from surveillance distortion.

### 2. Observation Corruption Layer

Construct observed incidence from the clean simulated signal through three mechanisms:

- Random observation noise:
  - daily perturbation applied to observed counts
- Persistent under-reporting:
  - a constant reporting fraction below 1
- Variable reporting delay:
  - each observed count is shifted by a sampled delay rather than a fixed lag

This design should mimic realistic surveillance data without requiring real-world datasets.

### 3. Condition Grid

Build a two-dimensional experiment grid:

- Horizontal axis: noise / under-reporting severity
- Vertical axis: reporting-delay severity

For each epidemic regime, run repeated simulations across the grid and fit both inference methods on the corrupted observations.

### 4. Estimation Targets

For each condition:

- estimate beta
- estimate gamma
- reconstruct epidemic trajectory if needed for downstream evaluation
- derive predicted peak day
- derive predicted peak size

## Evaluation Plan

### Primary Metrics

- beta estimation error
- gamma estimation error
- peak timing error
- peak magnitude error

### Secondary Summaries

- mean error per grid cell
- variability across repeated runs
- failure rate above predefined error thresholds

The failure-rate summary is important because it helps distinguish "usually okay but occasionally catastrophic" from "consistently mediocre."

## Figure Plan

The paper should feel anchored by visuals rather than tables alone.

### Main Figures

1. Failure maps for parameter error
2. Failure maps for peak timing error
3. Failure maps for peak magnitude error
4. Side-by-side comparison of least squares vs likelihood-based inference

### Supporting Figures

1. Clean epidemic curves for the three regimes
2. Example corrupted observations under representative corruption settings
3. Reconstructed curves from both inference methods versus ground truth

### Visual Message

The figures should emphasize:

- safe regions where inference remains usable
- transition regions where reliability begins to erode
- danger regions where estimates or forecasts become practically misleading

## Writing Angle

This should not read like a generic benchmarking paper. The narrative should emphasize:

- surveillance corruption is not just a nuisance; it changes whether inferred quantities are interpretable
- average performance alone hides breakdown boundaries
- practical users need regime maps, not only global summary scores

Working title direction:

- When Does SIR Inference Break? Mapping Failure Regions Under Observation Noise and Reporting Delays
- Reliability Boundaries of SIR Inference Under Noisy and Delayed Observation
- Failure Maps for SIR Parameter Inference and Peak Prediction Under Surveillance Corruption

## Output and Run Organization

The user wants final outputs to be easy to cut out and reuse. The run setup should therefore optimize for artifact isolation.

### Workspace Strategy

- Use a dedicated git worktree rather than a fresh clone
- Reuse the repository-level tracked config files already present in the project
- Avoid polluting the main workspace with paper-specific iteration files when possible

### Run Organization

- Use a paper-specific project name in config
- Keep all paper outputs tied to a single named run family
- Treat `artifacts/<run-id>/deliverables/` as the primary export boundary

### Desired Final Bundle

- paper draft
- LaTeX source
- bibliography
- charts and failure maps
- experiment logs
- review artifacts
- any paper-specific config or notes needed for rerun

## Success Criteria

The design is successful if the run produces:

- a coherent paper centered on one clear question
- at least one strong failure-map figure that communicates the result immediately
- a defensible comparison between least squares and likelihood-based inference
- a deliverables folder that is easy to extract and inspect independently

## Out of Scope

To keep the first paper finishable, do not include:

- real epidemiological datasets
- SEIR or richer compartment models
- Bayesian inference in the first version
- intervention-policy optimization
- elaborate correction or deconvolution methods
- broad public-health policy claims

## Risks and Mitigations

- Risk: the study becomes too generic
  - Mitigation: center the paper on reliability boundaries and failure regions
- Risk: too many grid conditions lead to a long or unstable run
  - Mitigation: keep the condition grid moderate in the first pass and expand only if needed
- Risk: the paper drifts into abstract identifiability language
  - Mitigation: keep evaluation tied to parameter and peak-prediction usefulness
- Risk: outputs become hard to separate from the main repo
  - Mitigation: run in a dedicated worktree and keep export focus on `deliverables/`

## Next Transition

After this design is approved, the next step is:

1. create a paper-specific implementation plan
2. create an isolated worktree for the paper run
3. prepare a dedicated config strategy for reproducible outputs
4. execute the first AutoResearchClaw run against this paper design
