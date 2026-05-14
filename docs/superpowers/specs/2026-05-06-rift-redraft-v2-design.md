# RIFT Redraft V2 Design

## Goal

Produce a revised RIFT manuscript that addresses the Codex and Gemini reviews:
broken figures, ambiguous primary metric, insufficient mechanism isolation, low
replicate count, and over-strong claims about CDC WONDER fidelity.

## Approach

The original ARC artifacts remain untouched. A separate working bundle lives in
`artifacts/mortality-suppression-20260505-s01/redraft-v2/`.

The revised experiment uses documentation-inspired synthetic county tables, but
weakens external-validity language from "policy-faithful" to "stress-test".
The primary endpoint becomes a single operational target:
age-adjusted top-20 ranking distortion. A composite ranking distortion remains
available as a secondary metric.

## Experiment Changes

- Increase the main run from 50 to 1,000 synthetic replicates per scenario and
  release condition.
- Add release arms that isolate components:
  - `full_data`
  - `suppression_only`
  - `low_count_unavailable_only`
  - `denominator_only`
  - `suppression_plus_low_count_unavailable`
  - `suppression_plus_denominator`
  - `all_combined`
- Represent low-count unreliability as a conservative analyst-facing
  unavailability rule, not as proof of CDC behavior.
- Store compact scalar records for large runs so result files remain manageable.

## Manuscript Changes

- Rebuild figures from the revised result JSON.
- Remove stale ARC-generated figures.
- Insert the practical-method comparison figure into the PDF.
- Replace broad causal language with mechanism-specific and associational
  claims.
- Add replicate count and arm-isolation details to Methods and Limitations.
- Keep the privacy claim narrow: no real suppressed cells are reconstructed.

## Validation

- Run a small smoke experiment before the 1,000-replicate run.
- Verify all reported numbers are read from the revised result JSON.
- Compile LaTeX to PDF.
- Render PDF pages and inspect figures/tables for missing images, overlap, and
  overfull tables.
- Check citation and numeric integrity using local deterministic checks where
  available.
