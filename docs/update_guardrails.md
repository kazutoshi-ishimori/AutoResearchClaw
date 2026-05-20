# Update Guardrails

This project carries local guardrails that must survive AutoResearchClaw and
MetaClaw updates.

## Protected Behaviors

- Stage 16 must not start paper writing when experiment metrics are empty,
  failed, zero-variance, or identical across conditions.
- Stage 22 must fail the pipeline when LaTeX/PDF export fails or when
  `paper.pdf` is not produced.
- Stage 23 must fail citation verification when `references.bib` is missing
  or contains no BibTeX entries.
- Stage 9 must fail single-cell/FCA/h5ad experiment designs that select generic
  image or citation-graph benchmarks, while leaving vision and graph ML topics
  free to use those benchmarks.
- Domain detection must not let generic words such as "visualization" or
  "simulation" override high-confidence single-cell/FCA/h5ad markers.
- Stage 10 must fail sandbox/no-network single-cell code that reads undeclared
  data files (for example `.npz` or `.h5ad`). In sandbox mode it must also
  reject `scanpy`/`anndata` imports and require a runnable
  numpy/pandas/sklearn fallback path.
- Stage 9/10 must not pass heavy single-cell tool requirements such as hdWGCNA,
  MAGIC, ALRA, CellChat, UCell, or scVI into local sandbox execution. These
  ideas must be mapped to numpy/pandas/sklearn approximations before code
  generation and rejected as required imports if they reappear in Stage 10 code.
- Stage 9/10 must keep sandbox tabular synthetic/conformal experiments within
  local CPU limits: `n_samples <= 5000`, `n_estimators <= 50`, total shift
  regimes `<= 4`, and `seeds = 3`. Stage 9 records the applied plan constraint
  in `tabular_cpu_budget_guardrail.json`; Stage 10 rejects generated code that
  hard-codes heavier literal settings.
- Stage 10 must not treat a script that exits 0 with no metrics as successful.
  Generated experiments must have an executable `main.py` entry point and emit
  the configured primary metric (for example `coverage_gap: <float>`). CodeAgent
  must repair zero-output/empty-metric runs instead of accepting them.
- Stage 10 CodeAgent execution must have an explicit wall-clock timeout and
  fail with `code_agent_timeout.json` when exhausted, rather than hanging
  indefinitely during LLM repair/review calls.
- Stage 10 CodeAgent LLM calls must also have a per-call timeout because some
  OpenAI-compatible providers can keep an HTTP read open long enough that the
  outer stage timeout does not interrupt promptly.
- Stage 10 CodeAgent attempt directories must be recreated before each sandbox
  run. Resume/retry paths must not mix stale files from an earlier attempt with
  newly generated files.
- Stage 10 CodeAgent blueprint parsing must accept both closed and unclosed
  fenced YAML blocks. Some providers truncate long blueprint responses after
  the opening `````yaml` fence; this must not force a fallback to trivial
  single-shot code when the partial YAML is still parseable.
- MetaClaw's OpenAI-compatible proxy must retry transient upstream 5xx/HTML
  responses from OpenRouter and fail explicitly if retries are exhausted.

## After Updating ARC or MetaClaw

Run:

```bash
scripts/verify_update_guardrails.sh
```

This script intentionally keeps MetaClaw proxy contract tests in ARC's own
`tests/` directory. Tests inside `.external/MetaClaw/` are useful during local
MetaClaw development, but they can disappear when the external checkout is
replaced.

## If the Guard Fails After a MetaClaw Update

Check `.external/MetaClaw/metaclaw/api_server.py` and restore the behavior of
`MetaClawAPIServer._forward_to_openai_compat`:

- retry upstream HTTP 5xx responses before returning an error;
- treat HTML or other non-JSON success responses as upstream failures, not as
  raw JSON parser crashes;
- pass through upstream 4xx errors without retrying, so auth and quota errors
  remain visible;
- return HTTP 502 with a detail string mentioning the upstream status/body when
  retries are exhausted.

The contract tests in `tests/test_update_guardrails.py` are the source of truth.
