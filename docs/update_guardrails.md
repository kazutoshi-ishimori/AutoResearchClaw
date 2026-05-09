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
