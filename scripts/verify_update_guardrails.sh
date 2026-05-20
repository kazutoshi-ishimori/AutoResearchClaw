#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python3}"
fi

"$PYTHON_BIN" -m pytest \
  tests/test_domain_detector.py \
  tests/test_experiment_design_guardrail.py \
  tests/test_stage10_data_contract.py \
  tests/test_code_agent.py \
  tests/test_pipeline_guardrails.py \
  tests/test_update_guardrails.py \
  tests/test_rc_e2e_regression.py \
  tests/test_rc_citation_verify.py \
  tests/test_compiler.py \
  -q
