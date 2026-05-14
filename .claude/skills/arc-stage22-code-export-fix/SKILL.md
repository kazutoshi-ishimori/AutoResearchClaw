---
name: arc-stage22-code-export-fix
description: >
  Prevent AutoResearchClaw stage 22 (EXPORT_PUBLISH) from failing with
  "Missing output directory: code/" by materializing the experiment source
  code into `deliverables/code/` before stage 22 runs. Use whenever ARC
  pipeline reaches stage 22, or when stage 22 has already failed with that
  exact error and the run needs to be resumed. Triggers on: "EXPORT_PUBLISH",
  "stage 22", "Missing output directory: code", "deliverables/code missing",
  "pipeline failed at export".
---

# ARC Stage 22: Materialize `code/` Before Export

`stage-22/export_publish` enforces a `code/` directory in `deliverables/`
even when the experiment ran in `sandbox`/`docker` mode where the executed
script lives in a per-stage workspace, not at the project root. The pipeline
then marks `final_status: failed` even though the paper PDF compiled fine —
a one-line fix recovers the run.

## Diagnosis

Check whether the failure is *only* the export step:

```bash
RUN_DIR=artifacts/<run-id>
test -f "$RUN_DIR/deliverables/paper.pdf"  && echo "✓ paper produced"
test -d "$RUN_DIR/deliverables/code/"      || echo "✗ code/ missing — this is the bug"
cat "$RUN_DIR/evolution/lessons.jsonl" | grep -c "Missing output directory: code"
```

If paper.pdf exists and the lesson appears, this skill applies.

## Fix recipe — materialize `code/` post-hoc

```bash
RUN_DIR=artifacts/<run-id>
mkdir -p "$RUN_DIR/deliverables/code"

# 1. Pull the executed experiment script (stage 12 or 13)
SCRIPT=$(find "$RUN_DIR/stage-12" "$RUN_DIR/stage-13" \
  -name "*.py" -size +0 2>/dev/null | head -1)
[ -n "$SCRIPT" ] && cp "$SCRIPT" "$RUN_DIR/deliverables/code/experiment.py"

# 2. Pull requirements (look for any requirements.txt the agent created)
REQ=$(find "$RUN_DIR" -name "requirements.txt" 2>/dev/null | head -1)
if [ -n "$REQ" ]; then
    cp "$REQ" "$RUN_DIR/deliverables/code/requirements.txt"
else
    # Synthesize from the script's imports
    python -c "
import ast, sys
tree = ast.parse(open('$RUN_DIR/deliverables/code/experiment.py').read())
mods = set()
for n in ast.walk(tree):
    if isinstance(n, ast.Import):
        mods.update(a.name.split('.')[0] for a in n.names)
    elif isinstance(n, ast.ImportFrom) and n.module:
        mods.add(n.module.split('.')[0])
stdlib = {'os','sys','json','pathlib','re','math','itertools','collections','typing','functools','dataclasses'}
print('\n'.join(sorted(mods - stdlib)))
" > "$RUN_DIR/deliverables/code/requirements.txt"
fi

# 3. Add a brief README so the export validator sees a populated dir
cat > "$RUN_DIR/deliverables/code/README.md" <<'EOF'
# Experiment Source

`experiment.py` is the executed script from this run. Install with:

    pip install -r requirements.txt

EOF
```

## Resume the pipeline

```bash
cd ~/workspace/AutoResearchClaw
researchclaw run --from-stage 22 --output artifacts/<run-id>
```

Stage 22 should now find the `code/` artifact and complete successfully.

## Long-term fix (not in this skill's scope)

The pipeline should either:
- Auto-materialize `code/` at the end of stage 13, OR
- Allow stage 22 to proceed with a warning when `code/` is absent.

File this as an ARC enhancement separately. This skill is the workaround
for runs that are *already* in the failed state.

## Anti-patterns

- ❌ Re-running the entire pipeline from stage 1 to fix this — wastes
  hours when the actual failure is one missing directory.
- ❌ Manually editing `pipeline_summary.json` to mark stage 22 as
  succeeded — leaves the deliverable bundle incomplete.
- ❌ Writing a fake `code/experiment.py` with placeholder content — breaks
  the integrity of the deliverable.

## Provenance

Distilled from `artifacts/ecdysone-20260407-114250/`:
- `paper.pdf` was produced successfully (264 KB, 9 pages, 0 warnings)
- `evolution/lessons.jsonl` recorded the single error
- `pipeline_summary.json` showed `final_status: "failed"` purely due to
  this one missing artifact.
