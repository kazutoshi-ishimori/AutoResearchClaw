"""Run ResearchClaw Stages 1-8 only (literature review + hypothesis generation)."""

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from researchclaw.adapters import AdapterBundle
from researchclaw.config import RCConfig
from researchclaw.pipeline.executor import execute_stage
from researchclaw.pipeline.runner import _write_checkpoint
from researchclaw.pipeline.stages import STAGE_SEQUENCE, Stage, StageStatus

STOP_AFTER = Stage.HYPOTHESIS_GEN  # Stage 8

config = RCConfig.load("config.yaml", check_paths=False)
run_id = f"ecdysone-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
run_dir = Path("artifacts") / run_id
run_dir.mkdir(parents=True, exist_ok=True)

print(f"=== ResearchClaw Literature Pipeline (Stages 1-{int(STOP_AFTER)}) ===")
print(f"Run ID : {run_id}")
print(f"Output : {run_dir}")
print(f"Topic  : {config.research.topic}")
print()

results = []
for stage in STAGE_SEQUENCE:
    if int(stage) > int(STOP_AFTER):
        break

    prefix = f"[{run_id}] Stage {int(stage):02d}/{int(STOP_AFTER)}"
    print(f"{prefix} {stage.name} — running...")

    t0 = time.monotonic()
    result = execute_stage(
        stage,
        run_dir=run_dir,
        run_id=run_id,
        config=config,
        adapters=AdapterBundle(),
        auto_approve_gates=True,
    )
    elapsed = time.monotonic() - t0

    if result.status == StageStatus.DONE:
        arts = ", ".join(result.artifacts) if result.artifacts else "none"
        print(f"{prefix} {stage.name} — done ({elapsed:.1f}s) → {arts}")
        _write_checkpoint(run_dir, stage, run_id)
    elif result.status == StageStatus.FAILED:
        print(f"{prefix} {stage.name} — FAILED ({elapsed:.1f}s) — {result.error}")
        sys.exit(1)
    else:
        print(f"{prefix} {stage.name} — {result.status.value} ({elapsed:.1f}s)")

    results.append(result)

print()
print(f"=== Literature pipeline complete: {len(results)} stages ===")
print(f"Artifacts: {run_dir}")
