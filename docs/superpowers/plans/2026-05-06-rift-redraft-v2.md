# RIFT Redraft V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a revised RIFT experiment and manuscript bundle that addresses the review findings.

**Architecture:** Work in a new `redraft-v2` artifact directory. Patch the copied experiment code, run smoke and full experiments, generate figures and tables from revised JSON, then write and compile a new manuscript.

**Tech Stack:** Python standard library, NumPy, Matplotlib, LaTeX/pdfTeX, Poppler tools.

---

### Task 1: Prepare V2 Workspace

**Files:**
- Create: `artifacts/mortality-suppression-20260505-s01/redraft-v2/`
- Create: `docs/superpowers/specs/2026-05-06-rift-redraft-v2-design.md`
- Create: `docs/superpowers/plans/2026-05-06-rift-redraft-v2.md`

- [x] **Step 1: Create isolated artifact directories**

Run:
```bash
mkdir -p artifacts/mortality-suppression-20260505-s01/redraft-v2/code artifacts/mortality-suppression-20260505-s01/redraft-v2/charts artifacts/mortality-suppression-20260505-s01/redraft-v2/reviews docs/superpowers/plans docs/superpowers/specs
```

- [x] **Step 2: Copy baseline deliverable code**

Run:
```bash
cp artifacts/mortality-suppression-20260505-s01/deliverables/code/main.py artifacts/mortality-suppression-20260505-s01/redraft-v2/code/main.py
cp artifacts/mortality-suppression-20260505-s01/deliverables/code/estimators.py artifacts/mortality-suppression-20260505-s01/redraft-v2/code/estimators.py
cp artifacts/mortality-suppression-20260505-s01/deliverables/code/metrics.py artifacts/mortality-suppression-20260505-s01/redraft-v2/code/metrics.py
cp artifacts/mortality-suppression-20260505-s01/deliverables/code/mortality_sim.py artifacts/mortality-suppression-20260505-s01/redraft-v2/code/mortality_sim.py
```

### Task 2: Patch Experiment Code

**Files:**
- Modify: `artifacts/mortality-suppression-20260505-s01/redraft-v2/code/mortality_sim.py`
- Modify: `artifacts/mortality-suppression-20260505-s01/redraft-v2/code/estimators.py`
- Modify: `artifacts/mortality-suppression-20260505-s01/redraft-v2/code/metrics.py`
- Modify: `artifacts/mortality-suppression-20260505-s01/redraft-v2/code/main.py`

- [ ] **Step 1: Add isolated release arms and low-count unavailability**
- [ ] **Step 2: Set age-adjusted top-20 distortion as primary metric**
- [ ] **Step 3: Add compact result writing for large replicate runs**
- [ ] **Step 4: Add focused summaries needed by the manuscript**

### Task 3: Run Experiments

**Files:**
- Create: `artifacts/mortality-suppression-20260505-s01/redraft-v2/code/results.json`

- [ ] **Step 1: Run smoke experiment**

Run:
```bash
ARC_REPLICATES=5 python3 main.py
```

- [ ] **Step 2: Run full experiment**

Run:
```bash
ARC_REPLICATES=1000 python3 main.py
```

### Task 4: Generate Figures And Manuscript

**Files:**
- Create: `artifacts/mortality-suppression-20260505-s01/redraft-v2/code/make_figures_and_tables.py`
- Create: `artifacts/mortality-suppression-20260505-s01/redraft-v2/paper_redraft_v2.md`
- Create: `artifacts/mortality-suppression-20260505-s01/redraft-v2/paper_redraft_v2.tex`

- [ ] **Step 1: Generate figures from revised results**
- [ ] **Step 2: Write revised manuscript text with weaker claims and clearer metrics**
- [ ] **Step 3: Compile PDF**

### Task 5: Validate Deliverables

**Files:**
- Create: `artifacts/mortality-suppression-20260505-s01/redraft-v2/validation_report.json`

- [ ] **Step 1: Check reported numbers against JSON**
- [ ] **Step 2: Check LaTeX warnings and page count**
- [ ] **Step 3: Render PDF pages and inspect figures/tables**
- [ ] **Step 4: Record any residual risks**
