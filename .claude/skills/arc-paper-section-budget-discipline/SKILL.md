---
name: arc-paper-section-budget-discipline
description: >
  When drafting or revising an AutoResearchClaw paper (stages 16-19), hit the
  upper bound of each section's word-count budget on the first pass. Use when
  the ARC pipeline reports "severely_short" warnings in stage-17/draft_quality
  .json, or any time you are writing IMRAD sections under explicit word-count
  targets. Triggers on: "ARC paper draft", "severely_short", "section_analysis",
  "draft_quality.json", "EXPAND ... do NOT pad with filler", "paper_draft.md".
---

# ARC Paper Section Budget Discipline

ARC's `quality_assessor` enforces strict per-section word-count budgets:

| Section | Target (words) |
|---|---|
| Abstract | 180–220 |
| Introduction | 800–1000 |
| Related Work | 600–800 |
| Method | 1000–1500 |
| Experiments | 800–1200 |
| Results | 600–800 |
| Discussion | 400–600 |
| Limitations | 200–300 |
| Conclusion | 200–300 |

The default LLM behavior is to write ~30–60% of these targets, triggering
`status: severely_short` on every section and forcing a costly revision pass
(stage 19). Aim for the **upper bound** on the first draft.

## Strategy: 2-pass writing per section

1. **Skeleton pass** — write the headings + topic-sentence per paragraph
   ('what's the point of each paragraph'). Aim for ~30% of target word count.
2. **Substance pass** — for each topic sentence, add:
   - one concrete mechanism / dataset / number
   - one citation (only if a verified one exists in `references.bib`)
   - one limitation, alternative interpretation, or caveat

This nearly always pushes the section into target range without filler.

## What "do NOT pad with filler" actually means

**Filler (banned):**
- Restating the same point with synonyms
- "It is important to note that..." / "Furthermore, it should be mentioned..."
- Generic background that any paper in the field could include
- Restating the abstract or repeating content from another section

**Substance (encouraged):**
- Specific numbers from the experiment (with verified provenance)
- Named methods, datasets, tools (with version numbers)
- Concrete examples illustrating an abstract claim
- Acknowledgement of failure modes / negative results
- Quantitative comparisons to prior work
- Stated boundary conditions ("this holds when X but breaks when Y")

## Section-specific anchors

- **Abstract**: 1 sentence each for context, gap, method, key finding,
  implication. 180+ words = ~5 substantive sentences with numbers.
- **Method**: pseudocode block + 1 paragraph per algorithmic step + 1
  paragraph on validation strategy + 1 paragraph on failure modes.
- **Results**: every claim cites a specific table/figure number.
- **Discussion**: structure as Principal Findings → Relation to Prior Work →
  Mechanism Hypothesis → Caveats → Implications. ~80–120 words each.

## Validation

After drafting, re-run ARC's `draft_quality` check (or compute manually):

```bash
python -c "
import re, sys
text = open('paper_draft.md').read()
sections = re.split(r'\n#+\s+', text)
for s in sections[1:]:
    head, body = s.split('\n', 1) if '\n' in s else (s, '')
    print(f'{head[:30]:30s} {len(body.split())} words')
"
```

Every section should hit at least the lower bound; aiming for upper bound
gives margin for stage-19 revision cuts.

## Anti-patterns

- ❌ Writing to the lower bound — leaves no margin and stage 17 may still
  flag short sections.
- ❌ Padding with hedging ("might", "could be", "may potentially") to bump
  word count — reduces clarity and triggers reviewer suspicion.
- ❌ Writing the Discussion before the Results are finalized — forces
  rewriting both.

## Provenance

Distilled from `artifacts/ecdysone-20260407-114250/stage-17/draft_quality.json`:
9 of 10 sections marked `severely_short`, requiring a full stage-19 revision
that nearly doubled total word count. Avoiding this saves one full pipeline
cycle.
