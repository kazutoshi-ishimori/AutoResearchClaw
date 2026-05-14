---
name: arc-md-to-tex-polish
description: >
  Avoid two recurring defects when converting an AutoResearchClaw paper from
  Markdown to LaTeX (md2tex.py / pandoc / similar): duplicated figure caption
  prefixes ("Figure 3: Figure 3:..."), and missing italic for taxonomic names
  and gene symbols (`*Drosophila*`, `*D. melanogaster*` not rendered as
  \textit{}). Use when running md2tex, paper_v2_build, or any markdown→LaTeX
  step. Triggers on: "md2tex", "pandoc", "figure caption duplicated",
  "Drosophila italic", "neurips_2025.sty", "paper.tex", "Figure N: Figure N".
---

# ARC Markdown → LaTeX Polish

Two specific defects produce visible quality drops in the final PDF and
are mechanically preventable.

## Defect 1 — Figure caption prefix doubling

`\caption{}` automatically inserts "Figure N: " in the rendered PDF.
If the markdown source already starts the caption with "Figure N:", the
output reads **"Figure 3: Figure 3: nvd mRNA detection..."**.

**Rule:** in the markdown source, write the caption *content only*:

```markdown
<!-- ✗ WRONG — produces "Figure 3: Figure 3: ..." -->
![Figure 3: nvd mRNA detection across tissues.](figs/fig3.png)

<!-- ✓ RIGHT -->
![nvd mRNA detection across tissues.](figs/fig3.png)
```

If the markdown was already authored with the prefix, strip it during
conversion:

```python
# in md2tex.py or post-processing
import re
tex = re.sub(
    r'\\caption\{Figure\s+\d+:\s*',
    r'\\caption{',
    tex,
    flags=re.IGNORECASE,
)
```

## Defect 2 — Italic for taxa and gene symbols

Biology / pharmacology venues require italic for:
- Taxonomic names: *Drosophila*, *D. melanogaster*, *Homo sapiens*,
  *Escherichia coli*
- Gene symbols (italic) vs proteins (upright): *nvd* (gene), Nvd (protein);
  *shd* (gene), Shd (protein)
- Latin loan terms: *in vivo*, *in vitro*, *in situ*, *ex vivo*

Most md→tex converters DROP single-asterisk italic in environments where
they expect bold (`**text**`) and silently emit plain text inside captions
and section headings.

**Recipe — explicit safety net:**

```python
# Apply AFTER conversion, before saving paper.tex
TAXA = [
    "Drosophila", "D. melanogaster", "Drosophila melanogaster",
    "Homo sapiens", "H. sapiens", "Mus musculus", "M. musculus",
    "Escherichia coli", "E. coli", "Saccharomyces cerevisiae",
    "Caenorhabditis elegans", "C. elegans",
]
LATIN = ["in vivo", "in vitro", "in situ", "ex vivo", "et al",
         "et al.", "i.e.", "e.g."]

for term in TAXA + LATIN:
    # Only wrap if not already inside \textit{...}
    pattern = r'(?<!\\textit\{)\b' + re.escape(term) + r'\b(?![^\{]*\})'
    tex = re.sub(pattern, r'\\textit{' + term + '}', tex)
```

For gene symbols specifically, build a list from the paper's Methods
section (the gene set is usually 5-20 symbols) and apply the same
treatment.

## Validation checklist

Before declaring the LaTeX build done, grep the output:

```bash
# Find caption prefix doubles
grep -n 'Figure [0-9]\+: Figure [0-9]\+' paper.tex && echo "✗ DOUBLED"

# Find unitalicized taxa (replace TAXA list)
for t in Drosophila "D. melanogaster" "in vivo"; do
    grep -n "[^{]\b$t\b" paper.tex | grep -v textit && echo "✗ UNITALICIZED: $t"
done

# Find unresolved markdown markers that leaked through
grep -nE '\*[a-zA-Z]|\[.*\]\(.*\)' paper.tex && echo "✗ RAW MARKDOWN LEAKED"
```

All three should produce no matches.

## Anti-patterns

- ❌ Hand-fixing each occurrence in `paper.tex` after the fact — must be
  re-done every time md2tex re-runs.
- ❌ Putting `\textit{}` directly in markdown — loses round-trip
  compatibility with markdown previewers and ChatGPT-style review tools.
- ❌ Italicizing protein names (Nvd, Phm) — biological convention is
  upright for proteins, italic only for genes.

## Provenance

Distilled from
`artifacts/ecdysone-20260407-114250/paper_v2_build/2026-04-08-...txt`:
external review (ChatGPT) flagged "Figure 3: Figure 3:" doubling on
multiple figures and *Drosophila* failing to render in italic in Related
Work and Method sections.
