"""Tests for Stage 22 evolution-overlay stripping (_strip_evolution_overlay).

The evolution overlay injects "## Lessons from Prior Runs" / "## Learned Skills
from Prior Runs" sections into paper_draft / paper_revision system prompts (see
researchclaw/evolution.py:get_prompt_overlay). Mid-tier models sometimes echo
these into the paper body; Stage 22 strips them deterministically before writing
paper_final.md.
"""

from __future__ import annotations

from researchclaw.pipeline.stage_impls._review_publish import _strip_evolution_overlay


class TestStripEvolutionOverlay:
    def test_strips_lessons_from_prior_runs(self):
        text = (
            "## Introduction\n\nOur work studies X.\n"
            "\n## Lessons from Prior Runs\n"
            "- Run 1 failed because of Y.\n- Run 2 fixed Z.\n"
            "\n## Methods\n\nWe use a transformer.\n"
        )
        cleaned, n = _strip_evolution_overlay(text)
        assert n == 1
        assert "Lessons from Prior Runs" not in cleaned
        assert "Run 1 failed because of Y." not in cleaned
        # Surrounding real sections are preserved
        assert "## Introduction" in cleaned
        assert "## Methods" in cleaned
        assert "We use a transformer." in cleaned

    def test_strips_learned_skills_from_prior_runs(self):
        text = (
            "## Results\n\nAccuracy was 0.9.\n"
            "\n## Learned Skills from Prior Runs\n"
            "- Skill: always set seeds.\n"
            "\n## Discussion\n\nThe results suggest W.\n"
        )
        cleaned, n = _strip_evolution_overlay(text)
        assert n == 1
        assert "Learned Skills from Prior Runs" not in cleaned
        assert "always set seeds" not in cleaned
        assert "## Discussion" in cleaned

    def test_preserves_real_lessons_learned_section(self):
        """A legitimate '## Lessons Learned' paper section must NOT be stripped."""
        text = (
            "## Conclusion\n\nWe conclude X.\n"
            "\n## Lessons Learned\n"
            "This study taught us about generalization.\n"
        )
        cleaned, n = _strip_evolution_overlay(text)
        assert n == 0
        assert cleaned == text
        assert "## Lessons Learned" in cleaned

    def test_strips_section_at_end_of_text(self):
        """Section landing at EOF (no trailing '## ' heading) is handled via \\Z."""
        text = (
            "## Methods\n\nWe trained a model.\n"
            "\n## Lessons from Prior Runs\n"
            "- The final run worked.\n"
        )
        cleaned, n = _strip_evolution_overlay(text)
        assert n == 1
        assert "Lessons from Prior Runs" not in cleaned
        assert "The final run worked." not in cleaned
        assert "We trained a model." in cleaned

    def test_strips_both_overlay_sections(self):
        text = (
            "## Intro\n\nA.\n"
            "\n## Lessons from Prior Runs\n- L1\n"
            "\n## Learned Skills from Prior Runs\n- S1\n"
            "\n## Refs\n\nB.\n"
        )
        cleaned, n = _strip_evolution_overlay(text)
        assert n == 2
        assert "Prior Runs" not in cleaned
        assert "## Intro" in cleaned
        assert "## Refs" in cleaned

    def test_no_overlay_returns_unchanged(self):
        text = "## Introduction\n\nNo overlay here.\n\n## Methods\n\nPlain text.\n"
        cleaned, n = _strip_evolution_overlay(text)
        assert n == 0
        assert cleaned == text
