"""Phase 2 ②-c: prompt guidance the agentic sandbox injects for Biomni.

When ``BIOMNI_BRIDGE_URL`` is plumbed into the container, the agent still
needs to *know* how to call the bridge. This module covers the pure
helpers that build that guidance — no AgenticSandbox or Docker required.
"""

from __future__ import annotations

from researchclaw.experiment.verify.agent_prompt import (
    augment_prompt_with_biomni_guidance,
    biomni_tool_guidance,
)


def test_empty_allowlist_emits_empty_guidance() -> None:
    """No allowlist → no guidance: prompt must be returned untouched."""
    assert biomni_tool_guidance(()) == ""
    assert augment_prompt_with_biomni_guidance("solve it", ()) == "solve it"


def test_guidance_mentions_cli_and_bridge_env() -> None:
    text = biomni_tool_guidance(("database.query_uniprot",))
    assert "biomni_client.py" in text
    assert "$BIOMNI_BRIDGE_URL" in text
    # The agent must learn the flag shape so it does not improvise.
    assert "--tool" in text
    assert "--args-json" in text


def test_guidance_lists_every_allowlisted_tool() -> None:
    tools = (
        "database.query_uniprot",
        "database.query_kegg",
        "database.query_stringdb",
    )
    text = biomni_tool_guidance(tools)
    for t in tools:
        assert t in text, f"missing tool {t!r} in guidance"


def test_guidance_forbids_unknown_tools_explicitly() -> None:
    """The agent must STOP rather than call tools outside the allowlist."""
    text = biomni_tool_guidance(("database.query_uniprot",))
    lowered = text.lower()
    assert "stop" in lowered or "not permitted" in lowered or "do not" in lowered


def test_augment_prepends_guidance_and_keeps_original_prompt() -> None:
    original = "Run the COVID drug-repurposing experiment."
    augmented = augment_prompt_with_biomni_guidance(
        original, ("database.query_uniprot",)
    )
    assert augmented.endswith(original)
    assert augmented.index("biomni_client.py") < augmented.index(original)


def test_augment_is_idempotent_when_allowlist_empty() -> None:
    """Empty allowlist must NOT prepend separator/header to a clean prompt."""
    p = "hello"
    assert augment_prompt_with_biomni_guidance(p, ()) == p
