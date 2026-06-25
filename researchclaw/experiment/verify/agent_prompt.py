"""Prompt guidance for the in-container agent on how to call Biomni tools.

Phase 2 ②-c: the agentic flow runs Claude Code inside a container; the bridge
URL is plumbed in via ``BIOMNI_BRIDGE_URL`` and the thin client at
``/usr/local/bin/biomni_client.py`` (mounted read-only) is the only legal way
to reach the host-side Biomni venv. Without this guidance the agent has no
reason to know that path or that contract. Keeping the helper pure means
both :class:`AgenticSandbox` and any future pipeline-stage caller can prepend
the same text without diverging.
"""

from __future__ import annotations

from typing import Sequence


_TEMPLATE = """\
## Biomni tools (host bridge — allowlisted, ledger-bound)

You MUST call the tools below via the bundled CLI client at
``/usr/local/bin/biomni_client.py``. The bridge URL is in the environment
variable ``$BIOMNI_BRIDGE_URL``. Every call appends a sha256-stamped row to
the host's provenance ledger and is later replayed by ARC's verification
gates, so direct subprocess access to a Biomni venv is not permitted and
will be flagged as fabricated.

Invocation shape (run inside the container):

    python /usr/local/bin/biomni_client.py \\
        --bridge-url "$BIOMNI_BRIDGE_URL" \\
        --tool <TOOL_NAME> \\
        --args-json '<JSON_OBJECT>'

The tool's JSON result is printed to stdout.

Allowlisted tools (qualified ``module.tool`` — pass only the short tool name
to ``--tool``):

{tool_list}

If a tool you need is not on this list, STOP and report it; calling
unknown tools is not permitted.
"""


def biomni_tool_guidance(tools: Sequence[str]) -> str:
    """Return the in-container prompt snippet, or "" if the allowlist is empty."""
    if not tools:
        return ""
    listed = "\n".join(f"  - {t}" for t in tools)
    return _TEMPLATE.format(tool_list=listed)


def augment_prompt_with_biomni_guidance(
    prompt: str, tools: Sequence[str]
) -> str:
    """Prepend Biomni tool guidance to ``prompt`` (no-op for empty allowlist)."""
    guidance = biomni_tool_guidance(tools)
    if not guidance:
        return prompt
    return f"{guidance}\n---\n\n{prompt}"
