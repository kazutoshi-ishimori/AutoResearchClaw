#!/usr/bin/env python3
"""Container-side thin client for the host bridge (Phase 2, ②).

This script lives in the agentic execution container and is **I/O-compatible**
with :mod:`biomni_tool_runner`: same ``--module / --tool / --args-json`` flags
so the agent prompt stays identical between Phase-1 (host-only direct
subprocess) and Phase-2 (containerised, HTTP bridge). The client POSTs
``{tool, args}`` to ``<bridge_url>/tool`` and emits the tool's JSON return
to stdout. The provenance ledger lives **host-side** (the daemon owns it),
so the ``--ledger`` flag is accepted for back-compat and ignored here — this
keeps the executor's observer outside the container, per the design invariant.

Usage (from inside the agentic container)::

    python biomni_client.py \\
        --bridge-url http://host.docker.internal:8765 \\
        --tool database.query_uniprot \\
        --args-json '{"endpoint": "https://rest.uniprot.org/uniprotkb/Q9BYF1.json?fields=accession,id,sequence"}'

Use the deterministic direct-``endpoint`` form, not the natural-language
``prompt`` form: ``prompt`` routes through an in-tool LLM (needs a langchain
backend in the Biomni venv and is non-deterministic), which breaks the
deterministic-replay gate (layer ③). The qualified ``module.tool`` name and the
bare ``tool`` name are both accepted by the bridge.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from typing import Any, Callable


def _default_http(req: urllib.request.Request, *, timeout: float):
    return urllib.request.urlopen(req, timeout=timeout)


def call_bridge(
    *,
    bridge_url: str,
    tool: str,
    args: dict,
    timeout: float = 120.0,
    http: Callable[..., Any] | None = None,
) -> Any:
    """POST ``{tool, args}`` to ``<bridge_url>/tool`` and return the parsed result.

    Raises :class:`RuntimeError` with the upstream status + body on HTTP errors,
    so the caller (CLI ``main``) can surface a clear failure to the agent.
    """
    sender = http or _default_http
    body = json.dumps({"tool": tool, "args": args}).encode()
    url = bridge_url.rstrip("/") + "/tool"
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with sender(req, timeout=timeout) as resp:
            payload = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace") if exc.fp else ""
        raise RuntimeError(
            f"bridge error {exc.code}: {detail.strip() or exc.reason}"
        ) from exc
    return payload.get("result")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Bridge client (POSTs to host bridge).")
    p.add_argument("--bridge-url", required=True)
    p.add_argument("--tool", required=True)
    p.add_argument("--args-json", default="{}")
    p.add_argument("--timeout", type=float, default=120.0)
    # Back-compat with biomni_tool_runner.py — accepted but ignored.
    p.add_argument("--module", default="", help="ignored (bridge owns the module map)")
    p.add_argument("--ledger", default="", help="ignored (ledger lives host-side)")
    ns = p.parse_args(argv)

    try:
        args = json.loads(ns.args_json)
    except json.JSONDecodeError as exc:
        print(f"ERROR: bad --args-json: {exc}", file=sys.stderr)
        return 2

    try:
        result = call_bridge(
            bridge_url=ns.bridge_url,
            tool=ns.tool,
            args=args,
            timeout=ns.timeout,
        )
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    json.dump(result, sys.stdout, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
