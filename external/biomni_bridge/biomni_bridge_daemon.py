#!/usr/bin/env python3
"""Host-side HTTP bridge to the isolated Biomni venv (Phase 2, ②).

The agentic execution flow runs Claude Code inside a Docker container, but the
Biomni venv (``~/workspace/Biomni/.venv``) and 15 GB data lake live on the
host. This daemon listens on a host port and proxies tool calls into the Biomni
venv via :mod:`biomni_tool_runner`, which writes the provenance ledger.
Container-side agents POST ``{tool, args}`` to ``/tool`` and receive the tool's
JSON return. Allowlist gating is enforced here so a compromised container
cannot escape to arbitrary Biomni tools.

Run host-side (under the ARC venv, not the Biomni venv)::

    .venv/bin/python external/biomni_bridge/biomni_bridge_daemon.py \\
        --ledger run.jsonl \\
        --allowlist database.query_uniprot,database.query_kegg

The daemon stays out of the Biomni venv on purpose: it only knows how to spawn
the runner there as a subprocess (see ``server_cmd``), keeping its own
dependencies to stdlib + the researchclaw helpers used to build
``server_cmd`` / ``tool_modules``.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable


@dataclass
class BridgeConfig:
    """Bridge runtime configuration.

    Parameters
    ----------
    server_cmd:
        Base command launching the Biomni-venv runner — split with
        :func:`shlex.split` before per-call argv is appended.
    tool_modules:
        ``{tool: module}`` map (e.g. ``{"query_uniprot": "database"}``). Acts
        as the allowlist: any ``tool`` not in this map is rejected with 403.
    ledger_path:
        Host-side provenance ledger the runner appends to.
    timeout:
        Per-call subprocess timeout in seconds.
    run:
        Injected subprocess runner (defaults to :func:`subprocess.run`), so
        the daemon stays unit-testable without a live Biomni environment.
    """

    server_cmd: str
    tool_modules: dict[str, str]
    ledger_path: Path
    timeout: float = 120.0
    run: Callable[..., Any] = field(default=subprocess.run)


def process_tool_request(payload: Any, cfg: BridgeConfig) -> tuple[int, dict]:
    """Pure-function ``POST /tool`` handler.

    Returns ``(status_code, response_body)`` so the HTTP layer can stay thin
    and the bulk of the logic is unit-testable without spinning up sockets.
    """
    if not isinstance(payload, dict) or "tool" not in payload:
        return 400, {"error": "body must be a JSON object with a 'tool' key"}
    tool = payload["tool"]
    args = payload.get("args", {})
    if not isinstance(args, dict):
        return 400, {"error": "'args' must be a JSON object"}
    if tool not in cfg.tool_modules:
        return 403, {"error": f"tool '{tool}' not in allowlist"}

    module = cfg.tool_modules[tool]
    argv = shlex.split(cfg.server_cmd) + [
        "--module", module,
        "--tool", tool,
        "--args-json", json.dumps(args),
        "--ledger", str(cfg.ledger_path),
    ]
    try:
        proc = cfg.run(argv, capture_output=True, text=True, timeout=cfg.timeout)
    except subprocess.TimeoutExpired:
        return 504, {"error": "tool execution timed out"}

    if getattr(proc, "returncode", 0) != 0:
        msg = (getattr(proc, "stderr", "") or "").strip() or "tool failed"
        return 500, {"error": msg}

    try:
        result = json.loads(proc.stdout)
    except Exception as exc:  # noqa: BLE001 — surface the parse error verbatim
        return 500, {"error": f"bad tool output: {exc}"}

    return 200, {"result": result}


def _make_handler(cfg: BridgeConfig) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _emit(self, code: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 — stdlib API
            if self.path == "/health":
                self._emit(200, {"ok": True})
            else:
                self._emit(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802 — stdlib API
            if self.path != "/tool":
                self._emit(404, {"error": "not found"})
                return
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b""
            try:
                payload = json.loads(raw) if raw else {}
            except json.JSONDecodeError as exc:
                self._emit(400, {"error": f"invalid JSON body: {exc}"})
                return
            code, body = process_tool_request(payload, cfg)
            self._emit(code, body)

        def log_message(self, *args, **kwargs) -> None:  # silence test runs
            return

    return Handler


def serve(cfg: BridgeConfig, *, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    """Build (but do not start) a threaded HTTP server bound to ``host:port``.

    Pass ``port=0`` to let the OS pick an ephemeral port; the caller can read
    the assigned port from ``server.server_address[1]``. Caller is responsible
    for ``server.serve_forever()`` and eventual ``server.shutdown()``.
    """
    return ThreadingHTTPServer((host, port), _make_handler(cfg))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Host-side bridge to the Biomni venv.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--ledger", required=True)
    p.add_argument(
        "--allowlist",
        required=True,
        help="comma-separated qualified tool names (e.g. database.query_uniprot)",
    )
    p.add_argument(
        "--server-cmd",
        default="",
        help="override Biomni runner command (default: ARC's default_server_cmd)",
    )
    p.add_argument("--timeout", type=float, default=120.0)
    ns = p.parse_args(argv)

    # Lazy-import so the daemon's unit tests don't need a researchclaw checkout.
    from researchclaw.experiment.verify.biomni_runner import (
        default_server_cmd,
        tool_modules_from_allowlist,
    )

    allow = tuple(t.strip() for t in ns.allowlist.split(",") if t.strip())
    cfg = BridgeConfig(
        server_cmd=ns.server_cmd or default_server_cmd(),
        tool_modules=tool_modules_from_allowlist(allow),
        ledger_path=Path(ns.ledger),
        timeout=ns.timeout,
    )
    server = serve(cfg, host=ns.host, port=ns.port)
    print(f"biomni bridge listening on http://{ns.host}:{server.server_address[1]}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
