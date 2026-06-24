"""Tests for the container-side bridge client (Phase 2, ②).

The client lives in the agentic container and is I/O-compatible with
``biomni_tool_runner.py``: same ``--module/--tool/--args-json`` flags so the
agent prompt stays identical between Phase-1 (host-only direct subprocess) and
Phase-2 (HTTP bridge). The client POSTs ``{tool, args}`` to the host bridge
and emits the result to stdout. The ledger lives host-side (the daemon owns
it), so ``--ledger`` is accepted for back-compat but ignored.

HTTP transport is injected so the unit tests don't touch the network.
"""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import urllib.error
from pathlib import Path
from unittest.mock import patch

_CLIENT = (
    Path(__file__).resolve().parent.parent
    / "external"
    / "biomni_bridge"
    / "biomni_client.py"
)


def _load_client():
    spec = importlib.util.spec_from_file_location("biomni_client", _CLIENT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class _FakeResponse:
    def __init__(self, body: bytes, status: int = 200) -> None:
        self._body = body
        self.status = status

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        return None


def test_call_bridge_posts_tool_and_args_to_bridge_url() -> None:
    c = _load_client()
    captured: dict = {}

    def fake_http(req, *, timeout):
        captured["url"] = req.full_url
        captured["body"] = req.data
        captured["method"] = req.get_method()
        captured["content_type"] = req.get_header("Content-type")
        return _FakeResponse(json.dumps({"result": {"length": 805}}).encode())

    result = c.call_bridge(
        bridge_url="http://host.docker.internal:8765",
        tool="query_uniprot",
        args={"id": "Q9BYF1"},
        http=fake_http,
    )
    assert result == {"length": 805}
    assert captured["url"] == "http://host.docker.internal:8765/tool"
    assert captured["method"] == "POST"
    assert captured["content_type"] == "application/json"
    assert json.loads(captured["body"]) == {
        "tool": "query_uniprot",
        "args": {"id": "Q9BYF1"},
    }


def test_call_bridge_raises_runtime_error_on_http_error() -> None:
    c = _load_client()

    def fake_http(req, *, timeout):
        raise urllib.error.HTTPError(
            req.full_url, 403, "Forbidden", hdrs={},
            fp=io.BytesIO(json.dumps({"error": "not in allowlist"}).encode()),
        )

    try:
        c.call_bridge(
            bridge_url="http://x:1",
            tool="bad",
            args={},
            http=fake_http,
        )
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "403" in str(exc)
        assert "allowlist" in str(exc)


def test_main_writes_tool_result_to_stdout() -> None:
    c = _load_client()

    def fake_http(req, *, timeout):
        return _FakeResponse(json.dumps({"result": {"hits": [1, 2, 3]}}).encode())

    buf = io.StringIO()
    with patch.object(c, "_default_http", fake_http), patch.object(sys, "stdout", buf):
        rc = c.main(
            [
                "--bridge-url", "http://host:8765",
                "--module", "database",  # back-compat with biomni_tool_runner.py
                "--tool", "query_uniprot",
                "--args-json", json.dumps({"id": "ACE2"}),
            ]
        )
    assert rc == 0
    assert json.loads(buf.getvalue()) == {"hits": [1, 2, 3]}


def test_main_returns_nonzero_on_bridge_error() -> None:
    c = _load_client()

    def fake_http(req, *, timeout):
        raise urllib.error.HTTPError(
            req.full_url, 500, "Internal",
            hdrs={}, fp=io.BytesIO(b'{"error":"boom"}'),
        )

    with patch.object(c, "_default_http", fake_http):
        rc = c.main(
            [
                "--bridge-url", "http://host:8765",
                "--tool", "x",
            ]
        )
    assert rc != 0


def test_main_accepts_and_ignores_ledger_flag_for_back_compat() -> None:
    """Agent prompts written for biomni_tool_runner pass --ledger; client ignores it."""
    c = _load_client()

    def fake_http(req, *, timeout):
        return _FakeResponse(json.dumps({"result": 1}).encode())

    buf = io.StringIO()
    with patch.object(c, "_default_http", fake_http), patch.object(sys, "stdout", buf):
        rc = c.main(
            [
                "--bridge-url", "http://host:8765",
                "--tool", "x",
                "--ledger", "/anywhere/host-side-only.jsonl",
            ]
        )
    assert rc == 0
    assert json.loads(buf.getvalue()) == 1
