"""Tests for the host-side Biomni HTTP bridge daemon (Phase 2, ②).

The daemon listens on a host port, accepts ``POST /tool {tool, args}`` from
agentic-mode container clients (or a curl smoke test), checks the tool against
its allowlist, and shells out to the Biomni venv's standalone tool runner. The
ledger is written host-side by the runner — keeping the executor's observer
(layer ①) on the host and out of the container, per the design invariant.

Unit tests exercise the pure ``process_tool_request`` function (subprocess is
injected). Integration tests boot a real ``ThreadingHTTPServer`` on an
ephemeral port and round-trip JSON over real HTTP.
"""

from __future__ import annotations

import importlib.util
import json
import shlex
import sys
import threading
import urllib.request
from pathlib import Path

_DAEMON = (
    Path(__file__).resolve().parent.parent
    / "external"
    / "biomni_bridge"
    / "biomni_bridge_daemon.py"
)


def _load_daemon():
    spec = importlib.util.spec_from_file_location("biomni_bridge_daemon", _DAEMON)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so @dataclass can resolve forward-referenced types
    # via ``sys.modules[cls.__module__].__dict__`` (otherwise NoneType.__dict__).
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _runner_ok(result_obj):
    class _Proc:
        returncode = 0
        stdout = json.dumps(result_obj)
        stderr = ""

    return lambda argv, **kw: _Proc()


# -----------------------------------------------------------------------------
# Unit: pure process_tool_request
# -----------------------------------------------------------------------------

def test_process_allowlisted_tool_returns_200(tmp_path: Path) -> None:
    d = _load_daemon()
    cfg = d.BridgeConfig(
        server_cmd="PY runner.py",
        tool_modules={"query_uniprot": "database"},
        ledger_path=tmp_path / "p.jsonl",
        run=_runner_ok({"length": 805}),
    )
    code, body = d.process_tool_request(
        {"tool": "query_uniprot", "args": {"id": "Q9BYF1"}}, cfg
    )
    assert code == 200
    assert body == {"result": {"length": 805}}


def test_process_unknown_tool_returns_403(tmp_path: Path) -> None:
    d = _load_daemon()
    cfg = d.BridgeConfig(
        server_cmd="PY runner.py",
        tool_modules={"query_uniprot": "database"},
        ledger_path=tmp_path / "p.jsonl",
        run=_runner_ok({}),
    )
    code, body = d.process_tool_request({"tool": "nope"}, cfg)
    assert code == 403
    assert "allowlist" in body["error"].lower()


def test_process_missing_tool_key_returns_400(tmp_path: Path) -> None:
    d = _load_daemon()
    cfg = d.BridgeConfig(
        server_cmd="PY",
        tool_modules={},
        ledger_path=tmp_path / "p.jsonl",
        run=lambda argv, **kw: None,
    )
    code, _ = d.process_tool_request({"args": {}}, cfg)
    assert code == 400


def test_process_subprocess_failure_returns_500(tmp_path: Path) -> None:
    d = _load_daemon()

    class _Proc:
        returncode = 2
        stdout = ""
        stderr = "boom"

    cfg = d.BridgeConfig(
        server_cmd="PY",
        tool_modules={"x": "database"},
        ledger_path=tmp_path / "p.jsonl",
        run=lambda argv, **kw: _Proc(),
    )
    code, body = d.process_tool_request({"tool": "x"}, cfg)
    assert code == 500
    assert "boom" in body["error"]


def test_process_builds_argv_with_module_tool_args_ledger(tmp_path: Path) -> None:
    d = _load_daemon()
    captured: dict = {}

    class _Proc:
        returncode = 0
        stdout = json.dumps({"ok": 1})
        stderr = ""

    def fake_run(argv, **kw):
        captured["argv"] = argv
        return _Proc()

    ledger = tmp_path / "p.jsonl"
    cfg = d.BridgeConfig(
        server_cmd="PY runner.py",
        tool_modules={"query_uniprot": "database"},
        ledger_path=ledger,
        run=fake_run,
    )
    d.process_tool_request({"tool": "query_uniprot", "args": {"id": "ACE2"}}, cfg)
    argv = captured["argv"]
    assert argv[:2] == shlex.split("PY runner.py")
    i = argv.index("--module")
    assert argv[i + 1] == "database"
    i = argv.index("--tool")
    assert argv[i + 1] == "query_uniprot"
    i = argv.index("--args-json")
    assert json.loads(argv[i + 1]) == {"id": "ACE2"}
    i = argv.index("--ledger")
    assert argv[i + 1] == str(ledger)


# -----------------------------------------------------------------------------
# Integration: real HTTP server on ephemeral port
# -----------------------------------------------------------------------------

def _spin_up(server) -> threading.Thread:
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return t


def test_http_server_serves_health_endpoint(tmp_path: Path) -> None:
    d = _load_daemon()
    cfg = d.BridgeConfig(
        server_cmd="PY",
        tool_modules={},
        ledger_path=tmp_path / "p.jsonl",
        run=lambda argv, **kw: None,
    )
    server = d.serve(cfg, host="127.0.0.1", port=0)
    port = server.server_address[1]
    _spin_up(server)
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/health", timeout=2
        ) as resp:
            assert resp.status == 200
            assert json.loads(resp.read()) == {"ok": True}
    finally:
        server.shutdown()


def test_http_server_routes_post_tool_through_handler(tmp_path: Path) -> None:
    d = _load_daemon()
    cfg = d.BridgeConfig(
        server_cmd="PY",
        tool_modules={"query_uniprot": "database"},
        ledger_path=tmp_path / "p.jsonl",
        run=_runner_ok({"length": 805}),
    )
    server = d.serve(cfg, host="127.0.0.1", port=0)
    port = server.server_address[1]
    _spin_up(server)
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/tool",
            data=json.dumps(
                {"tool": "query_uniprot", "args": {"id": "Q9BYF1"}}
            ).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=2) as resp:
            assert resp.status == 200
            assert json.loads(resp.read()) == {"result": {"length": 805}}
    finally:
        server.shutdown()


def test_http_server_rejects_unallowlisted_with_403(tmp_path: Path) -> None:
    d = _load_daemon()
    cfg = d.BridgeConfig(
        server_cmd="PY",
        tool_modules={"query_uniprot": "database"},
        ledger_path=tmp_path / "p.jsonl",
        run=_runner_ok({}),
    )
    server = d.serve(cfg, host="127.0.0.1", port=0)
    port = server.server_address[1]
    _spin_up(server)
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/tool",
            data=json.dumps({"tool": "evil"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urllib.request.urlopen(req, timeout=2)
            assert False, "expected HTTPError 403"
        except urllib.error.HTTPError as e:
            assert e.code == 403
            body = json.loads(e.read())
            assert "allowlist" in body["error"].lower()
    finally:
        server.shutdown()
