"""Host-side adapter that drives the isolated Biomni venv (Phase 1, ③ activation).

``replay_ledger`` (layer ③) needs a ``(tool, args) -> raw_return`` callable.
:class:`BiomniSubprocessRunner` is that callable: it shells out to the Biomni
venv's standalone runner (``server_cmd``) and parses the tool's JSON return.
The subprocess call is injected so command construction and stdout parsing are
unit-testable without a live Biomni environment.
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path

from researchclaw.experiment.verify.biomni_runner import (
    BiomniSubprocessRunner,
    default_server_cmd,
    tool_modules_from_allowlist,
)


def test_tool_modules_split_qualified_allowlist() -> None:
    mapping = tool_modules_from_allowlist(
        ("database.query_uniprot", "pharmacology.retrieve_topk")
    )
    assert mapping == {
        "query_uniprot": "database",
        "retrieve_topk": "pharmacology",
    }


def test_default_server_cmd_points_at_biomni_venv_and_runner() -> None:
    cmd = default_server_cmd()
    assert ".venv/bin/python" in cmd
    assert cmd.rstrip().endswith("biomni_tool_runner.py")


def test_runner_builds_argv_and_parses_stdout(tmp_path: Path) -> None:
    captured: dict = {}

    class _Proc:
        returncode = 0
        stdout = json.dumps({"length": 805})
        stderr = ""

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        return _Proc()

    runner = BiomniSubprocessRunner(
        server_cmd="PY runner.py",
        tool_modules={"query_uniprot": "database"},
        ledger_path=tmp_path / "replay.jsonl",
        run=fake_run,
    )
    result = runner("query_uniprot", {"id": "Q9BYF1"})

    assert result == {"length": 805}
    argv = captured["argv"]
    assert argv[:2] == shlex.split("PY runner.py")
    assert "--module" in argv and argv[argv.index("--module") + 1] == "database"
    assert "--tool" in argv and argv[argv.index("--tool") + 1] == "query_uniprot"
    args_json = argv[argv.index("--args-json") + 1]
    assert json.loads(args_json) == {"id": "Q9BYF1"}


def test_runner_raises_on_nonzero_exit(tmp_path: Path) -> None:
    class _Proc:
        returncode = 2
        stdout = ""
        stderr = "tool 'x' not in allowlist"

    runner = BiomniSubprocessRunner(
        server_cmd="PY runner.py",
        tool_modules={"x": "database"},
        ledger_path=tmp_path / "replay.jsonl",
        run=lambda argv, **kw: _Proc(),
    )
    try:
        runner("x", {})
        assert False, "expected RuntimeError on nonzero exit"
    except RuntimeError as exc:
        assert "allowlist" in str(exc)


def test_runner_raises_on_unknown_tool(tmp_path: Path) -> None:
    runner = BiomniSubprocessRunner(
        server_cmd="PY runner.py",
        tool_modules={"query_uniprot": "database"},
        ledger_path=tmp_path / "replay.jsonl",
        run=lambda argv, **kw: None,
    )
    try:
        runner("unmapped_tool", {})
        assert False, "expected KeyError/ValueError for unmapped tool"
    except (KeyError, ValueError):
        pass
