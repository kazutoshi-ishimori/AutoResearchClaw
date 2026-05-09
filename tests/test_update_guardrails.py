from __future__ import annotations

import asyncio
import queue
import sys
import threading
from pathlib import Path

import httpx
import pytest
from fastapi import HTTPException


_REPO_ROOT = Path(__file__).resolve().parents[1]
_METACLAW_ROOT = _REPO_ROOT / ".external" / "MetaClaw"


def _import_metaclaw_api():
    if not _METACLAW_ROOT.is_dir():
        pytest.skip("MetaClaw checkout is not present under .external/MetaClaw")
    sys.path.insert(0, str(_METACLAW_ROOT))
    try:
        from metaclaw.api_server import MetaClawAPIServer
        from metaclaw.config import MetaClawConfig
    except ModuleNotFoundError as exc:
        pytest.skip(f"MetaClaw dependencies are not importable: {exc}")
    return MetaClawAPIServer, MetaClawConfig


def test_metaclaw_openrouter_html_503_is_retried_from_arc_test_suite(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    MetaClawAPIServer, MetaClawConfig = _import_metaclaw_api()
    config = MetaClawConfig(
        mode="skills_only",
        llm_api_base="https://openrouter.ai/api/v1",
        llm_api_key="test-key",
        llm_model_id="openai/gpt-oss-120b",
        record_dir=str(tmp_path / "records"),
    )
    server = MetaClawAPIServer(config, queue.Queue(), threading.Event())

    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    responses = [
        httpx.Response(
            503,
            text="<!DOCTYPE html><html><title>Service Unavailable</title></html>",
            headers={"content-type": "text/html"},
            request=request,
        ),
        httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ]
            },
            request=request,
        ),
    ]

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, url, **kwargs):
            return responses.pop(0)

    async def fake_sleep(delay):
        return None

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setattr("asyncio.sleep", fake_sleep)

    result = asyncio.run(
        server._forward_to_openai_compat(
            {"model": "ignored", "messages": [{"role": "user", "content": "hi"}]}
        )
    )

    assert result["choices"][0]["message"]["content"] == "ok"
    assert responses == []


def test_metaclaw_openrouter_repeated_html_503_fails_explicitly(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    MetaClawAPIServer, MetaClawConfig = _import_metaclaw_api()
    config = MetaClawConfig(
        mode="skills_only",
        llm_api_base="https://openrouter.ai/api/v1",
        llm_api_key="test-key",
        llm_model_id="openai/gpt-oss-120b",
        record_dir=str(tmp_path / "records"),
    )
    server = MetaClawAPIServer(config, queue.Queue(), threading.Event())

    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    responses = [
        httpx.Response(
            503,
            text="<!DOCTYPE html><html><title>Service Unavailable</title></html>",
            headers={"content-type": "text/html"},
            request=request,
        )
        for _ in range(3)
    ]

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, url, **kwargs):
            return responses.pop(0)

    async def fake_sleep(delay):
        return None

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setattr("asyncio.sleep", fake_sleep)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            server._forward_to_openai_compat(
                {"model": "ignored", "messages": [{"role": "user", "content": "hi"}]}
            )
        )

    assert exc_info.value.status_code == 502
    assert "upstream 503" in str(exc_info.value.detail)
    assert "Service Unavailable" in str(exc_info.value.detail)
