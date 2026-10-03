from __future__ import annotations

import httpx
import pytest

from rag_agent.api import create_app
from rag_agent.config import Settings
from rag_agent.models import (
    AvailableModel,
    ModelSelectionError,
    discover_unsloth_models,
    select_unsloth_model,
)


class _FakeAsyncClient:
    response: httpx.Response | None = None
    error: Exception | None = None

    def __init__(self, **_kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def get(self, url, headers):
        if self.error:
            raise self.error
        assert headers["Authorization"].startswith("Bearer ")
        assert self.response is not None
        return self.response


def _unsloth_settings(tmp_path, **overrides):
    values = {
        "data_dir": tmp_path,
        "database_path": tmp_path / "rag.db",
        "agent_provider": "unsloth",
        "unsloth_base_url": "http://unsloth.test/v1",
        "unsloth_api_key": "test-key",
        "unsloth_model_id": "old/model",
        "embedding_provider": "deterministic",
        "run_timeout_seconds": 5,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


@pytest.mark.asyncio
async def test_discovery_filters_downloaded_and_non_chat_models(monkeypatch, tmp_path):
    request = httpx.Request("GET", "http://unsloth.test/v1/models")
    _FakeAsyncClient.response = httpx.Response(
        200,
        request=request,
        json={
            "data": [
                {"id": "new/chat", "loaded": True},
                {"id": "old/chat", "loaded": False},
                {"id": "speech", "loaded": True, "task": "automatic-speech-recognition"},
                {"id": "new/chat", "loaded": True, "display_name": "duplicate"},
            ]
        },
    )
    _FakeAsyncClient.error = None
    monkeypatch.setattr("rag_agent.models.httpx.AsyncClient", _FakeAsyncClient)

    models = await discover_unsloth_models(_unsloth_settings(tmp_path))

    assert models == [AvailableModel("new/chat", "new/chat")]


@pytest.mark.asyncio
async def test_discovery_refuses_catalog_without_loaded_state(monkeypatch, tmp_path):
    request = httpx.Request("GET", "http://unsloth.test/v1/models")
    _FakeAsyncClient.response = httpx.Response(
        200,
        request=request,
        json={"data": [{"id": "ambiguous/model", "object": "model"}]},
    )
    _FakeAsyncClient.error = None
    monkeypatch.setattr("rag_agent.models.httpx.AsyncClient", _FakeAsyncClient)

    with pytest.raises(ModelSelectionError) as caught:
        await discover_unsloth_models(_unsloth_settings(tmp_path))

    assert caught.value.code == "UNSLOTH_LOADED_STATE_UNSUPPORTED"


@pytest.mark.parametrize(
    ("models", "configured", "expected_id", "expected_reason", "expected_error"),
    [
        ([], None, None, None, "UNSLOTH_NO_LOADED_MODEL"),
        (
            [AvailableModel("first/model", "First"), AvailableModel("preferred/model", "Preferred")],
            "preferred/model",
            "preferred/model",
            "configured_preference",
            None,
        ),
        (
            [AvailableModel("first/model", "First"), AvailableModel("second/model", "Second")],
            None,
            None,
            None,
            "UNSLOTH_MODEL_SELECTION_REQUIRED",
        ),
    ],
)
def test_automatic_model_selection_boundaries(
    models, configured, expected_id, expected_reason, expected_error
):
    if expected_error:
        with pytest.raises(ModelSelectionError) as caught:
            select_unsloth_model(models, None, configured)
        assert caught.value.code == expected_error
        return

    selected, reason = select_unsloth_model(models, None, configured)
    assert selected.id == expected_id
    assert reason == expected_reason


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "error", "code"),
    [
        (httpx.Response(401, request=httpx.Request("GET", "http://x")), None, "UNSLOTH_AUTH_FAILED"),
        (None, httpx.ReadTimeout("late"), "UNSLOTH_DISCOVERY_TIMEOUT"),
    ],
)
async def test_discovery_reports_safe_failures(monkeypatch, tmp_path, response, error, code):
    _FakeAsyncClient.response = response
    _FakeAsyncClient.error = error
    monkeypatch.setattr("rag_agent.models.httpx.AsyncClient", _FakeAsyncClient)

    with pytest.raises(ModelSelectionError) as caught:
        await discover_unsloth_models(_unsloth_settings(tmp_path))
    assert caught.value.code == code
    assert "test-key" not in str(caught.value)


def _ready_session(client):
    document = client.post(
        "/documents",
        files={"file": ("source.md", b"# Source\n\nA grounded source sentence.", "text/markdown")},
    ).json()
    session = client.post(
        "/sessions",
        json={"document_id": document["document_id"], "version_id": document["version_id"]},
    ).json()
    return session["session_id"]


def test_each_run_resolves_current_loaded_model_and_records_it(monkeypatch, tmp_path):
    loaded = [[AvailableModel("new/model", "New")], [AvailableModel("next/model", "Next")]]

    async def fake_discovery(_settings):
        return loaded.pop(0)

    seen = []

    async def fake_run_adk(self, run, budget, resolved_model_id=None):
        seen.append(resolved_model_id)
        return {"answer": "", "evidence_ids": [], "stop_reason": "insufficient_evidence"}

    monkeypatch.setattr("rag_agent.models.discover_unsloth_models", fake_discovery)
    monkeypatch.setattr("rag_agent.agent.AgentService._run_adk", fake_run_adk)
    app = create_app(_unsloth_settings(tmp_path))

    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        session_id = _ready_session(client)
        runs = []
        for _ in range(2):
            queued = client.post(
                f"/sessions/{session_id}/messages",
                json={"message": "Analyze the source", "request_type": "question"},
            ).json()
            runs.append(client.get(f"/runs/{queued['run_id']}").json())

    assert seen == ["new/model", "next/model"]
    assert [run["model"] for run in runs] == seen
    for run in runs:
        events = app.state.trace.list(run["run_id"])
        selected = next(event for event in events if event["event_type"] == "model_selected")
        assert selected["payload"]["model_id"] == run["model"]


def test_explicit_unloaded_model_fails_before_inference(monkeypatch, tmp_path):
    async def fake_discovery(_settings):
        return [AvailableModel("loaded/model", "Loaded")]

    calls = 0

    async def fake_run_adk(self, run, budget, resolved_model_id=None):
        nonlocal calls
        calls += 1
        return {}

    monkeypatch.setattr("rag_agent.models.discover_unsloth_models", fake_discovery)
    monkeypatch.setattr("rag_agent.agent.AgentService._run_adk", fake_run_adk)
    app = create_app(_unsloth_settings(tmp_path))

    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        session_id = _ready_session(client)
        queued = client.post(
            f"/sessions/{session_id}/messages",
            json={"message": "Analyze", "model_id": "stale/model"},
        ).json()
        run = client.get(f"/runs/{queued['run_id']}").json()

    assert calls == 0
    assert run["status"] == "failed"
    assert run["error_code"] == "UNSLOTH_MODEL_NOT_LOADED"
    assert run["model"] == "stale/model"


@pytest.mark.parametrize(
    ("provider", "model_id", "expected_configured"),
    [
        ("demo", "demo-extractive", None),
        ("gemini", "gemini-test-model", "gemini-test-model"),
    ],
)
def test_models_endpoint_stays_local_for_non_unsloth(
    monkeypatch, tmp_path, provider, model_id, expected_configured
):
    async def forbidden(_settings):
        raise AssertionError("non-Unsloth model listing must stay local")

    monkeypatch.setattr("rag_agent.api.discover_unsloth_models", forbidden)
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path / provider,
        database_path=tmp_path / provider / "rag.db",
        agent_provider=provider,
        model=model_id,
        google_api_key="test-key" if provider == "gemini" else None,
        embedding_provider="deterministic",
    )
    app = create_app(settings)

    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        response = client.get("/models")
        assert response.status_code == 200
        body = response.json()

    assert body["provider"] == provider
    assert body["models"] == [{"id": model_id, "label": model_id}]
    assert body["configured_model_id"] == expected_configured
    assert "api_key" not in str(body).casefold()
