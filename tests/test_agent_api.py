from __future__ import annotations

import json
from types import SimpleNamespace

from rag_agent.agent import Budget, _adk_event_payload
from rag_agent.validation import validate_agent_payload, validate_evidence_ids


def _run(client, session_id, message, request_type="question"):
    response = client.post(f"/sessions/{session_id}/messages", json={"message": message, "request_type": request_type})
    assert response.status_code == 202
    run_id = response.json()["run_id"]
    run = client.get(f"/runs/{run_id}").json()
    return run_id, run


def test_question_has_server_registered_citation_and_trace(client, session):
    run_id, run = _run(client, session["session_id"], "How long must approval records be retained?")
    assert run["status"] == "completed"
    assert "seven years" in run["answer"]
    assert run["citations"]
    evidence = client.get(f"/evidence/{run['citations'][0]['evidence_id']}").json()
    assert run["citations"][0]["excerpt"] in evidence["raw_text"]
    events = client.get(f"/runs/{run_id}/trace").json()["events"]
    assert {event["event_type"] for event in events} >= {"run_started", "tool_result", "citation_validation", "final_output"}
    assert client.get(f"/runs/{run_id}/trace?format=jsonl").headers["content-type"].startswith("application/x-ndjson")


def test_unanswerable_question_refuses_and_rewrites(client, session):
    run_id, run = _run(client, session["session_id"], "What is the mandatory satellite launch date?")
    assert run["status"] == "completed"
    assert not run["citations"]
    assert "未找到足够证据" in run["answer"]
    events = client.get(f"/runs/{run_id}/trace").json()["events"]
    assert any(event["event_type"] == "query_rewrite" for event in events)
    assert run["search_rounds"] <= 2
    assert run["tool_calls"] <= 6


def test_report_matches_schema_shape(client, session):
    _, run = _run(client, session["session_id"], "Generate a report about records access.", "report")
    assert run["status"] == "completed"
    report = run["report"]
    assert report["run_id"] == run["run_id"]
    assert report["document_version"] == session["version_id"]
    assert report["findings"]
    assert all(item["evidence_ids"] for item in report["findings"])


def test_report_without_evidence_removes_factual_content(client, session):
    _, run = _run(client, session["session_id"], "Generate a report about the satellite launch date.", "report")
    assert run["status"] == "completed"
    assert run["report"]["citations"] == []
    assert run["report"]["findings"] == []
    assert run["report"]["recommendations"] == []
    assert "未找到足够" in run["report"]["executive_summary"]


def test_fabricated_evidence_is_rejected(app, session):
    service = app.state.agent_service
    run = service.create_run(session["session_id"], "test", "question")
    citations, invalid = validate_evidence_ids(app.state.db, run["run_id"], session["version_id"], ["ev_fabricated"])
    assert citations == []
    assert invalid == ["ev_fabricated"]


def test_trace_redacts_secrets(app, session):
    service = app.state.agent_service
    run = service.create_run(session["session_id"], "test", "question")
    app.state.trace.record(run["run_id"], "test", {"api_key": "sk-super-secret-value", "content": "Bearer sk-leaked-value"})
    event = app.state.trace.list(run["run_id"])[0]
    dumped = json.dumps(event)
    assert "super-secret" not in dumped
    assert "leaked-value" not in dumped


def test_adk_event_serializes_tool_traffic_without_binary_data():
    call = SimpleNamespace(id="call-1", name="search_document", args={"query": "retention"})
    response = SimpleNamespace(id="call-1", name="search_document", response={"results": ["chunk"]})
    parts = [
        SimpleNamespace(text=None, function_call=call, function_response=None, thought=None, inline_data=None),
        SimpleNamespace(text=None, function_call=None, function_response=response, thought=None, inline_data=None),
        SimpleNamespace(text="final text", function_call=None, function_response=None, thought=False, inline_data=None),
    ]
    content = SimpleNamespace(role="model", parts=parts)

    class Event(SimpleNamespace):
        def is_final_response(self):
            return True

    payload = _adk_event_payload(Event(id="evt", invocation_id="inv", author="agent", content=content, partial=False, usage_metadata=None, error_code=None, error_message=None))
    assert payload["content"]["parts"][0]["function_call"]["args"] == {"query": "retention"}
    assert payload["content"]["parts"][1]["function_response"]["response"] == {"results": ["chunk"]}
    assert payload["content"]["parts"][2]["text"] == "final text"


def test_question_without_valid_citations_never_returns_model_claim(app, session):
    service = app.state.agent_service
    run = service.create_run(session["session_id"], "What is JEV?", "question")
    result = service._finalize_payload(
        run,
        {
            "answer": "Unsupported model claim",
            "evidence_ids": [],
            "stop_reason": "insufficient_evidence",
        },
        {"citations": [], "invalid_evidence_ids": [], "findings": []},
    )
    assert result["citations"] == []
    assert result["answer"] == "文档中未找到足够证据来回答该问题。"


def test_adk_search_limit_is_nonfatal_after_registered_evidence(app, session):
    """Mirror bootstrap + three model searches + reads + one excess search."""
    service = app.state.agent_service
    run = service.create_run(session["session_id"], "How should records access be implemented?", "question")
    budget = Budget(max_rounds=4, max_tool_calls=12)

    # Server-side bootstrap consumes one search and registers initial evidence.
    reserved, code = budget.reserve_search_tool()
    assert (reserved, code) == (True, None)
    bootstrap = service.retriever.search(
        session["version_id"], run["user_message"], "hybrid", 5, run_id=run["run_id"]
    )
    assert bootstrap["results"]
    assert budget.reserve_tool() == (True, None)
    initial_evidence = service.retriever.read_passage(
        session["version_id"], bootstrap["results"][0]["chunk_id"], run["run_id"], 0, 0
    )

    search_document, read_passage, _ = service._adk_tools(
        session["version_id"], run["run_id"], budget
    )
    searches = [search_document(query) for query in ("records access", "manager approval", "authentication")]
    assert all(result["ok"] for result in searches)
    assert searches[-1]["budget"]["searches_remaining"] == 0

    # Four reads match the observed local-model sequence and leave tool-call
    # capacity available even though search capacity is exhausted.
    candidate_chunk = searches[0]["results"][0]["chunk_id"]
    evidence = [read_passage(candidate_chunk, 0, 0) for _ in range(4)]
    assert all(result["ok"] for result in evidence)

    denied = search_document("one more broad search")
    assert denied["ok"] is False
    assert denied["error"]["code"] == "SEARCH_BUDGET_EXHAUSTED"
    assert denied["budget"] == {
        "searches_used": 4,
        "searches_remaining": 0,
        "tool_calls_used": 9,
        "tool_calls_remaining": 3,
        "denied_calls": 1,
    }
    assert "Produce the final JSON now" in denied["next_action"]

    # The already registered evidence remains valid and can still complete an
    # answer; citation validation is unchanged by the graceful limit response.
    checked = validate_agent_payload(
        app.state.db,
        run["run_id"],
        session["version_id"],
        {"answer": "Use controlled access.", "evidence_ids": [initial_evidence["evidence_id"]]},
    )
    assert len(checked["citations"]) == 1
    events = app.state.trace.list(run["run_id"])
    limit_event = next(event for event in events if event["event_type"] == "budget_limit_reached")
    assert limit_event["payload"]["error"]["code"] == "SEARCH_BUDGET_EXHAUSTED"


def test_adk_tool_limit_returns_guidance_instead_of_raising(app, session):
    service = app.state.agent_service
    run = service.create_run(session["session_id"], "test", "question")
    budget = Budget(max_rounds=4, max_tool_calls=1)
    search_document, read_passage, inspect_document_outline = service._adk_tools(
        session["version_id"], run["run_id"], budget
    )

    search = search_document("records")
    assert search["ok"] is True
    chunk_id = search["results"][0]["chunk_id"]
    read_denied = read_passage(chunk_id)
    outline_denied = inspect_document_outline()

    assert read_denied["error"]["code"] == "TOOL_BUDGET_EXHAUSTED"
    assert outline_denied["error"]["code"] == "TOOL_BUDGET_EXHAUSTED"
    assert budget.calls == 1
    assert budget.denied_calls == 2


def test_session_is_bound_to_exact_version(client, imported_markdown):
    response = client.post("/sessions", json={"document_id": "doc_wrong", "version_id": imported_markdown["version_id"]})
    assert response.status_code == 404


def test_failed_indexing_is_retried_by_reupload(client, app):
    delegate = app.state.ingestor.embedder

    class FlakyEmbedder:
        model_name = delegate.model_name
        dimension = delegate.dimension

        def __init__(self):
            self.calls = 0

        def embed(self, texts):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("synthetic transient failure")
            return delegate.embed(texts)

    flaky = FlakyEmbedder()
    app.state.ingestor.embedder = flaky
    app.state.retriever.embedder = flaky
    text = b"# Retention\n\nApproval audit records must be retained for nine years."

    first = client.post("/documents", files={"file": ("retry.md", text, "text/markdown")}).json()
    first_state = client.get(f"/documents/{first['document_id']}/versions/{first['version_id']}").json()
    assert first_state["status"] == "failed"
    assert first_state["error_code"] == "INDEXING_FAILED"

    second = client.post("/documents", files={"file": ("retry.md", text, "text/markdown")}).json()
    second_state = client.get(f"/documents/{second['document_id']}/versions/{second['version_id']}").json()
    assert second["version_id"] == first["version_id"]
    assert second_state["status"] == "ready"
    assert second_state["error_code"] is None
    assert flaky.calls == 2

    duplicate = client.post("/documents", files={"file": ("retry.md", text, "text/markdown")}).json()
    assert duplicate["status"] == "ready"
    assert flaky.calls == 2

    session = client.post(
        "/sessions", json={"document_id": second["document_id"], "version_id": second["version_id"]}
    ).json()
    _, run = _run(client, session["session_id"], "How long must approval audit records be retained?")
    assert run["status"] == "completed"
    assert "nine years" in run["answer"]
    assert run["citations"]

    evidence_id = run["citations"][0]["evidence_id"]
    chunks_before = app.state.db.all(
        "SELECT chunk_id, raw_text FROM chunks WHERE version_id=? ORDER BY ordinal", (second["version_id"],)
    )
    repeated = app.state.ingestor.process(second["version_id"])
    chunks_after = app.state.db.all(
        "SELECT chunk_id, raw_text FROM chunks WHERE version_id=? ORDER BY ordinal", (second["version_id"],)
    )
    assert repeated["status"] == "ready"
    assert chunks_after == chunks_before
    assert app.state.db.one("SELECT evidence_id FROM evidence WHERE evidence_id=?", (evidence_id,)) is not None
