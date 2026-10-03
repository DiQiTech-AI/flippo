from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import fitz

from rag_agent.ingestion import ParsedUnit, chunk_units
from rag_agent.retrieval import lexical_terms


def test_mixed_script_query_keeps_latin_entity_separate():
    assert lexical_terms("jev是什么？") == ["jev", "是什", "什么"]


def test_heading_boundary_does_not_carry_structural_overlap():
    units = [
        ParsedUnit("old section body", ["Old"], None, 0, 16),
        ParsedUnit("tail", ["Old"], None, 18, 22),
        ParsedUnit("new section body", ["New"], None, 24, 40),
    ]

    chunks = chunk_units(units, target_chars=100, overlap_chars=10)

    assert [(chunk.text, chunk.heading_path) for chunk in chunks] == [
        ("old section body\n\ntail", ["Old"]),
        ("new section body", ["New"]),
    ]


def test_page_boundary_does_not_carry_structural_overlap():
    units = [
        ParsedUnit("first page body", [], 1, 0, 15),
        ParsedUnit("tail", [], 1, 17, 21),
        ParsedUnit("second page body", [], 2, 22, 38),
    ]

    chunks = chunk_units(units, target_chars=100, overlap_chars=10)

    assert [(chunk.text, chunk.page) for chunk in chunks] == [
        ("first page body\n\ntail", 1),
        ("second page body", 2),
    ]


def test_length_boundary_keeps_overlap_within_same_structure():
    units = [
        ParsedUnit("a" * 20, ["Same"], 1, 0, 20),
        ParsedUnit("tail", ["Same"], 1, 22, 26),
        ParsedUnit("b" * 20, ["Same"], 1, 28, 48),
    ]

    chunks = chunk_units(units, target_chars=30, overlap_chars=10)

    assert [chunk.text for chunk in chunks] == [f"{'a' * 20}\n\ntail", f"tail\n\n{'b' * 20}"]
    assert all(chunk.heading_path == ["Same"] and chunk.page == 1 for chunk in chunks)


def test_markdown_import_preserves_heading_and_hybrid_search(app, imported_markdown):
    result = app.state.retriever.search(imported_markdown["version_id"], "approval records retained", "hybrid", 5)
    assert result["results"]
    assert result["results"][0]["location"]["heading_path"] == ["Records"]
    assert result["results"][0]["scores"]["keyword_rank"] is not None


def test_chinese_keyword_search_indexes_body_and_heading(client, app):
    text = """# 审批政策

审批记录的保留期限为七年。
"""
    uploaded = client.post("/documents", files={"file": ("policy-zh.md", text.encode(), "text/markdown")}).json()
    state = client.get(f"/documents/{uploaded['document_id']}/versions/{uploaded['version_id']}").json()
    assert state["status"] == "ready"

    body = app.state.retriever.search(state["version_id"], "保留期限", "keyword", 5)
    heading = app.state.retriever.search(state["version_id"], "审批政策", "keyword", 5)

    assert body["results"][0]["scores"]["keyword_rank"] == 1
    assert heading["results"][0]["location"]["heading_path"] == ["审批政策"]
    assert heading["results"][0]["scores"]["keyword_rank"] == 1


def test_initialize_upgrades_existing_fts_once(client, app):
    text = """# 旧版制度标题

旧版正文包含审批记录与保留期限。
"""
    uploaded = client.post("/documents", files={"file": ("legacy-zh.md", text.encode(), "text/markdown")}).json()
    version_id = uploaded["version_id"]
    chunks = app.state.db.all("SELECT * FROM chunks WHERE version_id=? ORDER BY ordinal", (version_id,))

    with app.state.db.transaction() as conn:
        conn.execute("DELETE FROM chunks_fts WHERE version_id=?", (version_id,))
        for chunk in chunks:
            conn.execute(
                "INSERT INTO chunks_fts(chunk_id, version_id, heading_text, normalized_text) VALUES (?, ?, ?, ?)",
                (chunk["chunk_id"], version_id, "旧版制度标题", chunk["normalized_text"]),
            )
        conn.execute("DELETE FROM schema_metadata WHERE key='fts_cjk_bigrams_v1'")

    assert app.state.retriever.search(version_id, "保留期限", "keyword", 5)["results"] == []

    app.state.db.initialize()
    app.state.db.execute(
        "INSERT INTO chunks_fts(chunk_id, version_id, heading_text, normalized_text) VALUES (?, ?, ?, ?)",
        ("sentinel", "sentinel", "sentinel", "sentinel"),
    )
    first_count = app.state.db.one("SELECT count(*) AS n FROM chunks_fts")["n"]
    app.state.db.initialize()
    second_count = app.state.db.one("SELECT count(*) AS n FROM chunks_fts")["n"]

    body = app.state.retriever.search(version_id, "保留期限", "keyword", 5)
    heading = app.state.retriever.search(version_id, "制度标题", "keyword", 5)
    assert body["results"]
    assert heading["results"]
    assert first_count == second_count
    assert app.state.db.one("SELECT chunk_id FROM chunks_fts WHERE chunk_id='sentinel'") is not None


def test_duplicate_processing_claim_embeds_once(app):
    ingestor = app.state.ingestor
    data = b"# Concurrent\n\nOnly one processing task should build this index."
    with ThreadPoolExecutor(max_workers=2) as pool:
        registrations = list(pool.map(lambda _: ingestor.register("concurrent.md", data), range(2)))
    assert registrations[0]["version_id"] == registrations[1]["version_id"]

    delegate = ingestor.embedder

    class CountingEmbedder:
        model_name = delegate.model_name
        dimension = delegate.dimension

        def __init__(self):
            self.calls = 0

        def embed(self, texts):
            self.calls += 1
            return delegate.embed(texts)

    counting = CountingEmbedder()
    ingestor.embedder = counting
    version_id = registrations[0]["version_id"]
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda _: ingestor.process(version_id), range(2)))

    assert counting.calls == 1
    assert app.state.db.one("SELECT status FROM documents WHERE version_id=?", (version_id,))["status"] == "ready"


def test_pdf_import_and_page_location(client):
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "The warranty period is twenty four months from purchase.")
    data = document.tobytes()
    document.close()
    uploaded = client.post("/documents", files={"file": ("warranty.pdf", data, "application/pdf")}).json()
    state = client.get(f"/documents/{uploaded['document_id']}/versions/{uploaded['version_id']}").json()
    assert state["status"] == "ready"
    assert state["page_count"] == 1


def test_chinese_definition_question_finds_english_pdf_evidence(client):
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "Jev Engineering reference document for coding agent architecture.")
    page = document.new_page()
    page.insert_text(
        (72, 72),
        "Jev is not the model that writes the code. It is the decision layer beside it. "
        "Jev returns typed choices, scores, or null decisions with probabilities.",
    )
    data = document.tobytes()
    document.close()

    uploaded = client.post("/documents", files={"file": ("jev.pdf", data, "application/pdf")}).json()
    state = client.get(f"/documents/{uploaded['document_id']}/versions/{uploaded['version_id']}").json()
    assert state["status"] == "ready"
    session = client.post(
        "/sessions",
        json={"document_id": uploaded["document_id"], "version_id": uploaded["version_id"]},
    ).json()

    response = client.post(
        f"/sessions/{session['session_id']}/messages",
        json={"message": "jev是什么？", "request_type": "question"},
    )
    assert response.status_code == 202
    run = client.get(f"/runs/{response.json()['run_id']}").json()
    assert run["status"] == "completed"
    assert "decision layer" in run["answer"]
    assert run["citations"]
    assert run["citations"][0]["location"]["page_start"] == 2

    unrelated = client.post(
        f"/sessions/{session['session_id']}/messages",
        json={"message": "What is the mandatory satellite launch date?", "request_type": "question"},
    )
    unrelated_run = client.get(f"/runs/{unrelated.json()['run_id']}").json()
    assert unrelated_run["citations"] == []
    assert "未找到足够证据" in unrelated_run["answer"]


def test_scanned_pdf_reports_ocr_required(client):
    document = fitz.open()
    document.new_page()
    data = document.tobytes()
    document.close()
    uploaded = client.post("/documents", files={"file": ("scan.pdf", data, "application/pdf")}).json()
    state = client.get(f"/documents/{uploaded['document_id']}/versions/{uploaded['version_id']}").json()
    assert state["status"] == "needs_ocr"
    assert state["error_code"] == "DOCUMENT_NEEDS_OCR"


def test_upload_type_and_magic_are_validated(client):
    bad = client.post("/documents", files={"file": ("fake.pdf", b"not pdf", "application/pdf")})
    assert bad.status_code == 400
    assert bad.json()["error"]["code"] == "INVALID_FILE"
