from __future__ import annotations

import fitz

from rag_agent.document_structure import inspect_image_structure


def _upload_and_session(client, filename: str, data: bytes, media_type: str) -> dict:
    uploaded = client.post("/documents", files={"file": (filename, data, media_type)}).json()
    state = client.get(f"/documents/{uploaded['document_id']}/versions/{uploaded['version_id']}").json()
    assert state["status"] == "ready"
    return client.post(
        "/sessions",
        json={"document_id": uploaded["document_id"], "version_id": uploaded["version_id"]},
    ).json()


def _ask(client, session_id: str, request_type: str = "question") -> tuple[dict, list[dict]]:
    response = client.post(
        f"/sessions/{session_id}/messages",
        json={"message": "文档里有几张图片？", "request_type": request_type},
    )
    assert response.status_code == 202
    run_id = response.json()["run_id"]
    run = client.get(f"/runs/{run_id}").json()
    trace = client.get(f"/runs/{run_id}/trace").json()["events"]
    return run, trace


def test_pdf_image_count_uses_structure_without_search_and_qualifies_vectors(client):
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "A document with raster images and a vector diagram." * 3)
    pixmap = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 4, 4), 0)
    pixmap.clear_with(0x336699)
    image = pixmap.tobytes("png")
    page.insert_image(fitz.Rect(72, 100, 112, 140), stream=image)
    page.insert_image(fitz.Rect(130, 100, 170, 140), stream=image)
    page.draw_rect(fitz.Rect(72, 170, 170, 220), color=(0, 0, 0))
    data = document.tobytes()
    document.close()

    session = _upload_and_session(client, "mixed.pdf", data, "application/pdf")
    run, trace = _ask(client, session["session_id"])

    assert run["status"] == "completed"
    assert "2 个嵌入式栅格图片显示实例" in run["answer"]
    assert "矢量图" in run["answer"]
    assert "不是所有可见图片的精确总数" in run["answer"]
    assert run["citations"] == []
    assert run["search_rounds"] == 0
    assert run["tool_calls"] == 0
    event = next(item for item in trace if item["event_type"] == "document_structure_inspection")
    assert event["payload"]["document_version"] == session["version_id"]
    assert event["payload"]["raster_image_placements"] == 2
    assert event["payload"]["vector_drawing_primitives"] >= 1
    assert not any(item["event_type"] == "tool_result" for item in trace)


def test_pdf_zero_rasters_does_not_claim_there_are_no_visible_graphics(client):
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "A vector-only document with enough searchable text." * 3)
    page.draw_line((72, 100), (200, 200), color=(0, 0, 0))
    data = document.tobytes()
    document.close()

    session = _upload_and_session(client, "vector.pdf", data, "application/pdf")
    run, _ = _ask(client, session["session_id"], request_type="report")

    assert run["status"] == "completed"
    assert run["error_code"] is None
    assert run["stop_reason"] == "document_structure_inspected"
    assert "0 个" in run["answer"]
    assert "不代表页面上没有可见图形" in run["answer"]
    assert run["report"]["executive_summary"] == run["answer"]
    assert run["report"]["document_version"] == session["version_id"]
    assert run["report"]["citations"] == []
    assert run["search_rounds"] == 0


def test_pdf_counts_distinct_figure_captions_with_pages_and_ignores_references(client):
    document = fitz.open()
    captions = {
        1: "Fig. 1. The harness.",
        4: "Fig. 2. The routing trap.",
        6: "Fig. 3. Retrieval dominates.",
        7: "Fig. 4. The visibility ladder.",
        8: "Fig. 5. Tiered disclosure.",
        9: "Fig. 6. Conditional instructions.",
        10: "Fig. 7. Background processing.",
    }
    for page_number in range(1, 13):
        page = document.new_page()
        page.insert_text((72, 72), f"Searchable content for page {page_number} and document indexing.")
        if page_number in captions:
            page.insert_text((72, 110), captions[page_number])
        if page_number == 2:
            page.insert_text((72, 110), "Figure 1: Repeated caption label from another location.")
            page.insert_text((72, 140), "Figure 8 shows a reference in prose, not a caption.")
            page.insert_text((72, 170), "The discussion also refers to Fig. 9. in prose.")
            page.insert_text((72, 200), "TABLE I IMAGE COUNTS")
    data = document.tobytes()
    document.close()

    session = _upload_and_session(client, "captioned-figures.pdf", data, "application/pdf")
    run, trace = _ask(client, session["session_id"])

    assert run["status"] == "completed"
    assert "检测到 7 幅带图注的图" in run["answer"]
    assert "Fig. 1（第 1 页）" in run["answer"]
    assert "Fig. 7（第 10 页）" in run["answer"]
    assert "未带图注" in run["answer"]
    assert "Fig. 8" not in run["answer"]
    assert "Fig. 9" not in run["answer"]
    assert run["search_rounds"] == 0
    event = next(item for item in trace if item["event_type"] == "document_structure_inspection")
    assert event["payload"]["captioned_figure_count"] == 7
    assert [item["page"] for item in event["payload"]["captioned_figures"]] == [1, 4, 6, 7, 8, 9, 10]


def test_pdf_recognizes_chinese_figure_captions_but_not_prose(tmp_path):
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "Chinese captions are inserted below.")
    page.insert_text((72, 110), "图1：系统架构", fontname="china-s")
    page.insert_text((72, 140), "图 2 数据流程", fontname="china-s")
    page.insert_text((72, 170), "如图3所示，这是正文引用。", fontname="china-s")
    page.insert_text((72, 200), "图4显示了另一处正文引用。", fontname="china-s")
    path = tmp_path / "chinese-captions.pdf"
    document.save(path)
    document.close()

    inspection = inspect_image_structure({"file_path": str(path), "media_type": "application/pdf"})

    assert inspection["captioned_figure_count"] == 2
    assert [item["label"] for item in inspection["captioned_figures"]] == ["图1", "图2"]


def test_markdown_counts_rendered_image_references_not_code(client):
    markdown = b"""# Images

![first](one.png)
![second][asset]

`![not an image](inline-code.png)`

```md
![also not an image](fenced.png)
```

[asset]: two.png

This text keeps the document indexable and ready for questions.
"""
    session = _upload_and_session(client, "images.md", markdown, "text/markdown")
    run, trace = _ask(client, session["session_id"])

    assert run["status"] == "completed"
    assert "2 处 Markdown 图片引用" in run["answer"]
    assert run["search_rounds"] == 0
    event = next(item for item in trace if item["event_type"] == "document_structure_inspection")
    assert event["payload"]["markdown_image_references"] == 2
