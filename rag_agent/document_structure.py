from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

import fitz
from markdown_it import MarkdownIt
from markdown_it.token import Token


_CHINESE_IMAGE_TERMS = re.compile(r"(?:图片|图像|照片|插图|配图|图表|(?:张|幅)图|图(?=[？?！!，,。.\s]|$))")
_CHINESE_COUNT_TERMS = re.compile(r"(?:几张|几幅|几个|多少|数量|数目|总数|计数|统计)")
_ENGLISH_IMAGE_TERMS = re.compile(r"\b(?:images?|pictures?|photos?|figures?|illustrations?|charts?)\b", re.I)
_ENGLISH_COUNT_TERMS = re.compile(r"\b(?:how\s+many|(?:total\s+)?number\s+of|count(?:\s+of)?|total)\b", re.I)

_ENGLISH_FIGURE_CAPTION = re.compile(
    r"^\s*(?:fig(?:ure)?\.?)[ \t]*"
    r"(?P<number>[A-Za-z]?[0-9]+(?:[.\-][0-9]+)*(?:[A-Za-z])?)"
    r"[ \t]*(?:[.:：．])[ \t]*(?P<title>\S.*)$",
    re.I,
)
_CHINESE_FIGURE_CAPTION = re.compile(
    r"^\s*图[ \t]*"
    r"(?P<number>[0-9０-９一二三四五六七八九十百]+(?:[.\-－—][0-9０-９一二三四五六七八九十百]+)*(?:[A-Za-z])?)"
    r"(?:[ \t]*(?:[.:：．、])[ \t]*|[ \t]+)(?P<title>\S.*)$",
)


def is_image_count_query(message: str) -> bool:
    """Return true only for an explicit request to count document images."""
    compact = re.sub(r"\s+", " ", message).strip()
    chinese = bool(_CHINESE_IMAGE_TERMS.search(compact) and _CHINESE_COUNT_TERMS.search(compact))
    english = bool(_ENGLISH_IMAGE_TERMS.search(compact) and _ENGLISH_COUNT_TERMS.search(compact))
    return chinese or english


def _walk_tokens(tokens: Iterable[Token]) -> Iterable[Token]:
    for token in tokens:
        yield token
        if token.children:
            yield from _walk_tokens(token.children)


def _figure_caption(block_text: str) -> tuple[str, str] | None:
    """Return a normalized figure label and caption for a caption-like text block."""
    text = " ".join(block_text.split())
    match = _ENGLISH_FIGURE_CAPTION.match(text)
    if match:
        number = match.group("number")
        return f"Fig. {number}", match.group("title").strip()
    match = _CHINESE_FIGURE_CAPTION.match(text)
    if match:
        number = match.group("number")
        return f"图{number}", match.group("title").strip()
    return None


def _inspect_pdf(path: Path) -> dict[str, Any]:
    document = fitz.open(path)
    try:
        if document.needs_pass:
            raise ValueError("The active PDF version is encrypted and cannot be inspected.")
        pages: list[dict[str, Any]] = []
        raster_placements = 0
        vector_primitives = 0
        captioned_figures: list[dict[str, Any]] = []
        seen_figure_labels: set[str] = set()
        for page_number, page in enumerate(document, 1):
            page_rasters = len(page.get_image_info())
            page_vectors = len(page.get_drawings())
            page_figures: list[str] = []
            for block in page.get_text("blocks"):
                caption = _figure_caption(str(block[4]))
                if caption is None:
                    continue
                label, title = caption
                deduplication_key = label.casefold()
                if deduplication_key in seen_figure_labels:
                    continue
                seen_figure_labels.add(deduplication_key)
                page_figures.append(label)
                captioned_figures.append({"label": label, "page": page_number, "caption": title})
            raster_placements += page_rasters
            vector_primitives += page_vectors
            pages.append(
                {
                    "page": page_number,
                    "raster_image_placements": page_rasters,
                    "vector_drawing_primitives": page_vectors,
                    "captioned_figures": page_figures,
                }
            )
        return {
            "media_type": "application/pdf",
            "method": "PyMuPDF text-block figure captions, Page.get_image_info, and Page.get_drawings",
            "page_count": document.page_count,
            "captioned_figure_count": len(captioned_figures),
            "captioned_figures": captioned_figures,
            "raster_image_placements": raster_placements,
            "vector_drawing_primitives": vector_primitives,
            "pages": pages,
        }
    finally:
        document.close()


def _inspect_markdown(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    tokens = MarkdownIt("commonmark", {"html": False}).parse(text)
    image_references = sum(token.type == "image" for token in _walk_tokens(tokens))
    return {
        "media_type": "text/markdown",
        "method": "markdown-it-py CommonMark image tokens",
        "markdown_image_references": image_references,
    }


def inspect_image_structure(document: dict[str, Any]) -> dict[str, Any]:
    """Count verifiable image placements in one immutable document version."""
    path = Path(str(document["file_path"]))
    if document["media_type"] == "application/pdf":
        return _inspect_pdf(path)
    if document["media_type"] == "text/markdown":
        return _inspect_markdown(path)
    raise ValueError(f"Image structure inspection is unsupported for {document['media_type']}")


def image_count_answer(inspection: dict[str, Any], message: str) -> str:
    chinese = bool(re.search(r"[\u3400-\u9fff]", message))
    if inspection["media_type"] == "application/pdf":
        captioned_figures = inspection.get("captioned_figures", [])
        if captioned_figures:
            locations = "、".join(
                f"{figure['label']}（第 {figure['page']} 页）" for figure in captioned_figures
            )
            count = len(captioned_figures)
            if chinese:
                return (
                    f"检测到 {count} 幅带图注的图：{locations}。"
                    "该数量按文档中的图注编号统计；未带图注的装饰图形或其他图形可能未计入。"
                )
            locations = ", ".join(
                f"{figure['label']} (page {figure['page']})" for figure in captioned_figures
            )
            return (
                f"I detected {count} captioned figures: {locations}. "
                "This count is based on figure-caption labels; unlabeled decorative or other graphics may not be included."
            )

        count = int(inspection["raster_image_placements"])
        if chinese:
            if count == 0:
                return (
                    "按 PDF 结构统计，未检测到嵌入式栅格图片显示实例（0 个）。"
                    "PDF 页面中的矢量图、图表和其他绘制元素不在此统计内，因此这不代表页面上没有可见图形，"
                    "也无法据此给出所有可见图片的精确总数。"
                )
            return (
                f"按 PDF 结构统计，文档共有 {count} 个嵌入式栅格图片显示实例。"
                "同一图片在不同位置重复显示会分别计数；矢量图、图表路径和其他页面绘制元素不在此统计内，"
                "因此这不是所有可见图片的精确总数。"
            )
        if count == 0:
            return (
                "The PDF contains 0 embedded raster-image placements. Vector diagrams, charts, and other "
                "drawing elements are outside this count, so this does not mean the pages contain no visible "
                "graphics and does not establish an exact total for all visible images."
            )
        return (
            f"The PDF contains {count} embedded raster-image placements. Repeated placements of the same image "
            "are counted separately. Vector diagrams, chart paths, and other page drawings are outside this "
            "count, so it is not an exact total of all visible images."
        )

    count = int(inspection["markdown_image_references"])
    if chinese:
        return (
            f"按 Markdown 结构统计，文档共有 {count} 处 Markdown 图片引用。"
            "同一图片的重复引用会分别计数；HTML <img>、扩展语法或运行时生成的图片不在此统计内。"
        )
    return (
        f"The document contains {count} Markdown image references. Repeated references are counted separately. "
        "HTML <img> elements, extension syntax, and dynamically generated images are outside this count."
    )
