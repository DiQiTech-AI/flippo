from __future__ import annotations

import hashlib
import json
import mimetypes
import re
from dataclasses import dataclass
from pathlib import Path

import fitz
from markdown_it import MarkdownIt

from .config import Settings
from .db import Database
from .embeddings import Embedder
from .ids import new_id
from .lexical import fts_index_text


class IngestionError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass
class ParsedUnit:
    text: str
    heading_path: list[str]
    page: int | None
    source_start: int
    source_end: int


@dataclass
class ParsedDocument:
    units: list[ParsedUnit]
    page_count: int | None
    parser_name: str
    parser_version: str
    character_count: int


def detect_media_type(filename: str, data: bytes) -> str:
    suffix = Path(filename).suffix.casefold()
    if suffix == ".pdf":
        if not data.startswith(b"%PDF-"):
            raise IngestionError("INVALID_FILE", "The uploaded .pdf does not have a PDF signature.")
        return "application/pdf"
    if suffix in {".md", ".markdown"}:
        if b"\x00" in data[:4096]:
            raise IngestionError("INVALID_FILE", "Markdown input must be UTF-8 text.")
        try:
            data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise IngestionError("INVALID_ENCODING", "Markdown input must be UTF-8.") from exc
        return "text/markdown"
    guessed, _ = mimetypes.guess_type(filename)
    raise IngestionError("UNSUPPORTED_MEDIA_TYPE", f"Unsupported file type: {guessed or suffix or 'unknown'}")


def parse_markdown(data: bytes) -> ParsedDocument:
    text = data.decode("utf-8")
    lines = text.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    tokens = MarkdownIt("commonmark", {"html": False}).parse(text)
    headings: list[str] = []
    units: list[ParsedUnit] = []
    seen_ranges: set[tuple[int, int]] = set()

    for index, token in enumerate(tokens):
        if token.type == "heading_open" and token.map:
            level = int(token.tag[1])
            title = tokens[index + 1].content.strip() if index + 1 < len(tokens) else ""
            headings = headings[: level - 1]
            headings.append(title)
            continue
        if token.type not in {"paragraph_open", "fence", "code_block", "blockquote_open", "bullet_list_open", "ordered_list_open"}:
            continue
        if not token.map:
            continue
        start_line, end_line = token.map
        key = (start_line, end_line)
        if key in seen_ranges:
            continue
        seen_ranges.add(key)
        start = offsets[min(start_line, len(offsets) - 1)]
        end = offsets[min(end_line, len(offsets) - 1)]
        raw = text[start:end].strip()
        if raw:
            units.append(ParsedUnit(raw, list(headings), None, start, end))

    if not units and text.strip():
        units = [ParsedUnit(text.strip(), [], None, 0, len(text))]
    return ParsedDocument(units, None, "markdown-it-py", "4", len(text))


def parse_pdf(data: bytes) -> ParsedDocument:
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise IngestionError("INVALID_PDF", "The PDF could not be opened.") from exc
    if doc.needs_pass:
        raise IngestionError("ENCRYPTED_PDF", "Encrypted PDFs are not supported in this POC.")
    units: list[ParsedUnit] = []
    offset = 0
    low_text_pages = 0
    total_chars = 0
    for page_index, page in enumerate(doc):
        blocks = page.get_text("blocks", sort=True)
        page_chars = 0
        for block in blocks:
            raw = re.sub(r"[ \t]+", " ", str(block[4])).strip()
            if not raw:
                continue
            start = offset
            offset += len(raw) + 1
            page_chars += len(raw)
            units.append(ParsedUnit(raw, [], page_index + 1, start, offset - 1))
        total_chars += page_chars
        if page_chars < 40:
            low_text_pages += 1
    page_count = doc.page_count
    doc.close()
    if page_count and (total_chars < max(20, page_count * 20) or low_text_pages / page_count >= 0.7):
        raise IngestionError("DOCUMENT_NEEDS_OCR", "The PDF has insufficient extractable text and requires OCR.")
    return ParsedDocument(units, page_count, "PyMuPDF", fitz.VersionBind, total_chars)


def normalize(text: str) -> str:
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _split_long(unit: ParsedUnit, max_chars: int, overlap: int) -> list[ParsedUnit]:
    if len(unit.text) <= max_chars:
        return [unit]
    pieces: list[ParsedUnit] = []
    cursor = 0
    while cursor < len(unit.text):
        target = min(cursor + max_chars, len(unit.text))
        if target < len(unit.text):
            boundary = max(unit.text.rfind(mark, cursor + max_chars // 2, target) for mark in ("。", ". ", "\n", "; "))
            if boundary > cursor:
                target = boundary + 1
        raw = unit.text[cursor:target].strip()
        if raw:
            pieces.append(
                ParsedUnit(
                    raw,
                    unit.heading_path,
                    unit.page,
                    unit.source_start + cursor,
                    min(unit.source_start + target, unit.source_end),
                )
            )
        if target >= len(unit.text):
            break
        cursor = max(cursor + 1, target - overlap)
    return pieces


def chunk_units(units: list[ParsedUnit], target_chars: int, overlap_chars: int) -> list[ParsedUnit]:
    expanded = [piece for unit in units for piece in _split_long(unit, target_chars, overlap_chars)]
    chunks: list[ParsedUnit] = []
    buffer: list[ParsedUnit] = []
    length = 0

    def flush(*, keep_overlap: bool = True) -> None:
        nonlocal buffer, length
        if not buffer:
            return
        text = "\n\n".join(item.text for item in buffer)
        chunks.append(
            ParsedUnit(
                text=text,
                heading_path=buffer[0].heading_path,
                page=buffer[0].page,
                source_start=buffer[0].source_start,
                source_end=buffer[-1].source_end,
            )
        )
        # Keep only the final unit as structural overlap when it is reasonably small.
        buffer = [buffer[-1]] if keep_overlap and len(buffer) > 1 and len(buffer[-1].text) <= overlap_chars else []
        length = sum(len(item.text) + 2 for item in buffer)

    for unit in expanded:
        boundary_change = buffer and (unit.heading_path != buffer[0].heading_path or unit.page != buffer[0].page)
        if boundary_change:
            flush(keep_overlap=False)
        elif buffer and length + len(unit.text) > target_chars:
            flush()
        buffer.append(unit)
        length += len(unit.text) + 2
    flush()
    return chunks


class DocumentIngestor:
    def __init__(self, db: Database, settings: Settings, embedder: Embedder):
        self.db = db
        self.settings = settings
        self.embedder = embedder

    def register(self, filename: str, data: bytes) -> dict:
        if not data:
            raise IngestionError("EMPTY_FILE", "The uploaded file is empty.")
        if len(data) > self.settings.max_upload_mb * 1024 * 1024:
            raise IngestionError("FILE_TOO_LARGE", f"Maximum upload size is {self.settings.max_upload_mb} MB.")
        media_type = detect_media_type(filename, data)
        sha = hashlib.sha256(data).hexdigest()
        with self.db.transaction(immediate=True) as conn:
            existing_row = conn.execute(
                "SELECT * FROM documents WHERE sha256=? AND embedding_model=?",
                (sha, self.embedder.model_name),
            ).fetchone()
            if existing_row:
                existing = dict(existing_row)
                if existing["status"] == "failed":
                    conn.execute(
                        """UPDATE documents SET status='uploaded', stage='uploaded', error_code=NULL, error_message=NULL
                           WHERE version_id=? AND status='failed'""",
                        (existing["version_id"],),
                    )
                    refreshed = conn.execute(
                        "SELECT * FROM documents WHERE version_id=?", (existing["version_id"],)
                    ).fetchone()
                    return dict(refreshed) if refreshed else existing
                return existing
            document_id = f"doc_{sha[:16]}"
            version_id = new_id("docv")
            suffix = ".pdf" if media_type == "application/pdf" else ".md"
            path = self.settings.upload_dir / f"{version_id}{suffix}"
            path.write_bytes(data)
            conn.execute(
                """INSERT INTO documents
                   (document_id, version_id, sha256, filename, media_type, status, stage,
                    embedding_model, embedding_dimension, file_path)
                   VALUES (?, ?, ?, ?, ?, 'uploaded', 'uploaded', ?, ?, ?)""",
                (document_id, version_id, sha, Path(filename).name, media_type,
                 self.embedder.model_name, self.embedder.dimension, str(path)),
            )
            row = conn.execute("SELECT * FROM documents WHERE version_id=?", (version_id,)).fetchone()
            return dict(row) if row else {}

    def process(self, version_id: str) -> dict:
        with self.db.transaction(immediate=True) as conn:
            existing = conn.execute("SELECT * FROM documents WHERE version_id=?", (version_id,)).fetchone()
            if not existing:
                raise IngestionError("DOCUMENT_NOT_FOUND", "Document version does not exist.")
            if existing["status"] != "uploaded":
                return dict(existing)
            claimed = conn.execute(
                "UPDATE documents SET status='parsing', stage='parsing' WHERE version_id=? AND status='uploaded'",
                (version_id,),
            )
            if claimed.rowcount != 1:
                current = conn.execute("SELECT * FROM documents WHERE version_id=?", (version_id,)).fetchone()
                return dict(current) if current else {}
            row = dict(existing)
        try:
            data = Path(row["file_path"]).read_bytes()
            parsed = parse_pdf(data) if row["media_type"] == "application/pdf" else parse_markdown(data)
            chunks = chunk_units(parsed.units, self.settings.chunk_target_chars, self.settings.chunk_overlap_chars)
            if not chunks:
                raise IngestionError("NO_TEXT", "The document contains no indexable text.")
            self.db.execute("UPDATE documents SET status='indexing', stage='embedding' WHERE version_id=?", (version_id,))
            vectors = self.embedder.embed([normalize(item.text) for item in chunks])
            with self.db.transaction() as conn:
                conn.execute("DELETE FROM chunks_fts WHERE version_id=?", (version_id,))
                conn.execute("DELETE FROM chunks WHERE version_id=?", (version_id,))
                for ordinal, (item, vector) in enumerate(zip(chunks, vectors, strict=True)):
                    chunk_id = f"chk_{version_id.removeprefix('docv_')}_{ordinal:05d}"
                    heading = json.dumps(item.heading_path, ensure_ascii=False)
                    normalized = normalize(item.text)
                    conn.execute(
                        """INSERT INTO chunks
                           (chunk_id, version_id, ordinal, heading_path, page_start, page_end,
                            source_start, source_end, raw_text, normalized_text, token_count, embedding)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (chunk_id, version_id, ordinal, heading, item.page, item.page,
                         item.source_start, item.source_end, item.text, normalized,
                         max(1, len(normalized) // 4), vector.astype("float32").tobytes()),
                    )
                    conn.execute(
                        "INSERT INTO chunks_fts(chunk_id, version_id, heading_text, normalized_text) VALUES (?, ?, ?, ?)",
                        (
                            chunk_id,
                            version_id,
                            fts_index_text(" > ".join(item.heading_path)),
                            fts_index_text(normalized),
                        ),
                    )
                conn.execute(
                    """UPDATE documents SET status='ready', stage='ready', parser_name=?, parser_version=?,
                       page_count=?, character_count=?, chunk_count=?, embedding_dimension=? WHERE version_id=?""",
                    (parsed.parser_name, parsed.parser_version, parsed.page_count,
                     parsed.character_count, len(chunks), int(vectors.shape[1]), version_id),
                )
        except IngestionError as exc:
            status = "needs_ocr" if exc.code == "DOCUMENT_NEEDS_OCR" else "failed"
            self.db.execute(
                "UPDATE documents SET status=?, stage='failed', error_code=?, error_message=? WHERE version_id=?",
                (status, exc.code, str(exc), version_id),
            )
        except Exception as exc:
            self.db.execute(
                "UPDATE documents SET status='failed', stage='failed', error_code='INDEXING_FAILED', error_message=? WHERE version_id=?",
                (str(exc)[:500], version_id),
            )
        return self.db.one("SELECT * FROM documents WHERE version_id=?", (version_id,)) or {}
