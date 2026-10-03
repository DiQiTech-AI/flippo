from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from .agent import AgentService
from .config import Settings
from .db import Database
from .embeddings import create_embedder
from .ids import new_id
from .ingestion import DocumentIngestor, IngestionError
from .models import ModelSelectionError, discover_unsloth_models, select_unsloth_model
from .retrieval import Retriever
from .schemas import MessageRequest, SessionRequest
from .trace import TraceRecorder


def _clean_document(row: dict[str, Any]) -> dict[str, Any]:
    keep = {
        "document_id", "version_id", "sha256", "filename", "media_type", "parser_name",
        "parser_version", "status", "stage", "page_count", "character_count", "chunk_count",
        "embedding_model", "embedding_dimension", "error_code", "error_message", "created_at",
    }
    return {key: value for key, value in row.items() if key in keep}


def _run_response(db: Database, row: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    result["citations"] = json.loads(result.pop("citations_json") or "[]")
    raw_report = result.pop("report_json")
    result["report"] = json.loads(raw_report) if raw_report else None
    return result


def create_app(settings: Settings | None = None) -> FastAPI:
    cfg = settings or Settings()
    cfg.prepare()
    db = Database(cfg.database_path)
    db.initialize()
    embedder = create_embedder(cfg)
    trace = TraceRecorder(db, cfg.trace_dir)
    retriever = Retriever(db, embedder, trace)
    ingestor = DocumentIngestor(db, cfg, embedder)
    agents = AgentService(db, cfg, retriever, trace)

    app = FastAPI(
        title="Single-document RAG Agent POC",
        version="0.1.0",
        description="Verifiable single-document Q&A and reports with complete content-level traces.",
    )
    app.state.settings = cfg
    app.state.db = db
    app.state.ingestor = ingestor
    app.state.agent_service = agents
    app.state.retriever = retriever
    app.state.trace = trace

    @app.exception_handler(IngestionError)
    async def ingestion_error(_: Request, exc: IngestionError) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content={"error": {"code": exc.code, "message": str(exc), "retryable": False, "request_id": new_id("req")}},
        )

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "agent_provider": cfg.agent_provider,
            "agent_ready": cfg.runtime_ready,
            "runtime_message": cfg.runtime_message,
            "embedding_provider": cfg.embedding_provider,
            "single_instance_worker": True,
        }

    @app.get("/models")
    async def available_models() -> dict[str, Any]:
        if cfg.agent_provider != "unsloth":
            model_id = "demo-extractive" if cfg.agent_provider == "demo" else cfg.model
            return {
                "provider": cfg.agent_provider,
                "models": [{"id": model_id, "label": model_id}],
                "configured_model_id": cfg.model if cfg.agent_provider == "gemini" else None,
                "automatic_selection": {"model_id": model_id, "reason": "configured"},
                "error": None,
            }
        models = []
        try:
            models = await discover_unsloth_models(cfg)
            selected, reason = select_unsloth_model(models, None, cfg.unsloth_model_id)
            automatic_selection: dict[str, Any] = {"model_id": selected.id, "reason": reason}
            error = None
        except ModelSelectionError as exc:
            automatic_selection = {"model_id": None, "reason": None}
            error = {"code": exc.code, "message": str(exc)}
        return {
            "provider": cfg.agent_provider,
            "models": [{"id": model.id, "label": model.label} for model in models],
            "configured_model_id": cfg.unsloth_model_id,
            "automatic_selection": automatic_selection,
            "error": error,
        }

    @app.post("/documents", status_code=202)
    async def upload_document(
        background_tasks: BackgroundTasks,
        file: Annotated[UploadFile, File(description="One PDF or UTF-8 Markdown file")],
    ) -> dict[str, Any]:
        declared = file.content_type or ""
        if declared and declared not in {"application/pdf", "text/markdown", "text/plain", "application/octet-stream"}:
            raise IngestionError("UNSUPPORTED_MEDIA_TYPE", f"Unsupported declared content type: {declared}")
        data = await file.read(cfg.max_upload_mb * 1024 * 1024 + 1)
        row = ingestor.register(file.filename or "document", data)
        if row["status"] == "uploaded":
            background_tasks.add_task(ingestor.process, row["version_id"])
        return _clean_document(row)

    @app.get("/documents/{document_id}/versions/{version_id}")
    async def document_version(document_id: str, version_id: str) -> dict[str, Any]:
        row = db.one("SELECT * FROM documents WHERE document_id=? AND version_id=?", (document_id, version_id))
        if not row:
            raise HTTPException(404, "Document version not found")
        return _clean_document(row)

    @app.post("/sessions", status_code=201)
    async def create_session(request: SessionRequest) -> dict[str, Any]:
        document = db.one(
            "SELECT * FROM documents WHERE document_id=? AND version_id=?",
            (request.document_id, request.version_id),
        )
        if not document:
            raise HTTPException(404, "Document version not found")
        if document["status"] != "ready":
            raise HTTPException(409, f"Document is not ready (status={document['status']})")
        session_id = new_id("ses")
        db.execute(
            "INSERT INTO sessions(session_id, document_id, version_id) VALUES (?, ?, ?)",
            (session_id, request.document_id, request.version_id),
        )
        return {"session_id": session_id, "document_id": request.document_id, "version_id": request.version_id}

    @app.post("/sessions/{session_id}/messages", status_code=202)
    async def send_message(session_id: str, request: MessageRequest, background_tasks: BackgroundTasks) -> dict[str, Any]:
        try:
            run = agents.create_run(session_id, request.message, request.request_type, request.model_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        background_tasks.add_task(agents.process_sync, run["run_id"])
        return {
            "run_id": run["run_id"],
            "status": run["status"],
            "agent_provider": cfg.agent_provider,
            "runtime_message": cfg.runtime_message,
        }

    @app.get("/runs/{run_id}")
    async def get_run(run_id: str) -> dict[str, Any]:
        row = db.one("SELECT * FROM runs WHERE run_id=?", (run_id,))
        if not row:
            raise HTTPException(404, "Run not found")
        result = _run_response(db, row)
        result["runtime_message"] = cfg.runtime_message
        return result

    @app.get("/runs/{run_id}/trace")
    async def get_trace(run_id: str, format: str = Query("json", pattern="^(json|jsonl)$")):
        if not db.one("SELECT run_id FROM runs WHERE run_id=?", (run_id,)):
            raise HTTPException(404, "Run not found")
        if format == "jsonl":
            return PlainTextResponse(trace.jsonl(run_id), media_type="application/x-ndjson")
        return {"schema_version": "1.0", "run_id": run_id, "events": trace.list(run_id)}

    @app.get("/evidence/{evidence_id}")
    async def get_evidence(evidence_id: str) -> dict[str, Any]:
        row = db.one("SELECT * FROM evidence WHERE evidence_id=?", (evidence_id,))
        if not row:
            raise HTTPException(404, "Evidence not found")
        return {
            "evidence_id": row["evidence_id"],
            "run_id": row["run_id"],
            "document_version": row["version_id"],
            "chunk_id": row["chunk_id"],
            "raw_text": row["raw_text"],
            "context_text": row["context_text"],
            "location": json.loads(row["location_json"]),
        }

    static_dir = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        return HTMLResponse((static_dir / "index.html").read_text(encoding="utf-8"))

    return app


app = create_app()
