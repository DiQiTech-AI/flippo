from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class Location(BaseModel):
    page_start: int | None = None
    page_end: int | None = None
    heading_path: list[str] = Field(default_factory=list)
    source_start: int | None = None
    source_end: int | None = None


class Citation(BaseModel):
    evidence_id: str
    chunk_id: str
    document_version: str
    location: Location
    excerpt: str


class Finding(BaseModel):
    finding_id: str
    claim: str
    evidence_ids: list[str]
    confidence: Literal["high", "medium", "low"]
    support: Literal["supported", "partially_supported", "unsupported", "conflicting"] = "supported"


class Recommendation(BaseModel):
    text: str
    basis_finding_ids: list[str] = Field(default_factory=list)


class Report(BaseModel):
    run_id: str
    document_version: str
    title: str
    executive_summary: str
    findings: list[Finding]
    recommendations: list[Recommendation] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    citations: list[Citation]


class MessageRequest(BaseModel):
    message: str = Field(min_length=1, max_length=20_000)
    request_type: Literal["auto", "question", "report"] = "auto"
    model_id: str | None = Field(default=None, min_length=1, max_length=500)


class SessionRequest(BaseModel):
    document_id: str
    version_id: str


class ErrorBody(BaseModel):
    code: str
    message: str
    retryable: bool = False
    request_id: str


class ErrorResponse(BaseModel):
    error: ErrorBody


class SearchFilters(BaseModel):
    heading_prefix: str | None = None
    page_from: int | None = None
    page_to: int | None = None


class EvalCase(BaseModel):
    id: str
    question: str
    request_type: Literal["question", "report"] = "question"
    expected_answer_terms: list[str] = Field(default_factory=list)
    acceptable_chunk_ids: list[str] = Field(default_factory=list)
    should_refuse: bool = False
    require_query_rewrite: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
