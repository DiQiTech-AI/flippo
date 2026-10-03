from __future__ import annotations

import json
import re
from typing import Any

from .db import Database
from .schemas import Citation


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def validate_evidence_ids(db: Database, run_id: str, version_id: str, evidence_ids: list[str]) -> tuple[list[Citation], list[str]]:
    citations: list[Citation] = []
    invalid: list[str] = []
    for evidence_id in dict.fromkeys(evidence_ids):
        row = db.one(
            """SELECT e.*, c.heading_path, c.page_start, c.page_end, c.source_start, c.source_end
               FROM evidence e JOIN chunks c ON c.chunk_id=e.chunk_id
               WHERE e.evidence_id=? AND e.run_id=? AND e.version_id=?""",
            (evidence_id, run_id, version_id),
        )
        if not row:
            invalid.append(evidence_id)
            continue
        location = json.loads(row["location_json"])
        excerpt = _normalized(row["raw_text"])[:500]
        # The excerpt is always rebuilt from immutable server evidence. Model-provided
        # location or quote text is deliberately ignored.
        citations.append(
            Citation(
                evidence_id=evidence_id,
                chunk_id=row["chunk_id"],
                document_version=version_id,
                location=location,
                excerpt=excerpt,
            )
        )
    return citations, invalid


def validate_agent_payload(db: Database, run_id: str, version_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    requested: list[str] = []
    if isinstance(payload.get("evidence_ids"), list):
        requested.extend(str(item) for item in payload["evidence_ids"])
    for finding in payload.get("findings", []) if isinstance(payload.get("findings"), list) else []:
        if isinstance(finding, dict) and isinstance(finding.get("evidence_ids"), list):
            requested.extend(str(item) for item in finding["evidence_ids"])
    citations, invalid = validate_evidence_ids(db, run_id, version_id, requested)
    valid = {citation.evidence_id for citation in citations}
    findings = []
    for index, finding in enumerate(payload.get("findings", []) if isinstance(payload.get("findings"), list) else [], 1):
        if not isinstance(finding, dict):
            continue
        ids = [str(item) for item in finding.get("evidence_ids", []) if str(item) in valid]
        support = "supported" if ids else "unsupported"
        findings.append(
            {
                "finding_id": str(finding.get("finding_id") or f"F{index}"),
                "claim": str(finding.get("claim") or "").strip(),
                "evidence_ids": ids,
                "confidence": str(finding.get("confidence") or ("high" if ids else "low")) if str(finding.get("confidence") or "") in {"high", "medium", "low"} else "low",
                "support": support,
            }
        )
    return {"citations": [item.model_dump() for item in citations], "invalid_evidence_ids": invalid, "findings": findings}

