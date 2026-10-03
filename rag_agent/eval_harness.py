from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import httpx

from .schemas import EvalCase


def load_cases(path: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                cases.append(EvalCase.model_validate_json(line))
            except Exception as exc:
                raise ValueError(f"Invalid evaluation case at line {number}: {exc}") from exc
    return cases


def run_eval(base_url: str, document_id: str, version_id: str, cases: list[EvalCase], timeout: float = 180) -> dict[str, Any]:
    client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)
    session = client.post("/sessions", json={"document_id": document_id, "version_id": version_id})
    session.raise_for_status()
    session_id = session.json()["session_id"]
    details = []
    for case in cases:
        response = client.post(
            f"/sessions/{session_id}/messages",
            json={"message": case.question, "request_type": case.request_type},
        )
        response.raise_for_status()
        run_id = response.json()["run_id"]
        deadline = time.monotonic() + timeout
        while True:
            run_response = client.get(f"/runs/{run_id}")
            run_response.raise_for_status()
            run = run_response.json()
            if run["status"] not in {"queued", "running"}:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Run {run_id} did not finish")
            time.sleep(0.25)
        trace = client.get(f"/runs/{run_id}/trace").json()["events"]
        answer_text = (run.get("answer") or "") + " " + json.dumps(run.get("report") or {}, ensure_ascii=False)
        cited_chunks = {item["chunk_id"] for item in run.get("citations", [])}
        refusal = not run.get("citations") and ("证据" in answer_text or "insufficient" in answer_text.casefold())
        term_hits = [term.casefold() in answer_text.casefold() for term in case.expected_answer_terms]
        location_valid = all(item.get("chunk_id") and item.get("document_version") == version_id and item.get("excerpt") for item in run.get("citations", []))
        detail = {
            "id": case.id,
            "run_id": run_id,
            "status": run["status"],
            "answer_terms_hit": sum(term_hits),
            "answer_terms_total": len(term_hits),
            "evidence_recalled": not case.acceptable_chunk_ids or bool(cited_chunks & set(case.acceptable_chunk_ids)),
            "refusal_correct": refusal == case.should_refuse,
            "query_rewrite_observed": any(event["event_type"] == "query_rewrite" for event in trace),
            "query_rewrite_expected": case.require_query_rewrite,
            "citation_locations_valid": location_valid,
            "trace_event_count": len(trace),
        }
        details.append(detail)
    answer_denominator = sum(item["answer_terms_total"] for item in details)
    metrics = {
        "cases": len(details),
        "answer_term_coverage": (sum(item["answer_terms_hit"] for item in details) / answer_denominator) if answer_denominator else None,
        "retrieval_recall": sum(bool(item["evidence_recalled"]) for item in details) / len(details) if details else None,
        "refusal_accuracy": sum(bool(item["refusal_correct"]) for item in details) / len(details) if details else None,
        "citation_location_validity": sum(bool(item["citation_locations_valid"]) for item in details) / len(details) if details else None,
        "trace_completeness": sum(item["trace_event_count"] > 0 for item in details) / len(details) if details else None,
    }
    return {"document_id": document_id, "version_id": version_id, "metrics": metrics, "cases": details}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an annotated JSONL evaluation set against the RAG API")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--document-id", required=True)
    parser.add_argument("--version-id", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_eval(args.base_url, args.document_id, args.version_id, load_cases(args.dataset))
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()

