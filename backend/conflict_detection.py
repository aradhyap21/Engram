"""
conflict_detection.py
---------------------
Production-grade conflict detection for temporal knowledge graph edges.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel, Field, model_validator
from dotenv import load_dotenv

load_dotenv()

NVIDIA_NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "meta/llama-3.1-70b-instruct"

CONFLICT_DETECTION_SYSTEM = """\
You are a contradiction-detection engine for a temporal knowledge graph. You
are given a new fact and existing edges for the same entity and the same
EXCLUSIVE relation (a relation where only one value should hold at a time,
e.g. current_employer, lives_in, marital_status). Decide, for each existing
edge, whether it CONTRADICTS, REFINES, or COEXISTS with the new fact, and
call the record_conflicts tool.

Definitions:
- CONTRADICTS: both cannot be true at the same time
  (e.g. "lives in Boston" vs "lives in Seattle" for the same relation).
- REFINES: the new fact adds detail without invalidating the existing edge
  (e.g. "works at Acme" already known; new fact is "works at Acme as a
  Senior Engineer").
- COEXISTS: no real conflict despite matching entity/relation - rare for
  exclusive relations, but valid if timeframes are disjoint and both are
  historically accurate
  (e.g. "lived in Boston 2018-2020" and "lives in Seattle since 2021").

For CONTRADICTS, determine which fact is more recent using, in priority
order: (1) explicit valid_at dates in either fact's text, (2) relative time
expressions ("now", "used to", "previously"), (3) ingestion order as a last
resort. The more recent fact wins - the older edge gets INVALIDATE_EXISTING
(or the new fact gets INVALIDATE_NEW if the new fact is actually the stale
one, e.g. a document being backfilled). Never discard information: an
invalidated edge stays in the graph with its invalid_at timestamp set, it is
never deleted.

If no existing edges are provided, return an empty conflicts list.
"""


class ExistingEdge(BaseModel):
    """An existing edge on the same entity + exclusive relation."""

    edge_id: str
    fact_text: str
    valid_at: str | None = None
    invalid_at: str | None = None
    created_at: str


class ConflictItem(BaseModel):
    existing_edge_id: str
    relationship: Literal["CONTRADICTS", "REFINES", "COEXISTS"]
    action: Literal["INVALIDATE_EXISTING", "INVALIDATE_NEW", "KEEP_BOTH", "MERGE_FACTS"]
    invalid_at: str | None = None
    reasoning: str

    @model_validator(mode="after")
    def require_timestamp_on_invalidate(self) -> "ConflictItem":
        # Bi-temporal correctness: an edge can be invalidated, never deleted,
        # and every invalidation must carry the timestamp it stopped being true.
        if self.action in ("INVALIDATE_EXISTING", "INVALIDATE_NEW") and not self.invalid_at:
            raise ValueError(f"action={self.action} requires an invalid_at timestamp")
        return self


class ProvenanceMeta(BaseModel):
    source_mention: str
    model_used: str
    decided_at: str


class ConflictDetectionResult(BaseModel):
    conflicts: list[ConflictItem] = Field(default_factory=list)
    provenance: ProvenanceMeta


class _RawConflictResponse(BaseModel):
    conflicts: list[ConflictItem] = Field(default_factory=list)


_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.environ.get("NVIDIA_API_KEY")
        if not api_key:
            raise RuntimeError("NVIDIA_API_KEY is not set.")
        _client = OpenAI(base_url=NVIDIA_NIM_BASE_URL, api_key=api_key)
    return _client


def detect_conflicts(
    entity_id: str,
    relation_type: str,
    new_fact_text: str,
    ingestion_timestamp: str,
    existing_edges: list[dict],
    extracted_valid_at: str | None = None,
    model: str = DEFAULT_MODEL,
    max_attempts: int = 2,
) -> ConflictDetectionResult:
    client = _get_client()
    schema = _RawConflictResponse.model_json_schema()

    tool = {
        "type": "function",
        "function": {
            "name": "record_conflicts",
            "description": "Record conflict decisions against existing edges.",
            "parameters": schema,
        },
    }

    edges_to_evaluate = []
    for e in existing_edges:
        edges_to_evaluate.append(
            ExistingEdge(
                edge_id=e["id"],
                fact_text=e.get("fact_text") or "(no original text)",
                valid_at=e.get("valid_at"),
                invalid_at=e.get("invalid_at"),
                created_at=e.get("created_at") or ingestion_timestamp,
            )
        )

    user_prompt = (
        f'New fact being ingested:\n'
        f'  Relation: "{relation_type}"\n'
        f'  Extracted valid_at: {extracted_valid_at}\n'
        f'  Ingestion timestamp: {ingestion_timestamp}\n'
        f'  Context (fact text): "{new_fact_text}"\n\n'
        f"Existing active edges for this entity and relation:\n"
        f"{json.dumps([e.model_dump() for e in edges_to_evaluate], indent=2)}\n\n"
        f"Call record_conflicts now."
    )
    messages: list[dict] = [
        {"role": "system", "content": CONFLICT_DETECTION_SYSTEM},
        {"role": "user", "content": user_prompt},
    ]
    last_error: Exception | None = None

    for _ in range(max_attempts):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=[tool],
            tool_choice={"type": "function", "function": {"name": "record_conflicts"}},
        )
        message = response.choices[0].message
        calls = message.tool_calls or []
        if not calls:
            last_error = RuntimeError("model did not return a tool call")
        else:
            try:
                args = json.loads(calls[0].function.arguments)
                raw = _RawConflictResponse.model_validate(args)
                provenance = ProvenanceMeta(
                    source_mention=new_fact_text,
                    model_used=model,
                    decided_at=datetime.now(timezone.utc).isoformat(),
                )
                return ConflictDetectionResult(
                    conflicts=raw.conflicts, provenance=provenance
                )
            except Exception as e:
                last_error = e
                messages.append(message.model_dump(exclude_none=True))
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": calls[0].id,
                        "content": f"Invalid input: {e}. Call record_conflicts again, fixed.",
                    }
                )
                continue

    raise RuntimeError(
        f"record_conflicts failed after {max_attempts} attempts: {last_error}"
    )
