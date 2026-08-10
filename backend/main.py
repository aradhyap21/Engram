"""
Engram — FastAPI application entry point.

Mounts CORS middleware, loads environment variables, and registers
all HTTP route handlers for memory storage, retrieval, and synthesis.

Run from the memorymesh/ directory:
    uvicorn main:app --reload
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Task 6.1 — Bootstrap: env loading and validation
# ---------------------------------------------------------------------------

load_dotenv()

from . import ai
from . import memory
from . import graph
from . import upload
from .entity_resolution_v2 import resolve_entity, EntityCandidate
from .conflict_detection import detect_conflicts
import json
from datetime import datetime, timezone

RELATION_CLASSIFICATION = {
    "causes": "cumulative",
    "developed": "cumulative",
    "relates": "cumulative",
    "related": "cumulative",
    "subentity_of": "cumulative",
    "lives_in": "exclusive",
    "current_employer": "exclusive",
    "marital_status": "exclusive",
    "job_title": "exclusive",
    "worked_at": "cumulative",
    "friend_of": "cumulative",
    "visited": "cumulative",
    "knows": "cumulative",
}

_SUPABASE_URL = os.getenv("SUPABASE_URL")
_SUPABASE_KEY = os.getenv("SUPABASE_KEY")
_NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

if not _SUPABASE_URL:
    raise RuntimeError(
        "SUPABASE_URL is not set. "
        "Add it to your .env file or set it as an environment variable."
    )
if not _SUPABASE_KEY:
    raise RuntimeError(
        "SUPABASE_KEY is not set. "
        "Add it to your .env file or set it as an environment variable."
    )
if not _NVIDIA_API_KEY:
    raise RuntimeError(
        "NVIDIA_API_KEY is not set. "
        "Add it to your .env file or set it as an environment variable."
    )

# ---------------------------------------------------------------------------
# App creation and CORS
# ---------------------------------------------------------------------------

app = FastAPI(title="Engram", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Pydantic request model
# ---------------------------------------------------------------------------


class MemoryRequest(BaseModel):
    text: str


# ---------------------------------------------------------------------------
# Internal helpers for store_memory
# ---------------------------------------------------------------------------


def _upsert_entities(entities: list[str], context: str) -> list[dict]:
    """
    Resolve and upsert each entity string as a node and return the list of node dicts.

    Args:
        entities: List of entity name strings from the LLM extraction.
        context: The original text context for resolution.

    Returns:
        List of node dicts as returned by ``memory.upsert_node`` or ``memory.insert_resolved_node``.
    """
    stored: list[dict] = []
    for entity in entities:
        # 1. Generate embedding
        try:
            emb = ai.extract_embedding(entity)
        except Exception as e:
            # Fallback to standard upsert if embedding fails
            node = memory.upsert_node(entity, "entity")
            stored.append(node)
            continue

        # 2. Find candidates
        raw_cands = memory.get_similar_candidates(emb, threshold=0.60)
        candidates = []
        for c in raw_cands:
            aliases_raw = c.get("aliases")
            if isinstance(aliases_raw, str):
                try:
                    aliases_list = json.loads(aliases_raw)
                except:
                    aliases_list = []
            else:
                aliases_list = aliases_raw or []
                
            candidates.append(
                EntityCandidate(
                    id=c["id"],
                    canonical_name=c.get("canonical_name") or c["content"],
                    type=c.get("entity_type", "entity"),
                    aliases=aliases_list,
                    sample_context=c.get("source_mention") or "",
                )
            )
            
        # 3. Resolve
        res = resolve_entity(
            mention_name=entity,
            mention_type="entity",
            context=context,
            candidates=candidates,
            model="meta/llama-3.1-70b-instruct"
        )
        
        # 4. Branch on decision
        if res.decision == "MERGE":
            memory.update_node_aliases(res.merge_target_id, res.aliases_to_add)
            existing_node = next((c for c in raw_cands if c["id"] == res.merge_target_id), None)
            if existing_node:
                existing_node["content"] = entity
                stored.append(existing_node)
        else:
            new_node = memory.insert_resolved_node(
                content=entity,
                entity_type="entity",
                embedding=emb,
                canonical_name=res.canonical_name,
                aliases=res.aliases_to_add,
                pending_review=(res.decision == "NEEDS_REVIEW"),
                source_mention=res.provenance.source_mention,
                model_used=res.provenance.model_used,
                decided_at=res.provenance.decided_at,
                reasoning=res.reasoning
            )
            stored.append(new_node)
            
            if res.decision == "RELATED_SUBENTITY" and res.related_subentity_of:
                try:
                    memory.insert_edge(new_node["id"], res.related_subentity_of, "subentity_of")
                except Exception:
                    pass

    return stored


def _insert_relationships(
    relationships: list[dict],
    content_to_id: dict[str, str],
    context: str,
) -> int:
    """
    Insert edges for each relationship dict and return the count of stored edges.

    Self-loops are silently skipped. Missing node UUIDs are also skipped.

    Args:
        relationships:  List of relationship dicts with keys ``from``, ``to``, ``type``.
        content_to_id:  Mapping from entity content string → node UUID.
        context:        The original context string (fact text).

    Returns:
        Number of edges successfully inserted.
    """
    edges_stored = 0
    now_iso = datetime.now(timezone.utc).isoformat()
    for rel in relationships:
        from_id = content_to_id.get(rel.get("from", ""))
        to_id = content_to_id.get(rel.get("to", ""))
        rel_type = rel.get("type", "related")

        if from_id is None or to_id is None:
            continue
            
        classification = RELATION_CLASSIFICATION.get(rel_type, "cumulative")

        try:
            if classification == "exclusive":
                active_edges = memory.get_active_edges(from_id, rel_type)
                if active_edges:
                    result = detect_conflicts(
                        entity_id=from_id,
                        relation_type=rel_type,
                        new_fact_text=context,
                        ingestion_timestamp=now_iso,
                        existing_edges=active_edges
                    )

                    inserted_new = False
                    for conflict in result.conflicts:
                        if conflict.action == "INVALIDATE_EXISTING":
                            memory.update_edge_invalid_at(conflict.existing_edge_id, conflict.invalid_at)
                            if not inserted_new:
                                memory.insert_edge(from_id, to_id, rel_type, fact_text=context)
                                inserted_new = True
                                edges_stored += 1
                        elif conflict.action == "INVALIDATE_NEW":
                            if not inserted_new:
                                memory.insert_edge(from_id, to_id, rel_type, fact_text=context, invalid_at=conflict.invalid_at)
                                inserted_new = True
                                edges_stored += 1
                        elif conflict.action == "KEEP_BOTH":
                            if not inserted_new:
                                memory.insert_edge(from_id, to_id, rel_type, fact_text=context)
                                inserted_new = True
                                edges_stored += 1
                        elif conflict.action == "MERGE_FACTS":
                            existing_edge = next((e for e in active_edges if e["id"] == conflict.existing_edge_id), None)
                            if existing_edge:
                                merged_text = f"{existing_edge.get('fact_text', '')} | {context}"
                                memory.update_edge_fact_text(conflict.existing_edge_id, merged_text)
                    
                    if not result.conflicts and not inserted_new:
                        memory.insert_edge(from_id, to_id, rel_type, fact_text=context)
                        edges_stored += 1
                else:
                    memory.insert_edge(from_id, to_id, rel_type, fact_text=context)
                    edges_stored += 1
            else:
                memory.insert_edge(from_id, to_id, rel_type, fact_text=context)
                edges_stored += 1
        except ValueError:
            # Self-loop — skip silently (Req 6.4)
            continue
    return edges_stored


# ---------------------------------------------------------------------------
# Task 6.1 — Health check
# ---------------------------------------------------------------------------


@app.get("/health")
def health_check():
    """Liveness probe. Returns {"status": "ok"}."""
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Task 6.2 — POST /memory  (store plain text as graph nodes + edges)
# ---------------------------------------------------------------------------


@app.post("/memory")
def store_memory(request: MemoryRequest):
    """
    Accept plain text, extract entities and relationships via the LLM,
    upsert nodes and insert edges into Supabase, and return counts.

    Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 7.3
    """
    if not request.text or not request.text.strip():
        return JSONResponse(
            status_code=400,
            content={"error": "Request body 'text' must be a non-empty string."},
        )

    try:
        extraction = ai.extract_entities(request.text)
        stored_nodes = _upsert_entities(extraction.get("entities", []), request.text)
        content_to_id: dict[str, str] = {n["content"]: n["id"] for n in stored_nodes}
        edges_stored = _insert_relationships(extraction.get("relationships", []), content_to_id, request.text)
        return {
            "nodes_stored": len(stored_nodes),
            "edges_stored": edges_stored,
            "node_ids": [n["id"] for n in stored_nodes],
        }
    except ValueError as exc:
        return JSONResponse(status_code=422, content={"error": f"Entity extraction failed: {exc}"})
    except RuntimeError as exc:
        return JSONResponse(status_code=503, content={"error": f"Database error: {exc}"})
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": f"Unexpected error: {exc}"})


# ---------------------------------------------------------------------------
# Task 6.3 — GET /memory/retrieve  (Dijkstra path retrieval)
# ---------------------------------------------------------------------------


@app.get("/memory/retrieve")
def retrieve_memory(query: str = ""):
    """
    Extract query entities, load the full graph from Supabase, run
    Dijkstra, and return the top-5 causal paths sorted by total cost.

    Requirements: 2.1–2.6, 7.3
    """
    try:
        # Step 1 — Extract query entities (Req 2.1)
        extraction = ai.extract_entities(query) if query.strip() else {"entities": [], "relationships": []}
        query_entities: list[str] = extraction.get("entities", [])

        # Step 2 — Load the full graph from Supabase (Req 2.2, 2.3)
        nodes = memory.get_all_nodes()
        edges = memory.get_all_edges()

        # Step 3 — Run Dijkstra via the graph facade (Req 2.3, 2.4, 2.5)
        paths = graph.top_paths(nodes, edges, query_entities)

        # Step 4 — Return paths (empty list is a valid HTTP 200 response, Req 2.6)
        return {"query": query, "paths": paths}

    except ValueError as exc:
        return JSONResponse(
            status_code=422,
            content={"error": f"Query processing failed: {exc}"},
        )
    except RuntimeError as exc:
        return JSONResponse(
            status_code=503,
            content={"error": f"Database error: {exc}"},
        )
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"error": f"Unexpected error: {exc}"},
        )


# ---------------------------------------------------------------------------
# Task 6.4 — GET /memory/synthesize  (LLM insight over retrieved paths)
# ---------------------------------------------------------------------------


@app.get("/memory/synthesize")
def synthesize_memory(query: str = ""):
    """
    Run the full retrieval pipeline and pass the resulting paths to the
    LLM for synthesis into a concise, non-obvious insight.

    Requirements: 3.1, 3.2, 3.3, 3.4, 7.3
    """
    try:
        # Step 1 — Full retrieval pipeline (Req 3.1)
        extraction = ai.extract_entities(query) if query.strip() else {"entities": [], "relationships": []}
        query_entities: list[str] = extraction.get("entities", [])

        nodes = memory.get_all_nodes()
        edges = memory.get_all_edges()

        paths = graph.top_paths(nodes, edges, query_entities)

        # Step 2 — Synthesize insight (Req 3.2); called even if paths is empty (Req 3.4)
        insight = ai.synthesize_insight(paths, query)

        # Step 3 — Return response (Req 3.3)
        return {
            "query": query,
            "insight": insight,
            "paths_used": len(paths),
        }

    except ValueError as exc:
        return JSONResponse(
            status_code=422,
            content={"error": f"Query processing failed: {exc}"},
        )
    except RuntimeError as exc:
        return JSONResponse(
            status_code=503,
            content={"error": f"Database error: {exc}"},
        )
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"error": f"Unexpected error: {exc}"},
        )
