"""
MemoryMesh — FastAPI application entry point.

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
import ai
import memory
import graph
import upload

# ---------------------------------------------------------------------------
# Task 6.1 — Bootstrap: env loading and validation
# ---------------------------------------------------------------------------

load_dotenv()

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

app = FastAPI(title="MemoryMesh", version="1.0.0")

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


def _upsert_entities(entities: list[str]) -> list[dict]:
    """
    Upsert each entity string as a node and return the list of node dicts.

    Args:
        entities: List of entity name strings from the LLM extraction.

    Returns:
        List of node dicts as returned by ``memory.upsert_node``.
    """
    stored: list[dict] = []
    for entity in entities:
        node = memory.upsert_node(entity, "entity")
        stored.append(node)
    return stored


def _insert_relationships(
    relationships: list[dict],
    content_to_id: dict[str, str],
) -> int:
    """
    Insert edges for each relationship dict and return the count of stored edges.

    Self-loops are silently skipped. Missing node UUIDs are also skipped.

    Args:
        relationships:  List of relationship dicts with keys ``from``, ``to``, ``type``.
        content_to_id:  Mapping from entity content string → node UUID.

    Returns:
        Number of edges successfully inserted.
    """
    edges_stored = 0
    for rel in relationships:
        from_id = content_to_id.get(rel.get("from", ""))
        to_id = content_to_id.get(rel.get("to", ""))
        if from_id is None or to_id is None:
            continue
        try:
            memory.insert_edge(from_id, to_id, rel.get("type", "related"))
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
        stored_nodes = _upsert_entities(extraction.get("entities", []))
        content_to_id: dict[str, str] = {n["content"]: n["id"] for n in stored_nodes}
        edges_stored = _insert_relationships(extraction.get("relationships", []), content_to_id)
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
