# Implementation Plan: MemoryMesh

## Overview

Build MemoryMesh incrementally: project scaffolding first, then the four Python modules from the inside out (data layer → graph engine → AI client → API routes), and finally the HTML frontend. Each task builds on the previous one and ends with everything wired together.

---

## Tasks

- [x] 1. Set up project structure, environment, and dependencies
  - Create the `memorymesh/` directory with all required files: `main.py`, `graph.py`, `memory.py`, `ai.py`, `index.html`, `.env`, `requirements.txt`
  - Populate `requirements.txt` with `fastapi`, `uvicorn`, `supabase`, `openai`, `python-dotenv`, `hypothesis`, `pytest`, `pytest-asyncio`, `httpx`
  - Add placeholder `.env` with the three required variable names (`SUPABASE_URL`, `SUPABASE_KEY`, `NVIDIA_API_KEY`) but no literal values
  - Add a `tests/` directory with `__init__.py`, `test_graph.py`, `test_ai.py`, `test_memory.py`, `test_routes.py`
  - _Requirements: 9.1, 9.2, 9.3, 9.4_

- [x] 2. Implement the Supabase data layer (`memory.py`)
  - [x] 2.1 Implement Supabase client initialisation and all CRUD functions
    - Load `SUPABASE_URL` and `SUPABASE_KEY` from `.env` using `python-dotenv`; raise a clear `RuntimeError` if either is missing
    - Implement `upsert_node(content, entity_type) -> dict` — insert or return existing node by `content`; new nodes get `strength=1.0`, `access_count=0`
    - Implement `insert_edge(from_id, to_id, relationship, weight=1.0) -> dict` — raise `ValueError` before any DB call when `from_id == to_id`
    - Implement `get_all_nodes() -> list[dict]` and `get_all_edges() -> list[dict]`
    - Implement `update_node_strength(node_id, new_strength, new_access_count) -> None`
    - Implement `update_edge_weight(edge_id, new_weight) -> None`
    - Wrap all `supabase-py` calls in try/except and re-raise as descriptive exceptions
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

  - [ ]* 2.2 Write unit tests for `memory.py`
    - Mock the `supabase-py` client; test `upsert_node` idempotency (same content → same UUID, no duplicate row)
    - Test `insert_edge` raises `ValueError` for self-loop (`from_id == to_id`)
    - Test that Supabase errors are re-raised as descriptive exceptions
    - _Requirements: 6.4, 6.5_

- [x] 3. Implement the Ebbinghaus decay engine and Dijkstra graph module (`graph.py`)
  - [x] 3.1 Implement `apply_decay(edges, nodes) -> list[dict]`
    - Use only `math`, `datetime`, `timezone` from the standard library
    - Formula: `decayed_weight = weight * exp(-days_elapsed / strength)`
    - Clamp `strength` to minimum `0.001`; derive `days_elapsed` from `edge.created_at`
    - Return a new list (do not mutate inputs); add `decayed_weight` field to each edge copy
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

  - [ ]* 3.2 Write property test for `apply_decay` — Property 1: Decay monotonicity
    - **Property 1: Decay monotonicity** — for any valid edge and positive `Δt`, `decayed_weight(t + Δt) <= decayed_weight(t)`
    - **Validates: Requirements 5.4, 2.2**
    - Use `hypothesis` strategies to generate arbitrary weights, strengths, and timestamps

  - [ ]* 3.3 Write property test for `apply_decay` — Property 5: Decay at t=0 equals original weight
    - **Property 5 (from Req 5.2):** Given `t = 0`, `decayed_weight == original_weight`
    - **Validates: Requirements 5.2**
    - Generate edges whose `created_at` equals `now` and verify no decay

  - [x] 3.4 Implement `build_adjacency(nodes, edges) -> dict`
    - Build adjacency list keyed by node UUID: `{node_id: [(neighbour_id, decayed_weight, edge_id), ...]}`
    - Use `decayed_weight` field (must call `apply_decay` before `build_adjacency`)
    - _Requirements: 4.1, 2.3_

  - [x] 3.5 Implement `dijkstra(adjacency, source_ids) -> dict`
    - Multi-source Dijkstra using only `heapq`; initialise all sources at cost `0.0`
    - Return `{node_id: (total_cost, path_list)}` for all reachable nodes; skip unreachable nodes silently
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

  - [ ]* 3.6 Write property test for `dijkstra` — Property 3: Dijkstra correctness
    - **Property 3: Dijkstra correctness** — for any generated connected graph with positive weights, no alternative path has lower total cost than the returned shortest path
    - **Validates: Requirements 4.2, 2.3**
    - Generate random small graphs with `hypothesis`; verify optimality by brute-force comparison on small inputs

  - [x] 3.7 Implement `top_paths(nodes, edges, query_entities, k=5) -> list[dict]`
    - Orchestrate: `apply_decay` → `build_adjacency` → `dijkstra` → rank → return top-k paths
    - Match `query_entities` (strings) to node UUIDs by `content` field; handle zero matches gracefully (return `[]`)
    - Increment `strength += 0.1` and `access_count += 1` for touched source nodes; call `update_node_strength` and `update_edge_weight` for each accessed edge
    - Return list of dicts with keys `path`, `edges`, `total_cost`; return `[]` if graph is empty or no paths found
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 4.1–4.5_

  - [ ]* 3.8 Write unit tests for `graph.py`
    - Test `apply_decay` at t=0 → weight unchanged; at `t = S * ln(2)` → weight halved
    - Test `dijkstra` on a simple 3-node chain; test with unreachable nodes; test multi-source selects minimum cost
    - Test `top_paths` returns `[]` for empty graph and for query with no matching entities
    - _Requirements: 5.2, 5.3, 4.2, 4.3, 4.4, 2.6_

- [x] 4. Checkpoint — core graph logic verified
  - Ensure all tests in `test_graph.py` pass. Ask the user if any algorithmic questions arise.

- [x] 5. Implement the NVIDIA NIM AI client (`ai.py`)
  - [x] 5.1 Implement `extract_entities(text: str) -> dict`
    - Load `NVIDIA_API_KEY` from environment; raise clear error if missing
    - Build the JSON-extraction prompt; call `client.chat.completions.create()` with `meta/llama-3.1-70b-instruct`
    - Parse JSON from response; attempt regex fallback to extract JSON substring if the model adds prose
    - Raise `ValueError("LLM returned non-JSON response")` if both attempts fail
    - Return `{"entities": [...], "relationships": [{"from": ..., "to": ..., "type": ...}, ...]}` — never a partial structure
    - _Requirements: 1.1, 7.4_

  - [ ]* 5.2 Write property test for `extract_entities` — Property 7: JSON contract
    - **Property 7: JSON contract** — `extract_entities` always returns a dict with both `"entities"` and `"relationships"` keys, or raises `ValueError`; it never returns `None` or a partial structure
    - **Validates: Requirements 1.1**
    - Mock the OpenAI client with `hypothesis`-generated strings (valid JSON, invalid JSON, empty); verify the contract holds

  - [x] 5.3 Implement `synthesize_insight(paths: list[dict], query: str) -> str`
    - Build a synthesis prompt that includes the serialised paths and query string; handle empty-paths case with explicit "no paths found" language
    - Call the NIM API and return the raw text response
    - _Requirements: 3.2, 3.4_

  - [ ]* 5.4 Write unit tests for `ai.py`
    - Mock `openai.OpenAI`; test valid JSON extraction, regex fallback path, pure-non-JSON raises `ValueError`
    - Test `synthesize_insight` with empty `paths` list produces a non-error string response
    - _Requirements: 1.1, 3.2, 3.4_

- [x] 6. Implement the FastAPI application (`main.py`)
  - [x] 6.1 Bootstrap FastAPI app with CORS middleware and environment loading
    - Create FastAPI app instance; mount `CORSMiddleware` with `allow_origins=["*"]`, `allow_methods=["*"]`, `allow_headers=["*"]`
    - Load all env vars via `python-dotenv` at startup; raise clear startup error if any are missing
    - Add `GET /health` route returning `{"status": "ok"}`
    - _Requirements: 7.1, 7.2, 7.4, 7.5_

  - [x] 6.2 Implement `POST /memory` route
    - Accept `{"text": str}` body; validate `text` is non-empty
    - Call `ai.extract_entities(text)` → upsert nodes via `memory.upsert_node` → insert edges via `memory.insert_edge`
    - Return `{"nodes_stored": int, "edges_stored": int, "node_ids": [...]}`
    - Wrap in try/except; return `{"error": "..."}` with appropriate HTTP status on any failure
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 7.3_

  - [x] 6.3 Implement `GET /memory/retrieve` route
    - Accept `query` query-param; call `ai.extract_entities(query)` then `graph.top_paths()`
    - Return `{"query": str, "paths": [...]}` — use HTTP 200 with empty paths list when no results found
    - Wrap in try/except with `{"error": "..."}` envelope
    - _Requirements: 2.1–2.6, 7.3_

  - [x] 6.4 Implement `GET /memory/synthesize` route
    - Call the retrieve pipeline internally; pass paths (even empty) to `ai.synthesize_insight()`
    - Return `{"query": str, "insight": str, "paths_used": int}`
    - Wrap in try/except with `{"error": "..."}` envelope
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 7.3_

  - [ ]* 6.5 Write route integration tests (`test_routes.py`)
    - Use FastAPI `TestClient` (via `httpx`); mock `ai`, `memory`, and `graph` modules
    - Test POST /memory happy path and error envelope (LLM failure, DB failure)
    - Test GET /memory/retrieve with paths and with empty result
    - Test GET /memory/synthesize with paths and with empty paths
    - Test GET /health returns `{"status": "ok"}`
    - _Requirements: 1.4, 1.5, 2.6, 3.3, 3.4, 7.1, 7.2, 7.3_

- [x] 7. Checkpoint — full backend wired and tested
  - Ensure all tests pass (`pytest tests/`). Ask the user if any route or integration questions arise.

- [ ] 8. Build the single-file HTML frontend (`index.html`)
  - [ ] 8.1 Create the HTML/CSS skeleton with dark theme
    - Write `<!DOCTYPE html>` page with inline `<style>` block; dark background (`#1a1a2e` or similar), light text, card component
    - Add `<textarea>` + "Store Memory" button, search `<input>` + "Retrieve" button + "Synthesize" button
    - Add a result `<div id="result-card">` that shows output below the controls
    - No external CSS frameworks, no npm, no bundler — pure HTML/CSS only
    - _Requirements: 8.1, 8.2, 8.4_

  - [ ] 8.2 Implement JavaScript API calls and result rendering
    - Write `storeMemory()`, `retrieveMemory()`, `synthesizeMemory()` functions using `fetch()` targeting `http://localhost:8000`
    - On success, render paths as readable `node → [relationship] → node` chains inside the result card; render synthesis insight as plain text
    - On fetch or API error, display the error message inside the result card — never silently swallow errors
    - Ensure the file opens correctly via `file://` protocol (no `type="module"` imports, no relative server paths)
    - _Requirements: 8.2, 8.3, 8.5, 8.6_

- [ ] 9. Final integration and wiring
  - [ ] 9.1 Verify end-to-end flow
    - Confirm `main.py` imports from `ai`, `memory`, `graph` with no circular dependencies
    - Confirm `.env` is not committed (add `.env` to `.gitignore` if a git repo exists)
    - Confirm `requirements.txt` lists every dependency used across all modules
    - _Requirements: 9.1, 9.2, 9.3, 9.4_

  - [ ]* 9.2 Write end-to-end smoke test
    - Using `TestClient` with real (or local) Supabase test credentials and mocked NVIDIA NIM, store a 3-entity text, retrieve one of those entities, verify a path is returned, verify `strength` increased after retrieval
    - _Requirements: 1.2, 1.3, 2.4, 2.5_

- [ ] 10. Final checkpoint — all tests pass, project is runnable
  - Run `pytest tests/` and confirm all tests pass
  - Verify the app starts with `uvicorn main:app --reload` from the `memorymesh/` directory
  - Ask the user if any remaining questions arise before handoff.

---

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP
- Each task references specific requirements for full traceability
- Property tests use the `hypothesis` library; unit/integration tests use `pytest` + `httpx`
- The NVIDIA NIM API is behind the OpenAI-compatible SDK — mock `openai.OpenAI` in all unit tests
- Strength increment and edge weight persistence (Req 2.5) happen inside `graph.top_paths()`, not inside the route handler
- All secrets must come from `.env`; never hardcode API keys or URLs

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1"] },
    { "id": 1, "tasks": ["2.1"] },
    { "id": 2, "tasks": ["2.2", "3.1"] },
    { "id": 3, "tasks": ["3.2", "3.3", "3.4"] },
    { "id": 4, "tasks": ["3.5"] },
    { "id": 5, "tasks": ["3.6", "3.7"] },
    { "id": 6, "tasks": ["3.8", "5.1"] },
    { "id": 7, "tasks": ["5.2", "5.3"] },
    { "id": 8, "tasks": ["5.4", "6.1"] },
    { "id": 9, "tasks": ["6.2", "6.3", "6.4"] },
    { "id": 10, "tasks": ["6.5", "8.1"] },
    { "id": 11, "tasks": ["8.2"] },
    { "id": 12, "tasks": ["9.1"] },
    { "id": 13, "tasks": ["9.2"] }
  ]
}
```
