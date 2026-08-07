# MemoryMesh Codebase Context
## Overview
MemoryMesh is an AI-powered memory engine that stores knowledge as a causal graph and retrieves it using Ebbinghaus forgetting-curve decay. Plain-text input is processed by an LLM (NVIDIA NIM / Llama-3.1-70b-instruct) to extract entities and relationships, which are persisted in Supabase (PostgreSQL). On retrieval, a pure-Python Dijkstra traversal finds the shortest causal paths through the graph, edge weights are decayed by `R = e^(-t/S)`, and the resulting paths are optionally synthesized by the LLM into non-obvious insights.

## Key Implementation Highlights
- ✅ **Graph Engine Implemented**: `graph.py` now contains:
  - Ebbinghaus decay algorithm with strength clamping
  - Multi-source Dijkstra's algorithm using heapq
  - Path retrieval with decay-aware edge weights
  - Side-effect tracking for node strength/edge weight updates
- ✅ **Comprehensive Testing**: `test_graph.py` now includes unit tests for all graph operations
- ✅ **Full Data Flow**: End-to-end functionality from text input → entity extraction → graph storage → decay-aware retrieval → insight synthesis

## File Structure
```
memorymesh/
├── main.py        # FastAPI entry point
├── ai.py          # NVIDIA NIM API client
├── graph.py       # Graph engine - apply_decay(), build_adjacency(), dijkstra(), top_paths()
├── memory.py      # Supabase data layer
├── index.html     # Single-file vanilla JS frontend
├── requirements.txt # fastapi, uvicorn[standard], supabase, openai, python-dotenv, hypothesis, pytest, pytest-asyncio, httpx
├── .gitignore     # Ignores .env, __pycache__, .pytest_cache, etc.
└── tests/
    ├── __init__.py
    ├── test_graph.py   # Unit tests for graph.py (comprehensive)
    ├── test_ai.py      # Empty stub (intentionally not implemented)
    ├── test_memory.py  # Empty stub (intentionally not implemented)
    └── test_routes.py  # Empty stub (intentionally not implemented)
```

## Module Responsibilities
### main.py (FastAPI Application)
- Entry point for running: `uvicorn main:app --reload` from memorymesh/ directory
- Loads DOTENV at startup - requires SUPABASE_URL, SUPABASE_KEY, NVIDIA_API_KEY
- Mounts CORSMiddleware with allow_origins/methods/headers = ["*"]
- Routes:
  - `GET /health` → `{"status": "ok"}`
  - `POST /memory` (MemoryRequest: text: str) → stores entities/relationships, returns `{nodes_stored, edges_stored, node_ids}`
  - `GET /memory/retrieve?query=str` → returns `{query, paths: [...]}`
  - `GET /memory/synthesize?query=str` → returns `{query, insight, paths_used}`

### ai.py (NVIDIA NIM Client)
- Uses OpenAI-compatible SDK with base_url = "https://integrate.api.nvidia.com/v1"
- Model: meta/llama-3.1-70b-instruct
- extract_entities(text) → `{entities: [...], relationships: [{from, to, type}]}`
  - Parses JSON from LLM response, with regex fallback for extracting JSON block
  - Raises ValueError if response cannot be parsed
- synthesize_insight(paths, query) → plain-text insight string

### memory.py (Supabase Data Layer)
- Uses supabase-py client exclusively - no raw SQL
- Table schemas:
  - **nodes**: id (UUID), content (text), entity_type (text), strength (float, default 1.0), access_count (int, default 0), created_at (timestamp)
  - **edges**: id (UUID), from_id (UUID→nodes), to_id (UUID→nodes), relationship (text), weight (float, default 1.0), created_at (timestamp)
- Key functions:
  - `upsert_node(content, entity_type)` - returns existing node by content or inserts new with strength=1.0, access_count=0
  - `insert_edge(from_id, to_id, relationship, weight=1.0)` - raises ValueError for self-loops (from_id == to_id)
  - `get_all_nodes()`, `get_all_edges()` - return all rows or empty list
  - `update_node_strength(node_id, new_strength, new_access_count)`
  - `update_edge_weight(edge_id, new_weight)`

### graph.py (Dijkstra + Decay Engine)
- Pure Python (heapq, math, datetime only - no networkx/scipy)
- Core algorithms:
  - **apply_decay(edges, nodes)**: Computes `decayed_weight = weight * exp(-days_elapsed / strength)`
    - days_elapsed from edge["created_at"] (ISO-8601) vs now(UTC)
    - Strength clamped to minimum 0.001 to prevent division by zero
    - Returns new list (doesn't mutate inputs), adds "decayed_weight" field
  - **build_adjacency(nodes, edges)**: Returns `{node_id: [(neighbour_id, decayed_weight, edge_id), ...]}`
  - **dijkstra(adjacency, source_ids)**: Multi-source Dijkstra
    - Initializes all source_ids with cost 0.0
    - Returns `{node_id: (total_cost, path_list)}` for reachable nodes only
    - Unreachable nodes silently omitted
  - **top_paths(nodes, edges, query_entities, k=5)**: Orchestration facade
    - Applies decay → builds adjacency → matches query entities to node UUIDs (case-insensitive) → runs Dijkstra
    - Side effects: increments strength += 0.1 and access_count += 1 for source nodes, persists decayed edge weights
    - Returns top-k paths sorted ascending by total_cost, each with `{path, edges, total_cost}`
    - Returns `[]` if no matches or empty graph

## Data Models
### Node (in Supabase)
```python
{
    "id": str (UUID),
    "content": str,       # Entity text
    "entity_type": str,   # person/concept/event/etc
    "strength": float,    # Ebbinghaus stability S, starts at 1.0
    "access_count": int,  # Incremented on retrieval
    "created_at": str     # ISO-8601
}
```

### Edge (in Supabase)
```python
{
    "id": str (UUID),
    "from_id": str (UUID),
    "to_id": str (UUID),
    "relationship": str,  # e.g., "causes", "influences"
    "weight": float,      # Default 1.0
    "created_at": str,    # ISO-8601
    "decayed_weight": float # Computed at read time, NOT stored
}
```

## API Response Shapes
```python
# POST /memory
{
    "nodes_stored": int,
    "edges_stored": int,
    "node_ids": list[str]
}

# GET /memory/retrieve
{
    "query": str,
    "paths": [{
        "path": [{"id", "content", "entity_type"}, ...],
        "edges": [{"relationship", "decayed_weight"}, ...],
        "total_cost": float
    }, ...]
}

# GET /memory/synthesize
{
    "query": str,
    "insight": str,
    "paths_used": int
}
```

## Key Algorithms
### Ebbinghaus Decay Formula
```
decayed_weight = weight * exp(-days_elapsed / S)
```
Where:
- days_elapsed = (now - edge.created_at) in days
- S = source node's strength (clamped to min 0.001)

### Dijkstra Implementation
- Multi-source initialization: all sources start at cost 0.0
- Uses heapq for min-heap priority queue
- Returns shortest path from nearest source to each reachable node
- Unreachable nodes silently omitted

## Environment Configuration
- Requires .env (never hardcode values):
```
SUPABASE_URL=<your-supabase-url>
SUPABASE_KEY=<your-supabase-anon-key>
NVIDIA_API_KEY=<your-nvidia-nim-api-key>
```

## Testing
- Run tests from memorymesh/ directory:
```bash
pytest tests/
```
- Test infrastructure uses hypothesis for property-based testing
- Mocks supabase-py and openai.OpenAI clients

## Key Requirements Traceability
| Req | Description | Implemented In |
|-----|-------------|----------------|
| 1.x | Memory Storage | main.py:store_memory, ai.py, memory.py |
| 2.x | Causal Graph Retrieval | main.py:retrieve_memory, graph.py:top_paths |
| 3.x | LLM Insight Synthesis | main.py:synthesize_memory, ai.py:synthesize_insight |
| 4.x | Pure-Python Dijkstra | graph.py:dijkstra |
| 5.x | Ebbinghaus Decay | graph.py:apply_decay |
| 6.x | Supabase Data Layer | memory.py:all functions |
| 7.x | FastAPI + CORS | main.py |
| 8.x | HTML Frontend | index.html |
| 9.x | Project Structure | All files |

## Common Patterns
### Error Handling in Routes
All route handlers wrap logic in try/except and return JSON errors with appropriate HTTP codes:
- ValueError → 422
- RuntimeError → 503
- Generic Exception → 500

### Circular Import Prevention
graph.py imports memory inside top_paths() function to avoid circular import at module load time.

## Frontend (index.html)
- Dark-themed single-file vanilla HTML/CSS/JS (no frameworks)
- Connects to http://localhost:8000
- Features:
  - Store Memory textarea + button
  - Search input + Retrieve/Synthesize buttons
  - Result card showing paths as node-edge-node chains
  - Animated Millennium Falcon easter egg
  - Responsive design with mobile breakpoints