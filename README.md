# Engram

AI-powered memory engine that stores knowledge as a causal graph and retrieves it using Ebbinghaus forgetting-curve decay.

## Overview

Engram transforms plain-text input into a structured knowledge graph. It uses an LLM to extract entities and relationships, persists them in PostgreSQL (via Supabase), and retrieves connected information through Dijkstra pathfinding with exponential decay weighting.

## Architecture

```
Browser (frontend/)
    └── HTTP API calls to FastAPI
        └── backend/main.py (FastAPI routes, CORS, error handling)
            ├── ai.py          (NVIDIA NIM — extraction, embeddings)
            ├── graph.py       (Dijkstra + Ebbinghaus decay)
            ├── memory.py      (Supabase / SQLite CRUD)
            ├── entity_resolution_v2.py  (multi-candidate entity resolution)
            └── upload.py      (PDF/PPT/TXT/DOC parsing)
```

## Features

- **Entity Extraction**: LLM-powered extraction of entities and relationships from text
- **Entity Resolution**: Multi-candidate resolution with MERGE / CREATE_NEW / RELATED_SUBENTITY / NEEDS_REVIEW outcomes — no silent duplicates
- **Graph Storage**: Nodes and edges stored in PostgreSQL (Supabase) with SQLite local fallback
- **Decay-Aware Retrieval**: Ebbinghaus exponential decay (`R = e^(-t/S)`) reduces edge weights over time
- **Shortest Path**: Dijkstra algorithm finds the most relevant causal paths
- **Insight Synthesis**: LLM generates non-obvious insights from retrieved paths

## Installation

```bash
# Clone the repository
git clone https://github.com/aradhyap21/Engram.git
cd Engram

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"
```

## Configuration

Create a `.env` file in the project root:

```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key
NVIDIA_API_KEY=your-nim-api-key
```

## Database Setup

Run the schema SQL in your Supabase SQL editor:

```sql
-- Enable UUID extension
create extension if not exists "uuid-ossp";

-- Nodes table
create table public.nodes (
    id uuid default uuid_generate_v4() primary key,
    content text not null,
    entity_type text,
    strength float default 1.0,
    access_count int default 0,
    created_at timestamp with time zone default timezone('utc'::text, now())
);

-- Edges table
create table public.edges (
    id uuid default uuid_generate_v4() primary key,
    from_id uuid references public.nodes(id),
    to_id uuid references public.nodes(id),
    relationship text,
    weight float default 1.0,
    created_at timestamp with time zone default timezone('utc'::text, now())
);

-- Indexes for performance
create index on public.nodes (content);
create index on public.edges (from_id);
create index on public.edges (to_id);
```

## Usage

```bash
# Start the server (from the project root)
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Liveness probe - returns `{"status": "ok"}` |
| `/memory` | POST | Store text as graph nodes and edges |
| `/memory/retrieve` | GET | Find top-5 causal paths for query |
| `/memory/synthesize` | GET | Generate insight from retrieved paths |

### Example

```bash
# Store a memory
curl -X POST http://localhost:8000/memory \
  -H "Content-Type: application/json" \
  -d '{"text": "The Industrial Revolution caused urbanization and led to the invention of the steam engine by James Watt."}'

# Retrieve related paths
curl http://localhost:8000/memory/retrieve?query=steam%20engine

# Synthesize an insight
curl http://localhost:8000/memory/synthesize?query=urbanization
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=backend

# Run integration test (requires NVIDIA_API_KEY)
python -m tests.test_resolution
```

## Project Structure

```
Engram/
├── backend/                    # Python FastAPI package
│   ├── __init__.py
│   ├── main.py                 # FastAPI application & routes
│   ├── ai.py                   # NVIDIA NIM (extraction + embeddings)
│   ├── graph.py                # Dijkstra + Ebbinghaus decay engine
│   ├── memory.py               # Supabase / SQLite data layer
│   ├── entity_resolution_v2.py # Multi-candidate entity resolution
│   ├── upload.py               # Document parsing (PDF/PPT/TXT/DOC)
│   ├── schema.sql              # Database schema
│   ├── .env.example            # Environment variable template
│   └── .gitignore
│
├── frontend/                   # Static UI
│   ├── index.html
│   ├── about.html
│   └── img_*.png
│
├── tests/                      # All test files
│   ├── __init__.py
│   ├── test_ai.py
│   ├── test_graph.py
│   ├── test_memory.py
│   ├── test_routes.py
│   └── test_resolution.py      # Entity resolution integration test
│
├── docs/
│   └── context.md              # Developer context & design notes
│
├── .env                        # Local secrets (gitignored)
├── .gitignore
├── pyproject.toml
└── README.md
```

## License

MIT License
