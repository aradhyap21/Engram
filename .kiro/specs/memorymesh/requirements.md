# Requirements Document

## Introduction

MemoryMesh is an AI memory engine that represents knowledge as a causal graph and retrieves it through Ebbinghaus decay-weighted Dijkstra traversal. Users store plain-text memories, which an LLM parses into entities and relationships, and later retrieve causally linked memory paths or synthesised insights. The system is built on FastAPI (Python), Supabase (PostgreSQL), NVIDIA NIM (Llama-3.1-70b-instruct), and a single-file vanilla HTML/JS frontend.

---

## Requirements

### Requirement 1: Memory Storage Pipeline

**User Story**: As a user, I want to submit plain text and have the system automatically extract entities and relationships, so that my knowledge is stored as a queryable causal graph.

#### Acceptance Criteria

1. Given a non-empty text body in a POST /memory request, the system SHALL call the NVIDIA NIM API with a JSON-extraction prompt and receive a response containing "entities" and "relationships" arrays.

2. Given a successful entity extraction, the system SHALL upsert each entity as a node in the nodes Supabase table with strength = 1.0 and access_count = 0 on first insertion; a second upsert of the same entity content SHALL NOT create a duplicate row.

3. Given a successful entity extraction, the system SHALL insert each relationship as an edge in the edges Supabase table with weight = 1.0, linking the correct from_id and to_id UUIDs.

4. Given a successful store operation, POST /memory SHALL return a JSON response containing nodes_stored (int), edges_stored (int), and node_ids (list of UUID strings).

5. Given any exception during storage such as Supabase failure, LLM failure, or JSON parse error, the endpoint SHALL return a JSON body with an "error" key and an appropriate HTTP error status code (4xx or 5xx); it SHALL NOT return an unstructured error or crash the server.

---

### Requirement 2: Causal Graph Retrieval with Ebbinghaus Decay

**User Story**: As a user, I want to retrieve the most causally relevant memory paths for a query, with older or less-accessed memories naturally fading, so that the most recently reinforced knowledge is prioritised.

#### Acceptance Criteria

1. Given a GET /memory/retrieve?query=... request, the system SHALL extract query entities via the NVIDIA NIM API and use them as Dijkstra source nodes.

2. Given all nodes and edges loaded from Supabase, the system SHALL compute a decayed weight for every edge using the formula decayed_weight = weight * exp(-days_elapsed / strength), where days_elapsed is derived from edge.created_at and strength is the source node's current strength value.

3. Given decayed weights, the system SHALL run a pure-Python multi-source Dijkstra using only heapq from the standard library, treating decayed_weight as the edge cost where lower means closer or stronger.

4. Given Dijkstra results, the system SHALL return the top-5 shortest-cost paths sorted ascending by total_cost, each containing the sequence of nodes and edges traversed.

5. Given a retrieval that touches node n, the system SHALL increment n.strength by 0.1 and n.access_count by 1 in Supabase, making the node more resistant to future decay.

6. Given a query that matches no known entities or an empty graph, the system SHALL return {"paths": []} with HTTP 200 and not an error response.

---

### Requirement 3: LLM Insight Synthesis

**User Story**: As a user, I want to ask a question and receive a synthesised insight that connects non-obvious dots across my stored memories, so that I can discover knowledge I did not know I had.

#### Acceptance Criteria

1. Given a GET /memory/synthesize?query=... request, the system SHALL internally run the full retrieval pipeline from Requirement 2 to obtain memory paths before calling the LLM.

2. Given retrieved paths, the system SHALL send a synthesis prompt to the NVIDIA NIM API that includes the serialised paths and the original query string.

3. Given a successful synthesis call, the system SHALL return a JSON response containing query (string), insight (non-empty string), and paths_used (int count of paths sent to the LLM).

4. Given zero retrieved paths, the system SHALL still call the LLM with an indication that no paths were found and return a response that explains the empty state rather than an error.

---

### Requirement 4: Pure-Python Dijkstra Implementation

**User Story**: As a developer, I want a graph shortest-path implementation with no third-party graph libraries, so that the codebase has minimal dependencies and the algorithm is fully auditable.

#### Acceptance Criteria

1. The Dijkstra implementation SHALL use only Python standard library modules (heapq, math, datetime) and SHALL NOT import networkx, scipy, or any other external graph library.

2. Given a connected graph with positive edge weights, the Dijkstra function SHALL return the minimum-cost path from any source node to every reachable node.

3. Given a disconnected graph, Dijkstra SHALL return results only for reachable nodes and SHALL NOT raise an exception for unreachable nodes.

4. Given multiple source nodes, the algorithm SHALL initialise all sources with cost 0.0 simultaneously and return the minimum distance from any source to each reachable node.

5. Given positive edge costs where all decayed weights are greater than 0 due to the exponential function, the algorithm SHALL always terminate.

---

### Requirement 5: Ebbinghaus Decay Engine

**User Story**: As a user, I want memories to fade naturally over time unless I revisit them, mirroring how human memory works, so that frequently accessed memories remain stronger.

#### Acceptance Criteria

1. The decay function SHALL compute decayed_weight = original_weight * exp(-t / S) where t is days elapsed since edge.created_at and S is the source node's strength value.

2. Given t = 0, the decayed weight SHALL equal the original weight, meaning no decay occurs at creation time.

3. Given t = S * ln(2), the decayed weight SHALL equal approximately half the original weight, satisfying the half-life property.

4. Given any t greater than 0 and a fixed S, the decayed weight SHALL be strictly less than the original weight, satisfying the monotonic decay property.

5. The strength value used in the decay calculation SHALL be clamped to a minimum of 0.001 to prevent division by zero.

---

### Requirement 6: Supabase Data Layer

**User Story**: As a developer, I want all database interactions encapsulated in a single module using the supabase-py client, so that the data layer is swappable and testable in isolation.

#### Acceptance Criteria

1. All Supabase interactions SHALL use the supabase-py client exclusively; no raw SQL strings or other database drivers SHALL be used.

2. The nodes table SHALL have columns: id as UUID primary key, content as text, entity_type as text, strength as float defaulting to 1.0, access_count as int defaulting to 0, and created_at as timestamp.

3. The edges table SHALL have columns: id as UUID primary key, from_id as UUID FK referencing nodes.id, to_id as UUID FK referencing nodes.id, relationship as text, weight as float defaulting to 1.0, and created_at as timestamp.

4. An insert_edge call with from_id equal to to_id SHALL raise a ValueError before any Supabase call is made, preventing self-loop edges.

5. All Supabase errors SHALL be caught in the data layer module and re-raised as descriptive exceptions that main.py can catch and convert to JSON error responses.

---

### Requirement 7: FastAPI Application and CORS

**User Story**: As a developer, I want a FastAPI app with CORS enabled so that the local single-file HTML frontend can call the API without browser security errors.

#### Acceptance Criteria

1. The FastAPI app SHALL mount CORSMiddleware with allow_origins=["*"], allow_methods=["*"], and allow_headers=["*"].

2. The app SHALL expose the following routes: POST /memory, GET /memory/retrieve, GET /memory/synthesize, and GET /health returning {"status": "ok"}.

3. Every route handler SHALL wrap its logic in a try/except block and return {"error": "message"} with an appropriate HTTP status code on any unhandled exception.

4. The app SHALL load SUPABASE_URL, SUPABASE_KEY, and NVIDIA_API_KEY from a .env file using python-dotenv; missing variables SHALL cause a clear startup error.

5. The app SHALL be runnable with uvicorn main:app --reload from the memorymesh/ directory.

---

### Requirement 8: Single-File HTML Frontend

**User Story**: As a user, I want a clean dark-themed web interface where I can store memories, retrieve related paths, and synthesise insights without installing any frontend tooling.

#### Acceptance Criteria

1. The frontend SHALL be a single file index.html using only vanilla HTML, CSS, and JavaScript with no npm packages, bundlers, or external JS frameworks.

2. The UI SHALL include a textarea and Store Memory button calling POST /memory, a search input and Retrieve button calling GET /memory/retrieve, and a Synthesize button calling GET /memory/synthesize.

3. Results SHALL be displayed in a styled card element below the controls, showing paths as readable node-edge-node chains or the synthesised insight text.

4. The UI SHALL use a dark colour scheme and be functional on modern desktop browsers including Chrome, Firefox, and Edge.

5. All API calls SHALL use fetch() with proper error handling; network or API errors SHALL be displayed to the user in the result card and not silently swallowed.

6. The index.html file SHALL be openable directly in a browser via file:// protocol and connect to the FastAPI server at http://localhost:8000.

---

### Requirement 9: Environment Configuration and Project Structure

**User Story**: As a developer, I want a clean project structure with all configuration in a .env file so that the project is easy to set up and credentials are never hardcoded.

#### Acceptance Criteria

1. The project SHALL have the following files: memorymesh/main.py, memorymesh/graph.py, memorymesh/memory.py, memorymesh/ai.py, memorymesh/index.html, memorymesh/.env, and memorymesh/requirements.txt.

2. The requirements.txt SHALL include: fastapi, uvicorn, supabase, openai, and python-dotenv.

3. The .env file SHALL define three variables: SUPABASE_URL, SUPABASE_KEY, and NVIDIA_API_KEY; their values SHALL NOT appear in any source file as literals.

4. No credentials, API keys, or environment-specific values SHALL appear as literals in any .py or .html source file.

---

## Glossary

| Term | Definition |
|------|-----------|
| Node | A memory entity stored in Supabase, representing a person, concept, or event extracted from input text. |
| Edge | A causal or relational link between two nodes, with a decaying weight. |
| Strength (S) | The Ebbinghaus stability parameter for a node; increases on access, making the node more decay-resistant. |
| Decayed weight | The current effective edge weight after applying the Ebbinghaus exponential decay formula. |
| Dijkstra | A shortest-path graph algorithm used to find causally closest memory nodes. |
| NVIDIA NIM | NVIDIA's inference microservices platform; provides LLM access via OpenAI-compatible API. |
| Supabase | A hosted PostgreSQL platform used as the persistent store for nodes and edges. |
| Synthesis | An LLM-generated insight derived from retrieved memory paths, connecting non-obvious relationships. |
