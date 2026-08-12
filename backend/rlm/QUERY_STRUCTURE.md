# RLM Query Structure

The Recursive Language Model (RLM) engine exposes a `/rlm/query` endpoint for handling iterative, autonomous data-gathering and synthesis tasks. This document explains the query and response structure expected by the API.

## Endpoint
**POST** `/rlm/query`

## Request Payload

The request body should be a JSON object conforming to the `RLMQueryRequest` structure:

```json
{
  "context": "Context data representing the environment or knowledge base the RLM has access to.",
  "query": "The instruction or question for the RLM to solve.",
  "max_recursion_depth": 5,
  "max_total_llm_subcalls": 10,
  "max_wall_clock_seconds": 60.0,
  "max_total_tokens": 16000
}
```

### Fields

- **`context`** *(string, required)*: The data or environment description that the RLM will use to answer the query. This is often fetched from a memory graph or external knowledge base.
- **`query`** *(string, required)*: The specific goal, instruction, or question you want the RLM to resolve based on the context.
- **`max_recursion_depth`** *(integer, optional)*: Limits how deep the recursive sub-agent chain can go. Useful to prevent runaway recursion.
- **`max_total_llm_subcalls`** *(integer, optional)*: Caps the total number of LLM invocations the RLM engine can make in a single query.
- **`max_wall_clock_seconds`** *(float, optional)*: Timeout in seconds for the entire RLM orchestration.
- **`max_total_tokens`** *(integer, optional)*: The maximum amount of tokens allowed to be consumed by the orchestration loop.

---

## Response Payload

The API responds with a JSON object conforming to the `RLMQueryResponse` structure:

```json
{
  "answer": "The final synthesized answer produced by the RLM.",
  "stats": {
    "subcalls_made": 4,
    "recursion_depth_reached": 2,
    "tokens_used": 1250,
    "elapsed_seconds": 3.45,
    "capped": false
  }
}
```

### Fields

- **`answer`** *(string)*: The final result or answer returned by the engine after completing its reasoning.
- **`stats`** *(object)*: Statistics about the execution of the query.
  - **`subcalls_made`** *(integer)*: Total number of LLM subcalls made.
  - **`recursion_depth_reached`** *(integer)*: The maximum recursion depth reached.
  - **`tokens_used`** *(integer)*: Number of tokens used during execution.
  - **`elapsed_seconds`** *(float)*: Time taken for the query to resolve.
  - **`capped`** *(boolean)*: True if the query hit any of the specified safety limits (time, depth, subcalls, or tokens).
