# Software Metrics Analysis Report

---

## Section 1 — Student & Project Details

| Field                 | Details                                               |
| --------------------- | ----------------------------------------------------- |
| **Project Title**     | Engram — AI-powered Memory Engine                 |
| **Source**            | Own project                                           |
| **GitHub Link**       | *(local workspace)*                                   |
| **Primary Languages** | Python 3.10+                                          |

### Project Description

Engram is an AI-powered memory engine that stores knowledge as a causal graph using Ebbinghaus forgetting-curve decay. It provides a FastAPI-based REST API for storing memories, retrieving memory paths using a custom Python implementation of Dijkstra's algorithm, and synthesizing insights via NVIDIA NIM (LLM).

The system consists of several core modules:
- **`graph.py`** — Pure-Python implementation of Ebbinghaus decay, graph adjacency building, Dijkstra shortest-path, and top-path orchestration.
- **`ai.py`** — NVIDIA NIM API client for entity extraction and LLM insight synthesis.
- **`main.py`** — FastAPI endpoints (`/memory`, `/memory/retrieve`, `/memory/synthesize`).
- **`memory.py`** — Data persistence and local caching abstraction.

---

## Section 2 — Tools & Metrics Used

### Tools Selected

| Tool      | Version | Role                                                                                   |
| --------- | ------- | -------------------------------------------------------------------------------------- |
| **Radon** | 6.0.1   | Primary analysis tool — Cyclomatic Complexity (CC), Raw Metrics (LOC/SLOC), Halstead   |

### Justification

- **Radon** natively supports Python and is the standard industry tool for measuring Cyclomatic Complexity (via `radon cc`), raw code metrics (via `radon raw`), and Halstead metrics (via `radon hal`). It provides direct output for the metrics required without needing to manually calculate token density.
- A custom PowerShell script (`generate_radon_report.ps1`) was used to automate the generation of all three reports simultaneously into a single file for reliable tracking.

### Metrics Captured

| Metric                               | Tool    | What it measures                                                |
| ------------------------------------ | ------- | --------------------------------------------------------------- |
| **LOC / SLOC**                       | Radon   | Lines of Code and Source Lines of Code (Raw Metrics)            |
| **CCN** (Cyclomatic Complexity)      | Radon   | Number of independent paths through a function / branching      |
| **Halstead Volume (V)**              | Radon   | The information content of the program (size of the vocabulary) |
| **Halstead Difficulty (D)**          | Radon   | The error proneness of the program / difficulty to write        |
| **Halstead Effort (E)**              | Radon   | Mental effort required to develop or maintain the function      |

---

## Section 3 — Results

### 3.1 Raw Metrics & Complexity — BEFORE Refactoring

**Command run:**
```powershell
python -m radon cc memorymesh -s -a
python -m radon raw memorymesh -s
python -m radon hal memorymesh
```

#### Per-file Summary (BEFORE)

| File                 | LOC | SLOC | Functions | Avg CCN | Max CCN |
| -------------------- | --- | ---- | --------- | ------- | ------- |
| `memorymesh/ai.py`   | 144 | 76   | 2         | ~6.5    | 10      |
| `memorymesh/graph.py`| 351 | 139  | 4         | ~10.5   | 24      |
| `memorymesh/main.py` | 259 | 147  | 4         | ~6.0    | 13      |
| `memorymesh/memory.py`| 353 | 266  | 14        | ~2.7    | 7       |
| **TOTAL (Overall)**  | **2395** | **1426** | **122** | **3.15** | **24** |

#### High-Complexity Functions (BEFORE) — CCN threshold > 5

| CCN | Function              | File                  |
| --- | --------------------- | --------------------- |
| **24** | `top_paths()`         | `memorymesh/graph.py` |
| **13** | `store_memory()`      | `memorymesh/main.py`  |
| **10** | `dijkstra()`          | `memorymesh/graph.py` |
| **10** | `extract_entities()`  | `memorymesh/ai.py`    |

---

### 3.2 Halstead Metrics (BEFORE Refactoring) — Key Files

| File                  | Length (N) | Vocab (η) | Volume (V) | Difficulty (D) | Effort (E) | Time (T) |
| --------------------- | ---------- | --------- | ---------- | -------------- | ---------- | -------- |
| `memorymesh/graph.py` | 82         | 60        | 484.3      | 7.33           | **3550.3** | 197.2    |
| `memorymesh/memory.py`| 29         | 21        | 127.3      | 2.81           | 358.2      | 19.9     |
| `memorymesh/ai.py`    | 27         | 19        | 114.7      | 2.26           | 259.9      | 14.4     |
| `memorymesh/main.py`  | 25         | 18        | 104.2      | 2.14           | 223.3      | 12.4     |

> **Interpretation:** The `graph.py` file stands out heavily. Its Halstead Effort is nearly 10x higher than any other core module, directly reflecting the cognitive density of the monolithic `top_paths` and `dijkstra` functions.

---

### 3.3 Raw Metrics & Complexity — AFTER Refactoring

**Files refactored:** `graph.py`, `ai.py`, `main.py`

#### Refactored File Summary (AFTER)

| File                  | LOC | SLOC | Functions | Avg CCN | Max CCN |
| --------------------- | --- | ---- | --------- | ------- | ------- |
| `memorymesh/ai.py`    | 168 | 67   | 5         | 2.8     | **5**   |
| `memorymesh/graph.py` | 332 | 134  | 7         | 6.1     | **10**  |
| `memorymesh/main.py`  | 263 | 139  | 6         | 3.6     | **8**   |

#### High-Complexity Functions (AFTER) — CCN Comparison

| Function              | File                  | CCN (Before) | CCN (After) | Change |
| --------------------- | --------------------- | ------------ | ----------- | ------ |
| `top_paths()`         | `memorymesh/graph.py` | **24 (D)**   | **9 (B)**   | **-62%** |
| `store_memory()`      | `memorymesh/main.py`  | **13 (C)**   | **8 (B)**   | **-38%** |
| `extract_entities()`  | `memorymesh/ai.py`    | **10 (B)**   | **1 (A)**   | **-90%** |
| `_validate_extraction`| `memorymesh/ai.py`    | *(new)*      | 5 (A)       | N/A    |

#### Project-Wide Averages

| Metric                        | Before | After   | Change |
| ----------------------------- | ------ | ------- | ------ |
| Total LOC                     | 2395   | 2404    | +9     |
| Average Complexity (All code) | 3.15   | **2.99**| ↓      |

---

## Section 4 — Analysis & Interpretation

### 4.1 Highly Complex Functions and Why

**`top_paths()` in `graph.py` — CCN 24 (Before)**
This function was a classic "God Function". It orchestrated the entire retrieval pipeline: applying decay, building adjacency lists, matching entities via case-insensitive loops, invoking Dijkstra, handling Supabase persistence for node strengths, handling Supabase persistence for edge weights, and mapping internal UUIDs back to content objects. The massive CCN of 24 came from multiple nested loops combined with inline `try/except` persistence blocks.

**`store_memory()` in `main.py` — CCN 13 (Before)**
This FastAPI route handler was directly executing business logic. It contained nested loops to process LLM extraction arrays, checked for self-referential graph edges, and handled DB insertion fallbacks all within the HTTP request boundary. 

**`extract_entities()` in `ai.py` — CCN 10 (Before)**
The complexity here was entirely tied to error handling and fallback parsing. Because LLMs sometimes output markdown ticks instead of raw JSON, the function had a nested `try/except` block falling back to a `re.search()` regex extraction, which itself had a nested `try/except` block. 

---

## Section 5 — Code Improvement

### 5.1 Changes Made

#### Change 1: Extracted persistence loops from `top_paths` (`graph.py`)
**Problem:** `top_paths` was performing graph orchestration *and* doing error-handled database writes. 
**Solution:** Extracted two helper functions: `_persist_source_strengths` and `_build_path_dict`. The main loop now just coordinates data, dropping the CCN from **24 (Grade D)** to **9 (Grade B)**.

#### Change 2: Separated LLM calling from JSON parsing (`ai.py`)
**Problem:** `extract_entities` was handling the HTTP request to NVIDIA NIM, parsing the JSON, doing regex fallbacks, and validating the schema.
**Solution:** Split into `_call_llm()`, `_parse_llm_json()`, and `_validate_extraction()`. The core `extract_entities` function is now a flat 5-line orchestrator with a CCN of **1 (Grade A)**.

#### Change 3: Removed business logic from HTTP handlers (`main.py`)
**Problem:** `store_memory` had a CCN of 13 due to iterating through relations and discarding self-loops inline.
**Solution:** Extracted `_upsert_entities()` and `_insert_relationships()`. The route handler is now clean and focuses solely on HTTP response mapping. CCN dropped from **13 (Grade C)** to **8 (Grade B)**.

---

## Section 6 — Inference & Conclusion

### Key Learnings

1. **Complexity hides in error handling:** In `ai.py`, over 80% of the Cyclomatic Complexity came from JSON parsing fallbacks (`try/except/regex/except`), not from the actual LLM integration. Pulling that into a pure function `_parse_llm_json` immediately solved the cognitive load of the main function.
2. **"God Functions" inflate CCN exponentially:** `top_paths` reached a CCN of 24 simply because it tried to do 6 things at once. By moving the `try/except` DB write blocks into helper functions, the branch count plummeted, making the orchestration flow obvious.
3. **Refactoring adds LOC but reduces CCN:** The project gained 9 lines of code overall, but the Average Complexity dropped from 3.15 to 2.99, and the maximum CCN dropped from 24 to 10. *Slightly longer code is acceptable if it significantly reduces branching.*

Running Radon as part of the CI/CD pipeline (e.g., `radon cc --min C .`) would be highly effective for this repository moving forward, ensuring that no new function is allowed to silently balloon past a CCN of 10.
