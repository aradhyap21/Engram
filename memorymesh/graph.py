"""
MemoryMesh — Dijkstra shortest-path and Ebbinghaus decay engine.

Pure-Python implementation using only heapq, math, and datetime
from the standard library. No third-party graph libraries used.
"""

import heapq
import math
from datetime import datetime, timezone


def apply_decay(edges: list[dict], nodes: list[dict]) -> list[dict]:
    """
    Apply Ebbinghaus exponential decay to each edge weight.

    For each edge computes:
        decayed_weight = weight * exp(-days_elapsed / S)

    where days_elapsed is derived from edge["created_at"] (ISO-8601) relative
    to the current UTC time, and S is the source node's strength value
    (clamped to a minimum of 0.001 to prevent division by zero).

    Args:
        edges: List of edge dicts with keys: id, from_id, to_id, weight,
               created_at, relationship.
        nodes: List of node dicts with keys: id, strength (and others).

    Returns:
        A new list of edge dicts (input is not mutated). Each dict in the
        returned list is a copy of the original edge dict with an additional
        'decayed_weight' key set to the computed decayed weight.

    Satisfies Requirements: 5.1, 5.2, 5.3, 5.4, 5.5
    """
    # Build a lookup from node id → clamped strength (min 0.001)
    node_strength = {n["id"]: max(n["strength"], 0.001) for n in nodes}

    result = []
    now = datetime.now(timezone.utc)

    for edge in edges:
        # Parse the ISO-8601 timestamp; treat naive timestamps as UTC
        created = datetime.fromisoformat(edge["created_at"])
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)

        days_elapsed = (now - created).total_seconds() / 86400.0

        # Fall back to strength=1.0 if the source node is not in the list
        S = node_strength.get(edge["from_id"], 1.0)

        decayed = edge["weight"] * math.exp(-days_elapsed / S)

        # Append a shallow copy of the edge dict with the extra field
        result.append({**edge, "decayed_weight": decayed})

    return result


def build_adjacency(nodes: list[dict], edges: list[dict]) -> dict:
    """
    Build adjacency list keyed by node UUID.

    Caller must invoke apply_decay before this function so that every edge
    in ``edges`` already has a ``decayed_weight`` field.

    Args:
        nodes: List of node dicts with at least the key ``id`` (UUID string).
        edges: List of edge dicts with keys: id, from_id, to_id,
               decayed_weight.  These are the *already-decayed* edges
               returned by apply_decay.

    Returns:
        A dict mapping each node UUID to a list of 3-tuples::

            {node_id: [(neighbour_id, decayed_weight, edge_id), ...]}

        Every node in ``nodes`` is guaranteed to have an entry, even if it
        has no outgoing edges (its value will be an empty list).

    Satisfies Requirements: 4.1, 2.3
    """
    # Seed every node with an empty neighbour list so isolated nodes are
    # always present in the result.
    adjacency: dict = {node["id"]: [] for node in nodes}

    for edge in edges:
        from_id = edge["from_id"]
        to_id = edge["to_id"]
        decayed_weight = edge["decayed_weight"]
        edge_id = edge["id"]

        # Add the forward direction. If from_id is somehow absent from the
        # nodes list, insert it defensively so the graph stays self-consistent.
        if from_id not in adjacency:
            adjacency[from_id] = []

        adjacency[from_id].append((to_id, decayed_weight, edge_id))

    return adjacency


def dijkstra(adjacency: dict, source_ids: list[str]) -> dict:
    """
    Run multi-source Dijkstra from all source_ids simultaneously.

    Initialises every source node with cost 0.0 and explores the graph
    using a min-heap.  Only ``heapq`` (already imported at module level)
    is used — no third-party graph libraries.

    Args:
        adjacency: Adjacency list as returned by ``build_adjacency``.
                   Format: {node_id: [(neighbour_id, cost, edge_id), ...]}
        source_ids: List of node UUIDs to treat as simultaneous sources.
                    All sources start with total_cost = 0.0.

    Returns:
        A dict mapping every *reachable* node UUID to a 2-tuple::

            {node_id: (total_cost, path_list)}

        where ``path_list`` is the ordered list of node UUIDs from the
        nearest source to ``node_id`` (inclusive on both ends).

        Unreachable nodes are silently omitted from the result.

    Satisfies Requirements: 4.1, 4.2, 4.3, 4.4, 4.5
    """
    dist: dict = {}
    prev: dict = {}
    heap: list = []

    for src in source_ids:
        dist[src] = 0.0
        prev[src] = None
        heapq.heappush(heap, (0.0, src))

    visited: set = set()

    while heap:
        cost, u = heapq.heappop(heap)
        if u in visited:
            continue
        visited.add(u)

        for (v, edge_cost, edge_id) in adjacency.get(u, []):
            new_cost = cost + edge_cost
            if v not in dist or new_cost < dist[v]:
                dist[v] = new_cost
                prev[v] = (u, edge_id)
                heapq.heappush(heap, (new_cost, v))

    # Reconstruct paths by walking prev pointers back to the source
    paths: dict = {}
    for node_id, total_cost in dist.items():
        path = []
        cur = node_id
        while cur is not None:
            path.append(cur)
            cur = prev[cur][0] if prev.get(cur) else None
        paths[node_id] = (total_cost, list(reversed(path)))

    return paths


def top_paths(
    nodes: list[dict],
    edges: list[dict],
    query_entities: list[str],
    k: int = 5,
) -> list[dict]:
    """
    Orchestration facade: decay → build → Dijkstra → rank → return top-k paths.

    Steps:
        1. Apply Ebbinghaus decay to every edge.
        2. Build adjacency list.
        3. Match query_entities (strings) to node UUIDs via case-insensitive
           comparison against each node's ``content`` field.
        4. Run multi-source Dijkstra from the matched source node IDs.
        5. Persist side-effects:
           - Increment ``strength += 0.1`` and ``access_count += 1`` for
             every source node touched by Dijkstra, then write to Supabase.
           - Persist the decayed weight for every edge traversed in the
             returned paths.
        6. Return the top-k path dicts sorted ascending by ``total_cost``.

    Args:
        nodes:           List of node dicts from Supabase.  Each dict must
                         have at least the keys: ``id``, ``content``,
                         ``entity_type``, ``strength``, ``access_count``.
        edges:           List of edge dicts from Supabase.  Each dict must
                         have at least the keys: ``id``, ``from_id``,
                         ``to_id``, ``weight``, ``created_at``,
                         ``relationship``.
        query_entities:  List of entity name strings extracted from the
                         user's query (e.g. ``["gravity", "Isaac Newton"]``).
        k:               Maximum number of paths to return (default 5).

    Returns:
        A list of at most *k* path dicts sorted ascending by ``total_cost``.
        Each dict has the shape::

            {
                "path": [
                    {"id": str, "content": str, "entity_type": str},
                    ...
                ],
                "edges": [
                    {"relationship": str, "decayed_weight": float},
                    ...
                ],
                "total_cost": float,
            }

        Returns ``[]`` when:
        - ``nodes`` or ``edges`` is empty,
        - no node's content matches any of the ``query_entities``, or
        - Dijkstra finds no multi-hop paths.

    Side effects:
        - Calls ``memory.update_node_strength`` for each source node.
        - Calls ``memory.update_edge_weight`` for each edge in the paths.

    Satisfies Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 4.1–4.5
    """
    # Import memory here to avoid any potential circular-import issues.
    # memory.py only imports supabase and dotenv — it does NOT import graph.py.
    import memory  # noqa: PLC0415

    # ------------------------------------------------------------------
    # Guard: empty graph
    # ------------------------------------------------------------------
    if not nodes or not edges:
        return []

    # ------------------------------------------------------------------
    # Step 1 — Apply Ebbinghaus decay
    # ------------------------------------------------------------------
    decayed_edges = apply_decay(edges, nodes)

    # ------------------------------------------------------------------
    # Step 2 — Build adjacency list
    # ------------------------------------------------------------------
    adjacency = build_adjacency(nodes, decayed_edges)

    # ------------------------------------------------------------------
    # Step 3 — Match query entities → source node UUIDs
    # ------------------------------------------------------------------
    query_lower = {q.lower() for q in query_entities}
    node_lookup: dict[str, dict] = {n["id"]: n for n in nodes}

    source_ids: list[str] = [
        n["id"]
        for n in nodes
        if n["content"].lower() in query_lower
    ]

    if not source_ids:
        return []

    # ------------------------------------------------------------------
    # Step 4 — Run multi-source Dijkstra
    # ------------------------------------------------------------------
    dijkstra_result = dijkstra(adjacency, source_ids)

    # ------------------------------------------------------------------
    # Step 5a — Persist strength increments for source nodes
    # ------------------------------------------------------------------
    for node_id in source_ids:
        node = node_lookup.get(node_id)
        if node is None:
            continue
        new_strength = node["strength"] + 0.1
        new_access_count = node["access_count"] + 1
        try:
            memory.update_node_strength(node_id, new_strength, new_access_count)
        except Exception:
            # Non-fatal: log and continue so retrieval still returns results
            pass

    # ------------------------------------------------------------------
    # Step 5b — Build an O(1) edge lookup for path reconstruction
    # (keyed by (from_id, to_id))
    # ------------------------------------------------------------------
    edge_lookup: dict[tuple, dict] = {
        (e["from_id"], e["to_id"]): e for e in decayed_edges
    }

    # ------------------------------------------------------------------
    # Step 6 — Reconstruct and rank paths; persist decayed edge weights
    # ------------------------------------------------------------------
    path_dicts: list[dict] = []

    for node_id, (total_cost, path_list) in dijkstra_result.items():
        # Only multi-hop paths (exclude source-only single-node entries)
        if len(path_list) <= 1:
            continue

        # Build human-readable node sequence
        path_nodes: list[dict] = []
        for nid in path_list:
            n = node_lookup.get(nid)
            if n is None:
                break
            path_nodes.append(
                {
                    "id": n["id"],
                    "content": n["content"],
                    "entity_type": n["entity_type"],
                }
            )
        else:
            # Build edge sequence and persist decayed weights
            path_edges: list[dict] = []
            for i in range(len(path_list) - 1):
                from_id = path_list[i]
                to_id = path_list[i + 1]
                edge = edge_lookup.get((from_id, to_id))
                if edge is None:
                    break
                path_edges.append(
                    {
                        "relationship": edge["relationship"],
                        "decayed_weight": edge["decayed_weight"],
                    }
                )
                # Persist the decayed weight back to Supabase
                try:
                    memory.update_edge_weight(edge["id"], edge["decayed_weight"])
                except Exception:
                    pass
            else:
                path_dicts.append(
                    {
                        "path": path_nodes,
                        "edges": path_edges,
                        "total_cost": total_cost,
                    }
                )

    if not path_dicts:
        return []

    # Sort ascending by total_cost and return top-k
    path_dicts.sort(key=lambda p: p["total_cost"])
    return path_dicts[:k]
