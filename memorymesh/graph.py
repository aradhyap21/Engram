"""
Engram — Dijkstra shortest-path and Ebbinghaus decay engine.

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


# ---------------------------------------------------------------------------
# top_paths helpers — extracted to keep each function at a single concern
# ---------------------------------------------------------------------------


def _match_source_nodes(nodes: list[dict], query_entities: list[str]) -> list[str]:
    """
    Return a list of node UUIDs whose ``content`` matches any query entity.

    Matching is case-insensitive and bidirectional (query substring of content
    OR content substring of query).

    Args:
        nodes:           Full node list from the graph.
        query_entities:  Entity strings from the user's query.

    Returns:
        List of matching node UUIDs (may be empty).
    """
    query_lower = {q.lower() for q in query_entities}
    return [
        n["id"]
        for n in nodes
        if any(
            q in n["content"].lower() or n["content"].lower() in q
            for q in query_lower
        )
    ]


def _persist_source_strengths(
    source_ids: list[str],
    node_lookup: dict[str, dict],
    memory_module: object,
) -> None:
    """
    Increment strength and access_count for every Dijkstra source node.

    Failures are swallowed so that retrieval always returns results even when
    the persistence layer is temporarily unavailable.

    Args:
        source_ids:    UUIDs of the nodes used as Dijkstra sources.
        node_lookup:   Mapping from node UUID → node dict.
        memory_module: The ``memory`` module (passed in to avoid circular import).
    """
    for node_id in source_ids:
        node = node_lookup.get(node_id)
        if node is None:
            continue
        new_strength = node["strength"] + 0.1
        new_access_count = node["access_count"] + 1
        try:
            memory_module.update_node_strength(node_id, new_strength, new_access_count)
        except Exception:
            # Non-fatal: retrieval still returns results
            pass


def _build_path_dict(
    path_list: list[str],
    total_cost: float,
    node_lookup: dict[str, dict],
    edge_lookup: dict[tuple, dict],
    memory_module: object,
) -> dict | None:
    """
    Construct a single path dict and persist decayed edge weights.

    Returns ``None`` if any node or edge in the path cannot be resolved,
    so the caller can safely skip incomplete paths.

    Args:
        path_list:     Ordered list of node UUIDs from source to destination.
        total_cost:    Dijkstra total cost for this path.
        node_lookup:   UUID → node dict mapping.
        edge_lookup:   (from_id, to_id) → decayed edge dict mapping.
        memory_module: The ``memory`` module for persisting decayed weights.

    Returns:
        A path dict with keys ``path``, ``edges``, ``total_cost``, or ``None``.
    """
    path_nodes: list[dict] = []
    for nid in path_list:
        n = node_lookup.get(nid)
        if n is None:
            return None
        path_nodes.append({"id": n["id"], "content": n["content"], "entity_type": n["entity_type"]})

    path_edges: list[dict] = []
    for i in range(len(path_list) - 1):
        from_id = path_list[i]
        to_id = path_list[i + 1]
        edge = edge_lookup.get((from_id, to_id))
        if edge is None:
            return None
        path_edges.append({"relationship": edge["relationship"], "decayed_weight": edge["decayed_weight"]})
        try:
            memory_module.update_edge_weight(edge["id"], edge["decayed_weight"])
        except Exception:
            pass

    return {"path": path_nodes, "edges": path_edges, "total_cost": total_cost}


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
        5. Persist side-effects via helper functions.
        6. Return the top-k path dicts sorted ascending by ``total_cost``.

    Args:
        nodes:           List of node dicts from Supabase.
        edges:           List of edge dicts from Supabase.
        query_entities:  List of entity name strings from the user's query.
        k:               Maximum number of paths to return (default 5).

    Returns:
        A list of at most *k* path dicts sorted ascending by ``total_cost``.
        Returns ``[]`` when nodes/edges are empty, no entities match, or
        Dijkstra finds no multi-hop paths.

    Satisfies Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 4.1–4.5
    """
    # Import memory here to avoid any potential circular-import issues.
    import memory  # noqa: PLC0415

    if not nodes or not edges:
        return []

    decayed_edges = apply_decay(edges, nodes)
    adjacency = build_adjacency(nodes, decayed_edges)

    source_ids = _match_source_nodes(nodes, query_entities)
    if not source_ids:
        return []

    node_lookup: dict[str, dict] = {n["id"]: n for n in nodes}
    _persist_source_strengths(source_ids, node_lookup, memory)

    dijkstra_result = dijkstra(adjacency, source_ids)

    edge_lookup: dict[tuple, dict] = {(e["from_id"], e["to_id"]): e for e in decayed_edges}

    path_dicts: list[dict] = []
    for node_id, (total_cost, path_list) in dijkstra_result.items():
        if len(path_list) <= 1:
            continue
        path_dict = _build_path_dict(path_list, total_cost, node_lookup, edge_lookup, memory)
        if path_dict is not None:
            path_dicts.append(path_dict)

    path_dicts.sort(key=lambda p: p["total_cost"])
    return path_dicts[:k]
