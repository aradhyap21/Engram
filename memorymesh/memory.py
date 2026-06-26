"""
MemoryMesh — Supabase data layer.

All CRUD interactions with the nodes and edges tables via the
supabase-py client. No raw SQL strings or other DB drivers are used.
"""

import os
from dotenv import load_dotenv
from supabase import create_client, Client

# ---------------------------------------------------------------------------
# Client initialisation
# ---------------------------------------------------------------------------

load_dotenv()

_SUPABASE_URL: str | None = os.getenv("SUPABASE_URL")
_SUPABASE_KEY: str | None = os.getenv("SUPABASE_KEY")

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

supabase: Client = create_client(_SUPABASE_URL, _SUPABASE_KEY)

# ---------------------------------------------------------------------------
# Node operations
# ---------------------------------------------------------------------------


def upsert_node(content: str, entity_type: str) -> dict:
    """
    Insert or return an existing node identified by *content*.

    If a node with the given content already exists, the existing row is
    returned unchanged (no duplicate is created).  New nodes are inserted
    with ``strength=1.0`` and ``access_count=0``.

    Args:
        content:     Entity text, e.g. "Albert Einstein".
        entity_type: Category string, e.g. "person", "concept", "event".

    Returns:
        The stored node row as a dict, always containing at least an ``id``
        (UUID string) field.

    Raises:
        RuntimeError: If the Supabase call fails.
    """
    # Check whether a node with this content already exists.
    try:
        response = (
            supabase.table("nodes")
            .select("*")
            .eq("content", content)
            .execute()
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to query nodes for content '{content}': {exc}"
        ) from exc

    if response.data:
        # Node already exists — return it without creating a duplicate.
        return response.data[0]

    # Node does not exist — insert a new one.
    try:
        insert_response = (
            supabase.table("nodes")
            .insert(
                {
                    "content": content,
                    "entity_type": entity_type,
                    "strength": 1.0,
                    "access_count": 0,
                }
            )
            .execute()
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to insert node with content '{content}': {exc}"
        ) from exc

    if not insert_response.data:
        raise RuntimeError(
            f"Insert of node with content '{content}' returned no data."
        )

    return insert_response.data[0]


def get_all_nodes() -> list[dict]:
    """
    Return every row from the nodes table.

    Returns:
        List of node dicts; empty list when the table has no rows.

    Raises:
        RuntimeError: If the Supabase call fails.
    """
    try:
        response = supabase.table("nodes").select("*").execute()
    except Exception as exc:
        raise RuntimeError(f"Failed to fetch all nodes: {exc}") from exc

    return response.data or []


def update_node_strength(
    node_id: str, new_strength: float, new_access_count: int
) -> None:
    """
    Persist updated strength and access_count for a node after retrieval.

    Args:
        node_id:          UUID string of the node to update.
        new_strength:     The new strength value (must be > 0).
        new_access_count: The new access count value (must be >= 0).

    Raises:
        RuntimeError: If the Supabase call fails.
    """
    try:
        supabase.table("nodes").update(
            {"strength": new_strength, "access_count": new_access_count}
        ).eq("id", node_id).execute()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to update strength for node '{node_id}': {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Edge operations
# ---------------------------------------------------------------------------


def insert_edge(
    from_id: str, to_id: str, relationship: str, weight: float = 1.0
) -> dict:
    """
    Insert a directed edge between two node UUIDs.

    Args:
        from_id:      UUID of the source node.
        to_id:        UUID of the destination node.
        relationship: Relationship label, e.g. "causes", "influences".
        weight:       Initial edge weight (default 1.0).

    Returns:
        The stored edge row as a dict.

    Raises:
        ValueError:   If ``from_id == to_id`` (self-loop guard, checked
                      before any Supabase call).
        RuntimeError: If the Supabase insert fails.
    """
    if from_id == to_id:
        raise ValueError(
            f"Self-loop detected: from_id and to_id are both '{from_id}'. "
            "Edges must connect two distinct nodes."
        )

    try:
        response = (
            supabase.table("edges")
            .insert(
                {
                    "from_id": from_id,
                    "to_id": to_id,
                    "relationship": relationship,
                    "weight": weight,
                }
            )
            .execute()
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to insert edge from '{from_id}' to '{to_id}': {exc}"
        ) from exc

    if not response.data:
        raise RuntimeError(
            f"Insert of edge from '{from_id}' to '{to_id}' returned no data."
        )

    return response.data[0]


def get_all_edges() -> list[dict]:
    """
    Return every row from the edges table.

    Returns:
        List of edge dicts; empty list when the table has no rows.

    Raises:
        RuntimeError: If the Supabase call fails.
    """
    try:
        response = supabase.table("edges").select("*").execute()
    except Exception as exc:
        raise RuntimeError(f"Failed to fetch all edges: {exc}") from exc

    return response.data or []


def update_edge_weight(edge_id: str, new_weight: float) -> None:
    """
    Persist a decayed edge weight back to Supabase.

    Args:
        edge_id:    UUID string of the edge to update.
        new_weight: The new weight value (should be > 0).

    Raises:
        RuntimeError: If the Supabase call fails.
    """
    try:
        supabase.table("edges").update({"weight": new_weight}).eq(
            "id", edge_id
        ).execute()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to update weight for edge '{edge_id}': {exc}"
        ) from exc
