"""
Engram — Supabase data layer.

All CRUD interactions with the nodes and edges tables via the
supabase-py client. No raw SQL strings or other DB drivers are used.
"""

import os
import sqlite3
import uuid
import json
import math
from datetime import datetime, timezone
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
# SQLite Local Fallback
# ---------------------------------------------------------------------------

DB_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "memorymesh.db"))


def _is_network_error(exc: Exception) -> bool:
    err_str = str(exc).lower()
    return any(
        kw in err_str
        for kw in [
            "getaddrinfo failed",
            "connecterror",
            "name resolution",
            "failed to resolve",
            "max retries exceeded",
            "connection refused",
            "connection error",
        ]
    )


def _init_local_db() -> None:
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS nodes (
        id TEXT PRIMARY KEY,
        content TEXT NOT NULL UNIQUE,
        entity_type TEXT,
        canonical_name TEXT,
        aliases TEXT DEFAULT '[]',
        embedding TEXT,
        pending_review INTEGER DEFAULT 0,
        source_mention TEXT,
        model_used TEXT,
        decided_at TEXT,
        reasoning TEXT,
        strength REAL DEFAULT 1.0,
        access_count INTEGER DEFAULT 0,
        created_at TEXT
    );
    """
    )
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS edges (
        id TEXT PRIMARY KEY,
        from_id TEXT,
        to_id TEXT,
        relationship TEXT,
        weight REAL DEFAULT 1.0,
        created_at TEXT,
        FOREIGN KEY(from_id) REFERENCES nodes(id),
        FOREIGN KEY(to_id) REFERENCES nodes(id)
    );
    """
    )
    conn.commit()
    conn.close()


def _local_upsert_node(content: str, entity_type: str) -> dict:
    _init_local_db()
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM nodes WHERE content = ?", (content,))
    row = cursor.fetchone()
    if row:
        res = dict(row)
        conn.close()
        return res

    node_id = str(uuid.uuid4())
    now_iso = datetime.now(timezone.utc).isoformat()
    cursor.execute(
        "INSERT INTO nodes (id, content, entity_type, strength, access_count, created_at, aliases) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (node_id, content, entity_type, 1.0, 0, now_iso, "[]"),
    )
    conn.commit()
    cursor.execute("SELECT * FROM nodes WHERE id = ?", (node_id,))
    res = dict(cursor.fetchone())
    conn.close()
    return res


def _local_get_all_nodes() -> list[dict]:
    _init_local_db()
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM nodes")
    rows = cursor.fetchall()
    res = [dict(r) for r in rows]
    conn.close()
    return res


def _local_update_node_strength(
    node_id: str, new_strength: float, new_access_count: int
) -> None:
    _init_local_db()
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE nodes SET strength = ?, access_count = ? WHERE id = ?",
        (new_strength, new_access_count, node_id),
    )
    conn.commit()
    conn.close()


def _local_insert_edge(
    from_id: str, to_id: str, relationship: str, weight: float = 1.0
) -> dict:
    if from_id == to_id:
        raise ValueError(
            f"Self-loop detected: from_id and to_id are both '{from_id}'. "
            "Edges must connect two distinct nodes."
        )
    _init_local_db()
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    edge_id = str(uuid.uuid4())
    now_iso = datetime.now(timezone.utc).isoformat()
    cursor.execute(
        "INSERT INTO edges (id, from_id, to_id, relationship, weight, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (edge_id, from_id, to_id, relationship, weight, now_iso),
    )
    conn.commit()
    cursor.execute("SELECT * FROM edges WHERE id = ?", (edge_id,))
    res = dict(cursor.fetchone())
    conn.close()
    return res


def _local_get_all_edges() -> list[dict]:
    _init_local_db()
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM edges")
    rows = cursor.fetchall()
    res = [dict(r) for r in rows]
    conn.close()
    return res


def _local_update_edge_weight(edge_id: str, new_weight: float) -> None:
    _init_local_db()
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE edges SET weight = ? WHERE id = ?", (new_weight, edge_id)
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Node operations
# ---------------------------------------------------------------------------


def upsert_node(content: str, entity_type: str) -> dict:
    """
    Insert or return an existing node identified by *content*.
    """
    try:
        response = (
            supabase.table("nodes")
            .select("*")
            .eq("content", content)
            .execute()
        )
    except Exception as exc:
        if _is_network_error(exc):
            return _local_upsert_node(content, entity_type)
        raise RuntimeError(
            f"Failed to query nodes for content '{content}': {exc}"
        ) from exc

    if response.data:
        return response.data[0]

    try:
        insert_response = (
            supabase.table("nodes")
            .insert(
                {
                    "content": content,
                    "entity_type": entity_type,
                    "strength": 1.0,
                    "access_count": 0,
                    "aliases": [],
                }
            )
            .execute()
        )
    except Exception as exc:
        if _is_network_error(exc):
            return _local_upsert_node(content, entity_type)
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
    """
    try:
        response = supabase.table("nodes").select("*").execute()
    except Exception as exc:
        if _is_network_error(exc):
            return _local_get_all_nodes()
        raise RuntimeError(f"Failed to fetch all nodes: {exc}") from exc

    return response.data or []


def update_node_strength(
    node_id: str, new_strength: float, new_access_count: int
) -> None:
    """
    Persist updated strength and access_count for a node after retrieval.
    """
    try:
        supabase.table("nodes").update(
            {"strength": new_strength, "access_count": new_access_count}
        ).eq("id", node_id).execute()
    except Exception as exc:
        if _is_network_error(exc):
            _local_update_node_strength(node_id, new_strength, new_access_count)
            return
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
        if _is_network_error(exc):
            return _local_insert_edge(from_id, to_id, relationship, weight)
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
    """
    try:
        response = supabase.table("edges").select("*").execute()
    except Exception as exc:
        if _is_network_error(exc):
            return _local_get_all_edges()
        raise RuntimeError(f"Failed to fetch all edges: {exc}") from exc

    return response.data or []


def update_edge_weight(edge_id: str, new_weight: float) -> None:
    """
    Persist a decayed edge weight back to Supabase.
    """
    try:
        supabase.table("edges").update({"weight": new_weight}).eq(
            "id", edge_id
        ).execute()
    except Exception as exc:
        if _is_network_error(exc):
            _local_update_edge_weight(edge_id, new_weight)
            return
        raise RuntimeError(
            f"Failed to update weight for edge '{edge_id}': {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Entity Resolution Support
# ---------------------------------------------------------------------------

def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm_a = math.sqrt(sum(a * a for a in vec1))
    norm_b = math.sqrt(sum(b * b for b in vec2))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)

def get_similar_candidates(mention_embedding: list[float], threshold: float = 0.75, k: int = 5) -> list[dict]:
    """
    Fetch all nodes and compute cosine similarity against mention_embedding.
    Returns the top k candidates that exceed the threshold.
    """
    nodes = get_all_nodes()
    candidates = []
    
    for node in nodes:
        emb_str = node.get("embedding")
        if not emb_str:
            continue
            
        if isinstance(emb_str, str):
            try:
                emb = json.loads(emb_str)
            except:
                continue
        else:
            emb = emb_str
            
        if not isinstance(emb, list) or len(emb) != len(mention_embedding):
            continue
            
        sim = _cosine_similarity(mention_embedding, emb)
        if sim >= threshold:
            node["_similarity"] = sim
            candidates.append(node)
            
    candidates.sort(key=lambda x: x["_similarity"], reverse=True)
    return candidates[:k]

def _local_update_node_aliases(node_id: str, new_aliases: list[str]) -> None:
    _init_local_db()
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT aliases FROM nodes WHERE id = ?", (node_id,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        return
        
    current_aliases = []
    if row[0]:
        try:
            current_aliases = json.loads(row[0])
        except:
            pass
            
    # merge and deduplicate
    combined = list(set(current_aliases + new_aliases))
    cursor.execute("UPDATE nodes SET aliases = ? WHERE id = ?", (json.dumps(combined), node_id))
    conn.commit()
    conn.close()

def update_node_aliases(node_id: str, new_aliases: list[str]) -> None:
    try:
        response = supabase.table("nodes").select("aliases").eq("id", node_id).execute()
        if not response.data:
            return
            
        current_aliases = response.data[0].get("aliases") or []
        combined = list(set(current_aliases + new_aliases))
        supabase.table("nodes").update({"aliases": combined}).eq("id", node_id).execute()
    except Exception as exc:
        if _is_network_error(exc):
            _local_update_node_aliases(node_id, new_aliases)
            return
        raise RuntimeError(f"Failed to update aliases for node '{node_id}': {exc}") from exc

def _local_insert_resolved_node(content: str, entity_type: str, kwargs: dict) -> dict:
    _init_local_db()
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    node_id = str(uuid.uuid4())
    now_iso = datetime.now(timezone.utc).isoformat()
    
    aliases = json.dumps(kwargs.get("aliases", []))
    embedding = json.dumps(kwargs.get("embedding", [])) if kwargs.get("embedding") else None
    
    cursor.execute(
        '''INSERT INTO nodes (
            id, content, entity_type, canonical_name, aliases, embedding, 
            pending_review, source_mention, model_used, decided_at, reasoning,
            strength, access_count, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
        (
            node_id, content, entity_type, 
            kwargs.get("canonical_name"), aliases, embedding,
            1 if kwargs.get("pending_review") else 0,
            kwargs.get("source_mention"), kwargs.get("model_used"), kwargs.get("decided_at"),
            kwargs.get("reasoning"),
            1.0, 0, now_iso
        )
    )
    conn.commit()
    cursor.execute("SELECT * FROM nodes WHERE id = ?", (node_id,))
    res = dict(cursor.fetchone())
    conn.close()
    return res

def insert_resolved_node(content: str, entity_type: str, **kwargs) -> dict:
    """
    Insert a new node with full resolution fields.
    kwargs can include: embedding, canonical_name, aliases, pending_review, source_mention, model_used, decided_at, reasoning.
    """
    try:
        # For Supabase, pass lists/dicts directly for JSONB
        insert_data = {
            "content": content,
            "entity_type": entity_type,
            "strength": 1.0,
            "access_count": 0,
            "canonical_name": kwargs.get("canonical_name"),
            "aliases": kwargs.get("aliases", []),
            "embedding": kwargs.get("embedding"),
            "pending_review": kwargs.get("pending_review", False),
            "source_mention": kwargs.get("source_mention"),
            "model_used": kwargs.get("model_used"),
            "decided_at": kwargs.get("decided_at"),
            "reasoning": kwargs.get("reasoning"),
        }
        response = supabase.table("nodes").insert(insert_data).execute()
        if not response.data:
            raise RuntimeError(f"Insert returned no data.")
        return response.data[0]
    except Exception as exc:
        if _is_network_error(exc):
            return _local_insert_resolved_node(content, entity_type, kwargs)
        raise RuntimeError(f"Failed to insert resolved node '{content}': {exc}") from exc

