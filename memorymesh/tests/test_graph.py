"""
Unit tests for memorymesh/graph.py

Tests cover:
  - apply_decay
  - build_adjacency
  - dijkstra
  - top_paths
"""

import math
import sys
import types
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Mock memory at sys.modules level so that top_paths() can do:
#   import memory
# without hitting Supabase connection code at import time.
# ---------------------------------------------------------------------------

def _make_memory_mock():
    """Return a module-like mock that satisfies graph.top_paths' usage."""
    m = MagicMock(spec=types.ModuleType)
    m.update_node_strength = MagicMock(return_value=None)
    m.update_edge_weight = MagicMock(return_value=None)
    return m

_MEMORY_MOCK = _make_memory_mock()

if "memory" not in sys.modules:
    sys.modules["memory"] = _MEMORY_MOCK


# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------

def _iso(days_ago: float = 0.0) -> str:
    """Return an ISO-8601 UTC timestamp *days_ago* days in the past."""
    ts = datetime.now(timezone.utc) - timedelta(days=days_ago)
    return ts.isoformat()


def _node(
    id_: str,
    strength: float = 1.0,
    content: str = "",
    entity_type: str = "concept",
    access_count: int = 0,
) -> dict:
    return {
        "id": id_,
        "content": content or id_,
        "entity_type": entity_type,
        "strength": strength,
        "access_count": access_count,
        "created_at": _iso(0),
    }


def _edge(
    id_: str,
    from_id: str,
    to_id: str,
    weight: float = 1.0,
    days_ago: float = 0.0,
    relationship: str = "relates",
) -> dict:
    return {
        "id": id_,
        "from_id": from_id,
        "to_id": to_id,
        "weight": weight,
        "created_at": _iso(days_ago),
        "relationship": relationship,
    }


# ===========================================================================
# Import graph functions AFTER sys.modules patching
# ===========================================================================

from ..graph import apply_decay, build_adjacency, dijkstra, top_paths  # noqa: E402


# ===========================================================================
# apply_decay
# ===========================================================================

class TestApplyDecay:
    def test_decay_at_t0_equals_original_weight(self):
        """Req 5.2: t=0 -> decayed_weight == original_weight."""
        node = _node("n1", strength=1.0)
        edge = _edge("e1", "n1", "n2", weight=2.5, days_ago=0.0)
        result = apply_decay([edge], [node])
        assert len(result) == 1
        assert pytest.approx(result[0]["decayed_weight"], rel=1e-6) == 2.5

    def test_decay_at_half_life(self):
        """Req 5.3: t = S*ln(2) -> decayed_weight ~ original_weight / 2."""
        S = 3.0
        t = S * math.log(2)
        node = _node("n1", strength=S)
        edge = _edge("e1", "n1", "n2", weight=4.0, days_ago=t)
        result = apply_decay([edge], [node])
        assert pytest.approx(result[0]["decayed_weight"], rel=1e-5) == 2.0

    def test_decay_strictly_less_for_positive_t(self):
        """Req 5.4: decayed_weight < original_weight for t > 0."""
        node = _node("n1", strength=1.0)
        edge = _edge("e1", "n1", "n2", weight=1.0, days_ago=1.0)
        result = apply_decay([edge], [node])
        assert result[0]["decayed_weight"] < 1.0

    def test_strength_clamped_to_minimum(self):
        """Req 5.5: strength=0 clamped to 0.001; no division-by-zero."""
        node = _node("n1", strength=0.0)
        edge = _edge("e1", "n1", "n2", weight=2.0, days_ago=0.0)
        result = apply_decay([edge], [node])
        dw = result[0]["decayed_weight"]
        assert math.isfinite(dw)
        assert dw > 0.0

    def test_missing_source_node_uses_default_strength(self):
        """Edge whose from_id has no matching node falls back to strength=1.0."""
        edge = _edge("e1", "unknown", "n2", weight=1.0, days_ago=0.0)
        result = apply_decay([edge], [])
        assert len(result) == 1
        assert pytest.approx(result[0]["decayed_weight"], rel=1e-6) == 1.0

    def test_input_not_mutated(self):
        """apply_decay must not mutate the original edge dicts."""
        node = _node("n1", strength=1.0)
        original_edge = _edge("e1", "n1", "n2", weight=1.5, days_ago=0.5)
        original_keys = set(original_edge.keys())
        apply_decay([original_edge], [node])
        assert set(original_edge.keys()) == original_keys
        assert "decayed_weight" not in original_edge

    def test_naive_timestamp_treated_as_utc(self):
        """Timestamps without tzinfo must not raise and decay normally."""
        node = _node("n1", strength=1.0)
        naive_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        edge = {**_edge("e1", "n1", "n2", weight=1.0), "created_at": naive_ts}
        result = apply_decay([edge], [node])
        assert result[0]["decayed_weight"] > 0.0


# ===========================================================================
# build_adjacency
# ===========================================================================

class TestBuildAdjacency:
    def test_every_node_present_even_if_no_edges(self):
        """Isolated nodes must appear in the adjacency dict with empty lists."""
        nodes = [_node("a"), _node("b")]
        adj = build_adjacency(nodes, [])
        assert "a" in adj and "b" in adj
        assert adj["a"] == [] and adj["b"] == []

    def test_forward_edge_added(self):
        nodes = [_node("a"), _node("b")]
        node = _node("a", strength=1.0)
        edge = _edge("e1", "a", "b", weight=1.0, days_ago=0.0)
        decayed = apply_decay([edge], [node])
        adj = build_adjacency(nodes, decayed)
        assert len(adj["a"]) == 1
        neighbour, cost, eid = adj["a"][0]
        assert neighbour == "b"
        assert eid == "e1"
        assert cost > 0.0

    def test_no_reverse_edge_by_default(self):
        """Graph is directed; no reverse edge should be added."""
        nodes = [_node("a"), _node("b")]
        node_a = _node("a")
        edge = apply_decay([_edge("e1", "a", "b")], [node_a])
        adj = build_adjacency(nodes, edge)
        assert adj["b"] == []


# ===========================================================================
# dijkstra
# ===========================================================================

class TestDijkstra:
    def _simple_graph(self):
        """a --(0.5)--> b --(0.3)--> c"""
        nodes = [_node("a"), _node("b"), _node("c")]
        edges_raw = [
            _edge("e1", "a", "b", weight=1.0, days_ago=0.0),
            _edge("e2", "b", "c", weight=1.0, days_ago=0.0),
        ]
        decayed = [
            {**edges_raw[0], "decayed_weight": 0.5},
            {**edges_raw[1], "decayed_weight": 0.3},
        ]
        adj = build_adjacency(nodes, decayed)
        return adj

    def test_single_source_shortest_path(self):
        """Req 4.2: Dijkstra finds minimum cost path in connected graph."""
        adj = self._simple_graph()
        result = dijkstra(adj, ["a"])
        assert "c" in result
        cost, path = result["c"]
        assert pytest.approx(cost, rel=1e-9) == 0.8
        assert path == ["a", "b", "c"]

    def test_source_node_cost_is_zero(self):
        adj = self._simple_graph()
        result = dijkstra(adj, ["a"])
        cost, path = result["a"]
        assert cost == 0.0
        assert path == ["a"]

    def test_unreachable_node_absent(self):
        """Req 4.3: unreachable nodes are silently omitted."""
        nodes = [_node("a"), _node("b"), _node("orphan")]
        decayed = [{"id": "e1", "from_id": "a", "to_id": "b", "decayed_weight": 1.0}]
        adj = build_adjacency(nodes, decayed)
        result = dijkstra(adj, ["a"])
        assert "orphan" not in result

    def test_multi_source_picks_minimum(self):
        """Req 4.4: multi-source Dijkstra picks minimum cost."""
        nodes = [_node("s1"), _node("s2"), _node("c")]
        decayed = [
            {"id": "e1", "from_id": "s1", "to_id": "c", "decayed_weight": 10.0},
            {"id": "e2", "from_id": "s2", "to_id": "c", "decayed_weight": 1.0},
        ]
        adj = build_adjacency(nodes, decayed)
        result = dijkstra(adj, ["s1", "s2"])
        cost, path = result["c"]
        assert pytest.approx(cost, rel=1e-9) == 1.0
        assert path[-1] == "c"

    def test_always_terminates_on_positive_weights(self):
        """Req 4.5: algorithm terminates on positive weights."""
        import random
        rng = random.Random(42)
        node_ids = [str(i) for i in range(20)]
        nodes = [_node(nid) for nid in node_ids]
        decayed = [
            {
                "id": f"e{i}",
                "from_id": node_ids[i],
                "to_id": node_ids[(i + 1) % 20],
                "decayed_weight": rng.uniform(0.01, 5.0),
            }
            for i in range(20)
        ]
        adj = build_adjacency(nodes, decayed)
        result = dijkstra(adj, [node_ids[0]])
        assert isinstance(result, dict)


# ===========================================================================
# top_paths
# ===========================================================================

class TestTopPaths:
    """
    Tests for the top_paths orchestration function.

    The module-level _MEMORY_MOCK in sys.modules["memory"] provides a
    safe no-op mock so the simpler tests (that don't inspect side-effect
    calls) still pass without touching Supabase.

    The two side-effect tests (strength increment + edge weight persist)
    swap in their own fresh mock via patch.dict so call_args_list is
    pristine and reliable.
    """

    def _make_graph(self):
        """
        3-node linear graph:
          gravity --(causes)--> orbital_mechanics --(enables)--> space_exploration
        """
        nodes = [
            _node("n1", strength=1.0, content="gravity", entity_type="concept"),
            _node("n2", strength=1.0, content="orbital mechanics", entity_type="concept"),
            _node("n3", strength=1.0, content="space exploration", entity_type="concept"),
        ]
        edges = [
            _edge("e1", "n1", "n2", weight=1.0, days_ago=0.0, relationship="causes"),
            _edge("e2", "n2", "n3", weight=1.0, days_ago=0.0, relationship="enables"),
        ]
        return nodes, edges

    # ------------------------------------------------------------------
    # tests that do NOT inspect mock calls -- module-level mock is fine
    # ------------------------------------------------------------------

    def test_returns_paths_for_matching_entity(self):
        nodes, edges = self._make_graph()
        result = top_paths(nodes, edges, ["gravity"])
        assert isinstance(result, list)
        assert len(result) >= 1
        for p in result:
            assert "path" in p
            assert "edges" in p
            assert "total_cost" in p

    def test_sorted_ascending_by_total_cost(self):
        nodes, edges = self._make_graph()
        result = top_paths(nodes, edges, ["gravity"])
        costs = [p["total_cost"] for p in result]
        assert costs == sorted(costs)

    def test_returns_empty_list_when_no_entity_matches(self):
        nodes, edges = self._make_graph()
        result = top_paths(nodes, edges, ["does not exist"])
        assert result == []

    def test_returns_empty_list_when_nodes_empty(self):
        _, edges = self._make_graph()
        result = top_paths([], edges, ["gravity"])
        assert result == []

    def test_returns_empty_list_when_edges_empty(self):
        nodes, _ = self._make_graph()
        result = top_paths(nodes, [], ["gravity"])
        assert result == []

    def test_case_insensitive_entity_matching(self):
        nodes, edges = self._make_graph()
        result_lower = top_paths(nodes, edges, ["gravity"])
        result_upper = top_paths(nodes, edges, ["GRAVITY"])
        result_mixed = top_paths(nodes, edges, ["Gravity"])
        assert len(result_lower) == len(result_upper) == len(result_mixed)

    def test_respects_k_limit(self):
        """Returns at most k paths."""
        nodes, edges = self._make_graph()
        result = top_paths(nodes, edges, ["gravity"], k=1)
        assert len(result) <= 1

    def test_path_node_keys(self):
        """Each node in a path must have id, content, entity_type."""
        nodes, edges = self._make_graph()
        result = top_paths(nodes, edges, ["gravity"])
        for path_dict in result:
            for node in path_dict["path"]:
                assert "id" in node
                assert "content" in node
                assert "entity_type" in node

    def test_path_edge_keys(self):
        """Each edge in a path must have relationship and decayed_weight."""
        nodes, edges = self._make_graph()
        result = top_paths(nodes, edges, ["gravity"])
        for path_dict in result:
            for edge in path_dict["edges"]:
                assert "relationship" in edge
                assert "decayed_weight" in edge
                assert edge["decayed_weight"] > 0.0

    def test_excludes_single_node_paths(self):
        """Source-only paths (len=1) must not appear in results."""
        nodes, edges = self._make_graph()
        result = top_paths(nodes, edges, ["gravity"])
        for path_dict in result:
            assert len(path_dict["path"]) > 1

    def test_returns_empty_list_when_query_entities_empty(self):
        nodes, edges = self._make_graph()
        result = top_paths(nodes, edges, [])
        assert result == []

    # ------------------------------------------------------------------
    # tests that inspect mock call_args_list -- fresh mock per test
    # ------------------------------------------------------------------

    def test_strength_increment_called_for_source_nodes(self):
        """Req 2.5: update_node_strength must be called for matched source nodes."""
        nodes, edges = self._make_graph()

        mock_mem = MagicMock()
        mock_mem.update_node_strength = MagicMock(return_value=None)
        mock_mem.update_edge_weight = MagicMock(return_value=None)

        with patch.dict(sys.modules, {"memory": mock_mem}):
            top_paths(nodes, edges, ["gravity"])

        called_ids = [
            call.args[0]
            for call in mock_mem.update_node_strength.call_args_list
        ]
        assert "n1" in called_ids

        for call in mock_mem.update_node_strength.call_args_list:
            node_id, new_strength, new_access_count = call.args
            if node_id == "n1":
                assert pytest.approx(new_strength, rel=1e-9) == 1.1
                assert new_access_count == 1

    def test_update_edge_weight_called_for_traversed_edges(self):
        """Req 2.6: update_edge_weight must be called for edges in paths."""
        nodes, edges = self._make_graph()

        mock_mem = MagicMock()
        mock_mem.update_node_strength = MagicMock(return_value=None)
        mock_mem.update_edge_weight = MagicMock(return_value=None)

        with patch.dict(sys.modules, {"memory": mock_mem}):
            result = top_paths(nodes, edges, ["gravity"])

        if result:
            called_ids = [
                call.args[0]
                for call in mock_mem.update_edge_weight.call_args_list
            ]
            assert len(called_ids) > 0