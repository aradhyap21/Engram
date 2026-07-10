"""
Unit tests for memorymesh/memory.py

Tests cover:
  - upsert_node
  - insert_edge
  - get_all_nodes
  - get_all_edges
  - update_node_strength
  - update_edge_weight
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Clean up any mock that test_graph.py may have left in sys.modules.
# We need the REAL memory module so we can patch its supabase attribute.
# ---------------------------------------------------------------------------
_mock_supabase_client = MagicMock()

# If test_graph.py injected a mock into sys.modules["memory"], remove it
# and reimport the real module.
if "memory" in sys.modules:
    _existing = sys.modules["memory"]
    # If it looks like a mock (no __file__), delete it
    if not hasattr(_existing, "__file__") or _existing.__file__ is None:
        del sys.modules["memory"]

# Do the same for supabase — test_graph might have put a mock there too
if "supabase" in sys.modules:
    _existing = sys.modules["supabase"]
    if not hasattr(_existing, "__file__") or _existing.__file__ is None:
        del sys.modules["supabase"]

# Now import the real memory module. We monkey-patch os.environ and
# supabase.create_client so that the module-level init doesn't fail.
with patch.dict(
    "os.environ", {"SUPABASE_URL": "http://test.supabase.co", "SUPABASE_KEY": "test-key"}
):
    import supabase as _supabase  # ensure it's loaded

    with patch.object(_supabase, "create_client", return_value=_mock_supabase_client):
        import memory  # noqa: E402

# ---------------------------------------------------------------------------
# upsert_node tests
# ---------------------------------------------------------------------------


class TestUpsertNode:
    """Tests for upsert_node function."""

    def test_inserts_new_node_when_not_exists(self):
        """New node is inserted when content doesn't exist."""
        mock_table = MagicMock()
        mock_table.select.return_value.eq.return_value.execute.return_value.data = []
        mock_table.insert.return_value.execute.return_value.data = [
            {"id": "new-uuid", "content": "Einstein"}
        ]
        _mock_supabase_client.table.return_value = mock_table

        with patch.object(memory, "supabase", _mock_supabase_client):
            result = memory.upsert_node("Einstein", "person")
            assert "id" in result
            assert result["content"] == "Einstein"

    def test_returns_existing_node_when_content_exists(self):
        """Existing node is returned without creating duplicate."""
        existing_node = {"id": "existing-uuid", "content": "Einstein", "strength": 1.0}
        mock_table = MagicMock()
        mock_table.select.return_value.eq.return_value.execute.return_value.data = [
            existing_node
        ]
        _mock_supabase_client.table.return_value = mock_table

        with patch.object(memory, "supabase", _mock_supabase_client):
            result = memory.upsert_node("Einstein", "person")
            assert result == existing_node

    def test_raises_on_insert_failure(self):
        """Raises RuntimeError when Supabase insert fails."""
        mock_table = MagicMock()
        mock_table.select.return_value.eq.return_value.execute.return_value.data = []
        mock_table.insert.return_value.execute.side_effect = Exception("DB error")
        _mock_supabase_client.table.return_value = mock_table

        with patch.object(memory, "supabase", _mock_supabase_client):
            with pytest.raises(RuntimeError, match="Failed to insert"):
                memory.upsert_node("Test", "entity")


# ---------------------------------------------------------------------------
# insert_edge tests
# ---------------------------------------------------------------------------


class TestInsertEdge:
    """Tests for insert_edge function."""

    def test_inserts_edge_successfully(self):
        """Edge is inserted with correct data."""
        mock_table = MagicMock()
        mock_table.insert.return_value.execute.return_value.data = [
            {"id": "edge-uuid", "from_id": "a", "to_id": "b", "relationship": "causes"}
        ]
        _mock_supabase_client.table.return_value = mock_table

        with patch.object(memory, "supabase", _mock_supabase_client):
            result = memory.insert_edge("a", "b", "causes", 1.0)
            assert result is not None

    def test_raises_on_self_loop(self):
        """Self-loop (from_id == to_id) raises ValueError."""
        with patch.object(memory, "supabase", _mock_supabase_client):
            with pytest.raises(ValueError, match="Self-loop"):
                memory.insert_edge("same-id", "same-id", "relates")

    def test_raises_on_insert_failure(self):
        """Raises RuntimeError when Supabase insert fails."""
        mock_table = MagicMock()
        mock_table.insert.return_value.execute.side_effect = Exception("DB error")
        _mock_supabase_client.table.return_value = mock_table

        with patch.object(memory, "supabase", _mock_supabase_client):
            with pytest.raises(RuntimeError, match="Failed to insert edge"):
                memory.insert_edge("a", "b", "relates")


# ---------------------------------------------------------------------------
# get_all_nodes tests
# ---------------------------------------------------------------------------


class TestGetAllNodes:
    """Tests for get_all_nodes function."""

    def test_returns_all_nodes(self):
        """Returns list of all nodes."""
        mock_nodes = [{"id": "1", "content": "A"}, {"id": "2", "content": "B"}]
        mock_table = MagicMock()
        mock_table.select.return_value.execute.return_value.data = mock_nodes
        _mock_supabase_client.table.return_value = mock_table

        with patch.object(memory, "supabase", _mock_supabase_client):
            result = memory.get_all_nodes()
            assert len(result) == 2

    def test_returns_empty_list_on_no_nodes(self):
        """Returns empty list when table is empty."""
        mock_table = MagicMock()
        mock_table.select.return_value.execute.return_value.data = []
        _mock_supabase_client.table.return_value = mock_table

        with patch.object(memory, "supabase", _mock_supabase_client):
            result = memory.get_all_nodes()
            assert result == []

    def test_raises_on_failure(self):
        """Raises RuntimeError on database error."""
        mock_table = MagicMock()
        mock_table.select.return_value.execute.side_effect = Exception("DB error")
        _mock_supabase_client.table.return_value = mock_table

        with patch.object(memory, "supabase", _mock_supabase_client):
            with pytest.raises(RuntimeError, match="Failed to fetch"):
                memory.get_all_nodes()


# ---------------------------------------------------------------------------
# get_all_edges tests
# ---------------------------------------------------------------------------


class TestGetAllEdges:
    """Tests for get_all_edges function."""

    def test_returns_all_edges(self):
        """Returns list of all edges."""
        mock_edges = [{"id": "1", "from_id": "a", "to_id": "b"}]
        mock_table = MagicMock()
        mock_table.select.return_value.execute.return_value.data = mock_edges
        _mock_supabase_client.table.return_value = mock_table

        with patch.object(memory, "supabase", _mock_supabase_client):
            result = memory.get_all_edges()
            assert len(result) == 1

    def test_raises_on_failure(self):
        """Raises RuntimeError on database error."""
        mock_table = MagicMock()
        mock_table.select.return_value.execute.side_effect = Exception("DB error")
        _mock_supabase_client.table.return_value = mock_table

        with patch.object(memory, "supabase", _mock_supabase_client):
            with pytest.raises(RuntimeError, match="Failed to fetch"):
                memory.get_all_edges()


# ---------------------------------------------------------------------------
# update_node_strength tests
# ---------------------------------------------------------------------------


class TestUpdateNodeStrength:
    """Tests for update_node_strength function."""

    def test_updates_strength_and_access_count(self):
        """Updates both strength and access_count for a node."""
        mock_table = MagicMock()
        _mock_supabase_client.table.return_value = mock_table

        with patch.object(memory, "supabase", _mock_supabase_client):
            memory.update_node_strength("node-uuid", 1.5, 3)

            # Verify update was called with correct data
            mock_table.update.assert_called_once()
            call_args = mock_table.update.call_args
            assert call_args[0][0]["strength"] == 1.5
            assert call_args[0][0]["access_count"] == 3

    def test_raises_on_failure(self):
        """Raises RuntimeError on database error."""
        mock_table = MagicMock()
        mock_table.update.return_value.eq.return_value.execute.side_effect = Exception(
            "DB error"
        )
        _mock_supabase_client.table.return_value = mock_table

        with patch.object(memory, "supabase", _mock_supabase_client):
            with pytest.raises(RuntimeError, match="Failed to update strength"):
                memory.update_node_strength("node-uuid", 1.0, 1)


# ---------------------------------------------------------------------------
# update_edge_weight tests
# ---------------------------------------------------------------------------


class TestUpdateEdgeWeight:
    """Tests for update_edge_weight function."""

    def test_updates_weight(self):
        """Updates weight for an edge."""
        mock_table = MagicMock()
        _mock_supabase_client.table.return_value = mock_table

        with patch.object(memory, "supabase", _mock_supabase_client):
            memory.update_edge_weight("edge-uuid", 0.75)

            mock_table.update.assert_called_once()
            call_args = mock_table.update.call_args
            assert call_args[0][0]["weight"] == 0.75

    def test_raises_on_failure(self):
        """Raises RuntimeError on database error."""
        mock_table = MagicMock()
        mock_table.update.return_value.eq.return_value.execute.side_effect = Exception(
            "DB error"
        )
        _mock_supabase_client.table.return_value = mock_table

        with patch.object(memory, "supabase", _mock_supabase_client):
            with pytest.raises(RuntimeError, match="Failed to update weight"):
                memory.update_edge_weight("edge-uuid", 0.5)