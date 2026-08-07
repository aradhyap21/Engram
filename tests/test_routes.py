"""
Unit tests for memorymesh/main.py (FastAPI routes)

Tests cover:
  - /health endpoint
  - /memory POST endpoint
  - /memory/retrieve GET endpoint
  - /memory/synthesize GET endpoint
"""

import json
import sys
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Mock dependencies at module level — FORCE inject regardless of what
# test_graph.py or other test modules may have left in sys.modules.
# ---------------------------------------------------------------------------

mock_memory = MagicMock()
mock_ai = MagicMock()
mock_graph = MagicMock()
mock_entity_resolution = MagicMock()

sys.modules["backend.memory"] = mock_memory
sys.modules["backend.ai"] = mock_ai
sys.modules["backend.graph"] = mock_graph
sys.modules["backend.upload"] = MagicMock()
sys.modules["backend.entity_resolution_v2"] = mock_entity_resolution

# Load main.py with env vars set so startup validation passes
with patch.dict("os.environ", {
    "SUPABASE_URL": "http://test.supabase.co",
    "SUPABASE_KEY": "test-key",
    "NVIDIA_API_KEY": "test-nvidia-key",
}):
    from backend.main import app  # noqa: E402

client = TestClient(app)


# ---------------------------------------------------------------------------
# /health endpoint tests
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_returns_ok(self):
        """GET /health returns status ok."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# /memory POST endpoint tests
# ---------------------------------------------------------------------------


class TestMemoryPostEndpoint:
    """Tests for the POST /memory endpoint."""

    def setup_method(self):
        """Reset mocks before each test — also clear child mock attributes."""
        mock_ai.extract_entities.side_effect = None
        mock_ai.extract_entities.return_value = MagicMock()
        mock_ai.extract_embedding.side_effect = None
        mock_ai.extract_embedding.return_value = [0.1] * 1024
        mock_ai.synthesize_insight.side_effect = None
        mock_ai.synthesize_insight.return_value = MagicMock()
        mock_memory.upsert_node.side_effect = None
        mock_memory.upsert_node.return_value = MagicMock()
        mock_memory.insert_resolved_node.side_effect = None
        mock_memory.insert_resolved_node.return_value = {"id": "uuid-mock", "content": "test"}
        mock_memory.get_similar_candidates.side_effect = None
        mock_memory.get_similar_candidates.return_value = []
        mock_memory.insert_edge.side_effect = None
        mock_memory.insert_edge.return_value = MagicMock()
        mock_memory.get_all_nodes.side_effect = None
        mock_memory.get_all_nodes.return_value = MagicMock()
        mock_memory.get_all_edges.side_effect = None
        mock_memory.get_all_edges.return_value = MagicMock()
        mock_graph.top_paths.side_effect = None
        mock_graph.top_paths.return_value = MagicMock()
        mock_entity_resolution.resolve_entity.return_value = MagicMock(
            decision="CREATE_NEW",
            canonical_name="test",
            aliases_to_add=[],
            reasoning="test",
            related_subentity_of=None,
            provenance=MagicMock(
                source_mention="test",
                model_used="test",
                decided_at="2024-01-01T00:00:00Z",
            ),
        )
        mock_ai.reset_mock(return_value=True, side_effect=True)
        mock_memory.reset_mock(return_value=True, side_effect=True)
        mock_graph.reset_mock(return_value=True, side_effect=True)
        # Re-apply defaults after reset
        mock_ai.extract_embedding.return_value = [0.1] * 1024
        mock_memory.insert_resolved_node.return_value = {"id": "uuid-mock", "content": "test"}
        mock_memory.get_similar_candidates.return_value = []

    def test_store_memory_success(self):
        """Successful memory storage returns counts and node IDs."""
        mock_ai.extract_entities.return_value = {
            "entities": ["Einstein", "Relativity"],
            "relationships": [{"from": "Einstein", "to": "Relativity", "type": "developed"}],
        }
        # Resolution pipeline calls insert_resolved_node (not upsert_node) for CREATE_NEW
        mock_memory.insert_resolved_node.side_effect = [
            {"id": "uuid-1", "content": "Einstein"},
            {"id": "uuid-2", "content": "Relativity"},
        ]
        mock_memory.insert_edge.return_value = {"id": "edge-1"}

        response = client.post("/memory", json={"text": "Einstein developed Relativity."})

        assert response.status_code == 200
        data = response.json()
        assert data["nodes_stored"] == 2
        assert data["edges_stored"] == 1
        assert len(data["node_ids"]) == 2

    def test_store_memory_empty_text_returns_400(self):
        """Empty text returns 400 error."""
        response = client.post("/memory", json={"text": ""})
        assert response.status_code == 400
        assert "error" in response.json()

    def test_store_memory_whitespace_only_returns_400(self):
        """Whitespace-only text returns 400 error."""
        response = client.post("/memory", json={"text": "   "})
        assert response.status_code == 400

    def test_store_memory_extraction_failure_returns_422(self):
        """LLM extraction failure returns 422 error."""
        mock_ai.extract_entities.side_effect = ValueError("Parse failed")

        response = client.post("/memory", json={"text": "some text"})
        assert response.status_code == 422
        assert "error" in response.json()

    def test_store_memory_database_failure_returns_503(self):
        """Supabase failure returns 503 error."""
        mock_ai.extract_entities.return_value = {"entities": ["test"], "relationships": []}
        mock_memory.upsert_node.side_effect = RuntimeError("DB error")

        response = client.post("/memory", json={"text": "test"})
        assert response.status_code == 503
        assert "error" in response.json()


# ---------------------------------------------------------------------------
# /memory/retrieve GET endpoint tests
# ---------------------------------------------------------------------------


class TestMemoryRetrieveEndpoint:
    """Tests for the GET /memory/retrieve endpoint."""

    def setup_method(self):
        """Reset mocks before each test — also clear child mock attributes."""
        mock_ai.extract_entities.side_effect = None
        mock_ai.extract_entities.return_value = MagicMock()
        mock_memory.get_all_nodes.side_effect = None
        mock_memory.get_all_nodes.return_value = MagicMock()
        mock_memory.get_all_edges.side_effect = None
        mock_memory.get_all_edges.return_value = MagicMock()
        mock_graph.top_paths.side_effect = None
        mock_graph.top_paths.return_value = MagicMock()
        mock_ai.reset_mock()
        mock_memory.reset_mock()

    def test_retrieve_returns_paths(self):
        """Retrieve returns paths for matching entities."""
        mock_ai.extract_entities.return_value = {"entities": ["Einstein"], "relationships": []}
        mock_memory.get_all_nodes.return_value = [
            {"id": "1", "content": "Einstein", "entity_type": "person", "strength": 1.0, "access_count": 0},
        ]
        mock_memory.get_all_edges.return_value = []
        mock_graph.top_paths.return_value = []

        response = client.get("/memory/retrieve?query=Einstein")
        assert response.status_code == 200
        data = response.json()
        assert "query" in data
        assert "paths" in data

    def test_retrieve_empty_query_returns_empty_paths(self):
        """Empty query returns empty paths without error."""
        mock_memory.get_all_nodes.return_value = []
        mock_memory.get_all_edges.return_value = []
        mock_graph.top_paths.return_value = []

        response = client.get("/memory/retrieve?query=")
        assert response.status_code == 200
        data = response.json()
        assert data["paths"] == []

    def test_retrieve_missing_query_param_returns_empty(self):
        """Missing query parameter returns empty paths."""
        mock_memory.get_all_nodes.return_value = []
        mock_memory.get_all_edges.return_value = []
        mock_graph.top_paths.return_value = []

        response = client.get("/memory/retrieve")
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# /memory/synthesize GET endpoint tests
# ---------------------------------------------------------------------------


class TestMemorySynthesizeEndpoint:
    """Tests for the GET /memory/synthesize endpoint."""

    def setup_method(self):
        """Reset mocks before each test — also clear child mock attributes."""
        mock_ai.extract_entities.side_effect = None
        mock_ai.extract_entities.return_value = MagicMock()
        mock_ai.synthesize_insight.side_effect = None
        mock_ai.synthesize_insight.return_value = MagicMock()
        mock_memory.get_all_nodes.side_effect = None
        mock_memory.get_all_nodes.return_value = MagicMock()
        mock_memory.get_all_edges.side_effect = None
        mock_memory.get_all_edges.return_value = MagicMock()
        mock_graph.top_paths.side_effect = None
        mock_graph.top_paths.return_value = MagicMock()
        mock_ai.reset_mock()
        mock_memory.reset_mock()
        mock_graph.reset_mock()

    def test_synthesize_returns_insight(self):
        """Synthesize returns insight and path count."""
        mock_ai.extract_entities.return_value = {"entities": ["Einstein"], "relationships": []}
        mock_memory.get_all_nodes.return_value = []
        mock_memory.get_all_edges.return_value = []
        mock_graph.top_paths.return_value = []
        mock_ai.synthesize_insight.return_value = "An insight about Einstein."

        response = client.get("/memory/synthesize?query=Einstein")
        assert response.status_code == 200
        data = response.json()
        assert "query" in data
        assert "insight" in data
        assert "paths_used" in data

    def test_synthesize_empty_query(self):
        """Empty query still calls synthesis."""
        mock_memory.get_all_nodes.return_value = []
        mock_memory.get_all_edges.return_value = []
        mock_ai.synthesize_insight.return_value = "No memories found."

        response = client.get("/memory/synthesize?query=")
        assert response.status_code == 200
        data = response.json()
        assert data["paths_used"] == 0

    def test_synthesize_handles_errors(self):
        """Errors during retrieval are handled gracefully."""
        mock_ai.extract_entities.side_effect = ValueError("Failed")

        response = client.get("/memory/synthesize?query=test")
        assert response.status_code == 422
        assert "error" in response.json()