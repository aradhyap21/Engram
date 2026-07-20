"""
Unit tests for memorymesh/ai.py

Tests cover:
  - extract_entities valid JSON path
  - extract_entities regex fallback path
  - extract_entities non-JSON raises ValueError
  - synthesize_insight with and without paths
"""

import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# Set required env vars before importing ai module
os.environ.setdefault("NVIDIA_API_KEY", "test-key")

# Create mock OpenAI client
_mock_client = MagicMock()

# ---------------------------------------------------------------------------
# Patch openai.OpenAI and dotenv at import time
# ---------------------------------------------------------------------------

with patch.dict(
    sys.modules,
    {
        "openai": MagicMock(),
        "dotenv": MagicMock(),
    },
):
    from ..ai import extract_entities, synthesize_insight  # noqa: E402

# Configure the mock client to be returned by OpenAI()
from openai import OpenAI as _OpenAI

# Get the actual mock client we created
_mock_client.chat.completions.create.return_value = None


def _reset_openai_mock():
    """Reset the mock's call history and response."""
    _mock_client.chat.completions.create.reset_mock()
    _mock_client.chat.completions.create.return_value = None


def _make_completion_response(content: str):
    """Build a fake chat.completions.create response with the given content."""
    mock_choice = MagicMock()
    mock_choice.message.content = content
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    return mock_response


class TestExtractEntities:
    """Tests for extract_entities function."""

    def test_valid_json_extraction(self):
        """Happy path: LLM returns valid JSON with entities and relationships."""
        _reset_openai_mock()
        response_data = {
            "entities": ["Alice", "Bob", "Company"],
            "relationships": [
                {"from": "Alice", "to": "Bob", "type": "knows"},
                {"from": "Bob", "to": "Company", "type": "works_at"},
            ],
        }
        _mock_client.chat.completions.create.return_value = _make_completion_response(
            json.dumps(response_data)
        )

        result = extract_entities("Alice knows Bob. Bob works at Company.")

        assert result == response_data
        _mock_client.chat.completions.create.assert_called_once()

    def test_regex_fallback_when_llm_adds_prose(self):
        """LLM wraps JSON in explanatory text; regex fallback extracts it."""
        _reset_openai_mock()
        raw_response = (
            "Here is the JSON you requested:\n"
            '{"entities": ["Newton", "gravity"], "relationships": [{"from": "Newton", "to": "gravity", "type": "discovered"}]}'
            "\nLet me know if you need anything else."
        )
        _mock_client.chat.completions.create.return_value = _make_completion_response(
            raw_response
        )

        result = extract_entities("Newton discovered gravity.")

        assert result["entities"] == ["Newton", "gravity"]
        assert len(result["relationships"]) == 1
        assert result["relationships"][0]["from"] == "Newton"
        assert result["relationships"][0]["to"] == "gravity"
        assert result["relationships"][0]["type"] == "discovered"

    def test_non_json_raises_value_error(self):
        """Pure non-JSON response raises ValueError with clear message."""
        _reset_openai_mock()
        _mock_client.chat.completions.create.return_value = _make_completion_response(
            "I cannot comply with that request. Here is some prose instead of JSON."
        )

        with pytest.raises(ValueError, match="LLM returned non-JSON response"):
            extract_entities("Some text")

    def test_missing_entities_key_raises(self):
        """Response missing 'entities' key raises ValueError."""
        _reset_openai_mock()
        _mock_client.chat.completions.create.return_value = _make_completion_response(
            json.dumps({"relationships": []})
        )

        with pytest.raises(ValueError, match='missing required key "entities"'):
            extract_entities("Some text")

    def test_missing_relationships_key_raises(self):
        """Response missing 'relationships' key raises ValueError."""
        _reset_openai_mock()
        _mock_client.chat.completions.create.return_value = _make_completion_response(
            json.dumps({"entities": ["a"]})
        )

        with pytest.raises(ValueError, match='missing required key "relationships"'):
            extract_entities("Some text")

    def test_entities_not_a_list_raises(self):
        """Response with 'entities' as non-list raises ValueError."""
        _reset_openai_mock()
        _mock_client.chat.completions.create.return_value = _make_completion_response(
            json.dumps({"entities": "not-a-list", "relationships": []})
        )

        with pytest.raises(ValueError, match='missing required key "entities"'):
            extract_entities("Some text")

    def test_relationships_not_a_list_raises(self):
        """Response with 'relationships' as non-list raises ValueError."""
        _reset_openai_mock()
        _mock_client.chat.completions.create.return_value = _make_completion_response(
            json.dumps({"entities": ["a"], "relationships": "not-a-list"})
        )

        with pytest.raises(ValueError, match='missing required key "relationships"'):
            extract_entities("Some text")

    def test_empty_response_raises(self):
        """Empty response raises ValueError."""
        _reset_openai_mock()
        _mock_client.chat.completions.create.return_value = _make_completion_response(
            ""
        )

        with pytest.raises(ValueError, match="LLM returned non-JSON response"):
            extract_entities("Some text")

    def test_malformed_json_raises(self):
        """Malformed JSON (e.g., trailing comma) raises ValueError."""
        _reset_openai_mock()
        _mock_client.chat.completions.create.return_value = _make_completion_response(
            '{"entities": ["a"], "relationships": [], }'
        )

        with pytest.raises(ValueError, match="LLM returned non-JSON response"):
            extract_entities("Some text")


class TestSynthesizeInsight:
    """Tests for synthesize_insight function."""

    def test_with_paths_returns_insight(self):
        """Non-empty paths list produces an insight string."""
        _reset_openai_mock()
        paths = [
            {
                "path": [
                    {"id": "1", "content": "Newton", "entity_type": "person"},
                    {"id": "2", "content": "gravity", "entity_type": "concept"},
                ],
                "edges": [
                    {"relationship": "discovered", "decayed_weight": 0.9}
                ],
                "total_cost": 0.9,
            }
        ]
        _mock_client.chat.completions.create.return_value = _make_completion_response(
            "Newton's discovery of gravity fundamentally changed our understanding of physics."
        )

        result = synthesize_insight(paths, "Who discovered gravity?")

        assert isinstance(result, str)
        assert len(result) > 0
        assert "Newton" in result or "gravity" in result

    def test_empty_paths_returns_non_error_string(self):
        """Empty paths list still returns a string (not an error)."""
        _reset_openai_mock()
        _mock_client.chat.completions.create.return_value = _make_completion_response(
            "No relevant memories were found to synthesize an insight."
        )

        result = synthesize_insight([], "unknown query")

        assert isinstance(result, str)
        assert len(result) > 0
        # Should not raise; should handle empty paths gracefully

    def test_passes_query_and_serialized_paths_to_prompt(self):
        """The prompt should include both the query and serialized paths."""
        _reset_openai_mock()
        paths = [{"path": [], "edges": [], "total_cost": 0.0}]
        query = "test query"

        _mock_client.chat.completions.create.return_value = _make_completion_response(
            "insight"
        )

        synthesize_insight(paths, query)

        # Verify the call was made with a prompt containing both query and paths
        call_args = _mock_client.chat.completions.create.call_args
        assert call_args is not None
        prompt = call_args[1]["messages"][0]["content"]
        assert "test query" in prompt
        assert "memory paths" in prompt.lower()