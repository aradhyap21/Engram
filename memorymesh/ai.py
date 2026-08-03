"""
MemoryMesh — NVIDIA NIM AI client.

Wraps the OpenAI-compatible NVIDIA NIM SDK to provide entity extraction
and LLM insight synthesis as pure, stateless functions.
"""

import os
import json
import re
from openai import OpenAI
from dotenv import load_dotenv

# Load .env at module level so NVIDIA_API_KEY is available
load_dotenv()

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
if not NVIDIA_API_KEY:
    raise RuntimeError(
        "NVIDIA_API_KEY is not set. "
        "Add it to your .env file or export it as an environment variable."
    )

# Module-level OpenAI-compatible client pointed at NVIDIA NIM
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY,
)

# ---------------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------------

_EXTRACTION_PROMPT = """\
Extract all entities and their relationships from this text.
Return JSON only with no additional text, explanation, or markdown:
{{"entities": ["entity1", "entity2"], "relationships": [{{"from": "entity1", "to": "entity2", "type": "relationship_type"}}]}}

Text: {text}"""


def _call_llm(prompt: str, temperature: float = 0.1, max_tokens: int = 1024) -> str:
    """
    Send *prompt* to the NVIDIA NIM LLM and return the raw response string.

    Args:
        prompt:      The full prompt string to send.
        temperature: Sampling temperature (default 0.1 for deterministic output).
        max_tokens:  Maximum tokens in the response.

    Returns:
        The raw text content from the first completion choice.
    """
    response = client.chat.completions.create(
        model="meta/llama-3.1-70b-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


def _parse_llm_json(content: str) -> dict:
    """
    Parse a JSON dict from *content*, with a regex fallback for wrapped JSON.

    Tries a direct ``json.loads`` first; if that fails, searches for the
    first ``{...}`` block using a regex and retries.

    Args:
        content: Raw string from the LLM that should contain a JSON object.

    Returns:
        Parsed dict.

    Raises:
        ValueError: If neither parse strategy succeeds.
    """
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    match = re.search(r'\{.*\}', content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    raise ValueError("LLM returned non-JSON response")


def _validate_extraction(result: dict) -> None:
    """
    Assert that *result* contains the required ``entities`` and
    ``relationships`` list keys.

    Args:
        result: Parsed dict from the LLM response.

    Raises:
        ValueError: If a required key is absent or has the wrong type.
    """
    if "entities" not in result or not isinstance(result["entities"], list):
        raise ValueError('LLM response is missing required key "entities" (list)')
    if "relationships" not in result or not isinstance(result["relationships"], list):
        raise ValueError('LLM response is missing required key "relationships" (list)')


def extract_entities(text: str) -> dict:
    """
    Call the NVIDIA NIM API to extract entities and relationships from *text*.

    Args:
        text: Non-empty input text to analyse.

    Returns:
        A dict with keys:
            "entities"      – list of entity strings
            "relationships" – list of dicts, each with "from", "to", "type"

    Raises:
        ValueError: If the LLM response cannot be parsed as valid JSON or
                    the expected keys are absent.
    """
    prompt = _EXTRACTION_PROMPT.format(text=text)
    content = _call_llm(prompt, temperature=0.1, max_tokens=1024)
    result = _parse_llm_json(content)
    _validate_extraction(result)
    return result


# ---------------------------------------------------------------------------
# Insight synthesis
# ---------------------------------------------------------------------------

_SYNTHESIS_PROMPT = """\
You are a knowledge synthesis assistant. Based on the memory paths below, \
provide a concise insight that connects non-obvious relationships and answers \
the query. Speak directly and do not repeat the raw path data.

Query: {query}

Memory paths:
{paths}

Insight:"""


def synthesize_insight(paths: list, query: str) -> str:
    """
    Generate a synthesised insight from the provided causal memory paths.

    Args:
        paths: List of path dicts returned by graph.top_paths().
        query: The original user query string.

    Returns:
        A plain-text insight string.
    """
    if paths:
        serialised = json.dumps(paths, indent=2)
    else:
        serialised = "(no memory paths were found for this query)"

    prompt = _SYNTHESIS_PROMPT.format(query=query, paths=serialised)
    return _call_llm(prompt, temperature=0.7, max_tokens=512).strip()
