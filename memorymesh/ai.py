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

    response = client.chat.completions.create(
        model="meta/llama-3.1-70b-instruct",
        messages=[
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=1024,
    )

    content = response.choices[0].message.content or ""

    # Primary parse attempt
    result = None
    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        # Regex fallback: grab the first {...} block in the response
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group())
            except json.JSONDecodeError:
                pass

    if result is None:
        raise ValueError("LLM returned non-JSON response")

    # Validate required keys
    if "entities" not in result or not isinstance(result["entities"], list):
        raise ValueError(
            'LLM response is missing required key "entities" (list)'
        )
    if "relationships" not in result or not isinstance(result["relationships"], list):
        raise ValueError(
            'LLM response is missing required key "relationships" (list)'
        )

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

    response = client.chat.completions.create(
        model="meta/llama-3.1-70b-instruct",
        messages=[
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=512,
    )

    return (response.choices[0].message.content or "").strip()
