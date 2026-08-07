"""
entity_resolution_v2.py
------------------------
Production-grade entity resolution for the Engram ingestion pipeline.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel, Field, model_validator
from dotenv import load_dotenv

load_dotenv()

NVIDIA_NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "meta/llama-3.1-70b-instruct"

# Thresholds - tune these against your own eval set once you have one.
# Starting values below are deliberately conservative (biased toward
# CREATE_NEW / NEEDS_REVIEW over false merges).
MERGE_THRESHOLD = 0.75       # top candidate must score at least this high
SUBENTITY_THRESHOLD = 0.60   # lower bar - a related-but-distinct link is lower risk than a merge
CREATE_THRESHOLD = 0.45      # below this, don't even consider the top candidate
AMBIGUITY_MARGIN = 0.15      # if top two scores are within this margin, escalate


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class EntityCandidate(BaseModel):
    """An existing graph node retrieved via embedding similarity search."""

    id: str
    canonical_name: str
    type: str
    aliases: list[str] = Field(default_factory=list)
    sample_context: str
    valid_timeframe: str | None = None  # e.g. "2021-2024", if known - enables temporal checks


class CandidateScore(BaseModel):
    """The model's independent evaluation of ONE candidate against the mention."""

    candidate_id: str
    match_score: float = Field(ge=0.0, le=1.0)
    relationship: Literal["SAME_ENTITY", "RELATED_SUBENTITY", "UNRELATED"]
    name_signal: str  # brief: what the name comparison shows
    context_signal: str  # brief: what the surrounding context shows
    temporal_signal: str = ""  # brief: any timeframe conflict/consistency noticed


class ProvenanceMeta(BaseModel):
    source_mention: str
    model_used: str
    decided_at: str  # ISO8601, set in code at call time - not trusted from the model


class EntityResolutionResult(BaseModel):
    """
    Final resolution result. `decision` here is the CODE-COMPUTED outcome
    (see _finalize_decision) - not taken directly from the model's opinion.
    The model's job is to score candidates; the threshold/margin logic
    below decides what happens with those scores.
    """

    decision: Literal["MERGE", "CREATE_NEW", "RELATED_SUBENTITY", "NEEDS_REVIEW"]
    candidate_scores: list[CandidateScore] = Field(default_factory=list)
    merge_target_id: str | None = None
    related_subentity_of: str | None = None
    canonical_name: str
    aliases_to_add: list[str] = Field(default_factory=list)
    reasoning: str
    provenance: ProvenanceMeta

    @model_validator(mode="after")
    def enforce_guardrails(self) -> "EntityResolutionResult":
        if self.decision == "MERGE" and not self.merge_target_id:
            raise ValueError("decision=MERGE requires merge_target_id")
        if self.decision == "RELATED_SUBENTITY" and not self.related_subentity_of:
            raise ValueError("decision=RELATED_SUBENTITY requires related_subentity_of")
        if self.decision in ("CREATE_NEW", "NEEDS_REVIEW"):
            # These decisions must not silently carry a merge/subentity target -
            # a downstream writer that only checks "is there a target id" instead
            # of checking `decision` first should still fail safe.
            if self.merge_target_id or self.related_subentity_of:
                raise ValueError(
                    f"decision={self.decision} must not carry merge_target_id "
                    f"or related_subentity_of"
                )
        return self


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

ENTITY_RESOLUTION_SYSTEM = """\
You are an entity resolution engine for a production knowledge graph. You do
NOT make the final merge/create decision yourself - your job is to
independently score how well the new mention matches EACH candidate given
to you, using the record_candidate_scores tool. A separate system applies
thresholds to your scores to make the final call, so score honestly and
independently per candidate rather than trying to guess which one "wins."

For EVERY candidate provided, output:
- match_score (0.0-1.0): how likely this candidate is the SAME real-world
  entity as the new mention, OR structurally related to it (see
  relationship below) - either way, score how strong the connection is.
- relationship: one of
    SAME_ENTITY       - this candidate IS the same real-world entity
    RELATED_SUBENTITY - genuinely different entity, but structurally linked
                        (e.g. a subsidiary, product line, department, or a
                        person who currently or formerly holds a named role)
    UNRELATED         - no meaningful connection
- name_signal: one short phrase on what the name comparison shows
  (exact match, nickname, abbreviation, transliteration, coincidental
  overlap, etc.)
- context_signal: one short phrase on what the surrounding context shows
  (consistent role/domain, contradicting attributes, insufficient
  information, etc.)
- temporal_signal: one short phrase ONLY if either the mention or the
  candidate has time information relevant to the decision (e.g. a role
  that changes hands over time) - otherwise leave empty.

Resolution principles:
1. Type must be compatible for SAME_ENTITY. A Person can never be the same
   entity as an Organization or Location, even on an exact name match
   (e.g. "Washington" the person vs "Washington" the state) - score such
   cases as UNRELATED, not SAME_ENTITY.
2. Name similarity alone is insufficient evidence. Weigh context and,
   where available, temporal consistency together with the name.
3. Distinguish SAME_ENTITY from RELATED_SUBENTITY carefully. "Apple TV+"
   mentioned in a new fact is NOT the same entity as an existing "Apple
   Inc." node - it is a product/subsidiary of it. Merging these would
   incorrectly fuse distinct facts (e.g. Apple TV+ subscriber counts
   would get attributed to Apple Inc. as a whole). Score this as
   RELATED_SUBENTITY, not SAME_ENTITY.
4. Roles are not people. "The CEO" or "the mayor" mentioned without a name
   may refer to different real people at different times even against a
   candidate with an identical role - check temporal_signal for this.
5. Recognize name-variation patterns as supporting SAME_ENTITY: nicknames
   ("Bob"/"Robert"), abbreviations ("IBM"/"International Business
   Machines"), honorific/title variations ("Dr. Smith"/"John Smith"), and
   transliteration/romanization variants of non-English names (e.g.
   "Muhammad"/"Mohammed", names romanized differently from Chinese,
   Arabic, Cyrillic, or Devanagari script).
6. Score conservatively. It is far cheaper to leave a genuine duplicate
   for a later pass than to fuse two different real-world entities into
   one node - a false SAME_ENTITY corrupts every fact attached to both
   entities going forward.

Also propose:
- canonical_name: the name to store if this turns out to be a new entity,
  or confirm the existing canonical name if it's likely the same entity.
- aliases_to_add: any alternate names, spellings, or transliterations
  from this specific mention worth storing for future matching.

Call record_candidate_scores with your evaluation of every candidate
provided (an empty list if none were provided).

--- EXAMPLES ---

Example 1 - SAME_ENTITY via nickname:
  Mention: "Obama" (Person), context "...Obama signed the bill in 2010..."
  Candidate: canonical_name="Barack Obama", type="Person",
             sample_context="...President Barack Obama addressed Congress..."
  -> match_score=0.93, relationship=SAME_ENTITY,
     name_signal="Surname match to a well-known full name",
     context_signal="Same role (president) and era, no contradicting attributes"

Example 2 - UNRELATED via type mismatch:
  Mention: "Washington" (Location), context "...moved to Washington last year..."
  Candidate: canonical_name="George Washington", type="Person"
  -> match_score=0.02, relationship=UNRELATED,
     name_signal="Exact string match but incompatible entity type",
     context_signal="Mention is clearly a place, candidate is a historical person"

Example 3 - RELATED_SUBENTITY, not a merge:
  Mention: "Apple TV+" (Organization/Product), context "...Apple TV+ renewed the show for a third season..."
  Candidate: canonical_name="Apple Inc.", type="Organization",
             sample_context="...Apple Inc. reported quarterly earnings..."
  -> match_score=0.55, relationship=RELATED_SUBENTITY,
     name_signal="Shares 'Apple' but refers to a specific product/service, not the parent company",
     context_signal="Different subject matter (streaming content vs corporate earnings) - distinct but related entity"

Example 4 - genuinely ambiguous, score both honestly (don't force a winner):
  Mention: "J. Smith" (Person), context "...J. Smith submitted the quarterly compliance report..."
  Candidate A: canonical_name="John Smith", type="Person", sample_context="...appointed as the new regional sales director..."
  Candidate B: canonical_name="Jane Smith", type="Person", sample_context="...compliance officer, flagged several discrepancies..."
  -> Candidate A: match_score=0.35, relationship=UNRELATED,
       name_signal="Initial matches but full name unconfirmed",
       context_signal="Sales role does not match a compliance-report context"
  -> Candidate B: match_score=0.60, relationship=SAME_ENTITY,
       name_signal="Initial matches but full name unconfirmed",
       context_signal="Compliance officer role is consistent with submitting a compliance report, but not conclusive"
  (Scores intentionally left close/moderate - insufficient evidence to be confident either way.)
"""


# ---------------------------------------------------------------------------
# OpenRouter -> NIM call
# ---------------------------------------------------------------------------

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.environ.get("NVIDIA_API_KEY")
        if not api_key:
            raise RuntimeError("NVIDIA_API_KEY is not set.")
        _client = OpenAI(base_url=NVIDIA_NIM_BASE_URL, api_key=api_key)
    return _client


class _RawScoresResponse(BaseModel):
    """What we ask the MODEL to produce. Separate from EntityResolutionResult,
    which includes the code-computed `decision` - the model never sets that
    field directly."""

    candidate_scores: list[CandidateScore] = Field(default_factory=list)
    canonical_name: str
    aliases_to_add: list[str] = Field(default_factory=list)
    reasoning: str


def _score_candidates_via_llm(
    mention_name: str,
    mention_type: str,
    context: str,
    candidates: list[EntityCandidate],
    model: str,
    max_attempts: int = 2,
) -> _RawScoresResponse:
    client = _get_client()
    # Fix properties schema generation as NIM may be strict
    schema = _RawScoresResponse.model_json_schema()
    
    tool = {
        "type": "function",
        "function": {
            "name": "record_candidate_scores",
            "description": "Record independent match scores for every candidate.",
            "parameters": schema,
        },
    }
    user_prompt = (
        f'New entity mention:\n'
        f'  Name: "{mention_name}"\n'
        f'  Type: "{mention_type}"\n'
        f'  Context: "{context}"\n\n'
        f"Candidates to score (may be empty):\n"
        f"{json.dumps([c.model_dump() for c in candidates], indent=2)}\n\n"
        f"Call record_candidate_scores now."
    )
    messages: list[dict] = [
        {"role": "system", "content": ENTITY_RESOLUTION_SYSTEM},
        {"role": "user", "content": user_prompt},
    ]
    last_error: Exception | None = None

    for _ in range(max_attempts):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=[tool],
            tool_choice={"type": "function", "function": {"name": "record_candidate_scores"}},
        )
        message = response.choices[0].message
        calls = message.tool_calls or []
        if not calls:
            last_error = RuntimeError("model did not return a tool call")
        else:
            try:
                args = json.loads(calls[0].function.arguments)
                return _RawScoresResponse.model_validate(args)
            except Exception as e:
                last_error = e
                # Prepare for a retry
                messages.append(message.model_dump(exclude_none=True))
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": calls[0].id,
                        "content": f"Invalid input: {e}. Call record_candidate_scores again, fixed.",
                    }
                )
                continue

    raise RuntimeError(f"record_candidate_scores failed after {max_attempts} attempts: {last_error}")


def _finalize_decision(
    raw: _RawScoresResponse,
    mention_context: str,
    model: str,
) -> EntityResolutionResult:
    """
    THE CORE PRODUCTION-GRADE LOGIC: turns independently-scored candidates
    into a final decision using thresholds + margin-based ambiguity
    detection, computed here in code rather than trusted from the model.
    """
    provenance = ProvenanceMeta(
        source_mention=mention_context,
        model_used=model,
        decided_at=datetime.now(timezone.utc).isoformat(),
    )
    scores = sorted(raw.candidate_scores, key=lambda c: c.match_score, reverse=True)

    if not scores or scores[0].match_score < CREATE_THRESHOLD:
        return EntityResolutionResult(
            decision="CREATE_NEW",
            candidate_scores=scores,
            canonical_name=raw.canonical_name,
            aliases_to_add=raw.aliases_to_add,
            reasoning=raw.reasoning or "No candidate scored above the create-new threshold.",
            provenance=provenance,
        )

    top = scores[0]
    margin = (top.match_score - scores[1].match_score) if len(scores) > 1 else 1.0

    if top.relationship == "RELATED_SUBENTITY" and top.match_score >= SUBENTITY_THRESHOLD:
        return EntityResolutionResult(
            decision="RELATED_SUBENTITY",
            candidate_scores=scores,
            related_subentity_of=top.candidate_id,
            canonical_name=raw.canonical_name,
            aliases_to_add=raw.aliases_to_add,
            reasoning=raw.reasoning,
            provenance=provenance,
        )

    if (
        top.relationship == "SAME_ENTITY"
        and top.match_score >= MERGE_THRESHOLD
        and margin >= AMBIGUITY_MARGIN
    ):
        return EntityResolutionResult(
            decision="MERGE",
            candidate_scores=scores,
            merge_target_id=top.candidate_id,
            canonical_name=raw.canonical_name,
            aliases_to_add=raw.aliases_to_add,
            reasoning=raw.reasoning,
            provenance=provenance,
        )

    # High-scoring but ambiguous (close margin) or in the gray zone between
    # thresholds - surface for human review rather than guessing either way.
    return EntityResolutionResult(
        decision="NEEDS_REVIEW",
        candidate_scores=scores,
        canonical_name=raw.canonical_name,
        aliases_to_add=raw.aliases_to_add,
        reasoning=(
            raw.reasoning
            + f" [escalated: top_score={top.match_score:.2f}, margin={margin:.2f}]"
        ),
        provenance=provenance,
    )


def resolve_entity(
    mention_name: str,
    mention_type: str,
    context: str,
    candidates: list[EntityCandidate],
    model: str = DEFAULT_MODEL,
) -> EntityResolutionResult:
    """
    Resolve a newly extracted entity mention against existing graph nodes.
    `candidates` should already be narrowed by embedding similarity search
    upstream (e.g. top-5 by cosine similarity) - don't make the model
    search the whole graph.
    """
    raw = _score_candidates_via_llm(mention_name, mention_type, context, candidates, model)
    return _finalize_decision(raw, mention_context=context, model=model)
