from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
from .engine import RLMEngine, RLMConfig, RLMStats

router = APIRouter()

class RLMQueryRequest(BaseModel):
    context: str
    query: str
    max_recursion_depth: Optional[int] = None
    max_total_llm_subcalls: Optional[int] = None
    max_wall_clock_seconds: Optional[float] = None
    max_total_tokens: Optional[int] = None

class RLMQueryResponse(BaseModel):
    answer: str
    stats: dict

@router.post("/query", response_model=RLMQueryResponse)
def rlm_query(request: RLMQueryRequest):
    # Construct config, applying overrides if provided
    config_kwargs = {}
    if request.max_recursion_depth is not None:
        config_kwargs["max_recursion_depth"] = request.max_recursion_depth
    if request.max_total_llm_subcalls is not None:
        config_kwargs["max_total_llm_subcalls"] = request.max_total_llm_subcalls
    if request.max_wall_clock_seconds is not None:
        config_kwargs["max_wall_clock_seconds"] = request.max_wall_clock_seconds
    if request.max_total_tokens is not None:
        config_kwargs["max_total_tokens"] = request.max_total_tokens
        
    config = RLMConfig(**config_kwargs)
    engine = RLMEngine(config)
    
    answer, stats = engine.query(request.context, request.query)
    
    return RLMQueryResponse(
        answer=answer,
        stats={
            "subcalls_made": stats.subcalls_made,
            "recursion_depth_reached": stats.recursion_depth_reached,
            "tokens_used": stats.tokens_used,
            "elapsed_seconds": stats.elapsed_seconds,
            "capped": stats.capped
        }
    )
