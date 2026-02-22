from pydantic import BaseModel, Field
from typing import Dict, List, Optional

class ModerateRequest(BaseModel):
    text: str = Field(..., max_length=5000)

class ModerateResponse(BaseModel):
    labels: Dict[str, float]
    overall: float

class HealthResponse(BaseModel):
    status: str
    classifier_loaded: bool
    vllm_reachable: bool

class CandidateResponse(BaseModel):
    text: str
    overall: float
    filtered: bool
    reasons: List[str]

class ScoreDetail(BaseModel):
    labels: Dict[str, float]
    overall: float

class RewriteRequest(BaseModel):
    text: str = Field(..., max_length=5000)
    n: int = 5
    max_tokens: int = 256
    temperature: float = 0.8
    top_p: float = 0.95
    tox_threshold: float = 0.2
    debug: bool = False

class RewriteResponse(BaseModel):
    original: ScoreDetail
    rewrite: str
    rewrite_scores: ScoreDetail
    selected_from_n: int
    candidates: Optional[List[CandidateResponse]] = None
    reduction_failed: bool = False
