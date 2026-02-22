import logging
import httpx
from fastapi import FastAPI, Depends, HTTPException
from api.schemas import ModerateRequest, ModerateResponse, HealthResponse, RewriteRequest, RewriteResponse, ScoreDetail
from api.moderation import classifier
from api.rewrite import generate_rewrite_candidates, pick_best_candidate
from api.config import settings
from api.logging_config import setup_logging, LoggingMiddleware

# Initialize logging
setup_logging(settings.LOG_LEVEL)

app = FastAPI(title="ToxScreen — Scalable Toxicity Detection & Rewrite Service")

# Add middleware
app.add_middleware(LoggingMiddleware)

@app.get("/health", response_model=HealthResponse)
async def health():
    # Check vLLM reachability
    vllm_reachable = False
    try:
        async with httpx.AsyncClient(timeout=1.0) as client:
            response = await client.get(f"{settings.VLLM_BASE_URL}/v1/models")
            if response.status_code == 200:
                vllm_reachable = True
    except Exception:
        pass

    return HealthResponse(
        status="ok",
        classifier_loaded=classifier.is_loaded,
        vllm_reachable=vllm_reachable
    )

@app.post("/moderate", response_model=ModerateResponse)
async def moderate(request: ModerateRequest):
    if not classifier.is_loaded:
        raise HTTPException(status_code=503, detail="Classifier not loaded")
    
    try:
        results = classifier.score_text(request.text)
        if "error" in results:
            raise HTTPException(status_code=500, detail=results["error"])
        return ModerateResponse(**results)
    except Exception as e:
        logging.error(f"Error during moderation: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during moderation")

@app.post("/rewrite_safe", response_model=RewriteResponse)
async def rewrite_safe(request: RewriteRequest):
    if not classifier.is_loaded:
        raise HTTPException(status_code=503, detail="Classifier not loaded")
    
    # 1. Score original
    original_scores = classifier.score_text(request.text)
    
    # 2. If already safe, we might skip or just do 1 candidate
    if original_scores["overall"] < request.tox_threshold:
        # Still do a rewrite for consistency, or return original
        # Let's generate 1 candidate anyway to 'polish' it
        pass
        
    # 3. Generate candidates
    try:
        candidates = await generate_rewrite_candidates(
            request.text, 
            n=request.n, 
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"vLLM service error: {str(e)}")
        
    # 4. Pick best
    best, scored_all = pick_best_candidate(request.text, candidates, request.tox_threshold)
    
    reduction_failed = best["overall"] > original_scores["overall"] and best["overall"] > request.tox_threshold

    return RewriteResponse(
        original=ScoreDetail(**original_scores),
        rewrite=best["text"],
        rewrite_scores=ScoreDetail(labels=best["labels"], overall=best["overall"]),
        selected_from_n=len(candidates),
        candidates=scored_all if request.debug else None,
        reduction_failed=reduction_failed
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=settings.API_PORT)
