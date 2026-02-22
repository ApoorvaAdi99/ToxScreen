import difflib
import logging
from api.vllm_client import vllm_client
from api.moderation import classifier
from api.filters import ContentFilter

content_filter = ContentFilter()

REWRITE_PROMPT_TEMPLATE = """You are a helpful and civil assistant. Your task is to rewrite the user's input to be polite, non-toxic, and safe while preserving the original meaning as much as possible.

Rules:
1. Remove all slurs, personal attacks, harassment, and threats.
2. Do not add any new facts or information.
3. Keep the length and style similar to the original if possible.
4. If the input is already safe, you can provide a slightly more polished version or keep it as is.

Original text: {text}

Rewrite:"""

def calculate_similarity(original: str, candidate: str) -> float:
    """Calculate a simple meaning preservation proxy (SequenceMatcher ratio)."""
    return difflib.SequenceMatcher(None, original.lower(), candidate.lower()).ratio()

async def generate_rewrite_candidates(text: str, n: int = 5, **kwargs) -> list[str]:
    """Generate N rewrite candidates from vLLM."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant that rewrites toxic text into safe, civil text."},
        {"role": "user", "content": REWRITE_PROMPT_TEMPLATE.format(text=text)}
    ]
    
    candidates = await vllm_client.generate_n_candidates(messages, n, **kwargs)
    # Deduplicate and normalize
    unique_candidates = list(set([c.strip() for c in candidates if c.strip()]))
    return unique_candidates

def pick_best_candidate(original_text: str, candidates: list[str], tox_threshold: float = 0.2):
    """
    Score candidates and pick the best one.
    Selection logic:
    1. Filter out candidates that fail heuristic filters.
    2. Score remaining with classifier.
    3. Select by: lowest toxicity, then highest similarity.
    """
    scored_candidates = []
    
    for cand in candidates:
        is_filtered, reasons = content_filter.check(cand)
        scores = classifier.score_text(cand)
        similarity = calculate_similarity(original_text, cand)
        
        # Scoring logic: penalize filtered candidates
        # We can still include them in the 'candidates' list for debug
        sort_score = scores["overall"]
        if is_filtered:
            sort_score += 10.0 # Large penalty
            
        scored_candidates.append({
            "text": cand,
            "overall": scores["overall"],
            "labels": scores["labels"],
            "filtered": is_filtered,
            "reasons": reasons,
            "similarity": similarity,
            "sort_score": sort_score
        })
    
    # Sort by sort_score (toxicity + penalty), then similarity (descending)
    scored_candidates.sort(key=lambda x: (x["sort_score"], -x["similarity"]))
    
    best = scored_candidates[0]
    return best, scored_candidates
