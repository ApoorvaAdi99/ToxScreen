import asyncio
import json
import os
import argparse
import pandas as pd
from datasets import load_dataset
from api.rewrite import generate_rewrite_candidates, pick_best_candidate, calculate_similarity
from api.moderation import classifier
from api.config import settings

async def run_evaluation(args):
    os.makedirs("reports", exist_ok=True)
    
    print("Loading test samples...")
    # Load test set and filter for toxic samples
    dataset = load_dataset("google/civil_comments", split="test")
    # Take a small sample for speed
    toxic_dataset = dataset.filter(lambda x: x["toxicity"] >= 0.5)
    
    if args.limit:
        toxic_dataset = toxic_dataset.select(range(min(args.limit, len(toxic_dataset))))
    
    results = []
    
    print(f"Evaluating {len(toxic_dataset)} toxic samples...")
    for i, example in enumerate(toxic_dataset):
        text = example["text"]
        original_tox = example["toxicity"]
        
        print(f"[{i+1}/{len(toxic_dataset)}] Rewriting: {text[:50]}...")
        
        try:
            # Generate and pick best
            candidates = await generate_rewrite_candidates(text, n=args.n)
            best, _ = pick_best_candidate(text, candidates, args.tox_threshold)
            
            rewrite_text = best["text"]
            rewrite_tox = best["overall"]
            similarity = best["similarity"]
            
            results.append({
                "original_text": text,
                "original_toxicity": original_tox,
                "rewrite_text": rewrite_text,
                "rewrite_toxicity": rewrite_tox,
                "toxicity_reduction": original_tox - rewrite_tox,
                "similarity": similarity,
                "below_threshold": rewrite_tox < args.tox_threshold
            })
        except Exception as e:
            print(f"Error evaluating sample {i}: {e}")
            
    # Compute aggregate metrics
    df = pd.DataFrame(results)
    
    metrics = {
        "total_samples": len(df),
        "avg_original_toxicity": float(df["original_toxicity"].mean()),
        "avg_rewrite_toxicity": float(df["rewrite_toxicity"].mean()),
        "avg_toxicity_reduction": float(df["toxicity_reduction"].mean()),
        "avg_similarity": float(df["similarity"].mean()),
        "below_threshold_rate": float(df["below_threshold"].mean()),
        "toxicity_reduction_rate_0.3": float((df["toxicity_reduction"] >= 0.3).mean())
    }
    
    # Save reports
    report_json = {
        "metrics": metrics,
        "samples": results[:20] # Save first 20 pairs for inspection
    }
    
    with open("reports/rewrite_eval.json", "w") as f:
        json.dump(report_json, f, indent=2)
        
    # Generate Markdown report
    with open("reports/rewrite_eval.md", "w") as f:
        f.write("# Rewrite Quality Evaluation Report\n\n")
        f.write("## Aggregate Metrics\n")
        for k, v in metrics.items():
            f.write(f"- **{k}**: {v:.4f}\n")
            
        f.write("\n## Example Rewrites\n")
        f.write("| Original | Rewrite | Original Tox | Rewrite Tox | Similarity |\n")
        f.write("| --- | --- | --- | --- | --- |\n")
        for s in results[:10]:
            # Mask sensitive tokens if needed, but here we just truncate for report
            orig = s["original_text"].replace("\n", " ")[:100]
            rewr = s["rewrite_text"].replace("\n", " ")[:100]
            f.write(f"| {orig} | {rewr} | {s['original_toxicity']:.3f} | {s['rewrite_toxicity']:.3f} | {s['similarity']:.3f} |\n")

    print(f"Evaluation complete. Reports saved to reports/rewrite_eval.json and reports/rewrite_eval.md")
    print(f"Summary: {metrics}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Rewrite Quality")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--tox_threshold", type=float, default=0.2)
    
    args = parser.parse_args()
    asyncio.run(run_evaluation(args))
