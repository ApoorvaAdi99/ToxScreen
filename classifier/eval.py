import os
import sys
# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding
from datasets import load_dataset
from tqdm import tqdm
import numpy as np

from classifier.dataset import preprocess_function
from classifier.model import load_model
from classifier.metrics import compute_metrics

def main(args):
    # Load configuration
    with open(os.path.join(args.artifacts_dir, "config.json"), "r") as f:
        config_data = json.load(f)
    
    with open(os.path.join(args.artifacts_dir, "label_map.json"), "r") as f:
        label_map = json.load(f)
        label_cols = [label_map[str(i)] for i in range(len(label_map))]
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
        
    print(f"Loading model from {args.artifacts_dir}")
    model = load_model(args.artifacts_dir, num_labels=len(label_cols))
    model.to(device)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(args.artifacts_dir)
    
    # Load test dataset
    print("Loading test dataset...")
    dataset = load_dataset("google/civil_comments", split="test")
    
    if args.limit_test:
        dataset = dataset.select(range(min(args.limit_test, len(dataset))))
        
    print("Preprocessing test dataset...")
    encoded_test = dataset.map(
        lambda x: preprocess_function(x, tokenizer, config_data["max_length"], label_cols),
        batched=True,
        remove_columns=dataset.column_names
    )
    encoded_test.set_format("torch")
    
    # Prepare DataLoader
    val_loader = DataLoader(
        encoded_test, 
        batch_size=args.batch_size, 
        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer)
    )
    
    # Evaluation
    print("Running evaluation...")
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Compute metrics
    metrics = compute_metrics((all_logits, all_labels))
    
    # Save test metrics
    with open(os.path.join(args.artifacts_dir, "test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
        
    print(f"Test Metrics: {json.dumps(metrics, indent=2)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Toxicity Classifier (Raw PyTorch)")
    parser.add_argument("--artifacts_dir", type=str, default="classifier/artifacts")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--limit_test", type=int, default=1000)
    
    args = parser.parse_args()
    main(args)
