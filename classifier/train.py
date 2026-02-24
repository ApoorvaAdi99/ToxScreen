import os
import sys
# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, DataCollatorWithPadding
from tqdm import tqdm
import numpy as np

from classifier.dataset import get_dataset
from classifier.model import get_model
from classifier.metrics import compute_metrics

def main(args):
    # Setup directories
    os.makedirs(args.artifacts_dir, exist_ok=True)
    
    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
        
    # Load dataset and tokenizer
    # Note: get_dataset returns encoded datasets (HF format)
    dataset, tokenizer, label_cols = get_dataset(
        model_name=args.model,
        max_length=args.max_length,
        limit_train=args.limit_train,
        limit_val=args.limit_val,
        label_cols=None # Use defaults
    )
    
    # Save label map and configuration
    label_map = {i: label for i, label in enumerate(label_cols)}
    with open(os.path.join(args.artifacts_dir, "label_map.json"), "w") as f:
        json.dump(label_map, f)
        
    config_data = {
        "model_name": args.model,
        "max_length": args.max_length,
        "num_labels": len(label_cols),
        "label_cols": label_cols,
        "seed": args.seed
    }
    with open(os.path.join(args.artifacts_dir, "config.json"), "w") as f:
        json.dump(config_data, f)
    
    # Initialize model
    model = get_model(
        args.model, 
        num_labels=len(label_cols),
        label2id={l: i for i, l in label_map.items()},
        id2label=label_map
    )
    model.to(device)
    
    # Prepare DataLoaders
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    train_loader = DataLoader(
        dataset["train"], 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=data_collator
    )
    val_loader = DataLoader(
        dataset["validation"], 
        batch_size=args.batch_size, 
        collate_fn=data_collator
    )
    
    # Optimizer and Loss
    optimizer = AdamW(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()
    
    # Training Loop
    best_roc_auc = 0.0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        model.train()
        train_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc="Training")
        for batch in progress_bar:
            optimizer.zero_grad()
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
            
        avg_train_loss = train_loss / len(train_loader)
        print(f"Average Training Loss: {avg_train_loss:.4f}")
        
        # Validation
        model.eval()
        all_logits = []
        all_labels = []
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                
                all_logits.append(logits.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                
        avg_val_loss = val_loss / len(val_loader)
        all_logits = np.concatenate(all_logits, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        # Compute metrics
        val_metrics = compute_metrics((all_logits, all_labels))
        val_metrics["val_loss"] = avg_val_loss
        
        print(f"Validation Loss: {avg_val_loss:.4f}")
        print(f"Validation Metrics: {val_metrics}")
        
        # Save best model
        if val_metrics["roc_auc_macro"] > best_roc_auc:
            best_roc_auc = val_metrics["roc_auc_macro"]
            print(f"New best model found (ROC-AUC: {best_roc_auc:.4f}). Saving...")
            torch.save(model.state_dict(), os.path.join(args.artifacts_dir, "model_state.pt"))
            tokenizer.save_pretrained(args.artifacts_dir)
            
            with open(os.path.join(args.artifacts_dir, "metrics.json"), "w") as f:
                json.dump(val_metrics, f, indent=2)

    print(f"Training complete. Artifacts saved to {args.artifacts_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Toxicity Classifier (Raw PyTorch)")
    parser.add_argument("--model", type=str, default="distilroberta-base")
    parser.add_argument("--artifacts_dir", type=str, default="classifier/artifacts")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--limit_train", type=int, default=None)
    parser.add_argument("--limit_val", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    main(args)
