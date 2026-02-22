import os
import argparse
import json
import torch
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from classifier.dataset import get_dataset
from classifier.model import get_model
from classifier.metrics import compute_metrics

def main(args):
    # Setup directories
    os.makedirs(args.artifacts_dir, exist_ok=True)
    
    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        
    # Load dataset and tokenizer
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
    
    # Data collator for dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.artifacts_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="roc_auc_macro",
        greater_is_better=bool(True),
        fp16=torch.cuda.is_available(),
        logging_dir=os.path.join(args.artifacts_dir, "logs"),
        logging_steps=100,
        report_to="none" # Disable integrations to avoid credentials issues
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save best model
    print(f"Saving artifacts to {args.artifacts_dir}")
    trainer.save_model(args.artifacts_dir)
    tokenizer.save_pretrained(args.artifacts_dir)
    
    # Final evaluation on validation set
    metrics = trainer.evaluate()
    with open(os.path.join(args.artifacts_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Final Validation Metrics: {metrics}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Toxicity Classifier")
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
