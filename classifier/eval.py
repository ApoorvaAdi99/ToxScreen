import os
import argparse
import json
import torch
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding, AutoTokenizer
from classifier.dataset import preprocess_function
from classifier.model import load_model
from classifier.metrics import compute_metrics
from datasets import load_dataset

def main(args):
    # Load configuration
    with open(os.path.join(args.artifacts_dir, "config.json"), "r") as f:
        config_data = json.load(f)
    
    with open(os.path.join(args.artifacts_dir, "label_map.json"), "r") as f:
        label_map = json.load(f)
        label_cols = [label_map[str(i)] for i in range(len(label_map))]
        
    print(f"Loading model from {args.artifacts_dir}")
    model = load_model(args.artifacts_dir)
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
    
    # Setup Trainer for evaluation
    training_args = TrainingArguments(
        output_dir=args.artifacts_dir,
        per_device_eval_batch_size=args.batch_size,
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics
    )
    
    # Evaluate
    print("Running evaluation...")
    metrics = trainer.evaluate(encoded_test)
    
    # Save test metrics
    with open(os.path.join(args.artifacts_dir, "test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
        
    print(f"Test Metrics: {json.dumps(metrics, indent=2)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Toxicity Classifier")
    parser.add_argument("--artifacts_dir", type=str, default="classifier/artifacts")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--limit_test", type=int, default=1000)
    
    args = parser.parse_args()
    main(args)
