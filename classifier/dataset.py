import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

def preprocess_function(examples, tokenizer, max_length, label_cols):
    """Tokenize text and prepare multi-label targets."""
    # Tokenize the comments
    result = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=max_length
    )
    
    # Prepare multi-label targets (binarize with 0.5 threshold)
    labels = []
    for i in range(len(examples["text"])):
        label_vec = [1.0 if examples[col][i] >= 0.5 else 0.0 for col in label_cols]
        labels.append(label_vec)
    
    result["labels"] = labels
    return result

def get_dataset(
    model_name="distilroberta-base",
    max_length=256,
    limit_train=None,
    limit_val=None,
    label_cols=None
):
    """Load and preprocess the Civil Comments dataset."""
    if label_cols is None:
        label_cols = [
            "toxicity", "severe_toxicity", "obscene", 
            "threat", "insult", "identity_attack", "sexual_explicit"
        ]
    
    print(f"Loading dataset 'google/civil_comments'...")
    # Using 'google/civil_comments' from HF Datasets
    # Note: This might take a while to download on first run.
    dataset = load_dataset("google/civil_comments")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Subset if requested
    if limit_train:
        dataset["train"] = dataset["train"].select(range(min(limit_train, len(dataset["train"]))))
    if limit_val:
        dataset["validation"] = dataset["validation"].select(range(min(limit_val, len(dataset["validation"]))))
        
    print(f"Preprocessing dataset...")
    encoded_dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer, max_length, label_cols),
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    
    encoded_dataset.set_format("torch")
    
    return encoded_dataset, tokenizer, label_cols

if __name__ == "__main__":
    # Smoke test
    encoded_ds, tokenizer, labels = get_dataset(limit_train=100, limit_val=50)
    print(f"Labels: {labels}")
    print(f"Dataset summary: {encoded_ds}")
    print(f"First sample labels: {encoded_ds['train'][0]['labels']}")
