import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

def compute_metrics(eval_pred):
    """Compute metrics for multi-label classification."""
    logits, labels = eval_pred
    # Apply sigmoid to get probabilities
    probs = 1 / (1 + np.exp(-logits))
    # Binarize predictions with 0.5 threshold
    preds = (probs >= 0.5).astype(int)
    
    # Macro metrics
    roc_auc_macro = roc_auc_score(labels, probs, average="macro")
    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    precision_macro = precision_score(labels, preds, average="macro", zero_division=0)
    recall_macro = recall_score(labels, preds, average="macro", zero_division=0)
    
    metrics = {
        "roc_auc_macro": roc_auc_macro,
        "f1_macro": f1_macro,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
    }
    
    # Add per-label ROC-AUC if possible
    for i in range(labels.shape[1]):
        try:
            metrics[f"roc_auc_label_{i}"] = roc_auc_score(labels[:, i], probs[:, i])
        except ValueError:
            # If all samples in a batch are of the same class
            metrics[f"roc_auc_label_{i}"] = 0.0
            
    return metrics
