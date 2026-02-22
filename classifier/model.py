from transformers import AutoModelForSequenceClassification, AutoConfig

def get_model(model_name, num_labels, label2id=None, id2label=None):
    """Initialize a transformer model for multi-label classification."""
    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type="multi_label_classification"
    )
    
    if label2id:
        config.label2id = label2id
    if id2label:
        config.id2label = id2label
        
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=config
    )
    return model

def load_model(artifacts_dir):
    """Load model from local artifacts directory."""
    model = AutoModelForSequenceClassification.from_pretrained(artifacts_dir)
    return model
