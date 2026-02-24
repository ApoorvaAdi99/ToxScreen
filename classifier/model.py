import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class ToxicityClassifierModel(nn.Module):
    def __init__(self, model_name, num_labels, dropout_prob=0.1):
        super(ToxicityClassifierModel, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        
        # Classification head
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        # Pass inputs through the encoder (e.g., RoBERTa, BERT)
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # We use the [CLS] token representation (or mean pooling)
        # For BERT/RoBERTa, usually index 0 is the [CLS] equivalent
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits

def get_model(model_name, num_labels, label2id=None, id2label=None):
    """Factory function for initialization."""
    model = ToxicityClassifierModel(model_name, num_labels)
    # Storing id2label/label2id in config for downstream compatibility
    model.config.id2label = id2label
    model.config.label2id = label2id
    return model

def load_model(artifacts_dir, num_labels):
    """Load model from local artifacts directory."""
    # We first initialize with a dummy name or the base name
    # Then we load the state dict
    # In practice, we should save the base model name in config.json
    import json
    import os
    
    with open(os.path.join(artifacts_dir, "config.json"), "r") as f:
        config = json.load(f)
    
    model = ToxicityClassifierModel(config["model_name"], num_labels)
    state_dict = torch.load(os.path.join(artifacts_dir, "model_state.pt"), map_location="cpu")
    model.load_state_dict(state_dict)
    return model
