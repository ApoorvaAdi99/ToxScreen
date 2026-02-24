import os
import json
import torch
from transformers import AutoTokenizer
from api.config import settings
from classifier.model import ToxicityClassifierModel

class ToxicityClassifier:
    def __init__(self, artifacts_dir: str):
        self.artifacts_dir = artifacts_dir
        self.tokenizer = None
        self.model = None
        self.label_map = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_artifacts()

    def load_artifacts(self):
        """Load model, tokenizer, and label map from artifacts directory (Raw PyTorch version)."""
        if not os.path.exists(self.artifacts_dir):
            print(f"Artifacts directory {self.artifacts_dir} not found.")
            return

        try:
            # Load basic artifacts
            self.tokenizer = AutoTokenizer.from_pretrained(self.artifacts_dir)
            
            with open(os.path.join(self.artifacts_dir, "label_map.json"), "r") as f:
                self.label_map = json.load(f)
            
            with open(os.path.join(self.artifacts_dir, "config.json"), "r") as f:
                config_data = json.load(f)
                
            # Initialize custom model architecture
            self.model = ToxicityClassifierModel(
                config_data["model_name"], 
                num_labels=len(self.label_map)
            )
            
            # Load state dict
            state_dict_path = os.path.join(self.artifacts_dir, "model_state.pt")
            if os.path.exists(state_dict_path):
                self.model.load_state_dict(torch.load(state_dict_path, map_location=self.device))
                print("Loaded model state dict.")
            else:
                print("Warning: model_state.pt not found. Model is using random initialization.")

            self.model.to(self.device)
            self.model.eval()
            print(f"Classifier loaded successfully on {self.device}")
        except Exception as e:
            print(f"Error loading classifier artifacts: {e}")

    @property
    def is_loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def score_text(self, text: str) -> dict:
        """Score a single piece of text."""
        if not self.is_loaded:
            return {"error": "Classifier not loaded"}

        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=256, 
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            logits = self.model(
                input_ids=inputs["input_ids"], 
                attention_mask=inputs["attention_mask"]
            )
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        labels = {self.label_map[str(i)]: float(probs[i]) for i in range(len(probs))}
        # Overall is typically the 'toxicity' label or max
        overall = labels.get("toxicity", max(labels.values()))
        
        return {
            "labels": labels,
            "overall": overall
        }

    def score_texts(self, texts: list[str]) -> list[dict]:
        """Score a list of texts (batch inference)."""
        if not self.is_loaded:
            return [{"error": "Classifier not loaded"}] * len(texts)

        inputs = self.tokenizer(
            texts, 
            return_tensors="pt", 
            truncation=True, 
            max_length=256, 
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            logits = self.model(
                input_ids=inputs["input_ids"], 
                attention_mask=inputs["attention_mask"]
            )
            probs = torch.sigmoid(logits).cpu().numpy()

        results = []
        for p in probs:
            labels = {self.label_map[str(i)]: float(p[i]) for i in range(len(p))}
            overall = labels.get("toxicity", max(labels.values()))
            results.append({
                "labels": labels,
                "overall": overall
            })
        return results

# Singleton instance
classifier = ToxicityClassifier(settings.CLASSIFIER_ARTIFACTS_DIR)
