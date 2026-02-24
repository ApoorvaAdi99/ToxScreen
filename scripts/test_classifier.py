# Ensure the project root is in the path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.moderation import ToxicityClassifier
from api.config import settings

def main():
    print("Initializing ToxScreen Classifier Test...")
    classifier = ToxicityClassifier(settings.CLASSIFIER_ARTIFACTS_DIR)
    
    if not classifier.is_loaded:
        print("Error: Classifier could not be loaded. Please ensure artifacts exist in", settings.CLASSIFIER_ARTIFACTS_DIR)
        return

    test_samples = [
        "This is a wonderful and helpful comment. Have a great day!",
        "You are a total idiot and nobody likes you. Go away!",
        "I will find you and I will hurt you.",
        "The weather is quite nice today in Seattle."
    ]

    print(f"\n{'='*20} SCORING RESULTS {'='*20}")
    for text in test_samples:
        result = classifier.score_text(text)
        print(f"\nTEXT: \"{text}\"")
        print(f"OVERALL TOXICITY: {result['overall']:.4f}")
        print("LABELS:")
        for label, score in result['labels'].items():
            print(f"  - {label}: {score:.4f}")
    print(f"\n{'='*56}")

if __name__ == "__main__":
    main()
