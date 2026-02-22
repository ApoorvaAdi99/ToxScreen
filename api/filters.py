import re
import logging
import os

class ContentFilter:
    def __init__(self, slurs_file: str = "data/slurs.txt"):
        self.slurs = []
        if os.path.exists(slurs_file):
            with open(slurs_file, "r") as f:
                self.slurs = [line.strip().lower() for line in f if line.strip()]
        
        # Simple threat patterns
        self.threat_patterns = [
            r"i will kill",
            r"i'm going to hurt",
            r"i'll kill",
            r"death to",
            r"hope you die"
        ]

    def check(self, text: str) -> tuple[bool, list[str]]:
        """Check text for slurs and threats. Returns (is_filtered, reasons)."""
        reasons = []
        text_lower = text.lower()
        
        # Check slurs
        for slur in self.slurs:
            if slur in text_lower:
                reasons.append(f"Contains slur: {slur}")
        
        # Check threats
        for pattern in self.threat_patterns:
            if re.search(pattern, text_lower):
                reasons.append(f"Contains threat pattern: {pattern}")
        
        return len(reasons) > 0, reasons
