import logging
import re
import string

logger = logging.getLogger("COG-6")

class ReflectiveValidator:
    def __init__(self, router):
        self.router = router

    def _salvage(self, text: str, valid_labels: list) -> str:
        if not text:
            return None
            
        # 1. Lowercase and normalize whitespace
        clean_text = " ".join(str(text).lower().split())
        # Strip punctuation from edges
        clean_text = clean_text.strip(string.punctuation)

        # 2. Exact match check
        for label in valid_labels:
            if clean_text == label.lower():
                return label

        # 3. Substring keyword matching
        # Sort labels by length descending to match "Not sad" before "Sad"
        sorted_labels = sorted(valid_labels, key=len, reverse=True)
        
        matches = []
        for label in sorted_labels:
            pattern = r'\b' + re.escape(label.lower()) + r'\b'
            if re.search(pattern, clean_text):
                # Avoid overlapping matches (if 'Not sad' is matched, don't also match 'Sad')
                # Since we sorted by length, we only append if it's the first match or entirely distinct
                if not any(label.lower() in m.lower() for m in matches):
                    matches.append(label)

        # If exactly one distinct label concept was found, it's a safe salvage
        if len(matches) == 1:
            return matches[0]
            
        # 4. Fallback matching without word boundaries (just in case)
        if not matches:
            for label in sorted_labels:
                if label.lower() in clean_text:
                    if not any(label.lower() in m.lower() for m in matches):
                        matches.append(label)
            if len(matches) == 1:
                return matches[0]

        return None

    def validate(self, raw_output: str, valid_labels: list) -> str:
        logger.info("[COG-6] Validating output...")
        
        # Phase 1: Deterministic Salvage
        salvaged = self._salvage(raw_output, valid_labels)
        
        repair_triggered = False
        final_output = raw_output
        
        if salvaged:
            logger.info("[COG-6] Deterministic salvage successful")
            logger.info("[COG-6] Output normalized")
            final_output = salvaged
        else:
            logger.info("[COG-6] Repair inference triggered")
            repair_triggered = True
            
            # Phase 2: Minimal Repair Prompting
            repair_prompt = (
                f"Previous output: {raw_output}\n\n"
                f"Return ONLY one valid label from: {valid_labels}"
            )
            
            # Reuse router for repair
            repaired_raw = self.router.generate(repair_prompt)
            
            # Attempt to salvage the repaired output
            repaired_salvage = self._salvage(repaired_raw, valid_labels)
            if repaired_salvage:
                final_output = repaired_salvage
            else:
                final_output = repaired_raw # fallback to whatever it produced
                
            logger.info("[COG-6] Output normalized")
        
        # Debug Table Requirement
        print("\n--- VALIDATION DEBUG ---")
        print(f"Raw Output: {raw_output}")
        print(f"Normalized Output: {final_output}")
        print(f"Salvage Success: {salvaged is not None}")
        print(f"Repair Triggered: {repair_triggered}")
        print("------------------------\n")
        
        return final_output
