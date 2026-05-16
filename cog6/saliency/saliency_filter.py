import logging
import re
import copy

logger = logging.getLogger("COG-6")

class SaliencyFilter:
    def __init__(self):
        logger.info("[COG-6] Saliency filtering initialized")

    def _approx_token_count(self, text: str) -> int:
        return len(text.split())

    def _clean_text(self, text: str) -> str:
        # 1. Remove URLs
        text = re.sub(r'http[s]?://\S+', '', text)
        
        # 2. Remove RT markers
        text = re.sub(r'\bRT\b', '', text)
        
        # 3. Remove @mentions
        text = re.sub(r'@\w+', '', text)
        
        # 4. Remove tokenizer garbage (e.g. from splitted contractions)
        text = re.sub(r'\b(ll|ve|re|don|didn|doesn|isn|aren|won)\b', '', text, flags=re.IGNORECASE)
        
        # 5. Deduplicate excessive punctuation, preserve structure
        text = re.sub(r'([!?.])\1+', r'\1', text)
        
        # 6. Clean up excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def compress(self, examples: list) -> list:
        logger.info("[COG-6] Compressing retrieved context...")
        
        compressed_examples = []
        debug_info = []
        total_original = 0
        total_filtered = 0
        
        for ex in examples:
            # Create a deep copy to avoid mutating the original retrieved object
            comp_ex = copy.deepcopy(ex)
            
            original_text = comp_ex.get("input", "")
            original_count = self._approx_token_count(original_text)
            
            cleaned_text = self._clean_text(original_text)
            filtered_count = self._approx_token_count(cleaned_text)
            
            comp_ex["input"] = cleaned_text
            compressed_examples.append(comp_ex)
            
            # Metrics
            total_original += original_count
            total_filtered += filtered_count
            
            ratio = 0.0
            if original_count > 0:
                ratio = ((original_count - filtered_count) / original_count) * 100.0
                
            debug_info.append({
                "id": comp_ex.get("id", "unknown"),
                "orig": original_count,
                "filt": filtered_count,
                "ratio": ratio
            })
            
        logger.info("[COG-6] Signal-dense context generated")
        
        # Debug Table Requirement
        print("\n--- SALIENCY COMPRESSION DEBUG ---")
        for info in debug_info:
            print(f"Example ID: {info['id']}")
            print(f"Original Tokens: {info['orig']}")
            print(f"Filtered Tokens: {info['filt']}")
            print(f"Compression Ratio: {info['ratio']:.1f}%")
            print("-")
            
        total_saved = total_original - total_filtered
        overall_ratio = 0.0
        if total_original > 0:
            overall_ratio = (total_saved / total_original) * 100.0
            
        print(f"TOTAL TOKENS SAVED: {total_saved} (Overall Reduction: {overall_ratio:.1f}%)")
        print("----------------------------------\n")
        
        return compressed_examples
