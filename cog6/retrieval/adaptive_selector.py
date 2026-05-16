import logging
import math
import re
from collections import Counter
from cog6.retrieval.stopwords import STOP_WORDS

logger = logging.getLogger("COG-6")

class AdaptiveSelector:
    def __init__(self):
        logger.info("[COG-6] Adaptive retrieval initialized")

    def _tokenize(self, text: str) -> list:
        # Simple whitespace and punctuation tokenization
        text = str(text).lower()
        tokens = re.findall(r'\b\w+\b', text)
        # Apply stop-word filtering
        return [t for t in tokens if t not in STOP_WORDS]

    def select(self, query: str, candidates: list, top_k: int = 2, return_scores: bool = False) -> list:
        logger.info("[COG-6] Computing TF-IDF statistics...")
        logger.info("[COG-6] Applying stop-word filtering...")
        
        query_tokens = self._tokenize(query)
        query_set = set(query_tokens)
        
        # If query is empty after filtering, fallback to static
        if not query_set:
            if return_scores:
                return [(0.0, ex) for ex in candidates[:top_k]]
            return candidates[:top_k]
            
        # 1. Compute Document Frequency (DF)
        df_counts = Counter()
        total_docs = len(candidates)
        
        candidates_tokens = []
        for ex in candidates:
            ex_input = ex.get("input", "")
            tokens = self._tokenize(ex_input)
            candidates_tokens.append(tokens)
            df_counts.update(set(tokens))
            
        # Precompute IDF for query tokens
        idf = {}
        for qt in query_set:
            doc_freq = df_counts.get(qt, 0)
            # standard IDF: log(N / (1 + DF))
            idf[qt] = math.log(total_docs / (1.0 + doc_freq)) if total_docs > 0 else 1.0
            
        logger.info("[COG-6] Weighted retrieval scoring active")
        logger.info("[COG-6] Ranking examples...")
        
        scored_examples = []
        for ex, tokens in zip(candidates, candidates_tokens):
            if not tokens:
                scored_examples.append((0.0, ex))
                continue
                
            ex_token_counts = Counter(tokens)
            ex_len = len(tokens)
            
            score = 0.0
            for qt in query_set:
                if qt in ex_token_counts:
                    tf = ex_token_counts[qt] / ex_len
                    score += tf * idf[qt]
                    
            scored_examples.append((score, ex))
            
        # Sort by score descending
        scored_examples.sort(key=lambda x: x[0], reverse=True)
        top_scored = scored_examples[:top_k]
        
        logger.info("[COG-6] Top-K examples selected")
        
        # Debugging Output
        print(f"--- TF-IDF RETRIEVAL DEBUG INFO ---")
        print(f"Filtered Query Tokens: {list(query_set)}")
        
        # Determine top weighted tokens for logging
        weighted_query = [(qt, idf[qt]) for qt in query_set]
        weighted_query.sort(key=lambda x: x[1], reverse=True)
        print(f"Top Weighted Query Tokens (by IDF):")
        for qt, weight in weighted_query[:5]:
            print(f"  - '{qt}': {weight:.4f}")
            
        print(f"Top-K requested: {top_k}")
        for rank, (score, ex) in enumerate(top_scored):
            print(f"Rank {rank+1} | ID: {ex.get('id', 'unknown')} | TF-IDF Score: {score:.4f}")
        print(f"-----------------------------------")
        
        if return_scores:
            return top_scored
        return [ex for score, ex in top_scored]
