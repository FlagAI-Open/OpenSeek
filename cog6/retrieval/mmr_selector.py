import logging
import math
import re
from collections import Counter
from cog6.retrieval.stopwords import STOP_WORDS

logger = logging.getLogger("COG-6")

class MMRSelector:
    def __init__(self):
        logger.info("[COG-6] MMR diversification initialized")

    def _tokenize(self, text: str) -> list:
        text = str(text).lower()
        tokens = re.findall(r'\b\w+\b', text)
        return [t for t in tokens if t not in STOP_WORDS]

    def _similarity(self, tokens_a: list, tokens_b: list, idf: dict) -> float:
        if not tokens_a or not tokens_b:
            return 0.0
        set_a = set(tokens_a)
        counts_b = Counter(tokens_b)
        len_b = len(tokens_b)
        score = 0.0
        for t in set_a:
            if t in counts_b:
                tf = counts_b[t] / len_b
                score += tf * idf.get(t, 1.0) # Default IDF to 1.0 if not found
        return score

    def select(self, query_text: str, ranked_candidates: list, top_k: int = 2, lambda_weight: float = 0.7) -> list:
        logger.info("[COG-6] Evaluating candidate diversity using TF-IDF...")
        
        if not ranked_candidates:
            return []
            
        candidates_data = []
        df_counts = Counter()
        total_docs = len(ranked_candidates)
        
        # Pre-tokenize and build Document Frequency mapping
        for rel_score, ex in ranked_candidates:
            tokens = self._tokenize(ex.get("input", ""))
            df_counts.update(set(tokens))
            candidates_data.append({
                "ex": ex,
                "relevance": rel_score,
                "tokens": tokens
            })
            
        # Compute IDF map for the candidates
        idf_map = {}
        for t, freq in df_counts.items():
            idf_map[t] = math.log(total_docs / (1.0 + freq))
            
        selected = []
        debug_info = []
        
        # 1. Select highest relevance candidate first
        best_first = max(candidates_data, key=lambda x: x["relevance"])
        selected.append(best_first)
        candidates_data.remove(best_first)
        
        debug_info.append({
            "rank": 1,
            "id": best_first["ex"].get("id", "unknown"),
            "relevance": best_first["relevance"],
            "penalty": 0.0,
            "final": best_first["relevance"]
        })
        
        # 2. Iteratively select remaining
        while len(selected) < top_k and candidates_data:
            best_mmr = -float('inf')
            best_candidate = None
            best_penalty = 0.0
            
            for cand in candidates_data:
                # Calculate maximum TF-IDF similarity to any already-selected example
                max_sim = 0.0
                for sel in selected:
                    # Treat the already selected example as "query" against candidate
                    sim = self._similarity(sel["tokens"], cand["tokens"], idf_map)
                    if sim > max_sim:
                        max_sim = sim
                
                penalty = max_sim
                mmr_score = (lambda_weight * cand["relevance"]) - ((1.0 - lambda_weight) * penalty)
                
                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_candidate = cand
                    best_penalty = penalty
                    
            selected.append(best_candidate)
            candidates_data.remove(best_candidate)
            
            debug_info.append({
                "rank": len(selected),
                "id": best_candidate["ex"].get("id", "unknown"),
                "relevance": best_candidate["relevance"],
                "penalty": best_penalty,
                "final": best_mmr
            })
            
        logger.info("[COG-6] Diversified examples selected")
        
        # Debug Table Requirement
        print("\n--- MMR DIVERSIFICATION DEBUG (TF-IDF WEIGHTED) ---")
        for info in debug_info:
            print(f"Rank {info['rank']} | ID: {info['id']} | Relevance: {info['relevance']:.4f} | Penalty: {info['penalty']:.4f} | Final: {info['final']:.4f}")
        print("---------------------------------------------------\n")
        
        return [item["ex"] for item in selected]
