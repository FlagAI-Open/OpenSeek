import json
import logging
import os
import time

logger = logging.getLogger("COG-6")

class MetricsLogger:
    def __init__(self):
        self.metrics = {
            "retrieval_latency_ms": [],
            "mmr_latency_ms": [],
            "validation_calls": 0,
            "salvage_successes": 0,
            "repair_triggers": 0,
            "total_tokens_original": 0,
            "total_tokens_filtered": 0,
            "failed_sample_ids": [],
            "start_time": time.time(),
            "end_time": None
        }

    def log_retrieval_latency(self, latency_ms: float):
        self.metrics["retrieval_latency_ms"].append(latency_ms)

    def log_mmr_latency(self, latency_ms: float):
        self.metrics["mmr_latency_ms"].append(latency_ms)

    def log_validation(self, salvaged: bool, repair_triggered: bool):
        self.metrics["validation_calls"] += 1
        if salvaged:
            self.metrics["salvage_successes"] += 1
        if repair_triggered:
            self.metrics["repair_triggers"] += 1

    def log_saliency(self, original_tokens: int, filtered_tokens: int):
        self.metrics["total_tokens_original"] += original_tokens
        self.metrics["total_tokens_filtered"] += filtered_tokens

    def log_failure(self, sample_id: str):
        self.metrics["failed_sample_ids"].append(sample_id)

    def save_metrics(self, filepath: str):
        self.metrics["end_time"] = time.time()
        
        # Calculate derived metrics
        avg_retrieval_ms = sum(self.metrics["retrieval_latency_ms"]) / max(1, len(self.metrics["retrieval_latency_ms"]))
        avg_mmr_ms = sum(self.metrics["mmr_latency_ms"]) / max(1, len(self.metrics["mmr_latency_ms"]))
        
        total_orig = self.metrics["total_tokens_original"]
        total_filt = self.metrics["total_tokens_filtered"]
        tokens_saved = total_orig - total_filt
        compression_ratio = (tokens_saved / total_orig * 100.0) if total_orig > 0 else 0.0
        
        val_calls = self.metrics["validation_calls"]
        salvage_rate = (self.metrics["salvage_successes"] / val_calls * 100.0) if val_calls > 0 else 0.0
        repair_freq = (self.metrics["repair_triggers"] / val_calls * 100.0) if val_calls > 0 else 0.0
        
        total_runtime_s = self.metrics["end_time"] - self.metrics["start_time"]
        
        summary = {
            "average_retrieval_latency_ms": avg_retrieval_ms,
            "average_mmr_latency_ms": avg_mmr_ms,
            "saliency_compression_ratio_pct": compression_ratio,
            "total_tokens_saved": tokens_saved,
            "validation_salvage_rate_pct": salvage_rate,
            "repair_trigger_frequency_pct": repair_freq,
            "failed_samples_count": len(self.metrics["failed_sample_ids"]),
            "failed_sample_ids": self.metrics["failed_sample_ids"],
            "total_runtime_seconds": total_runtime_s
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=4)
            # Let me fix this in the code string below, json.dump
        
        logger.info(f"[COG-6] Metrics saved to {filepath}")
