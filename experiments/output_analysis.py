import json
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("OutputAnalyzer")

def analyze():
    logger.info("================================================")
    logger.info("COG-6 OUTPUT ANALYSIS")
    logger.info("================================================")
    
    baseline_metrics_path = "experiments/baseline_run_metrics.json"
    cog6_metrics_path = "experiments/cog6_run_metrics.json"
    
    if os.path.exists(cog6_metrics_path):
        with open(cog6_metrics_path, 'r') as f:
            cog6_metrics = json.load(f)
            logger.info("\n[COG-6 METRICS]")
            logger.info(f"Avg Retrieval Latency: {cog6_metrics.get('average_retrieval_latency_ms', 0):.2f} ms")
            logger.info(f"Avg MMR Latency: {cog6_metrics.get('average_mmr_latency_ms', 0):.2f} ms")
            logger.info(f"Saliency Compression: {cog6_metrics.get('saliency_compression_ratio_pct', 0):.2f}%")
            logger.info(f"Validation Salvage Rate: {cog6_metrics.get('validation_salvage_rate_pct', 0):.2f}%")
            logger.info(f"Repair Trigger Freq: {cog6_metrics.get('repair_trigger_frequency_pct', 0):.2f}%")
            logger.info(f"Total Failures: {cog6_metrics.get('failed_samples_count', 0)}")
    
    # Analyze predictions consistency
    baseline_preds = {}
    cog6_preds = {}
    
    if os.path.exists("outputs/openseek-5_baseline_predictions.jsonl"):
        with open("outputs/openseek-5_baseline_predictions.jsonl", "r") as f:
            for line in f:
                data = json.loads(line)
                baseline_preds[data["sample_id"]] = data["prediction"]
                
    if os.path.exists("outputs/openseek-5_cog6_predictions.jsonl"):
        with open("outputs/openseek-5_cog6_predictions.jsonl", "r") as f:
            for line in f:
                data = json.loads(line)
                cog6_preds[data["sample_id"]] = data["prediction"]
                
    common_keys = set(baseline_preds.keys()).intersection(set(cog6_preds.keys()))
    if not common_keys:
        logger.warning("No overlapping samples between Baseline and COG-6.")
        return
        
    match_count = sum(1 for k in common_keys if baseline_preds[k] == cog6_preds[k])
    consistency = (match_count / len(common_keys)) * 100.0
    
    logger.info(f"\n[CONSISTENCY ANALYSIS]")
    logger.info(f"Overlapping Samples: {len(common_keys)}")
    logger.info(f"Prediction Consistency (Baseline vs COG-6): {consistency:.2f}%")
    
if __name__ == "__main__":
    analyze()
