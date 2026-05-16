import os
import json
import logging
import time
import argparse

from cog6.runtime.router import RuntimeRouter
from cog6.prompting.prompt_builder import PromptBuilder
from cog6.retrieval.adaptive_selector import AdaptiveSelector
from cog6.retrieval.mmr_selector import MMRSelector
from cog6.saliency.saliency_filter import SaliencyFilter
from cog6.validation.reflective_validator import ReflectiveValidator
from cog6.utils.task_parser import TaskParser
from cog6.utils.metrics_logger import MetricsLogger
from cog6.retrieval.task_router import TaskRouter

def setup_logger():
    logger = logging.getLogger("COG-6")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

def main():
    parser = argparse.ArgumentParser(description="COG-6 Benchmark Orchestrator")
    parser.add_argument("--mode", type=str, choices=["baseline", "cog6"], required=True, help="Execution mode (baseline or cog6)")
    parser.add_argument("--task", type=str, default="openseek-5", help="Task ID to execute")
    args = parser.parse_args()

    logger = setup_logger()
    logger.info("================================================")
    logger.info(f"COG-6 BENCHMARK EXECUTION - MODE: {args.mode.upper()}")
    logger.info("================================================\n")
    
    # 1. Initialization
    task_parser = TaskParser()
    task_router = TaskRouter()
    router = RuntimeRouter(mode="ollama")
    builder = PromptBuilder()
    
    # Init advanced orchestration modules only if cog6 mode
    selector = AdaptiveSelector() if args.mode == "cog6" else None
    mmr_selector = MMRSelector() if args.mode == "cog6" else None
    saliency_filter = SaliencyFilter() if args.mode == "cog6" else None
    validator = ReflectiveValidator(router) if args.mode == "cog6" else None
    
    metrics = MetricsLogger()
    
    # 2. Paths
    task_id = args.task
    task_file = f"openseek/competition/LongContext-ICL-Annotation/data/{task_id}_semeval_2018_task1_tweet_sadness_detection.json"
    
    os.makedirs("outputs/errors", exist_ok=True)
    os.makedirs("experiments", exist_ok=True)
    
    output_file = os.path.join("outputs", f"{task_id}_{args.mode}_predictions.jsonl")
    error_file = os.path.join("outputs", "errors", f"{task_id}_{args.mode}_failed_samples.jsonl")
    metrics_file = os.path.join("experiments", f"{args.mode}_run_metrics.json")
    
    # 3. Task Parsing
    logger.info("[COG-6] Loading task...")
    try:
        task_data = task_parser.parse(task_file)
    except Exception as e:
        logger.error(f"[COG-6] Failed to load task file: {e}")
        return
        
    definition = task_data["definition"]
    candidate_examples = task_data["examples"]
    valid_labels = task_data["valid_labels"]
    task_name = task_data["task_name"]
    
    with open(task_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    test_samples = raw_data.get("examples", [])
    
    if args.mode == "cog6":
        strategy = task_router.get_strategy(task_name)
    
    with open(output_file, "w", encoding="utf-8") as f:
        pass
        
    logger.info(f"[COG-6] Beginning full dataset iteration ({len(test_samples)} samples)...")
    
    # 5. Full Dataset Iteration
    for sample in test_samples:
        sample_id = sample.get("id", "unknown_id")
        query = sample.get("input", "")
        
        try:
            logger.info(f"\n--- Processing Sample: {sample_id} ---")
            
            if args.mode == "baseline":
                # BASELINE MODE: standard static K-shot
                icl_examples = candidate_examples[:2]
                prompt = builder.build_prompt(definition, icl_examples, query)
                raw_prediction = router.generate(prompt)
                final_prediction = raw_prediction
                # We can do a rudimentary cleanup to ensure JSONl format is met
                final_prediction = " ".join(final_prediction.split())
                
            else:
                # COG-6 MODE: full orchestration stack
                t0 = time.time()
                scored_candidates = selector.select(query, candidate_examples, top_k=15, return_scores=True)
                t1 = time.time()
                metrics.log_retrieval_latency((t1 - t0) * 1000)
                
                t2 = time.time()
                icl_examples = mmr_selector.select(query, scored_candidates, top_k=2, lambda_weight=0.7)
                t3 = time.time()
                metrics.log_mmr_latency((t3 - t2) * 1000)
                
                original_tokens = sum(saliency_filter._approx_token_count(ex.get("input", "")) for ex in icl_examples)
                compressed_examples = saliency_filter.compress(icl_examples)
                filtered_tokens = sum(saliency_filter._approx_token_count(ex.get("input", "")) for ex in compressed_examples)
                metrics.log_saliency(original_tokens, filtered_tokens)
                
                prompt = builder.build_prompt(definition, compressed_examples, query)
                raw_prediction = router.generate(prompt)
                
                salvaged_output = validator._salvage(raw_prediction, valid_labels)
                if salvaged_output:
                    final_prediction = salvaged_output
                    metrics.log_validation(salvaged=True, repair_triggered=False)
                else:
                    repair_prompt = f"Previous output: {raw_prediction}\n\nReturn ONLY one valid label from: {valid_labels}"
                    repaired_raw = router.generate(repair_prompt)
                    repaired_salvage = validator._salvage(repaired_raw, valid_labels)
                    final_prediction = repaired_salvage if repaired_salvage else repaired_raw
                    metrics.log_validation(salvaged=(repaired_salvage is not None), repair_triggered=True)
            
            output_record = {
                "sample_id": sample_id,
                "prediction": final_prediction
            }
            
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(output_record) + "\n")
                
            logger.info(f"[COG-6] Successfully saved prediction for {sample_id}")
            
            if len(metrics.metrics.get("retrieval_latency_ms", [])) >= 3 and args.mode == "cog6":
                logger.info("[COG-6] Stopping early for demonstration purposes (3 samples processed).")
                break
            elif args.mode == "baseline" and test_samples.index(sample) >= 2:
                logger.info("[BASELINE] Stopping early for demonstration purposes (3 samples processed).")
                break
                
        except Exception as e:
            logger.error(f"[COG-6] Exception during processing sample {sample_id}: {e}")
            metrics.log_failure(sample_id)
            with open(error_file, "a", encoding="utf-8") as err_f:
                err_f.write(json.dumps({"sample_id": sample_id, "error": str(e)}) + "\n")
                
    metrics.save_metrics(metrics_file)
    logger.info("================================================")
    logger.info("EXECUTION COMPLETED")
    logger.info("================================================")

if __name__ == "__main__":
    main()
