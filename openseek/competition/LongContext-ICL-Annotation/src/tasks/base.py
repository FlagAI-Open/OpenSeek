"""
BaseTask abstract class for all T1-T8 tasks.
"""
import json
import os
from abc import ABC, abstractmethod
from collections import Counter


class BaseTask(ABC):
    """Abstract base for task-specific processing.

    Each task subclass implements:
    - DATA_FILE: path to the task data JSON
    - DEFAULT_CFG: dict with task-specific config (postprocess_func excluded)
    - PROMPT_TEMPLATE: string template for prompts
    - split_icl_padding(all_examples): how to split into icl / padding
    - prepare(): build retrievers, precompute indices, filter examples
    - process_sample(): build prompt for a single test sample
    - postprocess(): extract/predict from model output
    - validate_prediction() (optional): post-hoc validation
    - fallback() (optional): fallback strategy
    """

    task_id: int = 0
    DATA_FILE: str = None
    DEFAULT_CFG: dict = {}
    PROMPT_TEMPLATE: str = None

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.cfg = None
        self.icl_examples = []
        self.padding_pool = []
        self.test_samples = []
        self.retriever = None
        self.min_icl_tokens = 30_000
        self.max_icl_tokens = 32_000

    def postprocess(self, prediction: str, raw_output: str = "") -> str:
        """Default postprocess: strip whitespace. Override in subclasses."""
        return prediction.strip() if prediction else ""

    def split_icl_padding(self, all_examples):
        """Default split: first 100 for ICL, rest for padding. Override for custom logic."""
        return all_examples[:100], all_examples[100:]

    def load_data(self):
        """Load data file and split into icl_examples / padding_pool."""
        if self.DATA_FILE is None:
            raise ValueError(f"Task {self.task_id} has no DATA_FILE defined")
        data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), self.DATA_FILE)
        with open(data_path, 'r') as f:
            task_dict = json.load(f)

        all_examples = task_dict['examples']
        self.test_samples = task_dict['test_samples']
        self.icl_examples, self.padding_pool = self.split_icl_padding(all_examples)
        return self

    def prepare(self):
        """Build retrievers, precompute indices, filter examples."""
        return self

    def process_sample(self, test_sample, first_sample=False):
        """Build prompt for a single test sample.

        Returns:
            (prompt, dynamic_examples_str, token_count, extra_debug_info)
        """
        raise NotImplementedError

    # ---- Inference hooks ----

    def should_retry(self):
        """Return False to skip retry loop (e.g. T4). Default True."""
        return True

    def get_retry_strategy(self):
        """Return (max_retries, mode) for retry behavior.

        mode='single': retry once with higher temperature (default)
        mode='vote':   multi-vote retry with fixed temperature
        """
        return 2, 'single'

    def run_inference(self, test_sample, call_model):
        """Run model inference with voting + retry logic.

        Returns:
            (final_prediction, prompt, raw_outputs, candidates, debug_log_entry)
        """
        prompt, _, _, _ = self.process_sample(test_sample)

        num_votes = self.cfg.get('num_votes', 1)
        candidates = []
        raw_outputs = []

        for _ in range(num_votes):
            prediction, raw_output = call_model(prompt, self.cfg)
            raw_outputs.append(raw_output)
            prediction = self.postprocess(prediction, raw_output)
            if prediction.strip():
                candidates.append(prediction.strip())

        if candidates:
            final_prediction = Counter(candidates).most_common(1)[0][0]
        else:
            final_prediction = ""

        # Retry
        if self.should_retry():
            max_retries, mode = self.get_retry_strategy()
            retry_count = 0
            while retry_count < max_retries:
                from shared_utils import needs_retry
                needs, reason = needs_retry(final_prediction, self.task_id)
                if not needs:
                    break
                retry_count += 1

                if mode == 'vote':
                    retry_votes = 3
                    retry_temp = 0.1
                    retry_candidates = []
                    for v in range(retry_votes):
                        prediction, raw_output = call_model(prompt, self.cfg)
                        raw_outputs.append(f"[RETRY attempt={retry_count} vote={v} {reason} temp={retry_temp}] {raw_output}")
                        prediction = self.postprocess(prediction, raw_output)
                        if prediction.strip():
                            retry_candidates.append(prediction.strip())
                    if retry_candidates:
                        final_prediction = Counter(retry_candidates).most_common(1)[0][0]
                        candidates.append(final_prediction)
                    else:
                        final_prediction = ""
                else:
                    retry_temp = max(min(self.cfg['temperature'] + 0.3, 0.8), 0.5)
                    print(f"  ⚠️ [{reason}] for sample {test_sample['id']}, retrying (attempt {retry_count}/{max_retries}, temp={retry_temp})...")
                    orig_temp = self.cfg['temperature']
                    self.cfg['temperature'] = retry_temp
                    prediction, raw_output = call_model(prompt, self.cfg)
                    self.cfg['temperature'] = orig_temp
                    raw_outputs.append(f"[RETRY attempt={retry_count} {reason} temp={retry_temp}] {raw_output}")
                    prediction = self.postprocess(prediction, raw_output)
                    if prediction.strip():
                        candidates.append(prediction.strip())
                        final_prediction = candidates[-1]

        # Validation + fallback
        debug_log_entry = self.run_post_inference(test_sample, final_prediction, call_model, raw_outputs)

        return final_prediction, prompt, raw_outputs, candidates, debug_log_entry

    def run_post_inference(self, test_sample, final_prediction, call_model, raw_outputs):
        """Post-inference hook for validation + fallback (used by T4). Default no-op."""
        return {}

    def validate_prediction(self, prediction, input_text):
        """Optional post-hoc validation. Override in T4."""
        return True, []

    def fallback(self, test_sample):
        """Optional fallback strategy. Override in T4."""
        return None
