"""Task 6: MNLI entailment — genre-filtered BM25 + balanced labels + encoding fix + multi-turn retry."""
import sys
import os
import random
import re
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from tasks.base import BaseTask
from retriever import BM25Retriever
from shared_utils import fix_broken_encoding, pad_to_min_tokens_dynamic, balance_by_label


PROMPT_TEMPLATE = (
    "You are a natural language inference expert. Determine if the hypothesis is entailed by the premise.\n\n"
    "Examples:\n"
    "{examples_str}\n"
    "Input: {input_text}\n\n"
    "Output only 'Y' or 'N' in <label> tags.\n"
    "Answer: "
)

RETRY_PROMPT = (
    "\n\n{feedback}\n\n"
    "Output ONLY: <label>Y</label> or <label>N</label>\n\n"
    "Answer: "
)


def _extract_genre(text):
    m = re.search(r'Genre:\s*(\w+)', text)
    return m.group(1) if m else None


def _validate(prediction, input_text):
    """Validate: output must be 'Y' or 'N'."""
    if not prediction or not prediction.strip():
        return False, "Your output was empty. Please answer 'Y' or 'N'."
    pred = prediction.strip()
    if pred not in ("Y", "N"):
        return False, f"Your answer '{pred}' is not valid. Output only 'Y' or 'N'."
    return True, ""


class Task6(BaseTask):
    task_id = 6
    DATA_FILE = '../data/openseek-6_mnli_same_genre_classification.json'
    PROMPT_TEMPLATE = PROMPT_TEMPLATE

    DEFAULT_CFG = {
        "name": "mnli_same_genre_classification",
        "temperature": 0.3,
        "top_k": 400,
        "num_votes": 1,
        "max_tokens": 5000,
        "stop_tokens": None,
        "system_prompt": "You are a natural language inference expert.",
        "min_icl_tokens": 30_000,
    }

    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self.cfg = None

    def postprocess(self, prediction: str, raw_output: str = "") -> str:
        pred = prediction.strip() if prediction else ""
        if pred in ("Y", "N"):
            return pred
        upper = pred.upper().strip()
        if upper.startswith("Y"):
            return "Y"
        elif upper.startswith("N"):
            return "N"
        if raw_output:
            matches = re.findall(r'\b([YN])\b', raw_output)
            if matches:
                return matches[-1]
        return pred

    def prepare(self):
        # Balance padding pool Y/N
        y_pool = [ex for ex in self.padding_pool if ex['output'][0] == 'Y']
        n_pool = [ex for ex in self.padding_pool if ex['output'][0] == 'N']
        min_count = min(len(y_pool), len(n_pool))
        random.seed(42)
        random.shuffle(y_pool)
        random.shuffle(n_pool)
        self.padding_pool = y_pool[:min_count] + n_pool[:min_count]
        random.shuffle(self.padding_pool)

        # Build per-genre BM25 retrievers (each genre ~1000+ examples, enough to fill 30K)
        genre_pools = {}
        for ex in self.icl_examples:
            g = _extract_genre(ex['input'])
            if g:
                genre_pools.setdefault(g, []).append(ex)

        self.genre_retrievers = {g: BM25Retriever(pool) for g, pool in genre_pools.items()}
        # Fallback: all examples retriever for unknown genre
        self.all_retriever = BM25Retriever(self.icl_examples)
        return self

    def _get_retriever(self, text):
        """Get the genre-specific retriever, or fallback to all."""
        g = _extract_genre(text)
        if g and g in self.genre_retrievers:
            return self.genre_retrievers[g]
        return self.all_retriever

    def process_sample(self, test_sample, first_sample=False):
        text = fix_broken_encoding(test_sample['input'])

        # Genre-filtered retrieval
        retriever = self._get_retriever(text)
        retrieved = retriever.retrieve_top_k(text, top_k=self.cfg['top_k'])

        # Balance Y/N from same-genre pool: use_k=300 → 150 Y + 150 N
        # Smallest genre Y count is 341 (slate), so 150 per label is always available
        selected = balance_by_label(retrieved, use_k=300)

        # Build examples string via prepend: most relevant (last in selected) closest to question
        examples_str = ""
        for i, ex in enumerate(reversed(selected)):
            output_val = ex['output']
            label_str = output_val[0] if isinstance(output_val, list) and len(output_val) > 0 else output_val
            clean_input = fix_broken_encoding(ex['input'])
            examples_str += f"Example {i+1}:\nInput:\n{clean_input}\nOutput:\n<label>{label_str}</label>\n\n"

        selected_ids = {id(ex) for ex in selected}
        dynamic_examples_str, token_count = pad_to_min_tokens_dynamic(
            examples_str, selected_ids, self.padding_pool,
            self.min_icl_tokens, self.max_icl_tokens, self.tokenizer
        )

        # Fallback: most-similar example's label from same genre pool
        output = retrieved[0]['output'] if retrieved else []
        if isinstance(output, list) and len(output) > 0:
            fallback = output[0]
        elif isinstance(output, str) and output:
            fallback = output
        else:
            fallback = 'N'

        prompt = self.PROMPT_TEMPLATE.format(
            examples_str=dynamic_examples_str,
            input_text=text,
        )
        return prompt, examples_str, token_count, {'fallback': fallback}

    def should_retry(self):
        """Use custom validation + retry in run_inference."""
        return False

    def run_inference(self, test_sample, call_model):
        """Round 1 → validate → retry with feedback → fallback to most-similar genre BM25 label."""
        prompt, _, _, info = self.process_sample(test_sample)
        fallback = info.get('fallback', 'N')

        prediction, raw_output = call_model(prompt, self.cfg)
        raw_outputs = [raw_output]
        prediction = self.postprocess(prediction, raw_output)

        valid, feedback = _validate(prediction, test_sample['input'])
        if valid:
            return prediction, prompt, raw_outputs, [prediction], {}

        # Round 2: retry with feedback, strip labels
        print(f"  ⚠️ [{feedback}] for sample {test_sample['id']}, retrying...")
        prev_output = raw_output.replace('<label>', '').replace('</label>', '').strip()
        retry_prompt = prompt + prev_output + RETRY_PROMPT.format(feedback=feedback)
        prediction2, raw_output2 = call_model(retry_prompt, self.cfg)
        raw_outputs.append(f"[RETRY with feedback] {raw_output2}")
        prediction2 = self.postprocess(prediction2, raw_output2)

        valid2, feedback2 = _validate(prediction2, test_sample['input'])
        if valid2:
            prediction = prediction2
            raw_outputs[-1] = f"[RETRY SUCCESS] {raw_output2}"
            return prediction, prompt, raw_outputs, [prediction], {}

        # Round 3: second retry
        print(f"  ⚠️ [{feedback2}] still invalid, second retry...")
        prev_output2 = raw_output2.replace('<label>', '').replace('</label>', '').strip()
        retry_prompt2 = retry_prompt + prev_output2 + RETRY_PROMPT.format(feedback=feedback2)
        prediction3, raw_output3 = call_model(retry_prompt2, self.cfg)
        raw_outputs.append(f"[RETRY2 with feedback] {raw_output3}")
        prediction3 = self.postprocess(prediction3, raw_output3)

        valid3, _ = _validate(prediction3, test_sample['input'])
        if valid3:
            prediction = prediction3
            raw_outputs[-1] = f"[RETRY2 SUCCESS] {raw_output3}"
        else:
            # Fallback to most-similar genre BM25 example's label
            prediction = fallback
            raw_outputs.append(f"[FALLBACK to most-similar label: {fallback} after 3 failed rounds]")

        return prediction, prompt, raw_outputs, [prediction], {}
