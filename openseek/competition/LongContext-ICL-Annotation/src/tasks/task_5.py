"""Task 5: Tweet sadness detection — BM25 + balanced labels + emoji removal + difficulty sorting."""
import sys
import os
import random
import re
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from tasks.base import BaseTask
from retriever import BM25Retriever
from shared_utils import remove_emoji, pad_to_min_tokens_dynamic, balance_by_label_target_k


PROMPT_TEMPLATE = (
    "You are a sentiment analysis expert. Determine if the tweet expresses sadness.\n\n"
    "Examples:\n"
    "{examples_str}\n"
    "Input: {input_text}\n\n"
    "Output only 'Sad' or 'Not sad' in <label> tags.\n"
    "Answer: "
)

RETRY_PROMPT = (
    "\n\n{feedback}\n\n"
    "Output ONLY: <label>Sad</label> or <label>Not sad</label>\n\n"
    "Answer: "
)


def _difficulty(example):
    if 'pre_computed_difficulty' in example:
        diff = example['pre_computed_difficulty']
        if diff == 'easy':
            return 1.0
        elif diff == 'medium':
            return 2.0
        elif diff == 'hard':
            return 3.0

    input_val = example.get('input', '')
    text = input_val.lower()
    score = 0

    word_count = len(text.split())
    score += min(word_count / 10, 1.0)

    negations = ["not", "no", "never", "neither", "nobody", "nothing"]
    if any(n in text for n in negations):
        score += 0.5

    contrast_words = ["but", "however", "although", "though", "yet"]
    if any(w in text for w in contrast_words):
        score += 0.5

    positive = ["happy", "love", "great", "good", "wonderful", "best"]
    negative = ["sad", "hate", "bad", "terrible", "worst", "cry"]
    has_pos = any(p in text for p in positive)
    has_neg = any(n in text for n in negative)
    if has_pos and has_neg:
        score += 1.5

    return max(score, 0.1)


def _sort_by_difficulty(examples, reverse=False):
    scored = [(max(_difficulty(ex), 1), ex) for ex in examples]
    scored.sort(key=lambda x: x[0], reverse=reverse)
    return [ex for _, ex in scored]


def _validate(prediction, input_text):
    """Validate: output must be 'Sad' or 'Not sad'."""
    if not prediction or not prediction.strip():
        return False, "Your output was empty. Please answer 'Sad' or 'Not sad'."
    pred = prediction.strip()
    if pred not in ("Sad", "Not sad"):
        return False, f"Your answer '{pred}' is not valid. Output only 'Sad' or 'Not sad'."
    return True, ""


class Task5(BaseTask):
    task_id = 5
    DATA_FILE = '../data/openseek-5_semeval_2018_task1_tweet_sadness_detection.json'
    PROMPT_TEMPLATE = PROMPT_TEMPLATE

    DEFAULT_CFG = {
        "name": "semeval_2018_task1_tweet_sadness_detection",
        "temperature": 0.0,
        "top_k": 40,
        "target_k": 20,
        "num_votes": 1,
        "max_tokens": 5000,
        "stop_tokens": None,
        "system_prompt": "You are a sentiment analysis expert.",
        "balanced": True,
        "min_icl_tokens": 30_000,
    }

    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self.cfg = None

    def postprocess(self, prediction: str, raw_output: str = "") -> str:
        pred = prediction.strip() if prediction else ""
        if pred in ("Sad", "Not sad"):
            return pred
        lower = pred.lower()
        # Check negation variants first: "not sad", "not_sad", "nonsad"
        if re.search(r"not[_\s]*sad|nonsad", lower):
            return "Not sad"
        elif "sad" in lower:
            return "Sad"
        if raw_output:
            lower_raw = raw_output.lower()
            # Same negation check for raw output
            if re.search(r"not[_\s]*sad|nonsad", lower_raw):
                return "Not sad"
            last_sad = lower_raw.rfind("sad")
            if last_sad != -1:
                return "Sad"
        return pred

    def prepare(self):
        # Balance padding pool Sad/Not sad
        sad_pool = [ex for ex in self.padding_pool
                    if isinstance(ex['output'], list) and len(ex['output']) > 0 and ex['output'][0] == 'Sad']
        not_sad_pool = [ex for ex in self.padding_pool
                        if isinstance(ex['output'], list) and len(ex['output']) > 0 and ex['output'][0] == 'Not sad']
        min_count = min(len(sad_pool), len(not_sad_pool))
        random.seed(42)
        random.shuffle(sad_pool)
        random.shuffle(not_sad_pool)
        self.padding_pool = sad_pool[:min_count] + not_sad_pool[:min_count]
        random.shuffle(self.padding_pool)

        self.retriever = BM25Retriever(self.icl_examples)
        return self

    def process_sample(self, test_sample, first_sample=False):
        text = remove_emoji(test_sample['input'])

        retrieved = self.retriever.retrieve_top_k(text, top_k=self.cfg['top_k'])
        target_k = self.cfg.get('target_k', 20)
        selected = balance_by_label_target_k(retrieved, target_k)
        # Difficulty sorted + reverse (hard examples near attention focus)
        selected = _sort_by_difficulty(selected, reverse=True)

        # Build examples string with emoji removal: hardest examples closest to question
        examples_str = ""
        for i, ex in enumerate(selected):
            output_val = ex['output']
            label_str = output_val[0] if isinstance(output_val, list) and len(output_val) > 0 else output_val
            clean_input = remove_emoji(ex['input'])
            part = f"Example {i+1}:\nInput:\n{clean_input}\nOutput:\n<label>{label_str}</label>\n\n"
            examples_str += part


        # Fallback: most-similar BM25 example's label
        output = retrieved[0]['output'] if retrieved else []
        if isinstance(output, list) and len(output) > 0:
            fallback = output[0]
        elif isinstance(output, str) and output:
            fallback = output
        else:
            fallback = 'Not sad'

        selected_ids = {id(ex) for ex in selected}
        dynamic_examples_str, token_count = pad_to_min_tokens_dynamic(
            examples_str, selected_ids, self.padding_pool,
            self.min_icl_tokens, self.max_icl_tokens, self.tokenizer
        )

        prompt = self.PROMPT_TEMPLATE.format(
            examples_str=dynamic_examples_str,
            input_text=text,
        )
        return prompt, dynamic_examples_str, token_count, {'fallback': fallback}

    def should_retry(self):
        """Use custom validation + retry in run_inference."""
        return False

    def run_inference(self, test_sample, call_model):
        """Round 1 → validate → retry with feedback → fallback to most-similar BM25 label."""
        prompt, _, _, info = self.process_sample(test_sample)
        fallback = info.get('fallback', 'Not sad')

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
            # Fallback to most-similar BM25 example's label
            prediction = fallback
            raw_outputs.append(f"[FALLBACK to most-similar label: {fallback} after 3 failed rounds]")

        return prediction, prompt, raw_outputs, [prediction], {}
