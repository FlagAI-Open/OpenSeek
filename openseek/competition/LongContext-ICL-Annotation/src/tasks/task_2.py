"""Task 2: Count nouns/verbs — POS retriever, most relevant closest to question."""
import sys
import os
import re
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from tasks.base import BaseTask
from pos_retriever import POSRetriever

# 11.28 版本精选高质量示例 — 覆盖 noun(0~5) + verb(0~5)
TASK2_CURATED_EXAMPLES = [
    {'input': "Sentence: 'THIS IS A COOL BLACK AND WHITE PHOTO OF A SKATEBOARDER'. Count the number of nouns in this sentence.", 'output': ['0']},
    {'input': "Sentence: 'Three motorcycles parked next to each other with two men on one'. Count the number of nouns in this sentence.", 'output': ['2']},
    {'input': "Sentence: 'A small child looking in a refrigerator with her bottom showing'. Count the number of nouns in this sentence.", 'output': ['3']},
    {'input': "Sentence: 'Three mountain goats standing amongst the rocks on the mountain'. Count the number of nouns in this sentence.", 'output': ['4']},
    {'input': "Sentence: 'A computer on a table with all kinds of bottles and other items'. Count the number of nouns in this sentence.", 'output': ['5']},
    {'input': "Sentence: 'A bowl of stew, and several plates of English Muffins and bread'. Count the number of verbs in this sentence.", 'output': ['0']},
    {'input': "Sentence: 'A hot dog sitting on a plate and in a bun with other food items'. Count the number of verbs in this sentence.", 'output': ['1']},
    {'input': "Sentence: 'A man showing a boy with a helmet on how to get on a skateboard'. Count the number of verbs in this sentence.", 'output': ['2']},
    {'input': "Sentence: 'Pan of food that looks like stir fry being stirred'. Count the number of verbs in this sentence.", 'output': ['4']},
    {'input': "Sentence: 'People wearing old Western-styled dress stand aside the road as a horse-drawn carriage in a parade passes by'. Count the number of verbs in this sentence.", 'output': ['5']},
]

PROMPT_TEMPLATE = (
    "You are a precise linguist. Count the requested parts of speech.\n\n"
    "Rules for nouns: Count ALL nouns (NOUN) and proper nouns (PROPN). "
    "Modifier nouns count too: in 'train tracks', both 'train' and 'tracks' are nouns. "
    "Compound nouns count each word: 'fire hydrant' = 2 nouns, 'Chicago Board Of Trade' = 4 nouns. "
    "Adjectives do NOT count as nouns.\n\n"
    "Rules for verbs: Count ALL verbs (VERB), including gerunds ('walking', 'getting') "
    "and past participles ('made', 'shaped', 'wrapped'). "
    "Auxiliary verbs ('is', 'are', 'was', 'were', 'been') "
    "count as verbs if used in progressive/passive constructions.\n\n"
    "Examples:\n"
    "{examples_str}\n"
    "Input: {input_text}\n\n"
    "CRITICAL: Output only the number in this format: <label>count</label>\n\n"
    "Answer: "
)

RETRY_PROMPT = (
    "\n\n{feedback}\n\n"
    "Output ONLY: <label>your corrected answer as an integer</label>\n\n"
    "Answer: "
)


def _extract_sentence_words(input_text):
    """Extract sentence from 'Sentence: '...' Count...' format and count words."""
    m = re.search(r"Sentence:\s*'(.+?)'", input_text)
    sent = m.group(1) if m else input_text
    words = re.findall(r'\w+', sent)
    return sent, len(words)


def _validate(prediction, input_text):
    """Validate prediction against input-derived constraints.

    Returns (valid, feedback) where feedback is a human-readable
    error description for multi-turn correction.
    Only uses non-revealing checks.
    """
    if not prediction or not prediction.strip():
        return False, "Your output was empty. Please provide a numeric answer."
    pred = prediction.strip()
    if not re.fullmatch(r'-?\d+', pred):
        return False, f"Your answer '{pred}' is not a valid integer. Output only a number."
    val = int(pred)
    if val < 0:
        return False, "Word count cannot be negative."

    # Word count upper bound
    _, word_count = _extract_sentence_words(input_text)
    if val > word_count:
        return False, (
            f"Your answer ({val}) exceeds the total word count of the sentence ({word_count}). "
            "The count of nouns/verbs cannot exceed all words. Recalculate."
        )

    return True, ""


class Task2(BaseTask):
    task_id = 2
    DATA_FILE = '../data/openseek-2_count_nouns_verbs.json'
    PROMPT_TEMPLATE = PROMPT_TEMPLATE

    DEFAULT_CFG = {
        "name": "count_nouns_verbs",
        "temperature": 0.3,
        "num_votes": 1,
        "max_tokens": 8000,
        "stop_tokens": None,
        "system_prompt": "You are a precise linguist.",
        "min_icl_tokens": 30_000,
    }

    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self.cfg = None

    def postprocess(self, prediction: str, raw_output: str = "") -> str:
        pred = prediction.strip() if prediction else ""
        if pred and re.fullmatch(r'-?\d+', pred):
            return pred
        if pred:
            numbers = re.findall(r'(?<!\w)-?\d+(?!\w)', pred)
            if numbers:
                return numbers[-1]
        return pred

    def split_icl_padding(self, all_examples):
        return all_examples, []

    def should_retry(self):
        """Use custom validation + retry in run_inference."""
        return False

    def prepare(self):
        self.retriever = POSRetriever(self.icl_examples)
        return self

    def _build_curated_examples_str(self, text, token_budget):
        """Build examples string using 11.28-style fixed curated examples (U-shape).

        Structure: head(curated) + middle(POS fill) + tail(curated repeat)
        """
        curated_formatted = []
        for ex in TASK2_CURATED_EXAMPLES:
            curated_formatted.append({
                'input': ex['input'],
                'output': ex['output'],
            })

        def _build_curated_str(examples, start_id):
            s = ""
            for i, ex in enumerate(examples):
                label_str = ex['output'][0] if isinstance(ex['output'], list) else ex['output']
                s += f"Example {start_id + i}:\nInput:\n{ex['input']}\nOutput:\n<label>{label_str}</label>\n\n"
            return s

        head_str = _build_curated_str(curated_formatted, 1)
        tail_str = _build_curated_str(curated_formatted, len(curated_formatted) + 1)
        curated_inputs = {ex['input'] for ex in curated_formatted}

        # POS retrieval for middle fill
        retrieved = self.retriever.retrieve_top_k(text, top_k=2000)
        padding_candidates = [ex for ex in retrieved if ex['input'] not in curated_inputs]

        token_count = len(self.tokenizer.encode(head_str + tail_str, add_special_tokens=False))
        example_id = len(curated_formatted) * 2 + 1
        middle_parts = []

        for ex in padding_candidates:
            if token_count >= token_budget:
                break
            label_str = ex['output'][0] if isinstance(ex['output'], list) else ex['output']
            part = f"Example {example_id}:\nInput:\n{ex['input']}\nOutput:\n<label>{label_str}</label>\n\n"
            part_tokens = len(self.tokenizer.encode(part, add_special_tokens=False))
            if token_count + part_tokens > token_budget + 2000:
                break
            middle_parts.append(part)
            token_count += part_tokens
            example_id += 1

        if middle_parts:
            dynamic_examples_str = head_str + "".join(middle_parts) + tail_str
            token_count = len(self.tokenizer.encode(dynamic_examples_str, add_special_tokens=False))
        else:
            dynamic_examples_str = head_str + tail_str
            token_count = len(self.tokenizer.encode(dynamic_examples_str, add_special_tokens=False))

        return dynamic_examples_str, token_count

    def process_sample(self, test_sample, first_sample=False):
        text = test_sample['input']

        # POS retrieval: get similar examples from same type (noun/verb)
        retrieved = self.retriever.retrieve_top_k(text, top_k=2000)

        # Pick 10 most relevant (top similarity) — they go right next to input
        head_examples = retrieved[:10]
        head_inputs = {ex['input'] for ex in head_examples}

        # Pick 10 balanced examples from remaining (for the start of prompt)
        def _pick_balanced(candidates, n=10):
            """Pick n examples with balanced answer distribution, preserving order."""
            from collections import defaultdict
            groups = defaultdict(list)
            for i, ex in enumerate(candidates):
                label = str(ex['output'][0] if isinstance(ex['output'], list) else ex['output']).strip()
                groups[label].append((i, ex))

            labels = sorted(groups.keys(), key=lambda x: int(x) if x.isdigit() else 99)
            per_label = max(1, n // len(labels))

            picked = []
            used_indices = set()
            for label in labels:
                for idx, ex in groups[label][:per_label]:
                    picked.append(ex)
                    used_indices.add(idx)

            # Fill remaining with highest-similarity unused
            remaining = n - len(picked)
            if remaining > 0:
                for i, ex in enumerate(candidates):
                    if i not in used_indices:
                        picked.append(ex)
                        remaining -= 1
                        if remaining == 0:
                            break

            return picked

        remaining_after_head = [ex for ex in retrieved if ex['input'] not in head_inputs]
        tail_examples = _pick_balanced(remaining_after_head, n=10)
        tail_inputs = {ex['input'] for ex in tail_examples}

        # Middle fill from remaining (between tail and head)
        remaining = [ex for ex in retrieved if ex['input'] not in head_inputs and ex['input'] not in tail_inputs]

        def _build_example_str(examples, start_id):
            s = ""
            for i, ex in enumerate(examples):
                label_str = ex['output'][0] if isinstance(ex['output'], list) else ex['output']
                s += f"Example {start_id + i}:\nInput:\n{ex['input']}\nOutput:\n<label>{label_str}</label>\n\n"
            return s

        # Tail (balanced) goes at start, head (most relevant) goes at end next to input
        tail_str = _build_example_str(tail_examples, 1)
        head_str = _build_example_str(head_examples, len(tail_examples) + 1)

        # Calculate token budget for middle fill
        head_tail_tokens = len(self.tokenizer.encode(tail_str + head_str, add_special_tokens=False))
        token_count = head_tail_tokens
        example_id = len(tail_examples) + 1
        middle_parts = []

        for ex in remaining:
            if token_count >= self.min_icl_tokens:
                break
            label_str = ex['output'][0] if isinstance(ex['output'], list) else ex['output']
            part = f"Example {example_id}:\nInput:\n{ex['input']}\nOutput:\n<label>{label_str}</label>\n\n"
            part_tokens = len(self.tokenizer.encode(part, add_special_tokens=False))
            if token_count + part_tokens > self.min_icl_tokens + 2000:
                break
            middle_parts.append(part)
            token_count += part_tokens
            example_id += 1

        if middle_parts:
            dynamic_examples_str = tail_str + "".join(middle_parts) + head_str
        else:
            dynamic_examples_str = tail_str + head_str
            token_count = len(self.tokenizer.encode(dynamic_examples_str, add_special_tokens=False))

        prompt = self.PROMPT_TEMPLATE.format(
            examples_str=dynamic_examples_str,
            input_text=text,
        )
        fallback = ''
        if retrieved:
            out = retrieved[0]['output']
            fallback = out[0] if isinstance(out, list) else str(out).strip()
        return prompt, dynamic_examples_str, token_count, {'fallback': fallback}

    def run_inference(self, test_sample, call_model):
        """R0(curated U-shape) → validate → R1(POS dynamic) → R2(feedback) → R3(temp=0.5) → fallback."""
        text = test_sample['input']

        # ===== R0: 11.28-style fixed curated examples (U-shape) =====
        examples_str_r0, token_count_r0 = self._build_curated_examples_str(text, self.min_icl_tokens)
        prompt_r0 = self.PROMPT_TEMPLATE.format(
            examples_str=examples_str_r0,
            input_text=text,
        )
        prediction, raw_output = call_model(prompt_r0, self.cfg)
        raw_outputs = [f"[R0 curated] {raw_output}"]
        prediction = self.postprocess(prediction, raw_output)

        valid, feedback = _validate(prediction, test_sample['input'])
        if valid:
            return prediction, prompt_r0, raw_outputs, [prediction], {}

        # ===== R1: dynamic POS retrieval (original R1) =====
        print(f"  ⚠️ R0 [{feedback}] for sample {test_sample['id']}, trying POS dynamic retrieval...")
        prompt, _, _, info = self.process_sample(test_sample)
        fallback = info.get('fallback', '0')

        prediction1, raw_output1 = call_model(prompt, self.cfg)
        raw_outputs.append(f"[R1 POS dynamic] {raw_output1}")
        prediction1 = self.postprocess(prediction1, raw_output1)

        valid1, feedback1 = _validate(prediction1, test_sample['input'])
        if valid1:
            prediction = prediction1
            raw_outputs[-1] = f"[R1 SUCCESS] {raw_output1}"
            return prediction, prompt, raw_outputs, [prediction], {}

        # ===== R2: retry with feedback (original R2) =====
        print(f"  ⚠️ R1 [{feedback1}] for sample {test_sample['id']}, retrying with feedback...")
        prev_output = raw_output1.replace('<label>', '').replace('</label>', '').strip()
        retry_prompt = prompt + prev_output + RETRY_PROMPT.format(feedback=feedback1)
        prediction2, raw_output2 = call_model(retry_prompt, self.cfg)
        raw_outputs.append(f"[R2 with feedback] {raw_output2}")
        prediction2 = self.postprocess(prediction2, raw_output2)

        valid2, feedback2 = _validate(prediction2, test_sample['input'])
        if valid2:
            prediction = prediction2
            raw_outputs[-1] = f"[R2 SUCCESS] {raw_output2}"
            return prediction, prompt, raw_outputs, [prediction], {}

        # ===== R3: retry with higher temperature (original R3) =====
        print(f"  ⚠️ R2 [{feedback2}] still invalid, retrying with temp=0.5...")
        prev_output2 = raw_output2.replace('<label>', '').replace('</label>', '').strip()
        retry_prompt2 = retry_prompt + prev_output2 + RETRY_PROMPT.format(feedback=feedback2)
        high_temp_cfg = dict(self.cfg)
        high_temp_cfg['temperature'] = 0.5
        prediction3, raw_output3 = call_model(retry_prompt2, high_temp_cfg)
        raw_outputs.append(f"[R3 temp=0.5] {raw_output3}")
        prediction3 = self.postprocess(prediction3, raw_output3)

        valid3, _ = _validate(prediction3, test_sample['input'])
        if valid3:
            prediction = prediction3
            raw_outputs[-1] = f"[R3 SUCCESS] {raw_output3}"
        else:
            prediction = fallback
            raw_outputs.append(f"[FALLBACK to most-similar answer: {fallback} after 4 failed rounds]")

        return prediction, prompt, raw_outputs, [prediction], {}
