"""Task 3: Collatz conjecture — length-matched examples, most relevant closest to question."""
import sys
import os
import re
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from tasks.base import BaseTask

PROMPT_TEMPLATE = (
    "You are a precise calculator. Apply Collatz transformation to each number.\n\n"
    "Rules:\n"
    "- If even → divide by 2\n"
    "- If odd → multiply by 3 and add 1\n\n"
    "Examples:\n"
    "{examples_str}\n"
    "Input: {input_text}\n\n"
    "Output ONLY: <label>[result_list]</label>\n\n"
    "Answer: "
)


def _parse_int_list(text):
    """Extract integers from input string like '[1, 2, 3]'."""
    if isinstance(text, list):
        return text
    return [int(x) for x in re.findall(r'-?\d+', text)]


RETRY_PROMPT = (
    "\n\n{feedback}\n\n"
    "Output ONLY: <label>your corrected answer as a list</label>\n\n"
    "Answer: "
)


def _extract_answer_from_thought(text, input_text=None, validate_fn=None):
    """Extract the model's intended answer from thought blocks.

    Strategy: find all list patterns in reasoning text, return the first that passes validation.
    """
    ot = chr(60) + 'think' + chr(62)
    ct = chr(60) + '/' + 'think' + chr(62)
    m = re.search(re.escape(ot) + r'(.*?)' + re.escape(ct), text, re.DOTALL)
    thought = m.group(1) if m else text

    # Collect all list patterns from thought
    all_lists = re.findall(r'\[\d+(?:,\s*\d+)+\]', thought)
    if not all_lists:
        return None

    # If validator provided, return first candidate that passes validation (reverse order = most recent first)
    if validate_fn and input_text:
        for lst in reversed(all_lists):
            valid, _ = validate_fn(lst, input_text)
            if valid:
                return lst
        return None  # None passed validation

    # No validator: return most recent list
    return all_lists[-1]


def _validate(prediction, input_text):
    """Validate prediction with conservative rules:
    - Length must match
    - If input contains only even numbers, all outputs must be < corresponding inputs
    """
    if not prediction or not prediction.strip():
        return False, "Your output was empty. Please provide a list as answer."

    pred = prediction.strip()
    if not (pred.startswith('[') and pred.endswith(']')):
        return False, f"Your answer must be a list in brackets. Got: '{pred[:50]}...'"

    try:
        pred_nums = [int(x) for x in re.findall(r'-?\d+', pred)]
    except:
        return False, f"Your answer is not a valid list of integers."

    if not pred_nums:
        return False, "Your output contains no valid numbers."

    input_nums = _parse_int_list(input_text)
    if len(pred_nums) != len(input_nums):
        return False, (
            f"Your output has {len(pred_nums)} numbers, but the input has {len(input_nums)} numbers. "
            f"Each input number maps to exactly one output. Recalculate."
        )

    # Conservative check: even input → output must be smaller; odd input → output must be larger
    for inp, out in zip(input_nums, pred_nums):
        if inp % 2 == 0 and out >= inp:
            return False, (
                f"Your output {out} for input {inp} should be smaller. Recalculate."
            )
        if inp % 2 == 1 and out <= inp:
            return False, (
                f"Your output {out} for input {inp} should be larger. Recalculate."
            )

    return True, ""


class Task3(BaseTask):
    task_id = 3
    DATA_FILE = '../data/openseek-3_collatz_conjecture.json'
    PROMPT_TEMPLATE = PROMPT_TEMPLATE

    DEFAULT_CFG = {
        "name": "collatz_conjecture",
        "temperature": 0.1,
        "num_votes": 1,
        "max_tokens": 8000,
        "stop_tokens": None,
        "system_prompt": "You are a precise calculator.",
        "min_icl_tokens": 30_000,
    }

    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self.cfg = None

    def postprocess(self, prediction: str, raw_output: str = "") -> str:
        pred = prediction.strip() if prediction else ""
        if pred.startswith('[') and pred.endswith(']'):
            pred = re.sub(r',\s*', ', ', pred)
        return pred

    def split_icl_padding(self, all_examples):
        return all_examples, []

    def prepare(self):
        # Group examples by input length
        self.by_length = {}
        for ex in self.icl_examples:
            nums = _parse_int_list(ex['input'])
            n = len(nums)
            if n not in self.by_length:
                self.by_length[n] = []
            self.by_length[n].append(ex)
        return self

    def process_sample(self, test_sample, first_sample=False):
        text = test_sample['input']
        test_nums = _parse_int_list(text)
        test_len = len(test_nums)

        # Build examples: same-length closest to question, other lengths as fill.
        # Cap other_lens so same_len examples fit within the ~30k token budget.
        same_len = self.by_length.get(test_len, [])
        other_lens = []
        for n, exs in sorted(self.by_length.items()):
            if n != test_len:
                other_lens.extend(exs)

        # Compute same_len token count to reserve budget
        same_len_tokens = 0
        for ex in same_len:
            label_str = ex['output'][0] if isinstance(ex['output'], list) else ex['output']
            part = f"Input:\n{ex['input']}\nOutput:\n<label>{label_str}</label>\n\n"
            same_len_tokens += len(self.tokenizer.encode(part, add_special_tokens=False))

        # Fill remaining budget with other_lens (prepended first = farthest from question)
        remaining_budget = self.min_icl_tokens - same_len_tokens
        other_tokens = 0
        capped_other = []
        for ex in other_lens:
            label_str = ex['output'][0] if isinstance(ex['output'], list) else ex['output']
            part = f"Input:\n{ex['input']}\nOutput:\n<label>{label_str}</label>\n\n"
            part_tokens = len(self.tokenizer.encode(part, add_special_tokens=False))
            if other_tokens + part_tokens <= remaining_budget:
                capped_other.append(ex)
                other_tokens += part_tokens
            else:
                break

        ordered = capped_other + same_len

        # Build examples string via prepend: last item in ordered ends up closest to question
        examples_str = ""
        token_count = 0
        for ex in ordered:
            label_str = ex['output'][0] if isinstance(ex['output'], list) else ex['output']
            part = f"Input:\n{ex['input']}\nOutput:\n<label>{label_str}</label>\n\n"
            part_tokens = len(self.tokenizer.encode(part, add_special_tokens=False))

            examples_str = part + examples_str
            token_count += part_tokens
            if token_count > self.min_icl_tokens:
                break

        prompt = self.PROMPT_TEMPLATE.format(
            examples_str=examples_str,
            input_text=text,
        )
        return prompt, examples_str, token_count, {}

    def should_retry(self):
        """Use custom validation + retry in run_inference."""
        return False

    def run_inference(self, test_sample, call_model):
        """Round 1 → validate → retry with feedback → use last attempt on total failure."""
        prompt, _, _, _ = self.process_sample(test_sample)

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

        # Round 3: second retry with higher temperature
        print(f"  ⚠️ [{feedback2}] still invalid, second retry (temp=0.5)...")
        prev_output2 = raw_output2.replace('<label>', '').replace('</label>', '').strip()
        retry_prompt2 = retry_prompt + prev_output2 + RETRY_PROMPT.format(feedback=feedback2)
        high_temp_cfg = dict(self.cfg)
        high_temp_cfg['temperature'] = 0.5
        prediction3, raw_output3 = call_model(retry_prompt2, high_temp_cfg)
        raw_outputs.append(f"[RETRY2 with feedback temp=0.5] {raw_output3}")
        prediction3 = self.postprocess(prediction3, raw_output3)

        valid3, _ = _validate(prediction3, test_sample['input'])
        if valid3:
            prediction = prediction3
            raw_outputs[-1] = f"[RETRY2 SUCCESS] {raw_output3}"
        else:
            # 3 rounds failed: try to extract answer from thought blocks
            thought_answer = None
            for raw in raw_outputs:
                extracted = _extract_answer_from_thought(raw, test_sample['input'], _validate)
                if extracted:
                    thought_answer = extracted
                    break
            if thought_answer:
                prediction = thought_answer
                raw_outputs.append(f"[THOUGHT EXTRACTION SUCCESS] {thought_answer}")
            else:
                prediction = prediction3
                raw_outputs.append(f"[RETRY2 FAILED, using last attempt: {prediction3[:50]}]")

        return prediction, prompt, raw_outputs, [prediction], {}
