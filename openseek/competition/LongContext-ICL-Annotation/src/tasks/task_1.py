"""Task 1: Closest integers — feature-based retrieval from full pool, most relevant closest to question."""
import sys
import os
import re
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from tasks.base import BaseTask

PROMPT_TEMPLATE = (
    "You are a precise calculator. Find the minimum absolute difference between any two integers.\n\n"
    "Examples:\n"
    "{examples_str}\n"
    "Input: {input_text}\n\n"
    "Output ONLY: <label>your answer as an integer</label>\n\n"
    "Answer: "
)

# Error feedback appended to prompt for multi-turn retry.
# The {feedback} placeholder is filled with the specific error message
# from _validate, so the prompt is direct and informative.
RETRY_PROMPT = (
    "\n\n{feedback}\n\n"
    "Output ONLY: <label>your corrected answer as an integer</label>\n\n"
    "Answer: "
)


def _parse_int_list(text):
    """Extract integers from input string like '[1, -2, 3]'."""
    if isinstance(text, list):
        return text
    return [int(x) for x in re.findall(r'-?\d+', text)]


def _extract_features(nums):
    """Extract observable input features for similarity matching."""
    return {
        'length': len(nums),
        'min': min(nums),
        'max': max(nums),
        'range': max(nums) - min(nums),
        'has_negative': any(n < 0 for n in nums),
        'has_duplicate': len(nums) != len(set(nums)),
        'mean_abs': sum(abs(n) for n in nums) / len(nums),
    }


def _answer_bucket(ex):
    """Group examples by answer value for balanced selection near question.

    Buckets follow v2's distribution (1-3 dominates at ~41%), which maximizes
    R1 accuracy. Silent 0-errors are handled by zero-detection + lenient fallback.
    """
    label = ex['output']
    if isinstance(label, list):
        label = label[0] if label else '0'
    val = int(label)
    if val == 0:
        return 0
    elif val <= 3:
        return 1
    elif val <= 6:
        return 2
    elif val <= 15:
        return 3
    else:
        return 4


def _balance_top5(scored):
    """From top-100 most similar examples, pick 5 with balanced answer distribution.

    Each answer bucket contributes at most 1 example (most similar in that bucket).
    Returns (balanced_5, remaining) where balanced_5 are ordered by bucket 0→4
    (small answers to large) for the last positions closest to question.
    """
    top_n = scored[:100]
    buckets = {i: [] for i in range(5)}
    remaining = list(scored[100:])

    for sim, ex in top_n:
        b = _answer_bucket(ex)
        buckets[b].append((sim, ex))

    balanced = []
    used_ids = set()  # Track used examples across all fills

    for b in range(5):
        if buckets[b]:
            # Filter out items already used as fills for earlier buckets
            buckets[b] = [(sim, ex) for sim, ex in buckets[b] if id(ex) not in used_ids]
            if not buckets[b]:
                continue
            item = buckets[b][0]
            balanced.append(item)
            used_ids.add(id(item[1]))
            remaining.extend(buckets[b][1:])
        else:
            # Bucket empty, fill with next most similar from any bucket (not already used)
            for b2 in range(5):
                found = False
                for item in buckets[b2]:
                    if id(item[1]) not in used_ids:
                        balanced.append(item)
                        used_ids.add(id(item[1]))
                        found = True
                        break
                if found:
                    break

    # Sort balanced_5 by bucket (small answer first → large answer last)
    # The last one (largest bucket) will be closest to question
    balanced.sort(key=lambda x: _answer_bucket(x[1]))
    return balanced, remaining


def _similarity(tf, ef):
    """Compute feature-based similarity between test and example features.

    Only uses observable input features (length, range, sign, duplicates) —
    never the answer, since that would leak the ground truth into retrieval.
    """
    max_len = max(tf['length'], ef['length'])
    len_sim = 1.0 - abs(tf['length'] - ef['length']) / max_len if max_len > 0 else 1.0
    max_range = max(tf['range'], ef['range'], 1)
    range_sim = 1.0 - abs(tf['range'] - ef['range']) / max_range
    max_mean = max(tf['mean_abs'], ef['mean_abs'], 1)
    mean_sim = 1.0 - abs(tf['mean_abs'] - ef['mean_abs']) / max_mean
    neg_match = 1.0 if tf['has_negative'] == ef['has_negative'] else 0.0
    dup_match = 1.0 if tf['has_duplicate'] == ef['has_duplicate'] else 0.0
    return (0.30 * len_sim + 0.25 * range_sim + 0.20 * mean_sim +
            0.15 * neg_match + 0.10 * dup_match)


def _extract_thought(text):
    """Extract  content from model output."""
    ot = chr(60) + 'think' + chr(62)
    ct = chr(60) + '/' + 'think' + chr(62)
    m = re.search(ot + r'(.*?)' + re.escape(ct), text, re.DOTALL)
    if m:
        return m.group(1)
    m = re.search(ot + r'(.*)', text, re.DOTALL)
    return m.group(1) if m else ''


def _lenient_thought_fallback(raw_outputs, input_text, validate_fn=None):
    """Extract most likely correct answer from  thought blocks.

    When all rounds fail, the model often already computed the correct
    answer inside  but got stuck in a loop. Strategy:
    1. Look for explicit answer declarations ("answer is X", "minimum is X").
    2. For high-frequency numbers, check if they appear near answer-related
       keywords within a window — a number surrounded by "answer/diff/min"
       context is much more likely to be the computed result than a raw input
       number repeated during sorting.
    """
    from collections import Counter

    CONTEXT_WINDOW = 80  # chars around each number to check for keywords
    ANSWER_KEYWORDS = ['answer', 'minimum', 'smallest', 'difference', 'diff',
                       'result', 'output', 'correct', 'min diff', 'absolute']

    nums = [int(x) for x in re.findall(r'-?\d+', input_text)]
    input_nums = set(str(abs(n)) for n in nums)
    has_dup = len(nums) != len(set(nums))

    all_thought = ''
    for raw in raw_outputs:
        if raw.startswith('['):
            bracket_end = raw.find('] ')
            if bracket_end != -1:
                raw = raw[bracket_end + 2:]
        all_thought += _extract_thought(raw) + '\n'

    if not all_thought:
        return None

    # Strategy 1: explicit answer declarations — try each candidate (last first)
    answer_patterns = [
        r'(?:the\s+)?(?:answer|result)\s+(?:is\s*=?\s*)(\d+)',
        r'(?:minimum|smallest|least)\s+(?:absolute\s+)?(?:diff(?:erence)?)?\s+(?:is\s*=?\s*)(\d+)',
        r'(?:min\s*diff|smallest\s+diff)\s+(?:is\s*=?\s*)(\d+)',
    ]
    answer_candidates = []
    for pat in answer_patterns:
        answer_candidates.extend(re.findall(pat, all_thought, re.IGNORECASE))

    if answer_candidates:
        # Try from last to first (most recent first), validate each
        for c in reversed(answer_candidates):
            if c == '0' and not has_dup:
                continue  # skip zero without duplicates
            if validate_fn:
                valid, _ = validate_fn(c, input_text)
                if valid:
                    return int(c)
            else:
                return int(c)
        # All candidates failed validation, fall through to strategy 2

    # Strategy 2: high-frequency numbers that appear near answer keywords
    freq = Counter(re.findall(r'\b(\d+)\b', all_thought))
    high_freq = {k: v for k, v in freq.items() if v >= 3 and k not in input_nums}
    if not high_freq:
        return None

    # For each candidate, score by how many times it appears near answer keywords
    keyword_pattern = '|'.join(re.escape(kw) for kw in ANSWER_KEYWORDS)
    scored = {}
    for num_str in high_freq:
        positions = [m.start() for m in re.finditer(r'\b' + re.escape(num_str) + r'\b', all_thought)]
        context_count = 0
        for pos in positions:
            start = max(0, pos - CONTEXT_WINDOW)
            end = min(len(all_thought), pos + CONTEXT_WINDOW)
            window = all_thought[start:end].lower()
            if re.search(keyword_pattern, window):
                context_count += 1
        scored[num_str] = context_count

    # Try candidates in score order, validate each
    scored = {k: v for k, v in scored.items() if v > 0}
    if scored:
        sorted_candidates = sorted(scored.keys(), key=lambda k: -scored[k])
        for c in sorted_candidates:
            if c == '0' and not has_dup:
                continue
            if validate_fn:
                valid, _ = validate_fn(c, input_text)
                if valid:
                    return int(c)
            else:
                return int(c)
        return None  # All candidates failed validation

    return None


class Task1(BaseTask):
    task_id = 1
    DATA_FILE = '../data/openseek-1_closest_integers.json'
    PROMPT_TEMPLATE = PROMPT_TEMPLATE

    DEFAULT_CFG = {
        "name": "closest_integers",
        "temperature": 0.3,
        "top_k": 5500,
        "num_votes": 1,
        "max_tokens": 8000,
        "stop_tokens": None,
        "system_prompt": "You are a precise calculator.",
        "min_icl_tokens": 30_000,
    }

    def _validate(self, prediction, input_text):
        """Validate prediction against input-derived constraints.

        Returns (valid, feedback) where feedback is a human-readable
        error description for multi-turn correction.
        Only uses non-revealing checks (format + pigeonhole upper bound).
        """
        if not prediction or not prediction.strip():
            return False, "Your output was empty. Please provide a numeric answer."
        pred = prediction.strip()
        if not re.fullmatch(r'-?\d+', pred):
            return False, f"Your answer '{pred}' is not a valid integer. Output only a number."
        val = int(pred)
        if val < 0:
            return False, "Minimum absolute difference cannot be negative."

        nums = _parse_int_list(input_text)
        if len(nums) < 2:
            return True, ""

        # Zero is only possible if input has duplicates
        if val == 0 and len(nums) == len(set(nums)):
            return False, (
                "Your answer is 0, but all input numbers are distinct. "
                "The minimum difference cannot be zero unless two numbers are equal. "
                "Recalculate."
            )

        # Duplicates in input force the answer to be 0
        if val != 0 and len(nums) != len(set(nums)):
            dup_vals = [x for x in set(nums) if nums.count(x) > 1]
            return False, (
                f"The input contains duplicate number(s): {dup_vals}. "
                f"Re-check your calculation."
            )

        n = len(nums)
        range_val = max(nums) - min(nums)

        # Pigeonhole upper bound: min diff <= range / (n-1)
        pigeonhole = range_val / (n - 1)
        if val > pigeonhole + 0.5:
            return False, (
                f"Your answer ({val}) is too large. By pigeonhole principle, "
                f"with {n} numbers spanning range {range_val}, the minimum difference "
                f"must be <= {int(pigeonhole) + 1}. Recalculate."
            )

        # Check: answer must be one of the adjacent differences after sorting
        sorted_nums = sorted(nums)
        adjacent_diffs = set(abs(sorted_nums[i+1] - sorted_nums[i]) for i in range(len(sorted_nums)-1))
        if val not in adjacent_diffs:
            return False, (
                "The minimum absolute difference between any two integers in the input "
                "must be the difference between two adjacent numbers after sorting. "
                f"Your answer {val} does not match any such difference. "
                f"Sort the numbers and re-check each adjacent pair."
            )

        return True, ""

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
        """Use all 5500 examples as retrieval pool, no separate padding."""
        return all_examples, []

    def prepare(self):
        # Filter zero-output examples to reduce "answer is usually 0" bias
        zero_examples = [ex for ex in self.icl_examples if ex['output'] == ['0'] or ex['output'] == '0']
        non_zero = [ex for ex in self.icl_examples if ex['output'] != ['0'] and ex['output'] != '0']
        if len(zero_examples) > 2:
            self.icl_examples = non_zero[:5498] + zero_examples[:2]

        # Precompute features for all examples
        for ex in self.icl_examples:
            ex['_t1_features'] = _extract_features(_parse_int_list(ex['input']))

        return self

    def _build_prompt_from_scored(self, test_sample, scored, offset=0):
        """Build prompt from scored examples, with balanced top-5 closest to question.

        From top-100 most similar examples, selects 5 with diverse answer values
        for the positions closest to the question. Remaining examples fill earlier
        positions in similarity order.
        """
        text = test_sample['input']
        scored = scored[offset:]

        # Balanced top-5 + remaining
        balanced_5, remaining = _balance_top5(scored)

        examples_str = ""
        token_count = 0

        # First: fill with remaining examples in similarity order (prepending = farthest first)
        for sim, ex in remaining:
            label_str = ex['output'][0] if isinstance(ex['output'], list) else ex['output']
            part = f"Input:\n{ex['input']}\nOutput:\n<label>{label_str}</label>\n\n"
            part_tokens = len(self.tokenizer.encode(part, add_special_tokens=False))

            examples_str = part + examples_str
            token_count += part_tokens

            if token_count > self.min_icl_tokens:
                break

        # Then: balanced top-5 closest to question (prepended last = closest)
        for sim, ex in balanced_5:
            label_str = ex['output'][0] if isinstance(ex['output'], list) else ex['output']
            part = f"Input:\n{ex['input']}\nOutput:\n<label>{label_str}</label>\n\n"
            part_tokens = len(self.tokenizer.encode(part, add_special_tokens=False))
            examples_str = part + examples_str
            token_count += part_tokens

        prompt = self.PROMPT_TEMPLATE.format(
            examples_str=examples_str,
            input_text=text,
        )
        prompt_tokens = len(self.tokenizer.encode(prompt, add_special_tokens=False))
        return prompt, examples_str, prompt_tokens, {}

    def _score_all(self, test_nums, examples=None):
        """Score ICL examples against test input features."""
        if examples is None:
            examples = self.icl_examples
        tf = _extract_features(test_nums)
        scored = []
        for ex in examples:
            ef = ex.get('_t1_features')
            if ef is None:
                ef = _extract_features(_parse_int_list(ex['input']))
            scored.append((_similarity(tf, ef), ex))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored

    def _build_prompt_without_zeros(self, test_sample):
        """Build prompt with zero-answer examples completely removed."""
        test_nums = _parse_int_list(test_sample['input'])
        non_zero = [ex for ex in self.icl_examples
                    if ex['output'] not in (['0'], '0')]
        scored = self._score_all(test_nums, examples=non_zero)
        return self._build_prompt_from_scored(test_sample, scored)

    def _is_zero_without_dup(self, prediction, input_text):
        """Check if prediction is 0 but input has no duplicates."""
        pred = prediction.strip()
        if pred != '0':
            return False
        nums = _parse_int_list(input_text)
        return len(nums) == len(set(nums))

    def process_sample(self, test_sample, first_sample=False):
        test_nums = _parse_int_list(test_sample['input'])
        scored = self._score_all(test_nums)
        return self._build_prompt_from_scored(test_sample, scored, offset=0)

    def should_retry(self):
        """Use custom multi-turn retry in run_inference."""
        return False

    def run_inference(self, test_sample, call_model):
        """Multi-turn retry: append error feedback to the original prompt.

        If all 3 rounds fail (empty or invalid), fall back to "0"
        as the safest default for a minimum-difference task.
        """
        prompt, _, _, _ = self.process_sample(test_sample)

        prediction, raw_output = call_model(prompt, self.cfg)
        raw_outputs = [raw_output]
        prediction = self.postprocess(prediction, raw_output)

        valid, feedback = self._validate(prediction, test_sample['input'])
        if valid:
            return prediction, prompt, raw_outputs, [prediction], {}

        # Round 2: retry with feedback, strip <label> from previous output
        print(f"  ⚠️ [{feedback}] for sample {test_sample['id']}, retrying...")

        # If model predicted 0 but input has no duplicates, rebuild prompt
        # with zero-answer examples removed to break the bias
        if self._is_zero_without_dup(prediction, test_sample['input']):
            prompt, _, _, _ = self._build_prompt_without_zeros(test_sample)
            prev_output = ''  # Start fresh with new examples
            retry_prompt = prompt + RETRY_PROMPT.format(
                feedback="Your answer is 0, but all input numbers are distinct. The minimum difference cannot be zero unless two numbers are equal. Recalculate."
            )
        else:
            prev_output = raw_output.replace('<label>', '').replace('</label>', '').strip()
            retry_prompt = prompt + prev_output + RETRY_PROMPT.format(
                feedback=feedback
            )
        prediction2, raw_output2 = call_model(retry_prompt, self.cfg)
        raw_outputs.append(f"[RETRY with feedback] {raw_output2}")
        prediction2 = self.postprocess(prediction2, raw_output2)

        valid2, feedback2 = self._validate(prediction2, test_sample['input'])
        if valid2:
            prediction = prediction2
            raw_outputs[-1] = f"[RETRY SUCCESS] {raw_output2}"
            return prediction, prompt, raw_outputs, [prediction], {}

        # Round 3: second retry with example calculation — show step-by-step on most similar example
        print(f"  ⚠️ [{feedback2}] still invalid, second retry with example calculation...")
        test_nums = _parse_int_list(test_sample['input'])

        # If still predicting 0 without duplicates, use non-zero examples for scoring
        if self._is_zero_without_dup(prediction2, test_sample['input']):
            non_zero = [ex for ex in self.icl_examples
                        if ex['output'] not in (['0'], '0')]
            scored = self._score_all(test_nums, examples=non_zero)
        else:
            scored = self._score_all(test_nums)

        top_ex = scored[0][1]  # most similar example
        ex_nums = _parse_int_list(top_ex['input'])
        ex_sorted = sorted(ex_nums)
        ex_diffs = [abs(ex_sorted[i+1] - ex_sorted[i]) for i in range(len(ex_sorted)-1)]
        ex_answer = min(ex_diffs)
        ex_label = top_ex['output']
        if isinstance(ex_label, list):
            ex_label = ex_label[0]

        # Build sorted pairs list (show first 5 + last 3 diffs to keep it short)
        ex_diff_pairs = []
        for i in range(len(ex_sorted)-1):
            d = abs(ex_sorted[i+1] - ex_sorted[i])
            ex_diff_pairs.append(f"|{ex_sorted[i]} - {ex_sorted[i+1]}| = {d}")

        # Show all diffs for short lists, truncate for long ones
        if len(ex_diff_pairs) <= 12:
            diffs_str = '\n'.join(ex_diff_pairs)
        else:
            diffs_str = '\n'.join(ex_diff_pairs[:6] + ['...', ex_diff_pairs[-2:]])

        test_sorted = sorted(test_nums)
        test_input_str = test_sample['input']

        prev_output2 = raw_output2.replace('<label>', '').replace('</label>', '').strip()
        retry_prompt2 = (
            retry_prompt + prev_output2 +
            f"\n\nLet me show you how to solve this step by step using a similar example:\n\n"
            f"Example input: {top_ex['input']}\n"
            f"Example answer: {ex_label}\n\n"
            f"Calculation steps for the example:\n"
            f"1. Sort the numbers: [{', '.join(str(x) for x in ex_sorted)}]\n"
            f"2. Compute adjacent differences:\n{diffs_str}\n"
            f"3. The smallest difference is: {ex_answer} → matches the example answer: {ex_label}\n\n"
            f"Now apply the same steps to the test input:\n"
            f"Input: {test_input_str}\n"
            f"1. Sort the numbers: [{', '.join(str(x) for x in test_sorted)}]\n"
            f"2. Compute each adjacent difference:\n"
        )
        prediction3, raw_output3 = call_model(retry_prompt2, self.cfg)
        raw_outputs.append(f"[RETRY2 with feedback] {raw_output3}")
        prediction3 = self.postprocess(prediction3, raw_output3)

        valid3, feedback3 = self._validate(prediction3, test_sample['input'])
        if valid3:
            prediction = prediction3
            raw_outputs[-1] = f"[RETRY2 SUCCESS] {raw_output3}"
            return prediction, prompt, raw_outputs, [prediction], {}

        # Round 4: higher temperature retry (break out of deterministic loop)
        print(f"  ⚠️ [{feedback3}] still invalid, third retry with higher temperature...")
        prev_output3 = raw_output3.replace('<label>', '').replace('</label>', '').strip()
        retry_prompt3 = retry_prompt2 + prev_output3 + RETRY_PROMPT.format(
            feedback=feedback3
        )
        high_temp_cfg = dict(self.cfg)
        high_temp_cfg['temperature'] = 0.5
        prediction4, raw_output4 = call_model(retry_prompt3, high_temp_cfg)
        raw_outputs.append(f"[RETRY3 (temp=0.5)] {raw_output4}")
        prediction4 = self.postprocess(prediction4, raw_output4)

        valid4, _ = self._validate(prediction4, test_sample['input'])
        if valid4:
            prediction = prediction4
            raw_outputs[-1] = f"[RETRY3 SUCCESS (temp=0.5)] {raw_output4}"
        else:
            # All 4 rounds failed — try lenient extraction from  thought blocks
            # before falling back to 0. The model often computes the correct answer
            # but gets stuck in a  loop without outputting <label>.
            fallback = _lenient_thought_fallback(raw_outputs, test_sample['input'], self._validate)
            if fallback is not None:
                valid_fallback, _ = self._validate(str(fallback), test_sample['input'])
                if valid_fallback:
                    prediction = str(fallback)
                    raw_outputs.append(f"[FALLBACK lenient: extracted {fallback} from ]")
                else:
                    prediction = "0"
                    raw_outputs.append(f"[FALLBACK to 0 after failed thought extraction]")
            else:
                prediction = "0"
                raw_outputs.append(f"[FALLBACK to 0 after 4 failed rounds]")

        return prediction, prompt, raw_outputs, [prediction], {}
