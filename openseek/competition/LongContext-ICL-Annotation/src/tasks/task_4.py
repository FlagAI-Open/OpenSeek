"""Task 4: String concatenation — feature-based retrieval, most relevant closest to question."""
import sys
import os
import re
import string
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from tasks.base import BaseTask

# Common English words that cause interference in concatenation tasks
COMMON_WORDS = {
    'the', 'and', 'for', 'his', 'her', 'was', 'but', 'not', 'are', 'all',
    'with', 'that', 'this', 'from', 'he', 'we', 'do', 'so', 'my', 'me',
    'be', 'if', 'or', 'no', 'am', 'it', 'on', 'at', 'by', 'in', 'to', 'of',
    'an', 'is', 'as', 'had', 'have', 'has', 'been', 'being', 'were', 'would',
    'could', 'should', 'will', 'shall', 'can', 'may', 'might', 'must',
    'they', 'them', 'their', 'our', 'your', 'its', 'who', 'what', 'which',
    'when', 'where', 'why', 'how', 'one', 'two', 'three', 'said', 'each',
}


def _output_has_common_words(output_str):
    """Check if an output string contains common English words.

    A 'clean' output has no recognizable English words — only concatenated
    fragments that don't form common words.
    """
    if not output_str:
        return False
    # Split by non-alpha to get individual tokens
    tokens = re.findall(r'[a-zA-Z]+', output_str)
    for token in tokens:
        if token.lower() in COMMON_WORDS:
            return True
    return False

PROMPT_TEMPLATE = (
    "You are a precise text processor. Concatenate all strings in order.\n\n"
    "Rules:\n"
    "- Join all strings directly without any separator\n"
    "- Preserve EXACT case (uppercase/lowercase)\n"
    "- Preserve ALL characters including punctuation\n\n"
    "Examples:\n"
    "{examples_str}\n"
    "Input: {input_text}\n\n"
    "CRITICAL: Your response MUST contain the answer in this exact format:\n"
    "<label>exact_concatenated_string</label>\n\n"
    "Output ONLY: <label>result</label>\n\n"
    "Answer: "
)

RETRY_PROMPT = (
    "\n\n{feedback}\n\n"
    "Output ONLY: <label>the concatenated string</label>\n\n"
)


def _parse_str_list(text):
    """Extract list of strings from input like \"['a', 'b', 'c']\"."""
    import ast
    try:
        lst = ast.literal_eval(text)
        if isinstance(lst, list):
            return lst
    except:
        pass
    # Fallback: regex extraction
    parts = re.findall(r"'([^']*)'|\"([^\"]*)\"", text)
    return [p[0] or p[1] for p in parts]


def _extract_features(parts):
    """Extract observable input features for similarity matching.

    Never computes the full concatenated string — only derives statistics
    from individual parts.
    """
    total_chars = sum(len(p) for p in parts)
    has_punct = False
    has_upper = False
    has_lower = False
    has_digit = False
    has_special = False
    has_quotes = False
    has_brackets = False

    for p in parts:
        for c in p:
            if c in string.punctuation:
                has_punct = True
            if c.isupper():
                has_upper = True
            if c.islower():
                has_lower = True
            if c.isdigit():
                has_digit = True
            if c not in string.ascii_letters and c not in string.digits and c not in string.punctuation and c != ' ':
                has_special = True
            if c == "'" or c == '"':
                has_quotes = True
            if c in '[{':
                has_brackets = True
        if has_punct and has_upper and has_lower and has_digit and has_special and has_quotes and has_brackets:
            break

    return {
        'length': len(parts),
        'total_chars': total_chars,
        'has_punct': has_punct,
        'has_upper': has_upper,
        'has_lower': has_lower,
        'has_digit': has_digit,
        'has_special': has_special,
        'avg_len': total_chars / len(parts) if parts else 0,
        'has_quotes': has_quotes,
        'has_brackets': has_brackets,
    }


def _similarity(tf, ef):
    """Compute feature-based similarity between test and example features.

    All features are observable from input — no answer leakage.
    """
    max_len = max(tf['length'], ef['length'], 1)
    len_sim = 1.0 - abs(tf['length'] - ef['length']) / max_len

    max_chars = max(tf['total_chars'], ef['total_chars'], 1)
    chars_sim = 1.0 - abs(tf['total_chars'] - ef['total_chars']) / max_chars

    max_avg = max(tf['avg_len'], ef['avg_len'], 1)
    avg_sim = 1.0 - abs(tf['avg_len'] - ef['avg_len']) / max_avg

    punct_match = 1.0 if tf['has_punct'] == ef['has_punct'] else 0.0
    upper_match = 1.0 if tf['has_upper'] == ef['has_upper'] else 0.0
    lower_match = 1.0 if tf['has_lower'] == ef['has_lower'] else 0.0
    digit_match = 1.0 if tf['has_digit'] == ef['has_digit'] else 0.0
    special_match = 1.0 if tf['has_special'] == ef['has_special'] else 0.0
    quotes_match = 1.0 if tf['has_quotes'] == ef['has_quotes'] else 0.0
    brackets_match = 1.0 if tf['has_brackets'] == ef['has_brackets'] else 0.0

    return (0.25 * len_sim + 0.20 * chars_sim + 0.15 * avg_sim +
            0.10 * punct_match + 0.05 * upper_match + 0.05 * lower_match +
            0.05 * digit_match + 0.05 * special_match +
            0.05 * quotes_match + 0.05 * brackets_match)


def _validate(prediction, input_text):
    """Validate prediction against input-derived constraints.

    Only checks non-revealing properties (empty + length).
    Does NOT compute the concatenated string.
    """
    if not prediction or not prediction.strip():
        return False, "Your output was empty. Please provide the concatenated string."

    pred = prediction.strip()

    # Parse input parts to get expected length (without computing the answer)
    parts = _parse_str_list(input_text)
    if not parts:
        return True, ""

    expected_len = sum(len(p) for p in parts)
    if len(pred) != expected_len:
        breakdown = []
        for i, p in enumerate(parts):
            breakdown.append(f"  {i+1}. '{p}' -> {len(p)} chars")
        parts_detail = "\n".join(breakdown)
        return False, (
            f"Your answer has {len(pred)} characters, but should have {expected_len} characters.\n"
            f"Here is each input part:\n{parts_detail}\n"
            f"Total: {expected_len} characters.\n"
            f"Concatenate ALL parts in order without skipping any."
        )

    # Character composition check: pred must have same character counts as input
    input_chars = Counter(''.join(parts))
    pred_chars = Counter(pred)
    if input_chars != pred_chars:
        missing = input_chars - pred_chars
        extra = pred_chars - input_chars
        feedback_parts = []
        for ch, cnt in sorted(missing.items()):
            if cnt > 0:
                feedback_parts.append(f"'{ch}' x{cnt} missing")
        for ch, cnt in sorted(extra.items()):
            if cnt > 0:
                feedback_parts.append(f"'{ch}' x{cnt} extra")
        return False, (
            f"Your answer has wrong character composition: {', '.join(feedback_parts)}.\n"
            f"Re-check each character carefully and re-concatenate."
        )

    return True, ""


class Task4(BaseTask):
    task_id = 4
    DATA_FILE = '../data/openseek-4_conala_concat_strings.json'
    PROMPT_TEMPLATE = PROMPT_TEMPLATE

    DEFAULT_CFG = {
        "name": "conala_concat_strings",
        "temperature": 0.1,
        "num_votes": 1,
        "max_tokens": 8000,
        "stop_tokens": None,
        "system_prompt": "You are a precise text processor.",
        "min_icl_tokens": 30_000,
    }

    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self.cfg = None

    def postprocess(self, prediction: str, raw_output: str = "") -> str:
        if not prediction:
            return ""
        pred = prediction.strip()
        # Remove accidental spaces the model may have inserted during concatenation
        pred = pred.replace(' ', '')
        return pred

    def split_icl_padding(self, all_examples):
        return all_examples, []

    def should_retry(self):
        """Use custom validation + retry in run_inference."""
        return False

    def prepare(self):
        # Precompute features for all examples
        for ex in self.icl_examples:
            parts = _parse_str_list(ex['input'])
            ex['_t4_features'] = _extract_features(parts)
        return self

    def _build_clean_examples_str(self, test_parts, token_budget, n_clean_head=15, skip_head=0):
        """Build 30K examples with clean examples as head (closest to question).

        Structure: [general examples...] + [clean examples (most relevant)] + question
        Clean examples are placed at the END, closest to the question.

        skip_head: number of most-relevant clean examples to remove from the head.
        """
        # Separate clean vs general examples
        clean_examples = []
        general_examples = []
        for ex in self.icl_examples:
            output_val = ex['output']
            output_str = output_val[0] if isinstance(output_val, list) else output_val
            if not _output_has_common_words(output_str):
                clean_examples.append(ex)
            else:
                general_examples.append(ex)

        # Score by feature similarity
        tf = _extract_features(test_parts)

        scored_clean = []
        for ex in clean_examples:
            ef = ex.get('_t4_features')
            if ef is None:
                ef = _extract_features(_parse_str_list(ex['input']))
            scored_clean.append((_similarity(tf, ef), ex))
        scored_clean.sort(key=lambda x: x[0], reverse=True)

        scored_general = []
        for ex in general_examples:
            ef = ex.get('_t4_features')
            if ef is None:
                ef = _extract_features(_parse_str_list(ex['input']))
            scored_general.append((_similarity(tf, ef), ex))
        scored_general.sort(key=lambda x: x[0], reverse=True)

        # Step 1: Pick top n_clean_head clean examples, skip the top skip_head most relevant
        clean_head = scored_clean[skip_head : skip_head + n_clean_head]
        clean_parts = []
        clean_tokens = 0
        for sim, ex in clean_head:
            label_str = ex['output'][0] if isinstance(ex['output'], list) else ex['output']
            part = f"Input:\n{ex['input']}\nOutput:\n<label>{label_str}</label>\n\n"
            part_tokens = len(self.tokenizer.encode(part, add_special_tokens=False))
            clean_parts.append(part)
            clean_tokens += part_tokens

        # Step 2: Fill remaining budget with general examples (farthest from question)
        remaining_budget = token_budget - clean_tokens
        general_parts = []
        general_tokens = 0
        for sim, ex in scored_general:
            label_str = ex['output'][0] if isinstance(ex['output'], list) else ex['output']
            part = f"Input:\n{ex['input']}\nOutput:\n<label>{label_str}</label>\n\n"
            part_tokens = len(self.tokenizer.encode(part, add_special_tokens=False))
            general_parts.append(part)
            general_tokens += part_tokens
            if general_tokens > remaining_budget:
                break

        # Assemble: general first (far), clean last (near question)
        # Clean examples in reverse order so most relevant is closest
        examples_str = "".join(general_parts) + "".join(reversed(clean_parts))
        total_tokens = general_tokens + clean_tokens

        return examples_str, total_tokens

    def _score_all(self, test_parts):
        """Score all ICL examples against test input features."""
        tf = _extract_features(test_parts)
        scored = []
        for ex in self.icl_examples:
            ef = ex.get('_t4_features')
            if ef is None:
                ef = _extract_features(_parse_str_list(ex['input']))
            scored.append((_similarity(tf, ef), ex))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored

    def process_sample(self, test_sample, first_sample=False):
        text = test_sample['input']
        test_parts = _parse_str_list(text)
        scored = self._score_all(test_parts)

        # Build examples string: most relevant closest to question
        examples_str = ""
        token_count = 0
        for sim, ex in scored:
            label_str = ex['output'][0] if isinstance(ex['output'], list) else ex['output']
            part = f"Input:\n{ex['input']}\nOutput:\n<label>{label_str}</label>\n\n"
            part_tokens = len(self.tokenizer.encode(part, add_special_tokens=False))

            # Prepend: most similar closest to question
            examples_str = part + examples_str
            token_count += part_tokens
            if token_count > self.min_icl_tokens:
                break

        prompt = self.PROMPT_TEMPLATE.format(
            examples_str=examples_str,
            input_text=text,
        )
        return prompt, examples_str, token_count, {}

    def run_inference(self, test_sample, call_model):
        """Round 1 → validate → retry with feedback → use last attempt on total failure."""
        prompt, _, _, _ = self.process_sample(test_sample)
        text = test_sample['input']
        test_parts = _parse_str_list(text)

        prediction, raw_output = call_model(prompt, self.cfg)
        raw_outputs = [raw_output]
        prediction = self.postprocess(prediction, raw_output)

        valid, feedback = _validate(prediction, test_sample['input'])
        if valid:
            return prediction, prompt, raw_outputs, [prediction], {}

        # Round 2: retry with clean examples (no common words) closest to question
        print(f"  ⚠️ [{feedback}] for sample {test_sample['id']}, retrying with clean examples...")
        clean_examples_str, _ = self._build_clean_examples_str(test_parts, self.min_icl_tokens)
        retry_prompt = (
            "You are a precise text processor. Concatenate all strings in order.\n\n"
            "Rules:\n"
            "- Join all strings directly without any separator\n"
            "- Preserve EXACT case (uppercase/lowercase)\n"
            "- Preserve ALL characters including punctuation\n\n"
            "Examples:\n"
            f"{clean_examples_str}\n"
            f"Task: Concatenate all strings in this list in order, directly joined with no separators:\n"
            f"{text}\n\n"
            "CRITICAL: Your response MUST contain the answer in this exact format:\n"
            "<label>exact_concatenated_string</label>\n\n"
            "Output ONLY: <label>the_concatenated_string</label>\n\n"
            "Answer: "
        )
        prediction2, raw_output2 = call_model(retry_prompt, self.cfg)
        raw_outputs.append(f"[RETRY with feedback] {raw_output2}")
        prediction2 = self.postprocess(prediction2, raw_output2)

        valid2, feedback2 = _validate(prediction2, test_sample['input'])
        if valid2:
            prediction = prediction2
            raw_outputs[-1] = f"[RETRY SUCCESS] {raw_output2}"
            return prediction, prompt, raw_outputs, [prediction], {}

        # Round 3: second retry with clean examples + feedback, higher temperature
        print(f"  ⚠️ [{feedback2}] still invalid, second retry with clean examples (temp=0.3)...")
        retry_prompt2 = (
            retry_prompt
            + f"\n\nYour previous attempt was incorrect. {feedback2}\n\n"
            + "Output ONLY: <label>the_concatenated_string</label>\n\n"
            + "Answer: "
        )
        high_temp_cfg = dict(self.cfg)
        high_temp_cfg['temperature'] = 0.3
        prediction3, raw_output3 = call_model(retry_prompt2, high_temp_cfg)
        raw_outputs.append(f"[RETRY2 with feedback] {raw_output3}")
        prediction3 = self.postprocess(prediction3, raw_output3)

        valid3, _ = _validate(prediction3, test_sample['input'])
        if valid3:
            prediction = prediction3
            raw_outputs[-1] = f"[RETRY2 SUCCESS] {raw_output3}"
        else:
            # Round 4: remove top 5 head examples (closest to question) and retry
            print(f"  ⚠️ R3 failed, retrying R4 without top 5 head examples...")
            clean_examples_str_r4, _ = self._build_clean_examples_str(
                test_parts, self.min_icl_tokens, n_clean_head=12, skip_head=3
            )
            retry_prompt_r4 = (
                "You are a precise text processor. Concatenate all strings in order.\n\n"
                "Rules:\n"
                "- Join all strings directly without any separator\n"
                "- Preserve EXACT case (uppercase/lowercase)\n"
                "- Preserve ALL characters including punctuation\n\n"
                "Examples:\n"
                f"{clean_examples_str_r4}\n"
                f"Task: Concatenate all strings in this list in order, directly joined with no separators:\n"
                f"{text}\n\n"
                f"Your previous attempt was incorrect. {feedback2}\n\n"
                "Output ONLY: <label>the_concatenated_string</label>\n\n"
                "Answer: "
            )
            prediction4, raw_output4 = call_model(retry_prompt_r4, high_temp_cfg)
            raw_outputs.append(f"[RETRY3 R4-no-head5] {raw_output4}")
            prediction4 = self.postprocess(prediction4, raw_output4)

            valid4, _ = _validate(prediction4, test_sample['input'])
            if valid4:
                prediction = prediction4
                raw_outputs[-1] = f"[RETRY3 SUCCESS] {raw_output4}"
            else:
                # Round 5: higher temp + swap top 3 clean examples
                print(f"  ⚠️ R4 failed, retrying R5 (temp=0.5, swap top 3 clean examples)...")
                clean_examples_str_r5, _ = self._build_clean_examples_str(
                    test_parts, self.min_icl_tokens, n_clean_head=12, skip_head=5
                )
                retry_prompt_r5 = (
                    "You are a precise text processor. Concatenate all strings in order.\n\n"
                    "Rules:\n"
                    "- Join all strings directly without any separator\n"
                    "- Preserve EXACT case (uppercase/lowercase)\n"
                    "- Preserve ALL characters including punctuation\n\n"
                    "Examples:\n"
                    f"{clean_examples_str_r5}\n"
                    f"Task: Concatenate all strings in this list in order, directly joined with no separators:\n"
                    f"{text}\n\n"
                    f"Your previous attempt was incorrect. {feedback2}\n\n"
                    "Output ONLY: <label>the_concatenated_string</label>\n\n"
                    "Answer: "
                )
                r5_cfg = dict(high_temp_cfg)
                r5_cfg['temperature'] = 0.5
                prediction5, raw_output5 = call_model(retry_prompt_r5, r5_cfg)
                raw_outputs.append(f"[RETRY4 R5-temp0.5-swap3] {raw_output5}")
                prediction5 = self.postprocess(prediction5, raw_output5)

                valid5, _ = _validate(prediction5, test_sample['input'])
                if valid5:
                    prediction = prediction5
                    raw_outputs[-1] = f"[RETRY4 SUCCESS] {raw_output5}"
                else:
                    prediction = prediction5
                    raw_outputs.append(f"[RETRY4 FAILED, using last attempt: {prediction5[:50]}]")

        return prediction, prompt, raw_outputs, [prediction], {}
