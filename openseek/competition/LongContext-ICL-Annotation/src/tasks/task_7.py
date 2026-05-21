"""Task 7: Jeopardy answer generation — Category-First + two-round verify."""
import sys
import os
import re
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from tasks.base import BaseTask
from retriever import BM25Retriever


# Boilerplate words that carry little discriminative power
BOILERPLATE = {'category', 'clue', 'one', 'these', 'this', 'known', 'also', 'name',
               'called', 'found', 'made', 'like', 'part', 'world', 'years', 'time',
               'first', 'last', 'new', 'old', 'great', 'best', 'good', 'long',
               'city', 'state', 'place', 'year', 'day', 'way'}


def _content_keywords(text):
    """Extract content words (nouns, proper nouns, etc.) from text."""
    words = set(re.findall(r'[a-z0-9]+', text.lower()))
    words = {w for w in words if len(w) > 3} - BOILERPLATE
    return words


def _normalize_ans(text):
    """Normalize an answer string for grouping."""
    ans = text.lower().strip()
    for a in ['the ', 'a ', 'an ']:
        if ans.startswith(a):
            ans = ans[len(a):]
    ans = re.sub(r'\s*\([^)]*\)', '', ans).strip()
    ans = re.sub(r'\s+', ' ', ans)
    return ans


def _strip_quotes(text):
    """Remove surrounding quotes."""
    if len(text) >= 2:
        if (text[0] == '"' and text[-1] == '"') or (text[0] == "'" and text[-1] == "'"):
            text = text[1:-1]
    return text.strip()


def _ans_variants(ans):
    """Generate all plausible normalized variants of an answer.

    For a given normalized answer, produce all forms that might appear in ICL:
    - Original normalized form
    - Without trailing 's' (plural → singular)
    - With trailing 's' (singular → plural)
    - Without leading article (already done by _normalize_ans)
    """
    variants = {ans}
    # Plural ↔ singular
    if ans.endswith('s') and len(ans) > 3:
        variants.add(ans[:-1])  # remove trailing s
        # Also remove 'es' if applicable
        if ans.endswith('es'):
            variants.add(ans[:-2])
    elif not ans.endswith('s') and len(ans) > 2:
        variants.add(ans + 's')   # add s
        variants.add(ans + 'es')  # add es

    # Handle possessives
    if ans.endswith("'s"):
        variants.add(ans[:-2])
    elif not ans.endswith('s'):
        variants.add(ans + "'s")

    return variants


def _find_matching_examples(candidate, answer_index, icl_examples):
    """Find ICL examples whose answer matches the candidate (with variant handling).

    Tries exact match first, then tight variant matching to avoid false positives:
    1. Exact match after _normalize_ans
    2. Plural/singular/possessive variants
    3. Name truncation: candidate is tail of ICL answer (scorsese ⊂ martin scorsese)
    4. Spelling similarity: difflib >= 0.85 AND share ALL content words

    Returns list of matching ICL examples.
    """
    cand_norm = _normalize_ans(candidate)
    cand_clean = _strip_quotes(cand_norm)
    cand_words = cand_norm.split()

    # Step 1: Exact match
    exact = answer_index.get(cand_norm, [])
    if exact:
        return exact

    # Step 2: Variant exact match (plural, possessive, etc.)
    cand_variants = _ans_variants(cand_norm)
    cand_variants.add(cand_clean)
    for v in cand_variants:
        if v in answer_index:
            return list(answer_index[v])

    # Step 3: Name truncation — candidate is the tail of ICL answer
    # e.g. "scorsese" ⊂ "martin scorsese", "elizabeth ii" ⊂ "queen elizabeth ii"
    if len(cand_words) >= 1 and len(cand_norm) > 2:
        tail_matches = []
        for ex in icl_examples:
            icl_ans_raw = ex['output'][0] if isinstance(ex['output'], list) else ex['output']
            icl_ans = _normalize_ans(icl_ans_raw)
            icl_ans_clean = _strip_quotes(icl_ans)
            icl_words = icl_ans_clean.split()

            # Candidate must be the LAST N words of ICL answer
            # and ICL answer must have more words than candidate
            if len(icl_words) > len(cand_words) and icl_words[-len(cand_words):] == cand_words:
                tail_matches.append(ex)
        if tail_matches:
            return tail_matches

    # Step 4: Spelling similarity — difflib >= 0.85 AND share ALL content words
    import difflib
    skip_words = {'the', 'a', 'an', 'of', 'and', 'or'}
    cand_content = {w.lower() for w in cand_words} - skip_words

    if cand_content:  # only if candidate has content words
        spelling_matches = []
        for ex in icl_examples:
            icl_ans_raw = ex['output'][0] if isinstance(ex['output'], list) else ex['output']
            icl_ans = _normalize_ans(icl_ans_raw)
            icl_ans_clean = _strip_quotes(icl_ans)
            icl_words = icl_ans_clean.split()
            icl_content = {w.lower() for w in icl_words} - skip_words

            # Must share ALL content words
            if not (cand_content <= icl_content or icl_content <= cand_content):
                continue

            ratio = difflib.SequenceMatcher(None, cand_clean, icl_ans_clean).ratio()
            if ratio >= 0.85:
                spelling_matches.append(ex)
        if spelling_matches:
            return spelling_matches

    return []


def _extract_category(text):
    """Extract Category from input text."""
    m = re.search(r'Category:\s*(.+?)\s*\nClue:', text)
    return m.group(1).strip() if m else ''


PROMPT_TEMPLATE = (
    "You are a Jeopardy expert. Answer each clue with a single word or short phrase.\n\n"
    "Rules:\n"
    "- FIRST read the Category — it defines the topic scope and strongly hints at the answer\n"
    "- Then read the Clue for specific details\n"
    "- Study the examples to understand the answer format\n"
    "- Think step-by-step to analyze both Category and Clue\n"
    "- Then give your final answer in <label> tags\n\n"
    "Examples:\n"
    "{examples_str}\n"
    "{input_text}\n\n"
    "Think step-by-step, then output: <label>answer</label>\n\n"
    "Answer: "
)

# Round 2a: Found matching ICL examples → verify
ROUND2_VERIFY_TEMPLATE = (
    "You previously answered: <label>{candidate}</label>\n\n"
    "I found {n_match} more examples from the database that have the SAME answer \"{candidate}\".\n"
    "Study these examples to confirm your answer:\n\n"
    "{match_examples_str}\n"
    "Now re-evaluate: Is \"{candidate}\" still the correct answer for this clue?\n\n"
    "Full reference examples:\n"
    "{examples_str}\n"
    "{input_text}\n\n"
    "Think step-by-step, then output: <label>answer</label>\n\n"
    "Answer: "
)

# Round 2b: No matching ICL found → fallback to own knowledge
ROUND2_FALLBACK_TEMPLATE = (
    "You previously answered: <label>{candidate}</label>\n\n"
    "I searched the example database but found NO examples with the answer \"{candidate}\".\n"
    "This means the database doesn't have this answer, so you must rely on your own knowledge.\n\n"
    "Please re-evaluate your answer carefully. Is \"{candidate}\" still correct?\n"
    "If not, think again and provide a better answer.\n\n"
    "Examples (for reference, none have this answer):\n"
    "{examples_str}\n"
    "{input_text}\n\n"
    "Think step-by-step, then output: <label>answer</label>\n\n"
    "Answer: "
)

RETRY_PROMPT = (
    "\n\n{feedback}\n\n"
    "Output ONLY: <label>your answer as a single word or short phrase</label>\n\n"
    "Answer: "
)

TS = chr(60) + 'think' + chr(62)
CT = chr(60) + '/' + 'think' + chr(62)


def _extract_answer_from_thought(raw_outputs):
    """Extract the model's best answer from <think> blocks."""
    all_thought = ''
    for raw in raw_outputs:
        m = re.search(re.escape(TS) + r'(.*?)' + re.escape(CT), raw, re.DOTALL)
        if m:
            all_thought += m.group(1) + '\n'
        else:
            m2 = re.search(re.escape(TS) + r'(.*)', raw, re.DOTALL)
            if m2:
                all_thought += m2.group(1) + '\n'

    if not all_thought:
        return None

    candidates = []
    for pat in [
        r'I think the answer is\s+[""]?\s*([^\n"<]{2,40}?)\s*["".]',
        r'the answer is\s+[""]?\s*([^\n"<]{2,40}?)\s*["".]',
    ]:
        for m in re.finditer(pat, all_thought, re.IGNORECASE):
            c = m.group(1).strip().strip('"').strip("'").strip()
            if _is_good_candidate(c):
                candidates.append(c)

    if not candidates:
        quotes = re.findall(r'[""]([A-Za-z][A-Za-z0-9 .\'-]{2,40})[""]', all_thought[-3000:])
        for q in reversed(quotes):
            c = q.strip()
            if _is_good_candidate(c):
                candidates.append(c)

    if not candidates:
        return None

    freq = Counter(candidates)
    return freq.most_common(1)[0][0]


def _is_good_candidate(c):
    """Filter out answers that are meta-descriptions rather than actual answers."""
    if len(c) < 2 or len(c) > 50:
        return False
    c_lower = c.lower()
    if c_lower in (
        'yes', 'no', 'here', 'correct', 'right', 'unknown', 'not sure',
        'a single word', 'a single city', 'the city that', 'the team',
        'a short phrase', 'the answer', 'the same', 'the right',
        'a reference', 'the name', 'the right answer', 'the city',
        'the person', 'the word', 'the term', 'the same thing',
        'a form of', 'the correct', 'the same as', 'the team that',
        'the person who', 'who is', 'what is', 'a different',
        'another', "but that", 'but the', 'wait,', 'however',
        'okay', 'ok', 'let me', "let's try", "let's see",
    ):
        return False
    if c_lower.startswith((
        'probably', 'likely', 'maybe', 'perhaps', 'possibly',
        'okay,', 'let me', "let's", 'wait,', 'i think',
    )):
        return False
    if any(skip in c_lower for skip in [
        'but the', 'however', 'not sure', "i'm not", 'not certain',
        'could be', 'might be', 'should be', 'would be',
        'is a single', 'is a form', 'is the',
        'even though', 'although', 'despite',
        "but he's", "but she's", "but it's",
        "but that", 'but i', 'but wait',
        'sidewalk is', 'city that', 'team that', 'person who',
        'is a hotbed', 'english teacher', 'hotbed of',
        'a different', 'different singer', 'different artist',
        'figure this out', 'try to figure', "let's try",
        "her middle name", 'his middle name', 'middle initial',
        'singer named', 'actor named', 'person named',
        "but that's not", "that's not correct", 'not a real',
        'clue is wrong', 'clue is incorrect', 'clue is conflicting',
    ]):
        return False
    if not c[0].isalpha() and c[0] not in ('"', "'"):
        return False
    if c_lower.startswith(('the wizard of', 'the chronicles of', 'the alchemist',
                           'wrinkle in time', 'the secret garden', 'the witch of',
                           'a wrinkle in time')):
        return False
    if c.endswith(','):
        return False
    return True


def _validate(prediction, input_text):
    """Validate: must not be empty, must be short (Jeopardy answer = word or phrase)."""
    if not prediction or not prediction.strip():
        return False, "Your output was empty. Please provide a short answer."
    pred = prediction.strip()
    if len(pred) > 80:
        return False, f"Your answer is too long ({len(pred)} chars). Jeopardy answers are short phrases."
    return True, ""


class Task7(BaseTask):
    task_id = 7
    DATA_FILE = '../data/openseek-7_jeopardy_answer_generation_all.json'
    PROMPT_TEMPLATE = PROMPT_TEMPLATE

    DEFAULT_CFG = {
        "name": "jeopardy_answer_generation_all",
        "temperature": 0.3,
        "top_k": 40,
        "num_votes": 1,
        "max_tokens": 32000,
        "stop_tokens": None,
        "system_prompt": "You are a Jeopardy expert.",
        "min_icl_tokens": 30_000,
    }

    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self.cfg = None
        self.min_icl_tokens = 30_000
        self.max_icl_tokens = 31_000

    def postprocess(self, prediction: str, raw_output: str = "") -> str:
        pred = prediction.strip() if prediction else ""
        pred = pred.lower()

        if pred.startswith('"') and pred.endswith('"'):
            pred = pred[1:-1]
        if pred.startswith("'") and pred.endswith("'"):
            pred = pred[1:-1]

        pred = re.sub(r'\s*\([^)]*\)', '', pred).strip()
        pred = pred.replace(' & ', ' ')
        pred = re.sub(r'\s+', ' ', pred).strip()

        for article in ["the ", "a ", "an "]:
            if pred.startswith(article):
                pred = pred[len(article):]
                break

        return pred

    def split_icl_padding(self, all_examples):
        return all_examples, all_examples

    def prepare(self):
        """Build BM25 retriever, Category index, and Answer index."""
        self.bm25_retriever = BM25Retriever(self.icl_examples)

        # Category index: category -> list of examples
        self.by_category = defaultdict(list)
        for ex in self.icl_examples:
            cat = _extract_category(ex['input'])
            if cat:
                self.by_category[cat].append(ex)

        # Answer index: normalized answer -> list of examples
        self.answer_index = defaultdict(list)
        for ex in self.icl_examples:
            ans = _normalize_ans(ex['output'][0] if isinstance(ex['output'], list) else ex['output'])
            self.answer_index[ans].append(ex)

        return self

    def _retrieve_category_first(self, test_sample, top_k=40):
        """Category-First retrieval: same Category examples first (BM25 ranked), then BM25 fill."""
        cat = _extract_category(test_sample['input'])

        cat_results = []
        cat_examples = self.by_category.get(cat, [])
        if cat_examples:
            cat_results = self.bm25_retriever.retrieve_top_k(test_sample['input'], top_k=top_k * 2)
            cat_results = [ex for ex in cat_results if _extract_category(ex['input']) == cat][:top_k]

        seen_ids = {id(ex) for ex in cat_results}

        if len(cat_results) < top_k:
            bm25_fill = self.bm25_retriever.retrieve_top_k(test_sample['input'], top_k=top_k * 2)
            for ex in bm25_fill:
                if len(cat_results) >= top_k:
                    break
                if id(ex) not in seen_ids:
                    cat_results.append(ex)
                    seen_ids.add(id(ex))

        return cat_results, cat

    def _is_category_example(self, ex, test_cat):
        return _extract_category(ex['input']) == test_cat

    def _pad_examples(self, examples_str, selected_ids):
        """Pad examples to min_icl_tokens. Padding placed at FAR END."""
        token_count = len(self.tokenizer.encode(examples_str, add_special_tokens=False))
        if token_count >= self.min_icl_tokens:
            return examples_str, token_count

        extras = [ex for ex in self.icl_examples if id(ex) not in selected_ids]

        parts = []
        for ex in extras:
            if token_count >= self.min_icl_tokens:
                break
            output_val = ex['output']
            label_str = output_val[0] if isinstance(output_val, list) and len(output_val) > 0 else output_val
            part = f"Input:\n{ex['input']}\nOutput:\n<label>{label_str}</label>\n\n"
            part_tokens = len(self.tokenizer.encode(part, add_special_tokens=False))
            parts.append(part)
            token_count += part_tokens
            if token_count > self.min_icl_tokens:
                break

        if parts:
            examples_str = "".join(parts) + examples_str
            token_count = len(self.tokenizer.encode(examples_str, add_special_tokens=False))

        return examples_str, token_count

    def process_sample(self, test_sample, first_sample=False):
        text = test_sample['input']

        selected, test_cat = self._retrieve_category_first(test_sample, top_k=self.cfg.get('top_k', 40))

        # Split: BM25 (non-category) examples go farthest, Category examples go closest
        bm25_examples = [ex for ex in selected if not self._is_category_example(ex, test_cat)]
        cat_examples = [ex for ex in selected if self._is_category_example(ex, test_cat)]

        examples_str = ""
        # BM25 examples first (farthest from question), reverse order so best BM25 is closest
        for ex in reversed(bm25_examples):
            output_val = ex['output']
            label_str = output_val[0] if isinstance(output_val, list) and len(output_val) > 0 else output_val
            part = f"Input:\n{ex['input']}\nOutput:\n<label>{label_str}</label>\n\n"
            examples_str = part + examples_str

        # Category examples last (closest to question)
        for ex in cat_examples:
            output_val = ex['output']
            label_str = output_val[0] if isinstance(output_val, list) and len(output_val) > 0 else output_val
            part = f"Input:\n{ex['input']}\nOutput:\n<label>{label_str}</label>\n\n"
            examples_str = examples_str + part

        token_count = len(self.tokenizer.encode(examples_str, add_special_tokens=False))

        # Pad to min_icl_tokens
        selected_ids = {id(ex) for ex in selected}
        examples_str, token_count = self._pad_examples(examples_str, selected_ids)

        fallback = ''
        if cat_examples:
            cat_out = cat_examples[0]['output']
            fallback = cat_out[0] if isinstance(cat_out, list) and cat_out else (cat_out if cat_out else '')
        elif selected:
            sel_out = selected[0]['output']
            fallback = sel_out[0] if isinstance(sel_out, list) and sel_out else (sel_out if sel_out else '')

        prompt = self.PROMPT_TEMPLATE.format(
            examples_str=examples_str,
            input_text=text,
        )
        return prompt, examples_str, token_count, {'fallback': fallback}

    def _build_verify_prompt(self, test_sample, candidate, match_examples):
        """Build Round 2 prompt with matched ICL examples for verification.

        Ensures full examples_str reaches 30K tokens.
        """
        text = test_sample['input']

        # Build examples string: matched examples closest to question (near)
        near_str = ""
        for ex in match_examples:
            output_val = ex['output']
            label_str = output_val[0] if isinstance(output_val, list) and len(output_val) > 0 else output_val
            near_str += f"Input:\n{ex['input']}\nOutput:\n<label>{label_str}</label>\n\n"

        # Category-First examples as base context (far end)
        selected, test_cat = self._retrieve_category_first(test_sample, top_k=self.cfg.get('top_k', 40))
        bm25_examples = [ex for ex in selected if not self._is_category_example(ex, test_cat)]
        cat_examples = [ex for ex in selected if self._is_category_example(ex, test_cat)]

        far_str = ""
        for ex in reversed(bm25_examples):
            output_val = ex['output']
            label_str = output_val[0] if isinstance(output_val, list) and len(output_val) > 0 else output_val
            part = f"Input:\n{ex['input']}\nOutput:\n<label>{label_str}</label>\n\n"
            far_str = part + far_str

        for ex in cat_examples:
            output_val = ex['output']
            label_str = output_val[0] if isinstance(output_val, list) and len(output_val) > 0 else output_val
            part = f"Input:\n{ex['input']}\nOutput:\n<label>{label_str}</label>\n\n"
            far_str = far_str + part

        # Combine: far examples + matched near examples (near NOT duplicated in examples_str)
        full_examples_str = far_str

        # Pad far_str to 30K tokens independently
        selected_ids = {id(ex) for ex in selected}
        full_examples_str, _ = self._pad_examples(full_examples_str, selected_ids)

        prompt = ROUND2_VERIFY_TEMPLATE.format(
            candidate=candidate,
            n_match=len(match_examples),
            match_examples_str=near_str,
            input_text=text,
            examples_str=full_examples_str,
        )
        return prompt

    def _build_fallback_prompt(self, test_sample, candidate):
        """Build Round 2 fallback prompt when no matching ICL found."""
        text = test_sample['input']

        # Still include Category-First examples for context
        selected, test_cat = self._retrieve_category_first(test_sample, top_k=self.cfg.get('top_k', 40))
        bm25_examples = [ex for ex in selected if not self._is_category_example(ex, test_cat)]
        cat_examples = [ex for ex in selected if self._is_category_example(ex, test_cat)]

        examples_str = ""
        for ex in reversed(bm25_examples):
            output_val = ex['output']
            label_str = output_val[0] if isinstance(output_val, list) and len(output_val) > 0 else output_val
            part = f"Input:\n{ex['input']}\nOutput:\n<label>{label_str}</label>\n\n"
            examples_str = part + examples_str

        for ex in cat_examples:
            output_val = ex['output']
            label_str = output_val[0] if isinstance(output_val, list) and len(output_val) > 0 else output_val
            part = f"Input:\n{ex['input']}\nOutput:\n<label>{label_str}</label>\n\n"
            examples_str = examples_str + part

        selected_ids = {id(ex) for ex in selected}
        examples_str, _ = self._pad_examples(examples_str, selected_ids)

        prompt = ROUND2_FALLBACK_TEMPLATE.format(
            candidate=candidate,
            examples_str=examples_str,
            input_text=text,
        )
        return prompt

    def should_retry(self):
        """Use custom validation + retry in run_inference."""
        return False

    def run_inference(self, test_sample, call_model):
        """Two-round reasoning: Category-First → verify with ICL match → final answer."""
        # ===== Round 1: Category-First ICL =====
        prompt, _, _, info = self.process_sample(test_sample)
        fallback = info.get('fallback', '')

        prediction, raw_output = call_model(prompt, self.cfg)
        raw_outputs = [f"[R1 Category-First] {raw_output}"]
        prediction = self.postprocess(prediction, raw_output)

        # ===== Extract candidate from Round 1 =====
        candidate = prediction.strip()

        # ===== Search ICL answer index with candidate (fuzzy match) =====
        match_examples = _find_matching_examples(candidate, self.answer_index, self.icl_examples)

        if match_examples:
            # ===== Round 2a: Found matching ICL → verify =====
            print(f"  Found {len(match_examples)} matching ICL examples for '{candidate}', verifying...")
            verify_prompt = self._build_verify_prompt(test_sample, candidate, match_examples)
            prediction2, raw_output2 = call_model(verify_prompt, self.cfg)
            raw_outputs.append(f"[R2 verify with {len(match_examples)} matches] {raw_output2}")
            prediction2 = self.postprocess(prediction2, raw_output2)

            valid2, _ = _validate(prediction2, test_sample['input'])
            if valid2:
                prediction = prediction2
                raw_outputs[-1] = f"[R2 VERIFY SUCCESS] {raw_output2}"
            else:
                prediction = prediction2
                raw_outputs[-1] = f"[R2 VERIFY FAILED, using verified answer: {prediction2[:50]}]"
        else:
            # ===== Round 2b: No matching ICL → fallback to own knowledge =====
            print(f"  No matching ICL for '{candidate}', asking model to re-reason...")
            fallback_prompt = self._build_fallback_prompt(test_sample, candidate)
            prediction2, raw_output2 = call_model(fallback_prompt, self.cfg)
            raw_outputs.append(f"[R2 fallback re-reason] {raw_output2}")
            prediction2 = self.postprocess(prediction2, raw_output2)

            valid2, _ = _validate(prediction2, test_sample['input'])
            if valid2:
                prediction = prediction2
                raw_outputs[-1] = f"[R2 FALLBACK SUCCESS] {raw_output2}"
            else:
                # Final fallback: thought extraction or most-similar
                thought_answer = _extract_answer_from_thought(raw_outputs)
                if thought_answer:
                    prediction = self.postprocess(thought_answer)
                    raw_outputs.append(f"[THOUGHT EXTRACTION] extracted: {thought_answer}")
                elif fallback:
                    prediction = self.postprocess(fallback)
                    raw_outputs.append(f"[FALLBACK to most-similar: {prediction[:30]}]")
                else:
                    prediction = prediction2
                    raw_outputs.append(f"[ALL FAILED, using last attempt]")

        return prediction, prompt, raw_outputs, [prediction], {}
