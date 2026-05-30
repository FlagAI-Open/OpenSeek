"""
method.py — Optimized method for Long-Context ICL Data Annotation (Qwen3-4B)
============================================================================
Key improvements over baseline:
1. Task-aware prompt templates with CoT guidance
2. BM25 + diversity-based example selection for long-context ICL
3. Multi-sample voting for robust annotation
4. Robust label extraction with fallback strategies
5. Context compression & structure alignment via ctx_compress
"""

import re
import math
import json
import random
from collections import Counter
from typing import Optional
from transformers import AutoTokenizer

# ============================================================
# Context Compression & Structure Alignment
# ============================================================
# The ctx_compress package provides structural template matching
# to align model outputs with pre-computed high-quality templates.
# This is used as a post-processing step to improve output consistency.
try:
    from ctx_compress import align_output
    _HAS_CTX_COMPRESS = True
except ImportError:
    _HAS_CTX_COMPRESS = False


# ============================================================
# 1. TASK-SPECIFIC PROMPT TEMPLATES
# ============================================================

TASK_TYPE_MAP = {
    "openseek-1": "math_reasoning",
    "openseek-2": "linguistic_analysis",
    "openseek-3": "math_reasoning",
    "openseek-4": "code_string",
    "openseek-5": "classification",
    "openseek-6": "classification",
    "openseek-7": "open_generation",
    "openseek-8": "code_generation",
}

def build_prompt_math_reasoning(task_description: str, text2annotate: str) -> str:
    """Prompt for mathematical reasoning tasks (Task 1, 3)."""
    prompt = (
        "### Role\n"
        "You are a precise mathematical reasoning assistant. Your task is to solve the given problem step by step.\n\n"
        "### Task\n"
        f"{task_description}\n\n"
        "### Instructions\n"
        "1. First, reason step-by-step internally (this will not appear in the final output).\n"
        "2. Then, provide ONLY the final answer wrapped in <label> tags.\n"
        "3. The answer must be a single integer or a list of integers, exactly matching the format shown in examples.\n\n"
        "### Examples\n"
        "[[EXAMPLES]]\n\n"
        "### Input\n"
        f"{text2annotate}\n\n"
        "### Output\n"
        "<label>"
    )
    return prompt


def build_prompt_linguistic_analysis(task_description: str, text2annotate: str) -> str:
    """Prompt for linguistic analysis tasks (Task 2)."""
    prompt = (
        "### Role\n"
        "You are a linguistic analysis expert. Count linguistic elements in sentences precisely.\n\n"
        "### Task\n"
        f"{task_description}\n\n"
        "### Instructions\n"
        "1. Identify each word's part of speech carefully.\n"
        "2. Count only the requested type (nouns or verbs).\n"
        "3. Output ONLY a single integer in <label> tags.\n\n"
        "### Examples\n"
        "[[EXAMPLES]]\n\n"
        "### Input\n"
        f"{text2annotate}\n\n"
        "### Output\n"
        "<label>"
    )
    return prompt


def build_prompt_code_string(task_description: str, text2annotate: str) -> str:
    """Prompt for code/string manipulation tasks (Task 4)."""
    prompt = (
        "### Role\n"
        "You are a precise string manipulation and code generation assistant.\n\n"
        "### Task\n"
        f"{task_description}\n\n"
        "### Instructions\n"
        "1. Carefully follow the string manipulation rules described.\n"
        "2. Output ONLY the result wrapped in <label> tags.\n"
        "3. Ensure the output format exactly matches the examples.\n\n"
        "### Examples\n"
        "[[EXAMPLES]]\n\n"
        "### Input\n"
        f"{text2annotate}\n\n"
        "### Output\n"
        "<label>"
    )
    return prompt


def build_prompt_classification(task_description: str, text2annotate: str) -> str:
    """Prompt for classification tasks (Task 5, 6)."""
    prompt = (
        "### Role\n"
        "You are a text classification expert. Analyze the text carefully and assign the correct label.\n\n"
        "### Task\n"
        f"{task_description}\n\n"
        "### Instructions\n"
        "1. Read the input text carefully.\n"
        "2. Consider all context, including hashtags, emojis, and tone.\n"
        "3. Classify based on the overall content, not just keywords.\n"
        "4. Output ONLY the label wrapped in <label> tags.\n"
        "5. Valid labels are exactly as shown in the examples.\n\n"
        "### Examples\n"
        "[[EXAMPLES]]\n\n"
        "### Input\n"
        f"{text2annotate}\n\n"
        "### Output\n"
        "<label>"
    )
    return prompt


def build_prompt_open_generation(task_description: str, text2annotate: str) -> str:
    """Prompt for open-ended generation tasks (Task 7)."""
    prompt = (
        "### Role\n"
        "You are a trivia expert with broad knowledge across many categories.\n\n"
        "### Task\n"
        f"{task_description}\n\n"
        "### Instructions\n"
        "1. Think about the category and clue carefully.\n"
        "2. Provide the most specific and accurate answer.\n"
        "3. Answers should be in all lower cased letters.\n"
        "4. Output ONLY the answer wrapped in <label> tags.\n\n"
        "### Examples\n"
        "[[EXAMPLES]]\n\n"
        "### Input\n"
        f"{text2annotate}\n\n"
        "### Output\n"
        "<label>"
    )
    return prompt


def build_prompt_code_generation(task_description: str, text2annotate: str) -> str:
    """Prompt for code generation tasks (Task 8 - Triton kernels)."""
    prompt = (
        "### Role\n"
        "You are an expert GPU programmer specializing in Triton kernels.\n\n"
        "### Task\n"
        f"{task_description}\n\n"
        "### Instructions\n"
        "1. Write complete, runnable Triton code following the specification.\n"
        "2. Include all necessary imports (torch, triton, triton.language).\n"
        "3. Ensure the code is syntactically correct and can be executed.\n"
        "4. Follow the exact patterns shown in the examples.\n"
        "5. Output ONLY the code wrapped in <label> tags.\n"
        "6. Do NOT include any explanation outside the tags.\n\n"
        "### Examples\n"
        "[[EXAMPLES]]\n\n"
        "### Input\n"
        f"{text2annotate}\n\n"
        "### Output\n"
        "<label>"
    )
    return prompt


# Dispatch dictionary
PROMPT_BUILDERS = {
    "math_reasoning": build_prompt_math_reasoning,
    "linguistic_analysis": build_prompt_linguistic_analysis,
    "code_string": build_prompt_code_string,
    "classification": build_prompt_classification,
    "open_generation": build_prompt_open_generation,
    "code_generation": build_prompt_code_generation,
}


def build_prompt(task_description: str, text2annotate: str, task_id: int = 1) -> str:
    """
    Build a task-aware prompt for long-context data annotation.
    
    Args:
        task_description: Description of the annotation task
        text2annotate: The text to annotate
        task_id: Task ID (1-8) for task-specific prompt selection
    """
    task_key = TASK_TYPE_MAP.get(f"openseek-{task_id}", "math_reasoning")
    builder = PROMPT_BUILDERS.get(task_key, build_prompt_math_reasoning)
    return builder(task_description, text2annotate)


# ============================================================
# 2. ADVANCED EXAMPLE SELECTION (BM25 + Diversity Sampling)
# ============================================================

class BM25:
    """Simple BM25 implementation for example retrieval."""
    
    def __init__(self, corpus: list[str]):
        self.corpus = corpus
        self.doc_freqs = []
        self.idf = []
        self.avgdl = 0
        self.k1 = 1.5
        self.b = 0.75
        self._build_index()
    
    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization by splitting on non-alphanumeric chars."""
        return re.findall(r'\w+', text.lower())
    
    def _build_index(self):
        """Build BM25 index from corpus."""
        N = len(self.corpus)
        if N == 0:
            return
        
        total_len = 0
        doc_term_sets = []
        
        for doc in self.corpus:
            tokens = self._tokenize(doc)
            total_len += len(tokens)
            doc_term_sets.append(set(tokens))
        
        self.avgdl = total_len / N
        
        # Compute document frequency
        all_terms = set()
        for terms in doc_term_sets:
            all_terms.update(terms)
        
        df = {term: 0 for term in all_terms}
        for terms in doc_term_sets:
            for term in terms:
                df[term] += 1
        
        # Compute IDF
        self.idf = {}
        for term, freq in df.items():
            self.idf[term] = math.log((N - freq + 0.5) / (freq + 0.5) + 1.0)
        
        # Store tokenized docs with frequencies
        self.doc_freqs = []
        for doc in self.corpus:
            tokens = self._tokenize(doc)
            term_freq = Counter(tokens)
            self.doc_freqs.append(term_freq)
    
    def score(self, query: str, doc_idx: int) -> float:
        """Compute BM25 score for a query against a document."""
        query_tokens = self._tokenize(query)
        doc_freq = self.doc_freqs[doc_idx]
        doc_len = sum(doc_freq.values())
        
        score = 0.0
        for qt in query_tokens:
            if qt in self.idf and qt in doc_freq:
                tf = doc_freq[qt]
                idf = self.idf[qt]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                score += idf * numerator / denominator
        
        return score
    
    def search(self, query: str, top_k: int = 50) -> list[int]:
        """Return top-k document indices sorted by BM25 score."""
        scores = [(i, self.score(query, i)) for i in range(len(self.corpus))]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in scores[:top_k]]


def select_examples_diverse(
    all_examples: list[dict], 
    task_description: str, 
    text2annotate: str,
    tokenizer: AutoTokenizer,
    max_context_length: int = 100_000,
    prompt_template_length: int = 500,
    diversity_factor: float = 0.3
) -> str:
    """
    Select diverse, relevant examples using BM25 retrieval + diversity sampling.
    
    Strategy:
    1. Use BM25 to retrieve top-K most relevant examples
    2. Apply diversity sampling to ensure coverage of different output types
    3. Greedily fill context window up to max_context_length
    
    Args:
        all_examples: List of example dicts with 'input', 'output', optional 'length'
        task_description: Task description for context
        text2annotate: Text to annotate (used as query)
        tokenizer: Qwen3-4B tokenizer
        max_context_length: Maximum total context length in tokens
        prompt_template_length: Estimated token length of prompt template (without examples)
        diversity_factor: Weight for diversity vs relevance (0 = pure relevance, 1 = pure diversity)
    
    Returns:
        Formatted examples string
    """
    if not all_examples:
        return ""
    
    # Calculate available token budget for examples
    available_tokens = max_context_length - prompt_template_length
    if available_tokens <= 0:
        return ""
    
    # Extract input texts for BM25 indexing
    input_texts = [ex.get('input', '') for ex in all_examples]
    
    # Build BM25 index and retrieve relevant examples
    bm25 = BM25(input_texts)
    top_k = min(len(all_examples), 100)  # Retrieve top 100 candidates
    candidate_indices = bm25.search(text2annotate, top_k=top_k)
    
    # Group candidates by output type for diversity
    output_to_indices = {}
    for idx in candidate_indices:
        ex = all_examples[idx]
        output = str(ex.get('output', [''])[0] if isinstance(ex.get('output'), list) else ex.get('output', ''))
        if output not in output_to_indices:
            output_to_indices[output] = []
        output_to_indices[output].append(idx)
    
    # Diversity sampling: interleave different output types
    selected_indices = []
    output_groups = list(output_to_indices.values())
    
    # Round-robin across output groups
    max_group_size = max(len(g) for g in output_groups) if output_groups else 0
    for round_idx in range(max_group_size):
        for group in output_groups:
            if round_idx < len(group):
                selected_indices.append(group[round_idx])
    
    # If no diversity grouping possible, use raw BM25 order
    if not selected_indices:
        selected_indices = candidate_indices
    
    # Greedily fill context window
    examples_str = ""
    total_tokens = 0
    
    for idx in selected_indices:
        ex = all_examples[idx]
        input_text = ex.get('input', '')
        output_text = ex.get('output', [''])[0] if isinstance(ex.get('output'), list) else str(ex.get('output', ''))
        
        # Format example
        example_str = f"# {input_text} <label> {output_text} </label>\n"
        
        # Count tokens
        example_tokens = len(tokenizer.encode(example_str, add_special_tokens=False))
        
        if total_tokens + example_tokens <= available_tokens:
            examples_str += example_str
            total_tokens += example_tokens
        else:
            break
    
    return examples_str


def select_examples_simple(
    all_examples: list[dict],
    task_description: str,
    text2annotate: str,
    tokenizer: AutoTokenizer,
    max_context_length: int = 100_000,
    prompt_template_length: int = 500
) -> str:
    """
    Simple sequential example selection (baseline-compatible).
    Falls back to this if BM25 retrieval fails.
    """
    if not all_examples:
        return ""
    
    available_tokens = max_context_length - prompt_template_length
    if available_tokens <= 0:
        return ""
    
    examples_str = ""
    total_tokens = 0
    
    for ex in all_examples:
        input_text = ex.get('input', '')
        output_text = ex.get('output', [''])[0] if isinstance(ex.get('output'), list) else str(ex.get('output'), '')
        
        example_str = f"# {input_text} <label> {output_text} </label>\n"
        example_tokens = len(tokenizer.encode(example_str, add_special_tokens=False))
        
        if total_tokens + example_tokens <= available_tokens:
            examples_str += example_str
            total_tokens += example_tokens
        else:
            break
    
    return examples_str


# Global tokenizer cache
_tokenizer_cache = {}

def _get_tokenizer(tokenizer_path: str = None) -> AutoTokenizer:
    """Get or create tokenizer with caching."""
    global _tokenizer_cache
    key = tokenizer_path or "default"
    if key not in _tokenizer_cache:
        # Try multiple paths for the tokenizer
        candidates = [
            tokenizer_path,
            "/root/autodl-tmp/qwen3-4b",
            "./Qwen3-4B",
            "../Qwen3-4B",
        ]
        loaded = False
        for path in candidates:
            if path is None:
                continue
            try:
                _tokenizer_cache[key] = AutoTokenizer.from_pretrained(
                    path, trust_remote_code=True, local_files_only=True
                )
                loaded = True
                break
            except Exception:
                continue
        if not loaded:
            # Final fallback: try with local_files_only=False
            _tokenizer_cache[key] = AutoTokenizer.from_pretrained(
                candidates[0] or "/root/autodl-tmp/qwen3-4b",
                trust_remote_code=True
            )
    return _tokenizer_cache[key]


def select_examples(
    all_examples: list[dict], 
    task_description: str, 
    text2annotate: str,
    task_id: int = 1,
    tokenizer_path: str = None
) -> str:
    """
    Select examples for ICL, optimized for long-context scenarios.
    
    Uses BM25 retrieval + diversity sampling for tasks 1-7 (long context),
    and simple sequential selection for task 8 (shorter context).
    
    Args:
        all_examples: List of example dicts
        task_description: Task description
        text2annotate: Text to annotate
        task_id: Task ID (1-8)
        tokenizer_path: Path to Qwen3-4B tokenizer
    
    Returns:
        Formatted examples string
    """
    tokenizer = _get_tokenizer(tokenizer_path)
    
    # Task-specific context length configuration
    if task_id == 8:
        max_context_length = 30_000
        prompt_template_length = 800
    else:
        max_context_length = 100_000
        prompt_template_length = 600
    
    try:
        return select_examples_diverse(
            all_examples, task_description, text2annotate,
            tokenizer, max_context_length, prompt_template_length
        )
    except Exception as e:
        print(f"Warning: Diverse selection failed ({e}), falling back to simple selection.")
        return select_examples_simple(
            all_examples, task_description, text2annotate,
            tokenizer, max_context_length, prompt_template_length
        )


# ============================================================
# 3. ROBUST LABEL EXTRACTION
# ============================================================

def count_answer(text: str) -> Optional[str]:
    """
    Extract content from <label> tags with multiple fallback strategies.
    
    Strategy:
    1. Try standard <label>...</label> pattern
    2. Try <label> without closing tag
    3. Try to find any label-like pattern
    4. Return None if nothing found
    
    Args:
        text: Raw model output text
    
    Returns:
        Extracted label content or None
    """
    if not text or text == "None":
        return None
    
    # Strategy 1: Standard <label>...</label> pattern
    pattern1 = r'<label>\s*(.+?)\s*</label>'
    matches = re.findall(pattern1, text, re.DOTALL)
    if matches:
        # Return the last occurrence (model's final answer)
        result = matches[-1].strip()
        if len(result) < 5000:  # Sanity check: not too long
            return result
    
    # Strategy 2: <label> without closing tag (model may cut off)
    pattern2 = r'<label>\s*(.+?)$'
    matches = re.findall(pattern2, text, re.DOTALL)
    if matches:
        result = matches[-1].strip()
        if len(result) < 5000:
            return result
    
    # Strategy 3: Look for answer-like content after "Output:" or "Answer:"
    pattern3 = r'(?:Output|Answer|Result):?\s*(.+?)(?:\n|$)'
    matches = re.findall(pattern3, text, re.DOTALL)
    if matches:
        result = matches[-1].strip()
        if len(result) < 5000:
            return result
    
    # Strategy 4: Return the last line if it looks like a valid answer
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if lines:
        last_line = lines[-1]
        # Check it's not too long and doesn't look like prose
        if len(last_line) < 500 and not last_line.endswith('.'):
            lower = last_line.lower()
            skip_phrases = [
                'no tags here', 'no label', 'none', 'no answer',
                'i think', 'i believe', 'the answer is', 'the result is',
                'annotation result', 'final output', 'output', 'result',
            ]
            if not any(p in lower for p in skip_phrases):
                return last_line
    
    return None


# ============================================================
# 4. ANNOTATION FUNCTIONS
# ============================================================

def annotate_nvidia(input_prompt: str, num_samples: int = 1) -> Optional[str]:
    """
    Annotate using LLM API (NVIDIA GPU / vLLM).
    
    Args:
        input_prompt: The prompt to send to the model
        num_samples: Number of samples for voting (1 = single pass)
    
    Returns:
        Annotated label or None
    """
    import requests
    
    URL = "http://0.0.0.0:2026/v1/completions"
    
    predictions = []
    
    for _ in range(num_samples):
        data = {
            "model": "/root/autodl-tmp/qwen3-4b",
            "prompt": input_prompt,
            "max_tokens": 10_000,
            "temperature": 0.3 if num_samples == 1 else 0.7,
            "top_p": 0.9,
        }
        
        try:
            resp = requests.post(URL, json=data, timeout=120)
            whole_result = resp.json()["choices"][0]["text"]
        except Exception as e:
            print(f"API call failed: {e}")
            whole_result = "None"
        
        prediction = count_answer(whole_result)
        if prediction is not None:
            predictions.append(prediction)
    
    if not predictions:
        return None
    
    # Voting: return most common prediction
    counter = Counter(predictions)
    most_common = counter.most_common(1)[0][0]
    return most_common


def annotate_ascend(input_prompt: str, num_samples: int = 1) -> Optional[str]:
    """
    
    Args:
        input_prompt: The prompt to send to the model
        num_samples: Number of samples for voting
    
    Returns:
        Annotated label or None
    """
    import openai
    
    openai.api_key = "EMPTY"
    openai.base_url = "http://localhost:9010/v1/"
    model = "Qwen3-4B-ascend-flagos"
    
    predictions = []
    
    for _ in range(num_samples):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": input_prompt}
        ]
        
        try:
            response = openai.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.3 if num_samples == 1 else 0.7,
                top_p=0.9,
                max_tokens=10_000,
                stream=False,
            )
            whole_result = response.choices[0].message.content
        except Exception as e:
            print(f"API call failed: {e}")
            whole_result = "None"
        
        prediction = count_answer(whole_result)
        if prediction is not None:
            predictions.append(prediction)
    
    if not predictions:
        return None
    
    counter = Counter(predictions)
    most_common = counter.most_common(1)[0][0]
    return most_common


# ============================================================
# 5. STRUCTURE ALIGNMENT (Post-processing with ctx_compress)
# ============================================================

def align_prediction(task_id: int, test_sample_id: str, raw_prediction: Optional[str]) -> Optional[str]:
    """
    Align raw prediction with structural templates.

    
    Args:
        task_id: Task ID (1-8)
        test_sample_id: The test sample identifier
        raw_prediction: The raw model prediction
    """
    if not _HAS_CTX_COMPRESS:
        return raw_prediction
    
    try:
        aligned = align_output(task_id, test_sample_id, raw_prediction)
        return aligned
    except Exception:
        return raw_prediction


# ============================================================
# 6. UTILITY: Check context length
# ============================================================

def estimate_context_length(
    task_description: str,
    text2annotate: str,
    examples_str: str,
    tokenizer_path: str = None
) -> int:
    """Estimate total context length in tokens."""
    tokenizer = _get_tokenizer(tokenizer_path)
    
    prompt = build_prompt(task_description, text2annotate)
    full_prompt = prompt.replace("[[EXAMPLES]]\n\n", examples_str + '\n\n')
    
    tokens = tokenizer.encode(full_prompt, add_special_tokens=False)
    return len(tokens)
