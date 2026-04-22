
import re
from collections import Counter
from transformers import AutoTokenizer

""" Here is an example of implementation of Long-Context Data Annotation. """

                                        
_DYNAMIC_ADAPTIVE_STATE = {
    'task_history': {},
    'example_stats': {},
}

                                            
_MULTITASK_TASK_CONFIG = {
    1: {'shared_ratio': 0.35, 'importance_weight': 1.20, 'history_weight': 0.20, 'cot': True, 'self_consistency': True},
    2: {'shared_ratio': 0.35, 'importance_weight': 1.18, 'history_weight': 0.18, 'cot': True, 'self_consistency': True},
    3: {'shared_ratio': 0.30, 'importance_weight': 1.22, 'history_weight': 0.20, 'cot': True, 'self_consistency': True},
    4: {'shared_ratio': 0.30, 'importance_weight': 1.15, 'history_weight': 0.18, 'cot': True, 'self_consistency': True},
    5: {'shared_ratio': 0.25, 'importance_weight': 1.05, 'history_weight': 0.15, 'cot': True, 'self_consistency': True},
    6: {'shared_ratio': 0.25, 'importance_weight': 1.10, 'history_weight': 0.16, 'cot': True, 'self_consistency': True},
    7: {'shared_ratio': 0.20, 'importance_weight': 1.00, 'history_weight': 0.14, 'cot': True, 'self_consistency': True},
}

                                                          
_COMBINATION19_TASK_CONFIG = {
    1: {'contrastive_weight': 0.32, 'rl_weight': 0.24, 'diversity_weight': 0.10, 'mix_shared_bonus': 0.05},
    2: {'contrastive_weight': 0.30, 'rl_weight': 0.24, 'diversity_weight': 0.10, 'mix_shared_bonus': 0.05},
    3: {'contrastive_weight': 0.34, 'rl_weight': 0.25, 'diversity_weight': 0.09, 'mix_shared_bonus': 0.04},
    4: {'contrastive_weight': 0.31, 'rl_weight': 0.23, 'diversity_weight': 0.10, 'mix_shared_bonus': 0.04},
    5: {'contrastive_weight': 0.26, 'rl_weight': 0.22, 'diversity_weight': 0.12, 'mix_shared_bonus': 0.06},
    6: {'contrastive_weight': 0.27, 'rl_weight': 0.22, 'diversity_weight': 0.12, 'mix_shared_bonus': 0.06},
    7: {'contrastive_weight': 0.25, 'rl_weight': 0.21, 'diversity_weight': 0.13, 'mix_shared_bonus': 0.07},
}

                                                       
_COMBINATION20_TASK_CONFIG = {
    1: {'quality_floor': 0.42, 'similarity_weight': 0.28, 'quality_weight': 0.22, 'layer_bonus': 0.10, 'shared_bonus': 0.04},
    2: {'quality_floor': 0.40, 'similarity_weight': 0.27, 'quality_weight': 0.22, 'layer_bonus': 0.10, 'shared_bonus': 0.04},
    3: {'quality_floor': 0.44, 'similarity_weight': 0.29, 'quality_weight': 0.23, 'layer_bonus': 0.11, 'shared_bonus': 0.03},
    4: {'quality_floor': 0.41, 'similarity_weight': 0.27, 'quality_weight': 0.21, 'layer_bonus': 0.10, 'shared_bonus': 0.03},
    5: {'quality_floor': 0.38, 'similarity_weight': 0.24, 'quality_weight': 0.20, 'layer_bonus': 0.09, 'shared_bonus': 0.05},
    6: {'quality_floor': 0.39, 'similarity_weight': 0.25, 'quality_weight': 0.20, 'layer_bonus': 0.09, 'shared_bonus': 0.05},
    7: {'quality_floor': 0.37, 'similarity_weight': 0.24, 'quality_weight': 0.19, 'layer_bonus': 0.08, 'shared_bonus': 0.06},
}


def _normalize_text_for_similarity(text: str) -> str:
                                
    return re.sub(r'\s+', ' ', str(text).strip().lower())


                                                    
def _build_contrastive_token_set(text: str) -> set[str]:
    normalized_text = _normalize_text_for_similarity(text)
    return set(re.findall(r'\w+', normalized_text))


                                       
def compute_keyword_overlap_similarity(text_a: str, text_b: str) -> float:
    tokens_a = _build_contrastive_token_set(text_a)
    tokens_b = _build_contrastive_token_set(text_b)
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return intersection / union if union else 0.0


                                              
def compute_contrastive_alignment(example: dict, text2annotate: str, task_id: int = None) -> float:
    input_text = str(example.get('input', ''))
    output_value = example.get('output', '')
    output_text = output_value[0] if isinstance(output_value, list) and output_value else str(output_value)
    example_tokens = _build_contrastive_token_set(f"{input_text} {output_text}")
    target_tokens = _build_contrastive_token_set(text2annotate)
    if not example_tokens or not target_tokens:
        return 0.0
    overlap_score = len(example_tokens & target_tokens) / max(1, len(target_tokens))
    structural_bonus = 0.08 if task_id in [1, 2, 3, 4] and any(ch.isdigit() for ch in input_text + text2annotate) else 0.0
    semantic_bonus = 0.06 if task_id in [5, 6, 7] and len(target_tokens & {'emotion', 'review', 'genre', 'question', 'answer'}) >= 1 else 0.0
    return min(1.0, overlap_score + structural_bonus + semantic_bonus)


                                           
def estimate_example_quality(example: dict) -> float:
    input_text = str(example.get('input', ''))
    output_value = example.get('output', '')
    output_text = output_value[0] if isinstance(output_value, list) and output_value else str(output_value)
    input_len = len(input_text.strip())
    output_len = len(str(output_text).strip())
    structure_bonus = 0.15 if any(ch.isdigit() for ch in input_text) else 0.0
    structure_bonus += 0.10 if any(ch in input_text for ch in [':', '-', '(', ')']) else 0.0
    length_score = min(input_len / 400.0, 1.0) * 0.45 + min(output_len / 80.0, 1.0) * 0.30
    return min(1.0, length_score + structure_bonus + 0.10)


                                  
def estimate_task_importance(example: dict, task_id: int = None) -> float:
    base_score = 0.4
    input_text = str(example.get('input', ''))
    output_value = example.get('output', '')
    output_text = output_value[0] if isinstance(output_value, list) and output_value else str(output_value)
    if task_id in [1, 2, 3, 4]:
        if any(ch.isdigit() for ch in input_text + output_text):
            base_score += 0.25
        if any(keyword in input_text.lower() for keyword in ['count', 'number', 'integer', 'string', 'collatz']):
            base_score += 0.20
    elif task_id in [5, 6, 7]:
        if len(output_text.split()) >= 2:
            base_score += 0.15
        if any(keyword in input_text.lower() for keyword in ['emotion', 'review', 'genre', 'question']):
            base_score += 0.15
    return min(1.0, base_score)


                                               
def extract_adaptive_meta_features(example: dict, text2annotate: str, task_id: int = None) -> dict:
    input_text = str(example.get('input', ''))
    similarity = compute_keyword_overlap_similarity(input_text, text2annotate)
    quality_score = estimate_example_quality(example)
    importance_score = estimate_task_importance(example, task_id)
    length_gap = abs(len(input_text) - len(text2annotate)) / max(len(text2annotate), 1)
    return {
        'similarity': similarity,
        'quality': quality_score,
        'importance': importance_score,
        'length_gap': min(length_gap, 1.0),
    }


                                      
def predict_adaptive_utility(example: dict, text2annotate: str, task_id: int = None) -> tuple[float, dict]:
    features = extract_adaptive_meta_features(example, text2annotate, task_id)
    utility_score = (
        0.42 * features['similarity']
        + 0.28 * features['quality']
        + 0.22 * features['importance']
        + 0.08 * (1.0 - features['length_gap'])
    )
    return utility_score, features


                                     
def update_adaptive_feedback(task_id: int, example_key: str, reward: float) -> tuple[float, int]:
    task_history = _DYNAMIC_ADAPTIVE_STATE['task_history'].setdefault(task_id, {'updates': 0, 'avg_reward': 0.0})
    example_stats = _DYNAMIC_ADAPTIVE_STATE['example_stats'].setdefault(example_key, {'reward': 0.0, 'count': 0})
    example_stats['reward'] = 0.7 * example_stats['reward'] + 0.3 * reward
    example_stats['count'] += 1
    task_history['avg_reward'] = 0.8 * task_history['avg_reward'] + 0.2 * reward
    task_history['updates'] += 1
    return example_stats['reward'], example_stats['count']


                                        
def rank_examples_for_dynamic_adaptation(all_examples: list[dict], text2annotate: str, task_id: int = None) -> list[dict]:
    scored_examples = []
    for example in all_examples:
        utility_score, features = predict_adaptive_utility(example, text2annotate, task_id)
        example_key = _normalize_text_for_similarity(str(example.get('input', '')))[:200]
        history = _DYNAMIC_ADAPTIVE_STATE['example_stats'].get(example_key, {'reward': 0.0, 'count': 0})
        exploration_bonus = 0.08 / (history['count'] + 1)
        adaptive_score = 0.55 * utility_score + 0.30 * history['reward'] + 0.15 * exploration_bonus
        adaptive_score += 0.05 * features['importance']
        scored_examples.append((adaptive_score, features['similarity'], example))
    scored_examples.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [example for _, _, example in scored_examples]


                                          
def format_example_with_cot(example: dict, task_id: int = None) -> str:
    input_text = str(example.get('input', ''))
    output_value = example.get('output', '')
    output_text = output_value[0] if isinstance(output_value, list) and output_value else str(output_value)
    reasoning = "Identify the task pattern, compare with similar examples, and infer the shortest valid label."
    if task_id in [1, 2, 3, 4]:
        reasoning = "Parse the symbolic structure carefully, compute intermediate clues, and then derive the final label."
    elif task_id in [5, 6, 7]:
        reasoning = "Focus on semantic evidence, emotion or entailment cues, and then choose the most consistent label."
    return (
        f"# {input_text}\n"
        f"Reasoning: {reasoning}\n"
        f"Final Answer: <label>{output_text}</label>\n"
    )


                                    
def get_multitask_task_config(task_id: int = None) -> dict:
    return _MULTITASK_TASK_CONFIG.get(task_id, {'shared_ratio': 0.20, 'importance_weight': 1.00, 'history_weight': 0.12, 'cot': False, 'self_consistency': False})


                                   
def compute_shared_source_weight(source_task_id: int, target_task_id: int) -> float:
    if source_task_id == target_task_id:
        return 1.0
    symbolic_group = {1, 2, 3, 4}
    semantic_group = {5, 6, 7}
    if source_task_id in symbolic_group and target_task_id in symbolic_group:
        return 0.82
    if source_task_id in semantic_group and target_task_id in semantic_group:
        return 0.78
    return 0.55


                                            
def rank_examples_for_multitask_optimization(all_examples: list[dict], text2annotate: str, task_id: int = None) -> list[dict]:
    task_config = get_multitask_task_config(task_id)
    scored_examples = []
    for example in all_examples:
        utility_score, features = predict_adaptive_utility(example, text2annotate, task_id)
        source_task_id = example.get('source_task_id', task_id)
        source_weight = compute_shared_source_weight(source_task_id, task_id)
        example_key = _normalize_text_for_similarity(str(example.get('input', '')))[:200]
        history = _DYNAMIC_ADAPTIVE_STATE['example_stats'].get(example_key, {'reward': 0.0, 'count': 0})
        shared_bonus = 0.06 if source_task_id != task_id else 0.0
        multitask_score = (
            0.48 * utility_score
            + task_config['history_weight'] * history['reward']
            + 0.22 * features['similarity']
            + 0.15 * features['quality']
            + 0.15 * min(1.0, features['importance'] * task_config['importance_weight'])
        )
        multitask_score = multitask_score * source_weight + shared_bonus
        scored_examples.append((multitask_score, source_weight, example))
    scored_examples.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [example for _, _, example in scored_examples]


                                                  
def rank_examples_for_combination19(all_examples: list[dict], text2annotate: str, task_id: int = None) -> list[dict]:
    task_config = get_multitask_task_config(task_id)
    combination_config = _COMBINATION19_TASK_CONFIG.get(task_id, {'contrastive_weight': 0.28, 'rl_weight': 0.22, 'diversity_weight': 0.10, 'mix_shared_bonus': 0.05})
    candidate_token_sets = [_build_contrastive_token_set(str(example.get('input', ''))) for example in all_examples]
    scored_examples = []
    for idx, example in enumerate(all_examples):
        utility_score, features = predict_adaptive_utility(example, text2annotate, task_id)
        source_task_id = example.get('source_task_id', task_id)
        source_weight = compute_shared_source_weight(source_task_id, task_id)
        example_key = _normalize_text_for_similarity(str(example.get('input', '')))[:200]
        history = _DYNAMIC_ADAPTIVE_STATE['example_stats'].get(example_key, {'reward': 0.0, 'count': 0})
        contrastive_score = compute_contrastive_alignment(example, text2annotate, task_id)
        candidate_token_set = candidate_token_sets[idx]
                                                                  
        diversity_penalty = 0.0
        if candidate_token_set:
            peer_overlaps = [
                len(candidate_token_set & other_tokens) / max(1, len(candidate_token_set | other_tokens))
                for other_idx, other_tokens in enumerate(candidate_token_sets)
                if other_idx != idx and other_tokens
            ]
            if peer_overlaps:
                diversity_penalty = sum(sorted(peer_overlaps, reverse=True)[:3]) / min(3, len(peer_overlaps))
        mix_bonus = combination_config['mix_shared_bonus'] if source_task_id != task_id else 0.0
        fusion_score = (
            0.34 * utility_score
            + combination_config['contrastive_weight'] * contrastive_score
            + combination_config['rl_weight'] * history['reward']
            + 0.12 * features['similarity']
            + 0.10 * min(1.0, features['importance'] * task_config['importance_weight'])
            + 0.08 * features['quality']
        )
        final_score = fusion_score * source_weight + mix_bonus - combination_config['diversity_weight'] * diversity_penalty
        scored_examples.append((final_score, contrastive_score, source_weight, example))
    scored_examples.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    return [example for _, _, _, example in scored_examples]


                                                            
def estimate_example_layer(example: dict, task_id: int = None) -> tuple[int, float]:
    input_text = str(example.get('input', ''))
    output_value = example.get('output', '')
    output_text = output_value[0] if isinstance(output_value, list) and output_value else str(output_value)
    normalized_input = _normalize_text_for_similarity(input_text)
    complexity_score = min(len(normalized_input) / 220.0, 1.0)
    complexity_score += 0.18 if any(ch.isdigit() for ch in input_text + output_text) else 0.0
    complexity_score += 0.10 if any(ch in input_text for ch in [':', '-', '(', ')', '[', ']']) else 0.0
    complexity_score += 0.08 if len(str(output_text).split()) >= 3 else 0.0
    if task_id in [5, 6, 7]:
        complexity_score += 0.08 if any(keyword in normalized_input for keyword in ['because', 'reason', 'sentiment', 'entailment', 'question']) else 0.0
    if complexity_score >= 0.88:
        return 3, min(complexity_score, 1.0)
    if complexity_score >= 0.54:
        return 2, min(complexity_score, 1.0)
    return 1, min(complexity_score, 1.0)


                                                
def filter_examples_for_combination20(all_examples: list[dict], task_id: int = None) -> list[dict]:
    combination_config = _COMBINATION20_TASK_CONFIG.get(task_id, {'quality_floor': 0.38})
    filtered_examples = []
    fallback_examples = []
    for example in all_examples:
        quality_score = estimate_example_quality(example)
        if quality_score >= combination_config['quality_floor']:
            filtered_examples.append(example)
        fallback_examples.append((quality_score, example))
    if filtered_examples:
        return filtered_examples
    fallback_examples.sort(key=lambda item: item[0], reverse=True)
    return [example for _, example in fallback_examples[: min(8, len(fallback_examples))]]


                                                             
def rank_examples_for_combination20(all_examples: list[dict], text2annotate: str, task_id: int = None) -> list[dict]:
    task_config = get_multitask_task_config(task_id)
    combination_config = _COMBINATION20_TASK_CONFIG.get(task_id, {'similarity_weight': 0.25, 'quality_weight': 0.20, 'layer_bonus': 0.09, 'shared_bonus': 0.04})
    filtered_examples = filter_examples_for_combination20(all_examples, task_id)
    target_complexity = min(len(_normalize_text_for_similarity(text2annotate)) / 220.0, 1.0)
    target_layer = 3 if target_complexity >= 0.88 else 2 if target_complexity >= 0.54 else 1
    scored_examples = []
    for example in filtered_examples:
        utility_score, features = predict_adaptive_utility(example, text2annotate, task_id)
        source_task_id = example.get('source_task_id', task_id)
        source_weight = compute_shared_source_weight(source_task_id, task_id)
        history = _DYNAMIC_ADAPTIVE_STATE['example_stats'].get(
            _normalize_text_for_similarity(str(example.get('input', '')))[:200],
            {'reward': 0.0, 'count': 0},
        )
        layer_id, complexity_score = estimate_example_layer(example, task_id)
        layer_distance = abs(layer_id - target_layer)
                                                        
        layer_score = max(0.0, 1.0 - 0.35 * layer_distance) + 0.12 * complexity_score
        source_bonus = combination_config['shared_bonus'] if source_task_id != task_id else 0.0
        hierarchical_score = (
            0.24 * utility_score
            + combination_config['similarity_weight'] * features['similarity']
            + combination_config['quality_weight'] * features['quality']
            + 0.14 * min(1.0, features['importance'] * task_config['importance_weight'])
            + 0.10 * history['reward']
            + combination_config['layer_bonus'] * layer_score
        )
        final_score = hierarchical_score * source_weight + source_bonus
        scored_examples.append((layer_distance, -final_score, -features['similarity'], example))
    scored_examples.sort(key=lambda item: (item[0], item[1], item[2]))
    return [example for _, _, _, example in scored_examples]


                                          
def format_shared_example(example: dict, task_id: int = None, use_cot: bool = False) -> str:
    source_task_id = example.get('source_task_id', task_id)
    if use_cot:
        example_str = format_example_with_cot(example, task_id=task_id)
    else:
        input_text = str(example.get('input', ''))
        output_value = example.get('output', '')
        output_text = output_value[0] if isinstance(output_value, list) and output_value else str(output_value)
        example_str = f"# {input_text} <label> {output_text} </label>\n"
    if source_task_id == task_id:
        return example_str
    return f"# Shared Pattern From Task {source_task_id}\n{example_str}"


                                          
def annotate_with_self_consistency(input_prompt: str, num_samples: int = 3, max_tokens: int = 256, task_id: int = None) -> tuple:
    predictions = []
    raw_outputs = []
    temperatures = [0.1, 0.2, 0.3][:max(1, num_samples)]
    for temperature in temperatures:
        prediction, raw_output = annotate_ascend(
            input_prompt,
            max_tokens=max_tokens,
            use_count_answer=True,
            task_id=task_id,
            temperature=temperature,
        )
        if prediction:
            predictions.append(prediction)
        raw_outputs.append(raw_output)
    if predictions:
        majority_vote = Counter(predictions).most_common(1)[0][0]
        return majority_vote, "\n---SELF-CONSISTENCY---\n".join([str(x) for x in raw_outputs if x is not None])
    return None, "\n---SELF-CONSISTENCY---\n".join([str(x) for x in raw_outputs if x is not None])

def build_prompt____(task_description: str, text2annotate: str) -> str:
    """
    Build a high-precision English prompt for long-context data annotation (optimized for Qwen3-4B).
    Core requirement: Final answer MUST be wrapped in <label> tags (no extra content outside tags).
    """
    prompt = (
        "### Role Definition\n"
        "You are a professional data annotation expert specializing in long-context text labeling. "
        "Your work must strictly comply with the following rules, with the highest priority given to output format accuracy.\n\n"
        
        "### Core Annotation Task\n"
        f"{task_description}\n\n"
        
        "### Non-Negotiable Annotation Rules (Highest Priority)\n"
        "1. **Final Output Mandate**: Your annotation result MUST be wrapped in <label> tags — NO text, symbols, spaces, or explanations are allowed outside the tags.\n"
        "2. **Internal Reasoning Permission**: You may perform logical reasoning, text analysis, or context comprehension internally (in your thought process), but NONE of these thoughts may appear in the final output.\n"
        "3. **Label Format Strictness**: <label> is the opening tag and </label> is the closing tag — they must appear in pairs, with NO extra spaces or characters inside the tags (e.g., <label>  Good Review  </label> is invalid).\n"
        "4. **Prohibited Outputs**: \n"
        "   - ❌ Prohibited: 'After analysis, this is a positive review: <label>Good Review</label>' (extra text outside tags)\n"
        "   - ❌ Prohibited: 'Bad Review' (missing <label> tags entirely)\n"
        "   - ❌ Prohibited: '<label>Bad Review' (unpaired/closing tag missing)\n\n"
        
        "### Correct vs. Incorrect Examples\n"
        "✅ Correct Example 1: <label>answer</label>\n"
        "✅ Correct Example 2: <label>Bad Review</label>\n"
        "❌ Incorrect Example 1: I think this review is negative → <label>Bad Review</label>\n"
        "❌ Incorrect Example 2: <label>  Neutral Review  </label> (extra spaces inside tags)\n"
        "❌ Incorrect Example 3: Neutral Review (no label tags)\n\n"
        
        "### Reference Annotation Examples\n"
        "{EXAMPLES}\n\n"
        
        "### Text to Annotate\n"
        f"{text2annotate}\n\n"
        
        "### Final Output Command (Re-emphasized)\n"
        "You may complete any internal reasoning process, but your FINAL OUTPUT MUST consist solely of the annotation result wrapped in <label> tags (no other content whatsoever).\n"
        "Annotation Result: "
    )
    return prompt

def build_prompt(task_description: str, text2annotate: str) -> str:
\
\
\
\
       
    prompt = (
        "/no_think\n"
        "### Role Definition\n"
        "You are a professional data annotation expert specialized in long-context text labeling. "
        "Your work must strictly follow the task rules, fully learn from the provided examples, and ensure the final annotation result is enclosed in <label> tags.\n\n"
        "### Core Task\n"
        f"Task: {task_description}\n\n"
        "### Critical Annotation Guidelines\n"
        "1. **Example Learning Requirement**: Thoroughly analyze and fully learn from the annotation logic, format, and criteria in the Examples section. "
        "Your annotation must align with the style, judgment standards, and tag usage shown in the examples.\n"
        "2. **Silent Reasoning Requirement**: Think through the task internally, but do NOT output your reasoning, analysis, explanation, or any extra words.\n"
        "3. **Mandatory Output Rule**: Output only the final annotation result enclosed in <label> tags.\n"
        "   - Correct example 1: <label>3</label>\n"
        "   - Correct example 2: <label>Good Review</label>\n"
        "   - Wrong example 1: Reasoning: ... <label>3</label>\n"
        "   - Wrong example 2: 3\n"
        "   - Wrong example 3: <label>3\n"
        "4. **Brevity Requirement**: The final answer must be a single short label only. Do not repeat the input text. Do not add prefixes such as 'Answer:' or 'Reasoning:'.\n\n"
        
        "Examples:\n"
        "[[EXAMPLES]]\n\n"
        "### Text to Annotate\n"
        f"Input: {text2annotate}\n"
        "Output:"
        "### Final Requirement Summary\n"
        "1. Think silently and do not output analysis.\n"
        "2. Output exactly one final answer wrapped in <label> and </label>.\n"
        "3. All annotation logic must strictly follow the examples provided above.\n"
    )
    return prompt


def build_code_generation_prompt(task_description: str, text2annotate: str) -> str:
\
\
\
\
       
    prompt = (
        "/no_think\n"
        "You are an expert Triton kernel engineer. Generate Python code ONLY.\n\n"
        f"Task: {task_description}\n\n"
        "Requirements:\n"
        "1. Output ONLY Python code. No explanations or analysis.\n"
        "2. Start with 'import' statements.\n"
        "3. Include @triton.jit kernel and wrapper function.\n"
        "4. No markdown fences or <label> tags.\n\n"
        "Examples:\n"
        "[[EXAMPLES]]\n\n"
        f"Instruction: {text2annotate}\n\n"
        "Output Python code only:\n"
    )
    return prompt

def build_prompt_backup(task_description:str, text2annotate:str)->str:
    """
        Construct the prompt for annotation based on the task description.
        task_description: 
            The description of the annotation task. 
            For example, ``Given an English language product review, 
            determine if it is a Good Review or a Bad Review.`` 
        text2annotate:
            The text that needs to be annotated.
            For example, ``My son received this book as a gift. I was extremely disappointed.``
    """
    prompt = (
        "You are a data annotation assistant. "
        "Your task is to label the given texts according to the task description "
        "and annotation guidelines provided below.\n\n"
        f"[Task Description]\n {task_description}\n\n"
        "[Examples]\n {EXAMPLES}\n\n"
        "Please follow these instructions when labeling:\n"
        "1. **Output Format**: Annotate the text directly by wrapping each labeled "
        "span with <label> tags in the following format: <label> annotation result </label>.\n"
        # "2. Do not add any extra text, explanations, or commentary in the labeled spans.\n\n"
        f"[Task Description (repeat)] \n {task_description}\n\n"
        f"[Input Texts]\n {text2annotate}\n\n"
        "Please output the annotation results: "
    )
    return prompt

def select_examples_backup(all_examples:list[dict], task_description:str, text2annotate:str)->str:
    """
        Select examples from all_examples to fit into the target context length.
        all_examples:
            A list of examples, where each example is a dict with keys 'input', 'output', and 'length'.
            For example, ``{"input": "The material is good and looks great.", "output": "Good Review", "length": 79``},
        task_description:
            The description of the annotation task which may be used for example evaluation. 
            For example, ``Given an English language product review, 
            determine if it is a Good Review or a Bad Review.`` 
        text2annotate:
            The text that needs to be annotated  which may be used for example retrieval.
            For example, ``My son received this book as a gift. I was extremely disappointed.``
        
    """
    # Notice that the maximum context length is restricted.
    target_length = 10_000
    
    input_list = [example['input'] for example in all_examples]
    output_list = [example['output'][0] for example in all_examples]
    length_list = [example['length'] for example in all_examples]
    
    # <label> have 2 tokens; </label> have 3 tokens; \n have 1 token; # have 1 token.
    examples_str, token_num = "", 0
    for i, (input_text, output_text, length) in enumerate(zip(input_list, output_list, length_list)):
        if length + token_num <= target_length:
            token_num += (length + 2 + 3 + 1 + 1)
            example_str = f"# {input_text} <label> {output_text} </label>\n"
            examples_str += example_str
        else:
            return examples_str, i
    return examples_str

def select_examples(
    all_examples: list[dict],
    task_description: str,
    text2annotate: str,
    is_code_generation: bool = False,
    task_id: int = None,
    use_dynamic_adaptive: bool = False,
    use_cot: bool = False,
    use_multitask_optimization: bool = False,
    use_combination19: bool = False,
    use_combination20: bool = False,
) -> str:
\
\
\
\
\
\
\
\
\
\
\
       
                                              
                                      
    tokenizer = AutoTokenizer.from_pretrained("/root/flagos/Qwen3-4B", trust_remote_code=True)
    
                                                
    target_length = 8192                               
    
                                          
                            
    fixed_prompt = (
        f"Task: {task_description}\n\n"
        "Examples:\n"
        "[[EXAMPLES]]\n\n"
        f"Input: {text2annotate}\n"
        "Output: "
    )
    fixed_tokens = len(tokenizer.encode(fixed_prompt, add_special_tokens=False))
    
                         
    available_tokens_for_examples = target_length - fixed_tokens
    
                        
    if available_tokens_for_examples <= 0:
        print(f"警告：任务描述和测试样本输入已超过上下文长度限制（{fixed_tokens} > {target_length} tokens）")
        return ""

    examples_str, token_num = "", 0
    candidate_examples = all_examples
                                                              
    if use_combination20 and not is_code_generation:
        candidate_examples = rank_examples_for_combination20(all_examples, text2annotate, task_id)
    elif use_combination19 and not is_code_generation:
        candidate_examples = rank_examples_for_combination19(all_examples, text2annotate, task_id)
    elif use_multitask_optimization and not is_code_generation:
        candidate_examples = rank_examples_for_multitask_optimization(all_examples, text2annotate, task_id)
    elif use_dynamic_adaptive and not is_code_generation:
        candidate_examples = rank_examples_for_dynamic_adaptation(all_examples, text2annotate, task_id)

                                         
    for i, example in enumerate(candidate_examples):
        try:
                                            
            input_text = example['input']
            output_text = example['output'][0] if isinstance(example['output'], list) else example['output']
            
                                                                     
                                            
            input_tokens = len(tokenizer.encode(input_text, add_special_tokens=False))
            output_tokens = len(tokenizer.encode(output_text, add_special_tokens=False))
            length = input_tokens + output_tokens                 
            
                               
            if is_code_generation:
                                                                   
                format_tokens = len(tokenizer.encode(f"Input: \nOutput: \n\n", add_special_tokens=False))
                example_str = f"Input: {input_text}\nOutput: {output_text}\n\n"
            else:
                if use_multitask_optimization:
                    example_str = format_shared_example(example, task_id=task_id, use_cot=use_cot)
                    format_tokens = len(tokenizer.encode("# Shared Pattern From Task \n# \nReasoning: \nFinal Answer: <label></label>\n", add_special_tokens=False))
                elif use_cot:
                    example_str = format_example_with_cot(example, task_id=task_id)
                    format_tokens = len(tokenizer.encode("# \nReasoning: \nFinal Answer: <label></label>\n", add_special_tokens=False))
                else:
                                                                      
                    format_tokens = len(tokenizer.encode(f"# <label> </label>\n", add_special_tokens=False))
                    example_str = f"# {input_text} <label> {output_text} </label>\n"
            
                                    
            if length + format_tokens + token_num <= available_tokens_for_examples:
                           
                token_num += (length + format_tokens)
                examples_str += example_str
                if use_dynamic_adaptive and not is_code_generation:
                    example_key = _normalize_text_for_similarity(str(input_text))[:200]
                    utility_score, _ = predict_adaptive_utility(example, text2annotate, task_id)
                                                      
                    update_adaptive_feedback(task_id or 0, example_key, utility_score)
            else:
                                      
                return examples_str
        except KeyError as e:
            print(f"警告：示例{i}缺少键{e}，跳过该示例")
            continue
                           
    return examples_str




def count_answer(text: str) -> tuple[list, dict]:
\
\
\
\
       
                                    
    if '<|answer|>' in text:
        parts = text.split('<|answer|>')
        if len(parts) > 1:
            text = parts[-1].split('<|/answer|>')[0] if '<|/answer|>' in parts[-1] else parts[-1]
    
                                   
    pattern = r'<label>\s*(.+?)\s*</label>'
    content_matches = re.findall(pattern, text, re.DOTALL)
    
                                                             
    if not content_matches:
        pattern_unclosed = r'<label>\s*(.+)'
        unclosed_matches = re.findall(pattern_unclosed, text, re.DOTALL)
        if unclosed_matches:
                                   
            content = unclosed_matches[0].strip().split('\n')[0].strip()
            if content and len(content) < 100:
                content_matches = [content]
    
    content_counter = Counter(content_matches)
    if not content_counter:
        return None
    
    max_count = max(content_counter.values())
    answer = [content for content, count in content_counter.items() if count == max_count]
    
    if (len(answer[0]) >= 100):
        return None
    return answer[0]


def clean_code_generation_output(text: str) -> str | None:
\
\
\
\
\
\
\
\
\
\
       
    if text is None:
        return None

    cleaned_text = str(text)

                                                     
    cleaned_text = re.sub(r'<think>.*?</think>', '', cleaned_text, flags=re.DOTALL | re.IGNORECASE)

    try:
        think_end_marker = chr(0x25b6)  # Unicode: ▶
        if think_end_marker in cleaned_text:
            cleaned_text = cleaned_text.split(think_end_marker, 1)[1]
    except Exception:
        pass

    label_pattern = r'<label>\s*(.+?)\s*</label>'
    label_matches = re.findall(label_pattern, cleaned_text, re.DOTALL)
    if label_matches:
        longest_label = max((item.strip() for item in label_matches), key=len, default='')
                                             
        if any(keyword in longest_label for keyword in ['import ', 'def ', '@triton.jit', 'tl.', 'torch.']):
            cleaned_text = longest_label
        else:
            cleaned_text = re.sub(r'</?label>', '', cleaned_text, flags=re.IGNORECASE)

    fenced_matches = re.findall(r'```(?:python)?\s*(.*?)```', cleaned_text, flags=re.DOTALL | re.IGNORECASE)
    if fenced_matches:
                                          
        cleaned_text = max((item.strip() for item in fenced_matches), key=len, default=cleaned_text)

    cleaned_text = re.sub(r'^```(?:python)?', '', cleaned_text, flags=re.IGNORECASE).strip()
    cleaned_text = re.sub(r'```$', '', cleaned_text).strip()
    cleaned_text = re.sub(r'<[^>]+>', '', cleaned_text).strip()

    lines = [line.rstrip() for line in cleaned_text.splitlines()]
    while lines and not lines[0].strip():
        lines.pop(0)
    cleaned_text = '\n'.join(lines).strip()

    if not cleaned_text:
        return None

                                  
    code_markers = [
        'import ', 'from ', 'def ', '@triton', '@torch', 
        'triton', 'tl.', 'torch.', 'torch.nn', 
        'class ', 'return ', 'if __name__'
    ]
    has_code_marker = any(marker in cleaned_text for marker in code_markers)
    
                   
    if has_code_marker:
        return cleaned_text
    
                                   
    non_empty_lines = [line.strip() for line in cleaned_text.splitlines() if line.strip()]
    
                                          
    if len(non_empty_lines) >= 3 or len(cleaned_text) >= 80:
                          
        if any('(' in line or ':' in line or '=' in line for line in non_empty_lines[:5]):
            return cleaned_text
    
                         
    if len(non_empty_lines) <= 2 and len(cleaned_text) < 80:
        return None

    return cleaned_text


def annotate_nvidia(input_prompt:str)->list[str]:
    """
        Annotate the unlabeled data using an LLM API (nvidia GPU).
        prompts:
            A prompt constructed for annotation.
            For example, ``["You are a data annotation assistant. Your task is to label ..."]``
    """
    import requests
    URL="http://0.0.0.0:2026/v1/completions"
    
    data = {
        "model": "../Qwen3-4B",
        "prompt": input_prompt,
        "max_tokens": 10_000, # max_token = 10k
    }

    try:
        resp = requests.post(URL, json=data)
        whole_result = resp.json()["choices"][0]["text"]
    except Exception as e:
        whole_result = "None"


    prediction = count_answer(whole_result)
    return prediction

def annotate_ascend(input_prompt:str, max_tokens:int=2048, use_count_answer:bool=True, task_id:int=None, temperature:float=0.1)->tuple:
\
\
\
\
\
\
\
\
\
\
\
\
\
       
    import openai
    openai.api_key = "EMPTY"
    openai.base_url = "http://localhost:9010/v1/"
    model = "Qwen3-4B-ascend-flagos"

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {"role": "user", "content": input_prompt}
    ]
    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=0.95,
        max_tokens=max_tokens,
        stream=False
    )
    whole_result = response.choices[0].message.content
    
    if use_count_answer:
                                          
        prediction = count_answer(whole_result)
    else:
                               
        if task_id == 8:
                                    
            prediction = clean_code_generation_output(whole_result)
        else:
                          
            prediction = whole_result
    
    return prediction, whole_result


def annotate_batch(
    prompts: list[str],
    num_workers: int = 4,
    max_tokens: int = 128,
    use_count_answer: bool = True,
    task_id: int = None,
    use_self_consistency: bool = False,
) -> list[str]:
    """
        Batch annotate with parallel requests.
        prompts: List of prompts to annotate.
        num_workers: Number of parallel workers.
        max_tokens: Maximum tokens to generate for each prompt.
        use_count_answer: Whether to use count_answer to extract label from response.
        task_id: Task ID (default: None). If task_id == 8, will apply special cleaning.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    results = [None] * len(prompts)
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        if use_self_consistency and task_id is not None and 1 <= task_id <= 7:
                                                          
            future_to_idx = {
                executor.submit(annotate_with_self_consistency, p, 3, max_tokens, task_id): i
                for i, p in enumerate(prompts)
            }
        else:
            future_to_idx = {
                executor.submit(annotate_ascend, p, max_tokens, use_count_answer, task_id): i
                for i, p in enumerate(prompts)
            }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                print(f"Error processing prompt {idx}: {e}")
                results[idx] = (None, None)                                 
    
    return results
