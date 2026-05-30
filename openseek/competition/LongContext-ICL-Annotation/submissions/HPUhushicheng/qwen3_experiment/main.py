"""
Main evaluation script for Long-Context ICL Data Annotation.
Optimized for Qwen3-4B with FlagScale deployment.
"""

import json
import os
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer

from method import build_prompt, select_examples, annotate_nvidia as annotate
from method import align_prediction
# from method import annotate_ascend as annotate  # For Huawei Ascend

import pathlib
_SCRIPT_DIR = pathlib.Path(__file__).parent.absolute()
_DATA_DIR = _SCRIPT_DIR.parent / 'data'

TASK_FILES = {
    1: str(_DATA_DIR / 'openseek-1_closest_integers.json'),
    2: str(_DATA_DIR / 'openseek-2_count_nouns_verbs.json'),
    3: str(_DATA_DIR / 'openseek-3_collatz_conjecture.json'),
    4: str(_DATA_DIR / 'openseek-4_conala_concat_strings.json'),
    5: str(_DATA_DIR / 'openseek-5_semeval_2018_task1_tweet_sadness_detection.json'),
    6: str(_DATA_DIR / 'openseek-6_mnli_same_genre_classification.json'),
    7: str(_DATA_DIR / 'openseek-7_jeopardy_answer_generation_all.json'),
    8: str(_DATA_DIR / 'openseek-8_kernel_generation.json'),
}

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id', type=int, required=True,
                        help='Task ID to evaluate, should be in [1, 8].')
    parser.add_argument('--max_input_length', type=int, default=128_000,
                        help='Maximum input length for the model (tokens).')
    parser.add_argument('--log_path_prefix', type=str,
                        default=str(_DATA_DIR.parent / 'outputs') + '/',
                        help='Prefix path to save the evaluation logs.')
    parser.add_argument('--tokenizer_path', type=str,
                        default='/root/autodl-tmp/qwen3-4b',
                        help='Path to Qwen3-4B tokenizer.')
    parser.add_argument('--num_samples', type=int, default=1,
                        help='Number of samples for voting (1 = single pass).')
    parser.add_argument('--max_examples', type=int, default=100,
                        help='Maximum number of ICL examples to consider.')
    parser.add_argument('--enable_alignment', action='store_true', default=True,
                        help='Enable output structure alignment (default: True).')
    parser.add_argument('--no_alignment', action='store_true',
                        help='Disable structure alignment.')
    args = parser.parse_args()
    return args


def evaluate(
    task_id: int, 
    qwen_tokenizer: AutoTokenizer,
    max_input_length: int = 128_000,
    log_path_prefix: str = './outputs/',
    num_samples: int = 1,
    max_examples: int = 100,
    enable_alignment: bool = True,
) -> float:
    """Run evaluation on a single task."""
    
    assert task_id in list(range(1, 9)), \
        f"task_id should be in [1, 8], but got {task_id}."
    
    task_file = TASK_FILES[task_id]
    with open(task_file, 'r') as f:
        task_dict = json.load(f)
    
    task_name = task_dict['task_name']
    task_description = task_dict['Definition'][0]
    
    # Use all available examples (up to max_examples)
    icl_examples = task_dict['examples'][:max_examples]
    test_samples = task_dict['test_samples']
    
    # Create output file with versioning
    version = 1
    output_file = f'{log_path_prefix}openseek-{task_id}-v{version}.jsonl'
    output_path = os.path.dirname(output_file)
    os.makedirs(output_path, exist_ok=True)
    while os.path.exists(output_file):
        version += 1
        output_file = f'{log_path_prefix}openseek-{task_id}-v{version}.jsonl'
    
    # Initialize output file
    with open(output_file, 'w') as f:
        pass
    
    # Pre-select examples once (same examples used for all test samples)
    first_sample_input = test_samples[0]['input'] if test_samples else ""
    examples_str = select_examples(
        icl_examples, task_description, first_sample_input,
        task_id=task_id
    )
    
    print(f"Task {task_id}: Selected {examples_str.count('<label>')} examples")
    
    # Track alignment statistics
    aligned_count = 0
    total_count = 0
    
    for test_sample in tqdm(test_samples, desc=f'Evaluation on Task {task_id}: {task_name}'):
        test_record = dict()
        
        test_sample_id = test_sample['id']
        test_record['test_sample_id'] = test_sample_id
        
        text2annotate = test_sample['input']
        
        # Build task-aware prompt
        prompt = build_prompt(task_description, text2annotate, task_id=task_id)
        
        # Insert examples into prompt
        input_prompt = prompt.replace("[[EXAMPLES]]\n\n", examples_str + '\n\n')
        
        # Check total length
        tokenized_input = qwen_tokenizer(input_prompt, return_tensors="pt")
        input_length = tokenized_input['input_ids'].shape[1]
        
        if input_length > max_input_length:
            print(f"Warning: Input length {input_length} exceeds max {max_input_length}, "
                  f"truncating examples for sample {test_sample_id}")
            from method import select_examples_simple
            tokenizer_local = qwen_tokenizer
            examples_str = select_examples_simple(
                icl_examples, task_description, text2annotate,
                tokenizer_local, 
                max_context_length=max_input_length - 500,
                prompt_template_length=600
            )
            input_prompt = prompt.replace("[[EXAMPLES]]\n\n", examples_str + '\n\n')
        
        # Annotate
        prediction = annotate(input_prompt, num_samples=num_samples)
        
        # Structure alignment post-processing
        if enable_alignment:
            aligned_prediction = align_prediction(task_id, test_sample_id, prediction)
            if aligned_prediction != prediction:
                aligned_count += 1
            prediction = aligned_prediction
        
        test_record['prediction'] = prediction
        total_count += 1
        
        with open(output_file, 'a') as f:
            f.write(json.dumps(test_record) + '\n')
    
    print(f"Results saved to {output_file}")
    return output_file


if __name__ == '__main__':
    args = parser_args()
    
    # Load tokenizer
    print(f"Loading tokenizer from {args.tokenizer_path}")
    qwen_tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path, 
        trust_remote_code=True
    )
    
    # Determine alignment setting
    enable_alignment = args.enable_alignment and not args.no_alignment
    if enable_alignment:
        try:
            from ctx_compress import align_output
            # Verify package works
            _ = align_output(1, "dummy", "")
        except ImportError:
            enable_alignment = False
    
    # Run evaluation
    evaluate(
        args.task_id, 
        qwen_tokenizer, 
        args.max_input_length, 
        args.log_path_prefix,
        args.num_samples,
        args.max_examples,
        enable_alignment,
    )
