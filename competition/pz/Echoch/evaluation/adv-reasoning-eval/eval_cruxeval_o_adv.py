import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import argparse
import json
import os
import glob # For finding checkpoint directories
from datetime import datetime

# New task description
TASK_DESCRIPTION_CODE_OUTPUT_PREDICTION = """You are given a Python function and an assertion containing an input to the function. Complete the assertion with a literal (no unsimplified expressions, no function calls) containing the output when executing the provided code on the given input, even if the function is incorrect or incomplete. Do NOT output any extra information. Execute the program step by step before arriving at an answer, and provide the full assertion with the correct output in [ANSWER] and [/ANSWER] tags, following the examples."""

def doc_to_text_code_output_prediction(sample, use_messy_cot=False):
    """
    Constructs the text input for the code output prediction task.

    Args:
        sample (dict): A dictionary containing the dataset fields.
                       Required fields: 'code', 'input', 'output', 'cot', 
                                   'messy_cot', 'messy_cot_output'.
        use_messy_cot (bool): If True, uses messy_cot and messy_cot_output.
                              Otherwise, uses cot and output.

    Returns:
        str: The formatted text string.
    """
    code = sample.get('code', '')
    fn_input = sample.get('input', '') # This is the input provided to function f

    if use_messy_cot:
        cot = sample.get('messy_cot', '')
        # messy_cot_output is the (potentially incorrect) output derived from messy_cot
        predicted_output = sample.get('messy_cot_output', '') 
    else:
        cot = sample.get('cot', '')
        # output is the correct (or target) output when the code is executed on fn_input
        predicted_output = sample.get('output', '')

    # Ensure input and output are correctly represented as strings in the assertion
    def format_for_assertion(val):
        if val is None:
            return "None"
        
        # Check if val is a string and if it's not already "quoted"
        if isinstance(val, str) and \
           not (val.startswith("'") and val.endswith("'")) and \
           not (val.startswith('"') and val.endswith('"')):
            try:
                # For simple strings, repr() is usually the best way to get a literal.
                # e.g., repr("it's") -> "'it\\'s'"
                return repr(val)
            except: # Fallback if repr() fails (e.g., for some custom string-like object)
                # Corrected line to avoid SyntaxError with backslash in f-string expression
                escaped_val = val.replace("'", "\\'") # Escape single quotes
                return f"'{escaped_val}'"
        
        # For non-strings (int, float, bool, list, dict) or strings already quoted,
        # or if the above string handling wasn't applicable.
        # str() is generally fine for numbers/bools.
        # For lists/dicts, repr() is often better for a literal representation.
        # If the dataset consistently provides Python objects, using repr() more broadly
        # might be more robust for creating literals.
        # However, sticking to minimal change for the reported bug.
        if isinstance(val, (list, dict)): # repr() is better for these to look like literals
            return repr(val)
        return str(val)


    fn_input_str = format_for_assertion(fn_input)
    predicted_output_str = format_for_assertion(predicted_output)
    
    text = f"{TASK_DESCRIPTION_CODE_OUTPUT_PREDICTION}\n\n"
    text += "Function Code:\n```python\n"
    text += f"{code}\n```\n\n"
    text += f"Given Input for the function:\n{fn_input_str}\n\n" # Clearly state this is the function's input
    text += "Think step by step (reasoning process to predict the output):\n"
    text += f"{cot}\n\n"
    text += "[ANSWER]\n"
    # The assertion always uses the given fn_input and the (CoT-based) predicted output
    text += f"assert f({fn_input_str}) == {predicted_output_str}\n" 
    text += "[/ANSWER]"
    return text

def calculate_perplexity_batched(texts, model, tokenizer, device, max_len):
    """
    Calculates perplexity for a batch of texts.
    (This function remains unchanged)
    """
    if not texts:
        return []
    try:
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
            return_attention_mask=True
        )
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            batch_size, seq_length, vocab_size = shift_logits.shape
            flat_logits = shift_logits.view(-1, vocab_size)
            flat_labels = shift_labels.view(-1)
            token_losses = loss_fct(flat_logits, flat_labels)
            token_losses = token_losses.view(batch_size, seq_length)
            shift_attention_mask = attention_mask[..., 1:].contiguous().float()
            masked_token_losses = token_losses * shift_attention_mask
            sum_loss_per_sequence = masked_token_losses.sum(dim=1)
            num_tokens_per_sequence = shift_attention_mask.sum(dim=1)
            mean_loss_per_sequence = torch.zeros_like(sum_loss_per_sequence)
            valid_sequence_mask = num_tokens_per_sequence > 0
            mean_loss_per_sequence[valid_sequence_mask] = sum_loss_per_sequence[valid_sequence_mask] / num_tokens_per_sequence[valid_sequence_mask]
        perplexities = torch.exp(mean_loss_per_sequence)
        return perplexities.cpu().tolist()
    except Exception as e:
        tqdm.write(f"Error calculating batched perplexity: {e}. Texts: {str(texts)[:200]}...")
        return [float('inf')] * len(texts)

def evaluate_checkpoint(checkpoint_path, dataset_obj, batch_size, device):
    """
    Evaluates a single model checkpoint on the code output prediction dataset.
    """
    print(f"\n--- Evaluating checkpoint: {checkpoint_path} (Code Output Prediction Evaluation) ---")
    checkpoint_result_data = {
        "checkpoint_path": checkpoint_path,
        "status": "pending",
        "error_message": None,
        "evaluation_timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "evaluation_parameters_used": {
            "device_used": str(device),
            "actual_batch_size": batch_size,
            "max_sequence_length": None
        },
        "results": {
            "total_samples_in_dataset": len(dataset_obj),
            "evaluated_samples_count": 0,
            "passed_samples": 0,
            "accuracy_percent": 0.0
        }
    }
    # Required fields for this task
    required_fields_for_task = ['code', 'input', 'output', 'cot', 'messy_cot', 'messy_cot_output']


    try:
        # Load tokenizer for the current checkpoint
        print(f"Loading tokenizer from {checkpoint_path}...")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
                print(f"Tokenizer's pad_token was not set, set to eos_token: {tokenizer.eos_token}")
            else:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                print(f"Tokenizer's pad_token and eos_token were not set. Added a new pad_token: '[PAD]'")
        
        # Load model for the current checkpoint
        print(f"Loading model from {checkpoint_path}...")
        model = AutoModelForCausalLM.from_pretrained(checkpoint_path, trust_remote_code=True)
        
        if len(tokenizer) > model.config.vocab_size:
            print(f"Resizing model token embeddings from {model.config.vocab_size} to {len(tokenizer)}")
            model.resize_token_embeddings(len(tokenizer))
        
        model.to(device)
        model.eval()
        print("Model and tokenizer for checkpoint loaded successfully.")

        max_len = tokenizer.model_max_length
        if max_len is None and hasattr(model.config, 'max_position_embeddings') and model.config.max_position_embeddings is not None:
            max_len = model.config.max_position_embeddings
        elif max_len is None:
            max_len = 1024 # Default to 1024 for potentially longer code snippets
        print(f"Using max_len: {max_len} for tokenization for this checkpoint.")
        checkpoint_result_data["evaluation_parameters_used"]["max_sequence_length"] = max_len

        # Evaluation loop for this checkpoint
        passed_samples_ckpt = 0
        evaluated_samples_count_ckpt = 0
        
        for i in tqdm(range(0, len(dataset_obj), batch_size), desc=f"Evaluating {os.path.basename(checkpoint_path)} (Code Output Prediction)"):
            batch_slice = dataset_obj[i:i+batch_size]
            current_batch_texts_original = []
            current_batch_texts_messy = []
            
            is_dict_of_lists = isinstance(batch_slice, dict) and all(isinstance(v, list) for v in batch_slice.values())
            num_samples_in_slice = 0
            if is_dict_of_lists:
                if 'code' in batch_slice: 
                    num_samples_in_slice = len(batch_slice['code'])
            elif isinstance(batch_slice, list):
                num_samples_in_slice = len(batch_slice)

            for k in range(num_samples_in_slice):
                sample = {}
                if is_dict_of_lists:
                    for key_field in batch_slice.keys():
                        if k < len(batch_slice[key_field]):
                             sample[key_field] = batch_slice[key_field][k]
                        else:
                             sample[key_field] = None 
                else: 
                    sample = batch_slice[k]

                # Check for missing essential fields for this code output prediction task
                if not all(key in sample and sample[key] is not None for key in required_fields_for_task):
                    # Identify missing fields for debugging
                    missing_or_none_fields = [f for f in required_fields_for_task if f not in sample or sample[f] is None]
                    tqdm.write(f"Skipping sample (original index approx {i+k}) due to missing essential fields for code output prediction task: {missing_or_none_fields}.")
                    continue
                
                current_batch_texts_original.append(doc_to_text_code_output_prediction(sample, use_messy_cot=False))
                current_batch_texts_messy.append(doc_to_text_code_output_prediction(sample, use_messy_cot=True))

            if not current_batch_texts_original:
                continue

            ppls_original_batch = calculate_perplexity_batched(current_batch_texts_original, model, tokenizer, device, max_len)
            ppls_messy_batch = calculate_perplexity_batched(current_batch_texts_messy, model, tokenizer, device, max_len)

            for k_idx, (ppl_original, ppl_messy) in enumerate(zip(ppls_original_batch, ppls_messy_batch)):
                evaluated_samples_count_ckpt += 1
                if ppl_original == float('inf') or ppl_messy == float('inf'):
                    tqdm.write(f"PPL calculation failed for a code output prediction sample (checkpoint: {checkpoint_path}).")
                elif ppl_original < ppl_messy:
                    passed_samples_ckpt += 1
            
        checkpoint_result_data["results"]["evaluated_samples_count"] = evaluated_samples_count_ckpt
        checkpoint_result_data["results"]["passed_samples"] = passed_samples_ckpt
        if evaluated_samples_count_ckpt > 0:
            accuracy_ckpt = (passed_samples_ckpt / evaluated_samples_count_ckpt) * 100
            checkpoint_result_data["results"]["accuracy_percent"] = float(f"{accuracy_ckpt:.2f}")
        
        checkpoint_result_data["status"] = "success"
        print(f"Code output prediction evaluation successful for checkpoint: {checkpoint_path}. Accuracy: {checkpoint_result_data['results']['accuracy_percent']:.2f}%")

    except Exception as e:
        error_msg = f"Failed to evaluate checkpoint {checkpoint_path} (Code Output Prediction): {e}"
        print(error_msg)
        checkpoint_result_data["status"] = "failed"
        checkpoint_result_data["error_message"] = error_msg
    
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return checkpoint_result_data

def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM checkpoints for code output prediction ability using CoT perplexity.")
    model_spec_group = parser.add_mutually_exclusive_group(required=True)
    model_spec_group.add_argument("--models_base_dir", type=str, help="Base directory containing model checkpoint folders (each ending with _hf).")
    model_spec_group.add_argument("--model_path", type=str, help="Path to a single model checkpoint directory to evaluate.")
    
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save the JSON results file.")
    parser.add_argument("--dataset_name_or_path", type=str, default="EssentialAI/cruxeval_o_adv", help="Name or path of the dataset.")
    parser.add_argument("--dataset_config_name", type=str, default="default", help="Configuration name of the dataset.")
    parser.add_argument("--dataset_split", type=str, default="test", help="Dataset split to use (e.g., test, validation).")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation.")
    parser.add_argument("--cuda_device", type=str, default="cuda:0", help="CUDA device to use (e.g., 'cuda:0', 'cuda:1'). Default is 'cuda:0'.")
    
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device(args.cuda_device)
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    print(f"Loading dataset '{args.dataset_name_or_path}', config '{args.dataset_config_name}', split '{args.dataset_split}' for code output prediction evaluation...")
    try:
        dataset_obj = load_dataset(args.dataset_name_or_path, name=args.dataset_config_name, split=args.dataset_split)
        print(f"Dataset loaded successfully, containing {len(dataset_obj)} samples.")
    except Exception as e:
        print(f"Failed to load dataset: {e}. Exiting.")
        return

    checkpoint_paths_to_evaluate = []
    base_name_for_output_file = ""
    evaluation_mode = ""

    if args.model_path:
        if not os.path.isdir(args.model_path):
            print(f"Error: Provided --model_path '{args.model_path}' is not a valid directory. Exiting.")
            return
        checkpoint_paths_to_evaluate.append(args.model_path)
        base_name_for_output_file = os.path.basename(args.model_path.rstrip('/'))
        evaluation_mode = "single_checkpoint"
        print(f"Evaluating single checkpoint (Code Output Prediction): {args.model_path}")
    elif args.models_base_dir:
        if not os.path.isdir(args.models_base_dir):
            print(f"Error: Provided --models_base_dir '{args.models_base_dir}' is not a valid directory. Exiting.")
            return
        checkpoint_pattern = os.path.join(args.models_base_dir, '*_hf')
        all_matching_paths = glob.glob(checkpoint_pattern)
        checkpoint_paths_to_evaluate = sorted([p for p in all_matching_paths if os.path.isdir(p)])
        base_name_for_output_file = os.path.basename(args.models_base_dir.rstrip('/'))
        evaluation_mode = "multi_checkpoint_directory"
        if not checkpoint_paths_to_evaluate:
            print(f"No checkpoint directories found matching pattern '{checkpoint_pattern}' in '{args.models_base_dir}'. Exiting.")
            return
        print(f"Found {len(checkpoint_paths_to_evaluate)} checkpoints in {args.models_base_dir} for code output prediction evaluation: {checkpoint_paths_to_evaluate}")

    all_results = []
    for checkpoint_path in checkpoint_paths_to_evaluate:
        result = evaluate_checkpoint(checkpoint_path, dataset_obj, args.batch_size, device)
        all_results.append(result)

    dataset_name_for_file = os.path.basename(args.dataset_name_or_path.rstrip('/')) if os.path.exists(args.dataset_name_or_path) else args.dataset_name_or_path.replace("/", "_")
    current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Add "CodeOutputPredEval" to filename for distinction
    output_filename = f"{base_name_for_output_file}-{dataset_name_for_file}-CodeOutputPredEval-{current_time_str}.json"
    output_filepath = os.path.join(args.output_dir, output_filename)

    overall_info = {
        "evaluation_type": "code_output_prediction_perplexity", # Updated evaluation type
        "evaluation_mode": evaluation_mode,
        "dataset_info": {
            "path_or_name": args.dataset_name_or_path,
            "config_name": args.dataset_config_name,
            "split": args.dataset_split
        },
        "evaluation_run_timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "global_batch_size": args.batch_size,
        "specified_cuda_device": args.cuda_device if torch.cuda.is_available() else "N/A (CUDA not available)",
        "number_of_checkpoints_found": len(checkpoint_paths_to_evaluate),
        "number_of_checkpoints_processed": len(all_results)
    }
    if args.model_path:
        overall_info["evaluated_model_path"] = args.model_path
    else: 
        overall_info["models_base_directory"] = args.models_base_dir

    final_json_output = {
        "overall_evaluation_info": overall_info,
        "checkpoint_evaluations": all_results
    }

    try:
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(final_json_output, f, ensure_ascii=False, indent=4)
        print(f"\nAll code output prediction evaluations complete. Results saved to: {output_filepath}")
    except IOError as e:
        print(f"Error saving final results to JSON: {e}")

if __name__ == "__main__":
    main()
