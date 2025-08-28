import random
import os
import argparse
import time
import sys
# from vllm import LLM, SamplingParams
import sglang as sgl
from datetime import datetime, timedelta
from tqdm import tqdm
import json

import torch
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM

from evaluate_final import evaluate
from utils import set_seed, load_jsonl, save_jsonl, construct_prompt, load_data
from parser import *
from trajectory import *
from python_executor import PythonExecutor
from model_utils import load_hf_lm_and_tokenizer, generate_completions


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_names", default="gsm8k,math", type=str)
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument("--model_name_or_path", default="gpt-4", type=str)
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--prompt_type", default="tool-integrated", type=str)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--num_test_sample", default=-1, type=int)  # -1 for full data
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--n_sampling", default=1, type=int)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--max_tokens_per_call", default=4096, type=int)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--use_sglang", action="store_true")
    parser.add_argument("--save_outputs", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--use_safetensors", action="store_true")
    parser.add_argument("--num_shots", type=int, default=0)
    parser.add_argument(
        "--apply_chat_template",
        action="store_true",
        help="Apply chat template to prompt.",
    )
    parser.add_argument("--pipeline_parallel_size", type=int, default=1)
    parser.add_argument("--dp_size", type=int, default=1)
    parser.add_argument(
        "--adapt_few_shot",
        action="store_true",
        help="Few shot for multiple-choice questions, zero shot for others.",
    )
    # Distributed training arguments
    parser.add_argument("--distributed", action="store_true", help="Enable distributed evaluation")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    parser.add_argument("--world_size", type=int, default=1, help="World size for distributed training")
    parser.add_argument("--dist_url", default="env://", help="URL used to set up distributed training")
    parser.add_argument("--dist_backend", default="nccl", help="Distributed backend")
    
    args = parser.parse_args()
    args.top_p = (
        1 if args.temperature == 0 else args.top_p
    )  # top_p must be 1 when using greedy sampling (vllm)
    return args


def prepare_data(data_name, args, rank=0, world_size=1):
    examples = load_data(data_name, args.split, args.data_dir)

    # sample `num_test_sample` from dataset
    if args.num_test_sample > 0:
        # examples = random.sample(examples, min(args.num_test_sample, len(examples)))
        examples = examples[: args.num_test_sample]

    # shuffle
    if args.shuffle:
        random.seed(datetime.now().timestamp())
        random.shuffle(examples)

    # select start and end
    examples = examples[args.start : len(examples) if args.end == -1 else args.end]

    # Split data for distributed processing
    if args.distributed and world_size > 1:
        # Calculate chunk size and get this rank's chunk
        chunk_size = len(examples) // world_size
        remainder = len(examples) % world_size

        # Distribute the remainder across the first few ranks
        start_idx = rank * chunk_size + min(rank, remainder)
        end_idx = start_idx + chunk_size + (1 if rank < remainder else 0)

        # Get this rank's examples
        examples = examples[start_idx:end_idx]

        if rank == 0:
            print(f"Distributed processing: {len(examples)} examples per GPU (total: {world_size} GPUs)")

    # get out_file name
    dt_string = datetime.now().strftime("%m-%d_%H-%M")
    model_name = "/".join(args.model_name_or_path.split("/")[-2:])
    out_file_prefix = f"{args.split}_{args.prompt_type}_{args.num_test_sample}_seed{args.seed}_t{args.temperature}"
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        output_dir = f"outputs/{output_dir}"

    # Add rank to output file name for distributed processing
    if args.distributed and world_size > 1:
        out_file = f"{output_dir}/{data_name}/{out_file_prefix}_s{args.start}_e{args.end}_rank{rank}.jsonl"
    else:
        out_file = f"{output_dir}/{data_name}/{out_file_prefix}_s{args.start}_e{args.end}.jsonl"

    os.makedirs(f"{output_dir}/{data_name}", exist_ok=True)

    # load all processed samples
    processed_samples = []
    if not args.overwrite:
        processed_files = [
            f
            for f in os.listdir(f"{output_dir}/{data_name}/")
            if f.endswith(".jsonl") and f.startswith(out_file_prefix)
        ]
        for f in processed_files:
            processed_samples.extend(
                list(load_jsonl(f"{output_dir}/{data_name}/{f}"))
            )

    # dedepulicate
    processed_samples = {sample["idx"]: sample for sample in processed_samples}
    processed_idxs = list(processed_samples.keys())
    processed_samples = list(processed_samples.values())
    examples = [example for example in examples if example["idx"] not in processed_idxs]
    return examples, processed_samples, out_file


def setup(args):
    # Initialize distributed training if enabled
    rank = 0
    world_size = 1

    if args.distributed:
        if args.local_rank == -1:  # Not launched with torch.distributed.launch
            if "LOCAL_RANK" in os.environ:
                args.local_rank = int(os.environ["LOCAL_RANK"])
            else:
                args.local_rank = 0

        # Initialize the distributed environment
        torch.distributed.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            timeout=timedelta(minutes=30)
        )

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(args.local_rank)

        if rank == 0:
            print(f"Initialized distributed training with {world_size} GPUs")

    # load model
    available_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")
    data_list = args.data_names.split(",")
    need_eval_data_list = []

    if not args.overwrite:
        for data_name in data_list:
            out_prefix = f"{args.split}_{args.prompt_type}_{args.num_test_sample}_seed{args.seed}_t{args.temperature}"

            # Check for distributed output files
            if args.distributed and world_size > 1:
                out_file = f"{args.output_dir}/{data_name}/{out_prefix}_s{args.start}_e{args.end}_rank{rank}.jsonl"
            else:
                out_file = f"{args.output_dir}/{data_name}/{out_prefix}_s{args.start}_e{args.end}.jsonl"

            out_metric_json = out_file.replace(".jsonl", f"_metrics.json")

            if os.path.exists(out_metric_json) and not args.distributed:
                if rank == 0:
                    print(f"Skipping {data_name} because {out_metric_json} already exists.")
                continue
            else:
                need_eval_data_list.append(data_name)

        if len(need_eval_data_list) == 0:
            if rank == 0:
                print("All datasets already evaluated. Exiting.")
            if args.distributed:
                dist.destroy_process_group()
            exit(0)
        data_list = need_eval_data_list

    # Load model based on whether we're using vLLM or SGLang
    if args.use_vllm:
        from vllm import LLM, SamplingParams
        # For vLLM, we can use tensor parallelism directly
        llm = LLM(
            model=args.model_name_or_path,
            tensor_parallel_size=len(available_gpus) // args.pipeline_parallel_size,
            pipeline_parallel_size=args.pipeline_parallel_size,
            trust_remote_code=True,
            gpu_memory_utilization=0.85,
            max_model_len=args.max_tokens_per_call,
            max_num_seqs=32,
            enforce_eager=True,
        )
        tokenizer = None
        if args.apply_chat_template:
            tokenizer = AutoTokenizer.from_pretrained(
                args.model_name_or_path, trust_remote_code=True
            )
    elif args.use_sglang:
        llm = sgl.Engine(
            model_path=args.model_name_or_path, 
            trust_remote_code=True,
            max_running_requests=32,
            tp_size=len(available_gpus) // args.pipeline_parallel_size,
            mem_fraction_static=0.85,
            max_micro_batch_size=16,
            dp_size=args.dp_size,
            log_requests=True,
            log_level="info"
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, trust_remote_code=True
        )
    else:
        # For regular HF models, each process loads its own model
        llm, tokenizer = load_hf_lm_and_tokenizer(
            model_name_or_path=args.model_name_or_path,
            load_in_half=True,
            use_fast_tokenizer=True,
            use_safetensors=args.use_safetensors,
        )

    # infer & eval
    results = []
    for data_name in data_list:
        result = main(llm, tokenizer, data_name, args, rank, world_size)
        results.append(result)

        # 添加同步点，确保所有rank处理完当前数据集
        if args.distributed and world_size > 1:
            dist.barrier()
            if rank == 0:
                print(f"All ranks finished processing dataset: {data_name}")

        # For rank 0, collect and merge results
        if args.distributed and world_size > 1 and rank == 0 and result is not None:
            # Collect and merge results from all ranks for this dataset
            all_results = []
            all_results.append(result)

            # Check for results from other ranks
            for r in range(1, world_size):
                out_prefix = f"{args.split}_{args.prompt_type}_{args.num_test_sample}_seed{args.seed}_t{args.temperature}"
                rank_out_file = f"{args.output_dir}/{data_name}/{out_prefix}_s{args.start}_e{args.end}_rank{r}.jsonl"
                rank_metric_json = rank_out_file.replace(".jsonl", f"_metrics.json")

                if os.path.exists(rank_metric_json):
                    try:
                        with open(rank_metric_json, "r") as f:
                            rank_result = json.load(f)
                            all_results.append(rank_result)
                    except:
                        print(f"Warning: Could not read results from rank {r}")

            # Merge results if we have all ranks
            if len(all_results) == world_size:
                # Collect all samples for merged evaluation
                all_samples = []
                for r in range(world_size):
                    rank_out_file = f"{args.output_dir}/{data_name}/{out_prefix}_s{args.start}_e{args.end}_rank{r}.jsonl"
                    if os.path.exists(rank_out_file):
                        rank_samples = list(load_jsonl(rank_out_file))
                        all_samples.extend(rank_samples)

                if all_samples:
                    # Re-evaluate merged samples
                    _, merged_result = evaluate(
                        samples=all_samples,
                        data_name=data_name,
                        prompt_type=args.prompt_type,
                        execute=True,
                    )

                    # Save merged results
                    merged_out_file = f"{args.output_dir}/{data_name}/{out_prefix}_s{args.start}_e{args.end}_merged.jsonl"
                    merged_metric_json = merged_out_file.replace(".jsonl", f"_metrics.json")

                    save_jsonl(all_samples, merged_out_file)
                    with open(merged_metric_json, "w") as f:
                        json.dump(merged_result, f, indent=4)

                    print(f"Merged results saved for {data_name}: {merged_result}")

    # Only rank 0 prints the final summary
    if not args.distributed or rank == 0:
        if results:
            # add "avg" result to data_list and results
            data_list.append("avg")
            valid_results = [r for r in results if r is not None and "acc" in r]
            if valid_results:
                avg_acc = sum([result["acc"] for result in valid_results]) / len(valid_results)
            else:
                avg_acc = 0
            results.append({"acc": avg_acc})

            # print all results
            pad = max([len(data_name) for data_name in data_list])
            print("\t".join(data_name.ljust(pad, " ") for data_name in data_list))
            print("\t".join([f"{result['acc']:.1f}".ljust(pad, " ") for result in results]))

    # Clean up distributed environment
    if args.distributed:
        try:
            dist.destroy_process_group()
        except:
            pass  # Ignore errors if already destroyed


def is_multi_choice(answer):
    for c in answer:
        if c not in ["A", "B", "C", "D", "E"]:
            return False
    return True


def main(llm, tokenizer, data_name, args, rank=0, world_size=1):
    examples, processed_samples, out_file = prepare_data(data_name, args, rank, world_size)
    
    # Only rank 0 prints general information if distributed
    if not args.distributed or rank == 0:
        print("=" * 50)
        print("data:", data_name, " ,remain samples:", len(examples))
        if len(examples) > 0:
            print(examples[0])

    # If no examples to process, return early
    if len(examples) == 0:
        if args.distributed:
            # Create empty metrics file to indicate this rank has completed
            empty_result = {
                "acc": 0.0,
                "num_samples": 0,
                "time_use_in_second": 0,
                "time_use_in_minite": "0:00"
            }
            with open(out_file.replace(".jsonl", f"_metrics.json"), "w") as f:
                json.dump(empty_result, f, indent=4)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return empty_result
        else:
            return None

    # init python executor
    if "pal" in args.prompt_type:
        executor = PythonExecutor(get_answer_expr="solution()")
    else:
        executor = PythonExecutor(get_answer_from_stdout=True)

    samples = []
    for example in tqdm(examples, total=len(examples), disable=args.distributed and rank != 0):
        idx = example["idx"]

        # parse question and answer
        example["question"] = parse_question(example, data_name)
        if example["question"] == "":
            continue
        gt_cot, gt_ans = parse_ground_truth(example, data_name)
        example["gt_ans"] = gt_ans
        full_prompt = construct_prompt(example, data_name, args)

        if idx == args.start and (not args.distributed or rank == 0):
            print(full_prompt)

        sample = {
            "idx": idx,
            "question": example["question"],
            "gt_cot": gt_cot,
            "gt": gt_ans,
            "prompt": full_prompt,
        }

        # add remain fields
        for key in [
            "level",
            "type",
            "unit",
            "solution_type",
            "choices",
            "solution",
            "ques_type",
            "ans_type",
            "answer_type",
            "dataset",
            "subfield",
            "filed",
            "theorem",
            "answer",
        ]:
            if key in example:
                sample[key] = example[key]
        samples.append(sample)

    # repeat n times
    input_prompts = [
        sample["prompt"] for sample in samples for _ in range(args.n_sampling)
    ]
    if args.apply_chat_template:
        input_prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt.strip()}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in input_prompts
        ]
    remain_prompts = input_prompts
    remain_prompts = [(i, prompt) for i, prompt in enumerate(remain_prompts)]
    end_prompts = []

    max_func_call = 1 if args.prompt_type in ["cot", "pal"] else 4

    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]

    if args.prompt_type in ["cot", "aquila_cot", "aquila_cot_few_shot"]:
        stop_words.append("\n\nQuestion:")
        stop_words.append("\nQuestion:")
    if args.prompt_type in ["pal", "tool-integrated", "jiuzhang_tora"]:
        stop_words.extend(["\n\n---", "```output"])
    elif args.prompt_type in ["wizard_zs", "platypus_fs"]:
        stop_words.extend(["Instruction", "Response"])
    elif "jiuzhang" in args.prompt_type:
        stop_words.append("\n\n## Question")
    elif "numina" in args.prompt_type:
        stop_words.append("\n### Problem")
    elif "pure" in args.prompt_type:
        stop_words.append("\n\n\n")
    elif "direct" in args.prompt_type:
        stop_words.append("[Question]")
    
    # start inference
    # measure time use
    start_time = time.time()
    for epoch in range(max_func_call):
        if not args.distributed or rank == 0:
            print("-" * 20, "Epoch", epoch)
        current_prompts = remain_prompts
        if len(current_prompts) == 0:
            break

        # get all outputs
        prompts = [item[1] for item in current_prompts]
        if args.use_vllm:
            from vllm import SamplingParams
            outputs = llm.generate(
                prompts,
                SamplingParams(
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_tokens_per_call,
                    n=1,
                    stop=stop_words,
                    stop_token_ids=(
                        [151645, 151643]
                        if "qwen2" in args.model_name_or_path.lower()
                        else None
                    ),
                ),
            )

            outputs = sorted(
                outputs, key=lambda x: int(x.request_id)
            )  # sort outputs by request_id
            outputs = [output.outputs[0].text for output in outputs]
        elif args.use_sglang:
            outputs = llm.generate(
                prompts,
                {
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "max_new_tokens": args.max_tokens_per_call,
                    "n": 1,
                    "stop": stop_words,
                    "stop_token_ids": (
                        [151645, 151643]
                        if "qwen2" in args.model_name_or_path.lower()
                        else [tokenizer.eos_token_id]
                    ),
                    "skip_special_tokens": False if 'aquila' in args.model_name_or_path.lower() else True,
                },
            )

            # print(outputs)
            # outputs = sorted(
                # outputs, key=lambda x: int(x.request_id)
            # )  # sort outputs by request_id
            outputs = [output['text'] for output in outputs]
        
        else:
            outputs = generate_completions(
                model=llm,
                tokenizer=tokenizer,
                prompts=prompts,
                max_new_tokens=args.max_tokens_per_call,
                batch_size=16,
                stop_id_sequences=stop_words,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=True,
            )

        assert len(outputs) == len(current_prompts)

        # process all outputs
        remain_prompts = []
        remain_codes = []
        for (i, query), output in zip(current_prompts, outputs):
            output = output.rstrip()
            query += output
            if args.prompt_type == "pal":
                remain_prompts.append((i, query))
                if "```python" in output:
                    output = extract_program(query)
                remain_codes.append(output)
            elif args.prompt_type == "cot":
                end_prompts.append((i, query))
            elif "boxed" not in output and output.endswith("```"):
                program = extract_program(query)
                remain_prompts.append((i, query))
                remain_codes.append(program)
            else:
                end_prompts.append((i, query))

        # execute the remain prompts
        remain_results = executor.batch_apply(remain_codes)
        for k in range(len(remain_prompts)):
            i, query = remain_prompts[k]
            res, report = remain_results[k]
            exec_result = res if res else report
            if "pal" in args.prompt_type:
                exec_result = "\\boxed{" + exec_result + "}"
            exec_result = f"\n```output\n{exec_result}\n```\n"
            query += exec_result
            # not end
            if epoch == max_func_call - 1:
                query += "\nReach max function call limit."
            remain_prompts[k] = (i, query)

    # unsolved samples
    if not args.distributed or rank == 0:
        print("Unsolved samples:", len(remain_prompts))
    end_prompts.extend(remain_prompts)
    # sort by idx
    end_prompts = sorted(end_prompts, key=lambda x: x[0])

    # remove input_prompt from end_prompt
    codes = []
    assert len(input_prompts) == len(end_prompts)
    for i in range(len(input_prompts)):
        _, end_prompt = end_prompts[i]
        code = end_prompt.split(input_prompts[i])[-1].strip()
        for stop_word in stop_words:
            if stop_word in code:
                code = code.split(stop_word)[0].strip()
        codes.append(code)

    # extract preds
    results = [
        run_execute(executor, code, args.prompt_type, data_name) for code in codes
    ]
    time_use = time.time() - start_time

    # put results back to examples
    all_samples = []
    for i, sample in enumerate(samples):
        code = codes[i * args.n_sampling : (i + 1) * args.n_sampling]
        result = results[i * args.n_sampling : (i + 1) * args.n_sampling]
        preds = [item[0] for item in result]
        reports = [item[1] for item in result]
        for j in range(len(preds)):
            if sample["gt"] in ["A", "B", "C", "D", "E"] and preds[j] not in [
                "A",
                "B",
                "C",
                "D",
                "E",
            ]:
                preds[j] = choice_answer_clean(code[j])
            elif is_multi_choice(sample["gt"]) and not is_multi_choice(preds[j]):
                # remove any non-choice char
                preds[j] = "".join(
                    [c for c in preds[j] if c in ["A", "B", "C", "D", "E"]]
                )

        sample.pop("prompt")
        sample.update({"code": code, "pred": preds, "report": reports})
        all_samples.append(sample)

    # add processed samples
    all_samples.extend(processed_samples)
    all_samples, result_json = evaluate(
        samples=all_samples,
        data_name=data_name,
        file_path=out_file,
        execute=True,
    )

    # # save outputs
    # if len(processed_samples) < len(all_samples) and args.save_outputs:
    #     save_jsonl(all_samples, out_file)

    result_json["time_use_in_second"] = time_use
    result_json["time_use_in_minite"] = (
        f"{int(time_use // 60)}:{int(time_use % 60):02d}"
    )

    print(out_file)
    with open(
        out_file.replace(".jsonl", f"_metrics.json"), "w"
    ) as f:
        json.dump(result_json, f, indent=4)

    # Release GPU resources for this rank after processing
    if args.distributed and torch.cuda.is_available():
        torch.cuda.empty_cache()
        if rank == 0:
            print(f"Rank {rank}: Completed processing {data_name}, released GPU resources")
        
    return result_json


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    
    # Handle distributed training initialization
    if args.distributed:
        # If using torchrun or torch.distributed.launch, LOCAL_RANK is set automatically
        if "LOCAL_RANK" in os.environ:
            args.local_rank = int(os.environ["LOCAL_RANK"])

        # Set device before initializing distributed environment
        if torch.cuda.is_available():
            torch.cuda.set_device(args.local_rank)
            
        # Initialize with timeout to avoid hanging
        os.environ["NCCL_BLOCKING_WAIT"] = "0"  # Non-blocking wait
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"  # Enable async error handling
        
        try:
            # Run the main setup function
            setup(args)
        except Exception as e:
            print(f"Error in process {args.local_rank}: {str(e)}")
            # Try to clean up resources
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            try:
                if torch.distributed.is_initialized():
                    torch.distributed.destroy_process_group()
            except:
                pass
    else:
        # Run the main setup function
        setup(args)
