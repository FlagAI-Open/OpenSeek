from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse

import requests
from tqdm import tqdm

from method import (
    InferenceConfig,
    assess_task4_prediction,
    assess_task8_prediction,
    annotate,
    annotate_detailed,
    build_task_solver_debug_prompt,
    build_task_solver_repair_prompt,
    build_task_solver_synthesis_prompt,
    build_prompt,
    build_task4_review_prompt,
    build_task4_retry_prompt,
    build_task8_repair_prompt,
    build_task8_rewrite_prompt,
    execute_solver_code,
    extract_solver_code,
    load_tokenizer,
    postprocess_prediction,
    select_examples,
    validate_solver_code,
)
from task7_semantic_tag_retrieval import ANSWER_RULE_CHOICES, predict_task7_semantic_mmr


TASK_FILES: Dict[int, str] = {
    1: "openseek-1_closest_integers.json",
    2: "openseek-2_count_nouns_verbs.json",
    3: "openseek-3_collatz_conjecture.json",
    4: "openseek-4_conala_concat_strings.json",
    5: "openseek-5_semeval_2018_task1_tweet_sadness_detection.json",
    6: "openseek-6_mnli_same_genre_classification.json",
    7: "openseek-7_jeopardy_answer_generation_all.json",
    8: "openseek-8_kernel_generation.json",
}

DEFAULT_REQUEST_RETRIES = 2
DEFAULT_RETRY_BACKOFF_SECONDS = 1.5


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LongContext-ICL-Annotation pipeline.")
    parser.add_argument("--task-id", type=int, choices=range(1, 9), help="Run a single task id in [1,8].")
    parser.add_argument("--all-tasks", action="store_true", help="Run all 8 tasks in one command.")
    parser.add_argument(
        "--backend",
        type=str,
        choices=("vllm", "flagscale", "llamacpp"),
        default="flagscale",
        help="Inference backend. Default starts the local FlagScale serving stack.",
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="../data/raw",
        help="Primary data directory (can be empty initially).",
    )
    parser.add_argument(
        "--fallback-data-dir",
        type=str,
        default="../third_party/LongContext-ICL-Annotation/data",
        help="Fallback official dataset directory.",
    )
    parser.add_argument("--output-dir", type=str, default="../outputs/submissions", help="Directory for jsonl predictions.")

    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default="../models/Qwen3-4B",
        help="Tokenizer path used for context-length budgeting.",
    )
    parser.add_argument("--max-input-tokens", type=int, default=60000, help="Max prompt tokens.")
    parser.add_argument("--reserved-generation-tokens", type=int, default=2048, help="Reserved output tokens.")
    parser.add_argument("--min-context-tokens", type=int, default=30000, help="Target minimum ICL context for tasks 1-7.")
    parser.add_argument("--min-context-tokens-task8", type=int, default=16000, help="Target minimum ICL context for task 8.")
    parser.add_argument(
        "--official-min-context",
        action="store_true",
        help="Force answer-generating calls to keep the official minimum ICL context instead of task-specific short contexts.",
    )
    parser.add_argument(
        "--context-audit-dir",
        type=str,
        default="",
        help="Optional directory for context audit JSONL files. Defaults to each task output directory.",
    )
    parser.add_argument(
        "--cache-friendly-context",
        action="store_true",
        help=(
            "Experimental speed mode for dynamic long-context tasks: reuse one fixed ICL examples prefix "
            "within a task/mode so vLLM prefix caching can hit more tokens. Disabled by default because it "
            "trades off per-sample retrieval relevance."
        ),
    )
    parser.add_argument(
        "--cache-prefix-context-tokens",
        type=int,
        default=0,
        help=(
            "Experimental hybrid speed mode for dynamic long-context tasks. If >0, build this many "
            "tokens of fixed shared ICL prefix, then append per-sample dynamic examples for the remaining "
            "minimum-context budget. This preserves a long cacheable prefix while keeping relevant examples near the sample."
        ),
    )

    parser.add_argument("--api-base", type=str, default="http://127.0.0.1:2026", help="OpenAI-compatible base URL.")
    parser.add_argument("--model-name", type=str, default="Qwen3-4B", help="Model id exposed by the inference service.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-new-tokens-task8", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Optional Qwen3 thinking switch. Default leaves the server behavior unchanged.",
    )
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--timeout-task8", type=float, default=600.0)
    parser.add_argument("--client-concurrency", type=int, default=1, help="Concurrent client requests.")
    parser.add_argument(
        "--task4-experiment",
        type=str,
        choices=("baseline", "prompt", "structured_examples", "self_refine", "gated_retry", "prompt_gated_retry", "solver_synthesis", "solver_debug_loop"),
        default="solver_debug_loop",
        help="Task4-only experiment switch.",
    )
    parser.add_argument(
        "--task8-experiment",
        type=str,
        choices=("baseline", "compact", "repair_pass"),
        default="baseline",
        help="Task8-only experiment switch.",
    )
    parser.add_argument("--task8-max-samples", type=int, default=0, help="Limit Task8 test samples for quick experiments; 0 means all.")
    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=0,
        help="Limit test samples for smoke runs on any task; 0 means all.",
    )
    parser.add_argument("--task8-repair-retries", type=int, default=1)
    parser.add_argument("--task8-rewrite-retries", type=int, default=1)
    parser.add_argument("--task4-review-max-new-tokens", type=int, default=128)
    parser.add_argument("--task4-gate-retries", type=int, default=2)
    parser.add_argument(
        "--task2-experiment",
        type=str,
        choices=("baseline", "context_rerank", "noun_biased", "noun_rules", "bucket_balanced"),
        default="context_rerank",
        help="Task2-only experiment switch.",
    )
    parser.add_argument(
        "--task5-experiment",
        type=str,
        choices=("baseline", "strict_dynamic", "balanced_dynamic", "conservative_dynamic"),
        default="baseline",
        help="Task5-only experiment switch.",
    )
    parser.add_argument(
        "--task6-experiment",
        type=str,
        choices=(
            "baseline",
            "genre_dynamic",
            "genre_conservative",
            "genre_balanced",
            "genre_balanced_conservative",
            "genre_hypothesis",
            "genre_balanced_hypothesis",
            "genre_anchor_hypothesis",
            "genre_balanced_anchor_hypothesis",
            "genre_balanced_anchor_genrefirst",
        ),
        default="baseline",
        help="Task6-only experiment switch.",
    )
    parser.add_argument(
        "--task7-experiment",
        type=str,
        choices=("baseline", "semantic_mmr"),
        default="semantic_mmr",
        help="Task7-only experiment switch.",
    )
    parser.add_argument(
        "--task7-bank-path",
        type=str,
        default="../outputs/task7_reasoning_bank_scale1500_20260401/successful_reasoning_bank.jsonl",
        help="Task7 semantic retrieval success bank path.",
    )
    parser.add_argument(
        "--task7-bank-tags-path",
        type=str,
        default="../outputs/task7_semantic_tag_retrieval_full500_20260401/bank_semantic_tags.jsonl",
        help="Task7 semantic retrieval bank tag cache path.",
    )
    parser.add_argument(
        "--task7-test-tags-path",
        type=str,
        default="",
        help="Optional Task7 test semantic tag cache path for reproducible submission sweeps.",
    )
    parser.add_argument(
        "--task7-retrieval-mode",
        type=str,
        choices=("lexical", "semantic", "semantic_rerank", "semantic_mmr"),
        default="semantic_mmr",
        help="Task7 retrieval mode used inside the semantic retrieval experiment.",
    )
    parser.add_argument("--task7-base-few-shot", type=int, default=4)
    parser.add_argument("--task7-retrieval-k", type=int, default=3)
    parser.add_argument("--task7-reasoning-char-limit", type=int, default=800)
    parser.add_argument("--task7-compress-answer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--task7-compress-max-new-tokens", type=int, default=64)
    parser.add_argument("--task7-compress-temperature", type=float, default=0.0)
    parser.add_argument("--task7-compress-top-p", type=float, default=0.8)
    parser.add_argument("--task7-tag-max-new-tokens", type=int, default=256)
    parser.add_argument("--task7-tag-temperature", type=float, default=0.0)
    parser.add_argument("--task7-tag-top-p", type=float, default=0.9)
    parser.add_argument("--task7-semantic-rerank-pool", type=int, default=24)
    parser.add_argument("--task7-semantic-mmr-pool", type=int, default=24)
    parser.add_argument("--task7-semantic-mmr-lambda", type=float, default=0.7)
    parser.add_argument(
        "--task7-answer-rules",
        choices=ANSWER_RULE_CHOICES,
        default="shortest",
        help="Task7 semantic_mmr answer canonicalization prompt variant.",
    )
    parser.add_argument(
        "--task1-experiment",
        type=str,
        choices=("baseline", "solver_synthesis", "solver_debug_loop"),
        default="solver_debug_loop",
        help="Task1-only experiment switch.",
    )
    parser.add_argument(
        "--task3-experiment",
        type=str,
        choices=("baseline", "solver_synthesis", "solver_debug_loop"),
        default="solver_debug_loop",
        help="Task3-only experiment switch.",
    )
    parser.add_argument("--solver-prompt-examples", type=int, default=12)
    parser.add_argument("--solver-validation-examples", type=int, default=32)
    parser.add_argument("--solver-max-attempts", type=int, default=3)
    parser.add_argument(
        "--generated-solver-dir",
        type=str,
        default="../outputs/generated_solvers",
        help="Directory for generated solver artifacts.",
    )

    parser.add_argument(
        "--llama-server-bin",
        type=str,
        default="../third_party/llama.cpp/build/bin/llama-server",
        help="Path to llama-server binary.",
    )
    parser.add_argument(
        "--llama-model-path",
        type=str,
        default="../models/gguf/Qwen3-4B-GGUF/Qwen3-4B-Q8_0.gguf",
        help="GGUF model path for llama.cpp.",
    )
    parser.add_argument("--llama-ctx-size", type=int, default=65536)
    parser.add_argument("--llama-gpu-layers", type=int, default=999)
    parser.add_argument("--llama-parallel", type=int, default=1)
    parser.add_argument("--llama-cache-type-k", type=str, default="q8_0")
    parser.add_argument("--llama-cache-type-v", type=str, default="q8_0")
    parser.add_argument("--llama-cont-batching", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument(
        "--vllm-model-path",
        type=str,
        default="../models/Qwen3-4B",
        help="HF/modelscope model path used by vLLM on Ascend.",
    )
    parser.add_argument("--vllm-served-model-name", type=str, default="Qwen3-4B")
    parser.add_argument("--vllm-dtype", type=str, default="bfloat16")
    parser.add_argument("--vllm-tensor-parallel-size", type=int, default=2)
    parser.add_argument("--vllm-max-model-len", type=int, default=32768)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--vllm-max-num-seqs", type=int, default=8)
    parser.add_argument("--vllm-enable-chunked-prefill", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vllm-enable-prefix-caching", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--vllm-enforce-eager", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ascend-visible-devices", type=str, default="0,1")

    parser.add_argument(
        "--auto-start-api",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If API is unavailable, auto start inference service first.",
    )
    parser.add_argument(
        "--api-start-cmd",
        type=str,
        default="",
        help="Optional custom startup command. If empty, use an auto-generated command with absolute paths.",
    )
    parser.add_argument("--api-start-timeout", type=float, default=600.0, help="Seconds to wait for API to become ready.")
    parser.add_argument("--api-check-timeout", type=float, default=5.0, help="Per-request timeout for API availability checks.")
    parser.add_argument("--api-check-interval", type=float, default=5.0, help="Polling interval while waiting for API startup.")

    args = parser.parse_args()
    if not args.task_id and not args.all_tasks:
        parser.error("Please set --task-id or --all-tasks.")
    return args


def _api_host_port(api_base: str) -> tuple[str, int]:
    parsed = urlparse(api_base)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 2026
    return host, port


def resolve_data_path(task_id: int, data_dir: Path, fallback_data_dir: Path) -> Path:
    filename = TASK_FILES[task_id]
    candidate = data_dir / filename
    if candidate.exists():
        return candidate

    fallback = fallback_data_dir / filename
    if fallback.exists():
        return fallback

    raise FileNotFoundError(
        f"Dataset for task {task_id} not found. Checked: {candidate} and {fallback}."
    )


def load_task_data(task_file: Path) -> dict:
    try:
        with task_file.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        raw_text = task_file.read_text(encoding="utf-8")
        sanitized_chars: List[str] = []
        in_string = False
        escaped = False
        for ch in raw_text:
            if escaped:
                sanitized_chars.append(ch)
                escaped = False
                continue
            if ch == "\\":
                sanitized_chars.append(ch)
                if in_string:
                    escaped = True
                continue
            if ch == '"':
                sanitized_chars.append(ch)
                in_string = not in_string
                continue
            if in_string and ch in {"\n", "\r", "\t"}:
                sanitized_chars.append({"\n": "\\n", "\r": "\\r", "\t": "\\t"}[ch])
                continue
            sanitized_chars.append(ch)
        return json.loads("".join(sanitized_chars))


def next_output_file(output_dir: Path, task_id: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    version = 1
    while True:
        path = output_dir / f"openseek-{task_id}-v{version}.jsonl"
        if not path.exists():
            return path
        version += 1


def _is_api_available(api_base: str, timeout: float) -> bool:
    try:
        resp = requests.get(f"{api_base.rstrip('/')}/v1/models", timeout=timeout)
        if resp.ok:
            return True
    except requests.RequestException:
        pass

    try:
        resp = requests.get(f"{api_base.rstrip('/')}/health", timeout=timeout)
        if resp.ok:
            return True
    except requests.RequestException:
        pass

    return False


def resolve_model_name(api_base: str, preferred_model_name: str, timeout: float = 5.0) -> str:
    try:
        resp = requests.get(f"{api_base.rstrip('/')}/v1/models", timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        models = data.get("data") or []
        model_ids = [str(item.get("id", "")).strip() for item in models if item.get("id")]
        if not model_ids:
            return preferred_model_name

        if preferred_model_name in model_ids:
            return preferred_model_name

        preferred_name = Path(preferred_model_name).name
        for model_id in model_ids:
            if Path(model_id).name == preferred_name:
                print(
                    f"[API] model id mismatch, auto switch: "
                    f"{preferred_model_name} -> {model_id}"
                )
                return model_id

        print(
            f"[API] preferred model not found: {preferred_model_name}. "
            f"Fallback to first served model: {model_ids[0]}"
        )
        return model_ids[0]
    except requests.RequestException:
        return preferred_model_name


def _build_llamacpp_start_cmd(args: argparse.Namespace) -> List[str]:
    llama_server_bin = (Path(__file__).resolve().parent / args.llama_server_bin).resolve()
    llama_model_path = (Path(__file__).resolve().parent / args.llama_model_path).resolve()
    host, port = _api_host_port(args.api_base)

    cmd = [
        str(llama_server_bin),
        "-m",
        str(llama_model_path),
        "--host",
        host,
        "--port",
        str(port),
        "--ctx-size",
        str(args.llama_ctx_size),
        "--n-gpu-layers",
        str(args.llama_gpu_layers),
        "--parallel",
        str(args.llama_parallel),
        "--cache-type-k",
        args.llama_cache_type_k,
        "--cache-type-v",
        args.llama_cache_type_v,
        "--metrics",
    ]
    if args.llama_cont_batching:
        cmd.append("--cont-batching")
    return cmd


def _build_vllm_start_cmd(args: argparse.Namespace) -> List[str]:
    model_path = (Path(__file__).resolve().parent / args.vllm_model_path).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"vLLM model path not found: {model_path}")

    host, port = _api_host_port(args.api_base)
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(model_path),
        "--host",
        host,
        "--port",
        str(port),
        "--served-model-name",
        args.vllm_served_model_name,
        "--dtype",
        args.vllm_dtype,
        "--tensor-parallel-size",
        str(args.vllm_tensor_parallel_size),
        "--max-model-len",
        str(args.vllm_max_model_len),
        "--gpu-memory-utilization",
        str(args.vllm_gpu_memory_utilization),
        "--max-num-seqs",
        str(args.vllm_max_num_seqs),
        "--trust-remote-code",
    ]
    cmd.append("--enable-chunked-prefill" if args.vllm_enable_chunked_prefill else "--no-enable-chunked-prefill")
    cmd.append("--enable-prefix-caching" if args.vllm_enable_prefix_caching else "--no-enable-prefix-caching")
    cmd.append("--enforce-eager" if args.vllm_enforce_eager else "--no-enforce-eager")
    return cmd


def _build_vllm_env(args: argparse.Namespace) -> Dict[str, str]:
    env = os.environ.copy()
    env["ASCEND_RT_VISIBLE_DEVICES"] = args.ascend_visible_devices
    # 保留 CUDA_VISIBLE_DEVICES 以兼容仓库内仍然按旧变量推断设备数的代码路径。
    env["CUDA_VISIBLE_DEVICES"] = args.ascend_visible_devices
    env.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    env.setdefault("ASCEND_LAUNCH_BLOCKING", "0")
    env.setdefault("VLLM_ASCEND_ENABLE_FLASHCOMM1", "0")
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    return env


def _build_autostart_cmd(args: argparse.Namespace) -> List[str]:
    project_root = _project_root()
    if args.api_start_cmd.strip():
        return shlex.split(args.api_start_cmd)
    if args.backend == "vllm":
        return _build_vllm_start_cmd(args)
    if args.backend == "llamacpp":
        return _build_llamacpp_start_cmd(args)
    return [
        "bash",
        str(project_root / "scripts" / "start_service.sh"),
    ]


def _build_autostart_env(args: argparse.Namespace) -> Dict[str, str]:
    if args.backend == "vllm":
        return _build_vllm_env(args)
    return os.environ.copy()


def _task_generation_config(task_id: int, args: argparse.Namespace) -> tuple[int, float]:
    if task_id == 8:
        return args.max_new_tokens_task8, args.timeout_task8
    return args.max_new_tokens, args.timeout


def _task_stop_sequences(task_id: int, args: argparse.Namespace) -> List[str] | None:
    if task_id == 8:
        if args.task8_experiment in {"compact", "repair_pass"}:
            return [
                "\n[Sample To Annotate]",
                "\nFunctional Description:",
                "\nWrapper Entry Information:",
                "\nFinal answer:",
            ]
        return None
    return None


def ensure_api_ready(args: argparse.Namespace) -> None:
    if _is_api_available(args.api_base, args.api_check_timeout):
        print(f"[API] already available: {args.api_base}")
        return

    if not args.auto_start_api:
        raise RuntimeError(f"API unavailable: {args.api_base}. You can enable --auto-start-api.")

    script_dir = Path(__file__).resolve().parent
    project_root = _project_root()
    startup_log_dir = project_root / "outputs" / "logs"
    startup_log_dir.mkdir(parents=True, exist_ok=True)
    startup_log = startup_log_dir / "api_autostart.log"

    cmd = _build_autostart_cmd(args)
    env = _build_autostart_env(args)

    print(f"[API] unavailable, auto-starting service: {args.api_base}")
    print(f"[API] startup log: {startup_log}")

    with startup_log.open("a", encoding="utf-8") as logf:
        logf.write(f"\n[autostart] cmd={' '.join(shlex.quote(part) for part in cmd)}\n")
        if args.backend == "vllm":
            logf.write(
                "[autostart] env="
                f"ASCEND_RT_VISIBLE_DEVICES={env.get('ASCEND_RT_VISIBLE_DEVICES', '')} "
                f"CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', '')}\n"
            )
        logf.flush()
        subprocess.Popen(
            cmd,
            cwd=str(script_dir),
            env=env,
            stdout=logf,
            stderr=logf,
            start_new_session=True,
        )

    deadline = time.time() + args.api_start_timeout
    total_checks = max(1, int(args.api_start_timeout / args.api_check_interval))
    with tqdm(total=total_checks, desc="Waiting API", unit="check") as wait_bar:
        while time.time() < deadline:
            if _is_api_available(args.api_base, args.api_check_timeout):
                wait_bar.n = total_checks
                wait_bar.refresh()
                print(f"[API] service is ready: {args.api_base}")
                return

            remaining = max(0, int(deadline - time.time()))
            wait_bar.set_postfix_str(f"remaining={remaining}s")
            time.sleep(args.api_check_interval)
            if wait_bar.n < total_checks:
                wait_bar.update(1)

    raise TimeoutError(
        f"API did not become ready within {args.api_start_timeout}s. "
        f"Check startup log: {startup_log}"
    )


def _create_solver_run_dir(task_id: int, args: argparse.Namespace) -> Path:
    solver_root = (Path(__file__).resolve().parent / args.generated_solver_dir).resolve()
    solver_root.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    run_dir = solver_root / f"task{task_id}_{timestamp}"
    suffix = 1
    while run_dir.exists():
        suffix += 1
        run_dir = solver_root / f"task{task_id}_{timestamp}_{suffix}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _write_solver_artifact(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _estimate_prompt_tokens(text: str, tokenizer: object | None) -> int:
    if not text:
        return 0
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text))
        except Exception:
            pass
    return max(1, len(text) // 4)


def _solver_example_block(example: Dict[str, object], idx: int) -> str:
    example_input = str(example.get("input", "")).strip()
    raw_output = example.get("output", "")
    if isinstance(raw_output, list):
        example_output = str(raw_output[0] if raw_output else "").strip()
    else:
        example_output = str(raw_output).strip()
    return (
        f"[Example {idx}]\n"
        f"Input:\n{example_input}\n"
        f"Expected Output:\n{example_output}\n"
    )


def _select_solver_min_context_examples(
    examples: List[dict],
    tokenizer: object | None,
    target_tokens: int,
    max_input_tokens: int,
    reserved_generation_tokens: int,
) -> tuple[List[dict], int]:
    budget_for_examples = max(0, max_input_tokens - reserved_generation_tokens)
    selected: List[dict] = []
    used_tokens = 0
    for example in examples:
        block = _solver_example_block(example, len(selected) + 1)
        block_tokens = _estimate_prompt_tokens(block, tokenizer)
        if used_tokens + block_tokens > budget_for_examples:
            continue
        selected.append(example)
        used_tokens += block_tokens
        if used_tokens >= target_tokens:
            break
    return selected, used_tokens


def _write_context_audit(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as writer:
        for row in rows:
            writer.write(json.dumps(row, ensure_ascii=False) + "\n")


def _context_audit_path(audit_dir: Path, task_id: int, mode: str) -> Path:
    safe_mode = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(mode)).strip("_")
    if safe_mode:
        return audit_dir / f"context_audit_task{task_id}_{safe_mode}.jsonl"
    return audit_dir / f"context_audit_task{task_id}.jsonl"


def _run_rule_task_solver_synthesis(
    task_id: int,
    task_description: str,
    examples: List[dict],
    test_samples: List[dict],
    output_dir: Path,
    args: argparse.Namespace,
    mode: str = "solver_synthesis",
) -> Path:
    output_file = next_output_file(output_dir, task_id)
    run_dir = _create_solver_run_dir(task_id, args)

    solver_context_tokens = 0
    solver_context_target = args.min_context_tokens
    if getattr(args, "official_min_context", False):
        tokenizer_path = (Path(__file__).resolve().parent / args.tokenizer_path).resolve()
        tokenizer = load_tokenizer(str(tokenizer_path) if tokenizer_path.exists() else None)
        prompt_examples, solver_context_tokens = _select_solver_min_context_examples(
            examples=examples,
            tokenizer=tokenizer,
            target_tokens=solver_context_target,
            max_input_tokens=args.max_input_tokens,
            reserved_generation_tokens=args.reserved_generation_tokens,
        )
    else:
        prompt_examples = examples[: args.solver_prompt_examples]
    validation_start = len(prompt_examples)
    validation_end = validation_start + args.solver_validation_examples
    validation_examples = examples[validation_start:validation_end]
    if not validation_examples:
        validation_examples = examples[-args.solver_validation_examples :]
    if not validation_examples:
        raise RuntimeError(f"Task{task_id} solver_synthesis requires validation examples, but none were selected.")

    resolved_model_name = resolve_model_name(args.api_base, args.model_name, timeout=args.api_check_timeout)
    if getattr(args, "official_min_context", False):
        solver_max_new_tokens = min(args.reserved_generation_tokens, max(args.max_new_tokens, 1024))
    else:
        solver_max_new_tokens = max(args.max_new_tokens, 4096)

    solver_cfg = InferenceConfig(
        api_base=args.api_base,
        model_name=resolved_model_name,
        timeout=max(args.timeout, 300.0),
        max_new_tokens=solver_max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        stop=None,
        enable_thinking=args.enable_thinking,
    )

    print(
        f"[Task {task_id}] {mode} mode, prompt_examples={len(prompt_examples)}, "
        f"validation_examples={len(validation_examples)}, "
        f"official_context_tokens~{solver_context_tokens}, run_dir={run_dir}"
    )

    best_code: Optional[str] = None
    best_correct = -1
    best_total = 0
    repair_failures: List[Dict[str, str]] = []

    for attempt in range(1, args.solver_max_attempts + 1):
        if attempt == 1:
            prompt = build_task_solver_synthesis_prompt(
                task_id=task_id,
                task_description=task_description,
                examples=prompt_examples,
            )
        else:
            if mode == "solver_debug_loop":
                prompt = build_task_solver_debug_prompt(
                    task_id=task_id,
                    task_description=task_description,
                    examples=prompt_examples,
                    draft_code=best_code or "",
                    failures=repair_failures,
                )
            else:
                prompt = build_task_solver_repair_prompt(
                    task_id=task_id,
                    task_description=task_description,
                    examples=prompt_examples,
                    draft_code=best_code or "",
                    failures=repair_failures,
                )

        response = annotate_detailed(prompt, solver_cfg)
        raw_text = response.visible_text.strip() if response.visible_text else response.raw_text.strip()
        code_source = response.visible_text.strip() if response.visible_text else response.raw_text.strip()
        code = extract_solver_code(code_source)

        _write_solver_artifact(run_dir / f"attempt_{attempt:02d}_prompt.txt", prompt)
        _write_solver_artifact(run_dir / f"attempt_{attempt:02d}_raw.txt", response.raw_text)
        _write_solver_artifact(run_dir / f"attempt_{attempt:02d}_visible.txt", response.visible_text)
        _write_solver_artifact(run_dir / f"attempt_{attempt:02d}_reasoning.txt", response.reasoning_text)

        if not code:
            repair_failures = [
                {
                    "input": "<module generation>",
                    "expected": "a Python module with def solve(input_text: str) -> str",
                    "predicted": code_source.strip(),
                    "error": "failed to extract solver code",
                }
            ]
            continue

        _write_solver_artifact(run_dir / f"attempt_{attempt:02d}_solver.py", code)
        correct, total, failures = validate_solver_code(
            task_id=task_id,
            solver_code=code,
            examples=validation_examples,
            function_name="solve",
        )
        validation_report = {
            "attempt": attempt,
            "correct": correct,
            "total": total,
            "accuracy": (correct / total) if total else 0.0,
            "failures": failures,
        }
        _write_solver_artifact(
            run_dir / f"attempt_{attempt:02d}_validation.json",
            json.dumps(validation_report, ensure_ascii=False, indent=2),
        )

        if correct > best_correct:
            best_code = code
            best_correct = correct
            best_total = total
            repair_failures = failures

        print(f"[Task {task_id}] {mode} attempt {attempt}: validation {correct}/{total}")
        if total > 0 and correct == total:
            best_code = code
            best_correct = correct
            best_total = total
            break

    if not best_code:
        raise RuntimeError(f"Task {task_id} {mode} failed: no valid solver code generated.")
    if best_total > 0 and best_correct != best_total:
        raise RuntimeError(
            f"Task {task_id} {mode} validation failed: best={best_correct}/{best_total}. "
            f"See artifacts in {run_dir}"
        )

    _write_solver_artifact(run_dir / "selected_solver.py", best_code)

    with output_file.open("w", encoding="utf-8") as writer:
        for sample in tqdm(test_samples, desc=f"Task {task_id} solver", unit="sample"):
            sample_id = sample.get("id")
            sample_input = str(sample.get("input", ""))
            prediction = execute_solver_code(best_code, sample_input, function_name="solve")
            row = {"test_sample_id": sample_id, "prediction": prediction}
            writer.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "task_id": task_id,
        "mode": mode,
        "prompt_examples": len(prompt_examples),
        "official_min_context": bool(getattr(args, "official_min_context", False)),
        "target_min_context_tokens": solver_context_target,
        "used_example_tokens": solver_context_tokens,
        "pass_min_context": (solver_context_tokens >= solver_context_target)
        if getattr(args, "official_min_context", False)
        else None,
        "validation_examples": len(validation_examples),
        "solver_max_new_tokens": solver_max_new_tokens,
        "best_validation_correct": best_correct,
        "best_validation_total": best_total,
        "output_file": str(output_file),
    }
    _write_solver_artifact(run_dir / "summary.json", json.dumps(summary, ensure_ascii=False, indent=2))
    if getattr(args, "official_min_context", False):
        audit_dir = Path(args.context_audit_dir).resolve() if args.context_audit_dir else output_dir
        audit_path = _context_audit_path(audit_dir, task_id, mode)
        _write_context_audit(
            audit_path,
            [
                {
                    "task_id": task_id,
                    "stage": "solver_generation",
                    "mode": mode,
                    "target_min_context_tokens": solver_context_target,
                    "used_example_tokens": solver_context_tokens,
                    "selected_example_count": len(prompt_examples),
                    "pass_min_context": solver_context_tokens >= solver_context_target,
                    "output_file": str(output_file),
                    "solver_run_dir": str(run_dir),
                }
            ],
        )
        _write_context_audit(audit_dir / f"context_audit_task{task_id}.jsonl", [
            {
                "task_id": task_id,
                "stage": "solver_generation",
                "mode": mode,
                "target_min_context_tokens": solver_context_target,
                "used_example_tokens": solver_context_tokens,
                "selected_example_count": len(prompt_examples),
                "pass_min_context": solver_context_tokens >= solver_context_target,
                "output_file": str(output_file),
                "solver_run_dir": str(run_dir),
                "mode_audit_file": str(audit_path),
            }
        ])
    print(
        f"[Task {task_id}] {mode} finished, validation={best_correct}/{best_total}, "
        f"samples={len(test_samples)}, output={output_file}"
    )
    return output_file


def run_one_task(task_id: int, args: argparse.Namespace) -> Path:
    print(f"\n[Task {task_id}] preparing data and prompt context...")
    data_dir = (Path(__file__).resolve().parent / args.data_dir).resolve()
    fallback_data_dir = (Path(__file__).resolve().parent / args.fallback_data_dir).resolve()
    output_dir = (Path(__file__).resolve().parent / args.output_dir).resolve()

    task_file = resolve_data_path(task_id, data_dir, fallback_data_dir)
    print(f"[Task {task_id}] dataset: {task_file}")
    task_data = load_task_data(task_file)

    task_description_list = task_data.get("Definition") or [""]
    task_description = str(task_description_list[0])
    examples: List[dict] = task_data.get("examples", [])
    test_samples: List[dict] = task_data.get("test_samples", [])

    task2_mode = args.task2_experiment if task_id == 2 else "baseline"
    task5_mode = args.task5_experiment if task_id == 5 else "baseline"
    task6_mode = args.task6_experiment if task_id == 6 else "baseline"
    task7_mode = args.task7_experiment if task_id == 7 else "baseline"
    task1_mode = args.task1_experiment if task_id == 1 else "baseline"
    task3_mode = args.task3_experiment if task_id == 3 else "baseline"
    task4_mode = (
        "prompt_gated_retry"
        if task_id == 4 and args.task4_experiment == "baseline"
        else (args.task4_experiment if task_id == 4 else "baseline")
    )
    task8_mode = args.task8_experiment if task_id == 8 else "baseline"
    task2_dynamic_modes = {"context_rerank", "noun_biased", "noun_rules", "bucket_balanced"}
    task5_dynamic_modes = {"strict_dynamic", "balanced_dynamic", "conservative_dynamic"}
    if args.max_test_samples > 0:
        test_samples = test_samples[: args.max_test_samples]
    elif task_id == 8 and args.task8_max_samples > 0:
        test_samples = test_samples[: args.task8_max_samples]
    if task_id == 2:
        print(f"[Task {task_id}] experiment mode={task2_mode}")
    if task_id == 5:
        print(f"[Task {task_id}] experiment mode={task5_mode}")
    if task_id == 6:
        print(f"[Task {task_id}] experiment mode={task6_mode}")
    if task_id == 7:
        print(f"[Task {task_id}] experiment mode={task7_mode}, retrieval={args.task7_retrieval_mode}")
    if task_id == 1:
        print(f"[Task {task_id}] experiment mode={task1_mode}")
    if task_id == 3:
        print(f"[Task {task_id}] experiment mode={task3_mode}")
    if task_id == 4:
        print(f"[Task {task_id}] experiment mode={task4_mode}")

    if task_id == 1 and task1_mode in {"solver_synthesis", "solver_debug_loop"}:
        return _run_rule_task_solver_synthesis(
            task_id=task_id,
            task_description=task_description,
            examples=examples,
            test_samples=test_samples,
            output_dir=output_dir,
            args=args,
            mode=task1_mode,
        )

    if task_id == 3 and task3_mode in {"solver_synthesis", "solver_debug_loop"}:
        return _run_rule_task_solver_synthesis(
            task_id=task_id,
            task_description=task_description,
            examples=examples,
            test_samples=test_samples,
            output_dir=output_dir,
            args=args,
            mode=task3_mode,
        )

    if task_id == 4 and task4_mode in {"solver_synthesis", "solver_debug_loop"}:
        return _run_rule_task_solver_synthesis(
            task_id=task_id,
            task_description=task_description,
            examples=examples,
            test_samples=test_samples,
            output_dir=output_dir,
            args=args,
            mode=task4_mode,
        )

    if task_id == 7 and task7_mode == "semantic_mmr":
        output_file = next_output_file(output_dir, task_id)
        cache_dir = output_dir / f"task7_semantic_mmr_artifacts_{output_file.stem}"
        prediction_rows, metadata = predict_task7_semantic_mmr(
            task_description=task_description,
            examples=examples,
            test_samples=test_samples,
            args=args,
            cache_dir=cache_dir,
        )
        with output_file.open("w", encoding="utf-8") as writer:
            for row in prediction_rows:
                writer.write(json.dumps(row, ensure_ascii=False) + "\n")
        metadata["output_file"] = str(output_file)
        metadata["task_id"] = task_id
        (cache_dir / "task7_output_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            f"[Task {task_id}] retrieval finished, mode={metadata.get('mode')}, samples={len(test_samples)}, "
            f"output={output_file}, artifacts={cache_dir}"
        )
        return output_file

    tokenizer_path = (Path(__file__).resolve().parent / args.tokenizer_path).resolve()
    print(f"[Task {task_id}] loading tokenizer from: {tokenizer_path}")
    tokenizer = load_tokenizer(str(tokenizer_path) if tokenizer_path.exists() else None)

    context_target = args.min_context_tokens_task8 if task_id == 8 else args.min_context_tokens
    if task_id == 2 and task2_mode in task2_dynamic_modes and not args.official_min_context:
        context_target = min(context_target, 6000)
    if task_id == 5 and task5_mode in task5_dynamic_modes and not args.official_min_context:
        context_target = min(context_target, 6000)
    if task_id == 6 and task6_mode in {
        "genre_dynamic",
        "genre_conservative",
        "genre_balanced",
        "genre_balanced_conservative",
        "genre_hypothesis",
        "genre_balanced_hypothesis",
        "genre_anchor_hypothesis",
        "genre_balanced_anchor_hypothesis",
        "genre_balanced_anchor_genrefirst",
    } and not args.official_min_context:
        context_target = min(context_target, 6000)
    if task_id == 4 and task4_mode == "structured_examples" and not args.official_min_context:
        context_target = min(context_target, 12000)
    query_seed = str(test_samples[0].get("input", "")) if test_samples else ""
    examples_text, selected_count, used_tokens = select_examples(
        all_examples=examples,
        query_text=query_seed,
        tokenizer=tokenizer,
        max_input_tokens=args.max_input_tokens,
        target_context_tokens=context_target,
        reserved_generation_tokens=args.reserved_generation_tokens,
        task_id=task_id,
        strategy=(
            (
                "task2_bucket_balanced"
                if task2_mode == "bucket_balanced"
                else ("task2_noun_biased" if task2_mode in {"noun_biased", "noun_rules"} else "task2_aware")
            )
            if task_id == 2 and task2_mode in task2_dynamic_modes
            else (
                "task6_genre_balanced"
                if task_id == 6
                and task6_mode
                in {
                    "genre_balanced",
                    "genre_balanced_conservative",
                    "genre_balanced_hypothesis",
                    "genre_balanced_anchor_hypothesis",
                }
                else ("structured" if task_id == 4 and task4_mode == "structured_examples" else "semantic")
            )
        ),
    )
    print(
        f"[Task {task_id}] selected examples: {selected_count}, "
        f"example_tokens~{used_tokens}, test_samples={len(test_samples)}"
    )
    cache_prefix_enabled = (
        args.cache_prefix_context_tokens > 0
        and not args.cache_friendly_context
        and (
            (task_id == 2 and task2_mode in task2_dynamic_modes)
            or (task_id == 5 and task5_mode in task5_dynamic_modes)
            or (
                task_id == 6
                and task6_mode
                in {
                    "genre_dynamic",
                    "genre_conservative",
                    "genre_balanced",
                    "genre_balanced_conservative",
                    "genre_hypothesis",
                    "genre_balanced_hypothesis",
                    "genre_anchor_hypothesis",
                    "genre_balanced_anchor_hypothesis",
                    "genre_balanced_anchor_genrefirst",
                }
            )
        )
    )
    cache_prefix_examples_text = ""
    cache_prefix_selected_count = 0
    cache_prefix_used_tokens = 0
    if cache_prefix_enabled:
        cache_prefix_target = min(context_target, max(0, args.cache_prefix_context_tokens))
        cache_prefix_examples_text, cache_prefix_selected_count, cache_prefix_used_tokens = select_examples(
            all_examples=examples,
            query_text=query_seed,
            tokenizer=tokenizer,
            max_input_tokens=args.max_input_tokens,
            target_context_tokens=cache_prefix_target,
            reserved_generation_tokens=args.reserved_generation_tokens,
            task_id=task_id,
            strategy=(
                (
                    "task2_bucket_balanced"
                    if task2_mode == "bucket_balanced"
                    else ("task2_noun_biased" if task2_mode in {"noun_biased", "noun_rules"} else "task2_aware")
                )
                if task_id == 2 and task2_mode in task2_dynamic_modes
                else (
                    "task5_conservative"
                    if task_id == 5 and task5_mode == "conservative_dynamic"
                    else (
                        "task5_balanced"
                        if task_id == 5 and task5_mode == "balanced_dynamic"
                        else (
                            "task6_genre_balanced"
                            if task_id == 6
                            and task6_mode
                            in {
                                "genre_balanced",
                                "genre_balanced_conservative",
                                "genre_balanced_hypothesis",
                                "genre_balanced_anchor_hypothesis",
                                "genre_balanced_anchor_genrefirst",
                            }
                            else "semantic"
                        )
                    )
                )
            ),
        )
        print(
            f"[Task {task_id}] cache prefix examples: {cache_prefix_selected_count}, "
            f"prefix_tokens~{cache_prefix_used_tokens}, remaining_target~{max(0, context_target - cache_prefix_used_tokens)}"
        )

    resolved_model_name = resolve_model_name(args.api_base, args.model_name, timeout=args.api_check_timeout)
    max_new_tokens, timeout = _task_generation_config(task_id, args)
    infer_cfg = InferenceConfig(
        api_base=args.api_base,
        model_name=resolved_model_name,
        timeout=timeout,
        max_new_tokens=max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        stop=_task_stop_sequences(task_id, args),
        enable_thinking=args.enable_thinking,
    )

    client_concurrency = max(1, min(args.client_concurrency, len(test_samples) or 1))
    print(
        f"[Task {task_id}] client concurrency={client_concurrency}, "
        f"retries={DEFAULT_REQUEST_RETRIES}, max_new_tokens={max_new_tokens}, timeout={timeout}"
    )
    if task_id == 8:
        print(f"[Task {task_id}] experiment mode={task8_mode}")

    def _infer_one(sample_index: int, sample: dict) -> tuple[int, dict, bool, Dict[str, object]]:
        sample_id = sample.get("id")
        sample_input = str(sample.get("input", ""))
        sample_examples_text = examples_text
        sample_selected_count = selected_count
        sample_used_tokens = used_tokens
        if cache_prefix_enabled:
            sample_examples_text = cache_prefix_examples_text
            sample_selected_count = cache_prefix_selected_count
            sample_used_tokens = cache_prefix_used_tokens
            remaining_target = max(0, context_target - cache_prefix_used_tokens)
            if remaining_target > 0:
                suffix_max_input_tokens = max(
                    args.reserved_generation_tokens,
                    args.max_input_tokens - cache_prefix_used_tokens,
                )
                suffix_examples_text, suffix_selected_count, suffix_used_tokens = select_examples(
                    all_examples=examples,
                    query_text=sample_input,
                    tokenizer=tokenizer,
                    max_input_tokens=suffix_max_input_tokens,
                    target_context_tokens=remaining_target,
                    reserved_generation_tokens=args.reserved_generation_tokens,
                    task_id=task_id,
                    strategy=(
                        (
                            "task2_bucket_balanced"
                            if task2_mode == "bucket_balanced"
                            else ("task2_noun_biased" if task2_mode in {"noun_biased", "noun_rules"} else "task2_aware")
                        )
                        if task_id == 2 and task2_mode in task2_dynamic_modes
                        else (
                            "task5_conservative"
                            if task_id == 5 and task5_mode == "conservative_dynamic"
                            else (
                                "task5_balanced"
                                if task_id == 5 and task5_mode == "balanced_dynamic"
                                else (
                                    "task6_genre_balanced"
                                    if task_id == 6
                                    and task6_mode
                                    in {
                                        "genre_balanced",
                                        "genre_balanced_conservative",
                                        "genre_balanced_hypothesis",
                                        "genre_balanced_anchor_hypothesis",
                                        "genre_balanced_anchor_genrefirst",
                                    }
                                    else "semantic"
                                )
                            )
                        )
                    ),
                )
                if suffix_examples_text:
                    sample_examples_text = (sample_examples_text + "\n\n" + suffix_examples_text).strip()
                sample_selected_count += suffix_selected_count
                sample_used_tokens += suffix_used_tokens
        elif task_id == 2 and task2_mode in task2_dynamic_modes and not args.cache_friendly_context:
            sample_examples_text, sample_selected_count, sample_used_tokens = select_examples(
                all_examples=examples,
                query_text=sample_input,
                tokenizer=tokenizer,
                max_input_tokens=args.max_input_tokens,
                target_context_tokens=context_target,
                reserved_generation_tokens=args.reserved_generation_tokens,
                task_id=task_id,
                strategy=(
                    "task2_bucket_balanced"
                    if task2_mode == "bucket_balanced"
                    else ("task2_noun_biased" if task2_mode in {"noun_biased", "noun_rules"} else "task2_aware")
                ),
            )
        if task_id == 5 and task5_mode in task5_dynamic_modes and not args.cache_friendly_context and not cache_prefix_enabled:
            sample_examples_text, sample_selected_count, sample_used_tokens = select_examples(
                all_examples=examples,
                query_text=sample_input,
                tokenizer=tokenizer,
                max_input_tokens=args.max_input_tokens,
                target_context_tokens=context_target,
                reserved_generation_tokens=args.reserved_generation_tokens,
                task_id=task_id,
                strategy=(
                    "task5_conservative"
                    if task5_mode == "conservative_dynamic"
                    else ("task5_balanced" if task5_mode == "balanced_dynamic" else "semantic")
                ),
            )
        if task_id == 6 and task6_mode in {
            "genre_dynamic",
            "genre_conservative",
            "genre_balanced",
            "genre_balanced_conservative",
            "genre_hypothesis",
            "genre_balanced_hypothesis",
            "genre_anchor_hypothesis",
            "genre_balanced_anchor_hypothesis",
        } and not args.cache_friendly_context and not cache_prefix_enabled:
            sample_examples_text, sample_selected_count, sample_used_tokens = select_examples(
                all_examples=examples,
                query_text=sample_input,
                tokenizer=tokenizer,
                max_input_tokens=args.max_input_tokens,
                target_context_tokens=context_target,
                reserved_generation_tokens=args.reserved_generation_tokens,
                task_id=task_id,
                strategy=(
                    "task6_genre_balanced"
                    if task6_mode
                    in {
                        "genre_balanced",
                        "genre_balanced_conservative",
                        "genre_balanced_hypothesis",
                        "genre_balanced_anchor_hypothesis",
                        "genre_balanced_anchor_genrefirst",
                    }
                    else "semantic"
                ),
            )
        prompt_variant = "default"
        if task_id == 2 and task2_mode == "noun_rules":
            prompt_variant = "task2_count_rules"
        elif task_id == 2 and task2_mode in {"context_rerank", "noun_biased", "bucket_balanced"}:
            prompt_variant = "task2_strict"
        elif task_id == 4 and task4_mode in {"prompt", "prompt_gated_retry"}:
            prompt_variant = "strict"
        elif task_id == 5 and task5_mode == "strict_dynamic":
            prompt_variant = "task5_sadness_strict"
        elif task_id == 5 and task5_mode == "balanced_dynamic":
            prompt_variant = "task5_sadness_balanced"
        elif task_id == 5 and task5_mode == "conservative_dynamic":
            prompt_variant = "task5_sadness_conservative"
        elif task_id == 6 and task6_mode == "genre_dynamic":
            prompt_variant = "task6_genre_strict"
        elif task_id == 6 and task6_mode in {"genre_conservative", "genre_balanced_conservative"}:
            prompt_variant = "task6_genre_conservative"
        elif task_id == 6 and task6_mode == "genre_balanced":
            prompt_variant = "task6_genre_strict"
        elif task_id == 6 and task6_mode in {"genre_hypothesis", "genre_balanced_hypothesis"}:
            prompt_variant = "task6_genre_hypothesis"
        elif task_id == 6 and task6_mode in {"genre_anchor_hypothesis", "genre_balanced_anchor_hypothesis"}:
            prompt_variant = "task6_genre_anchor_hypothesis"
        elif task_id == 6 and task6_mode == "genre_balanced_anchor_genrefirst":
            prompt_variant = "task6_genre_anchor_genrefirst"
        elif task_id == 8 and task8_mode == "compact":
            prompt_variant = "task8_compact"
        prompt = build_prompt(
            task_id=task_id,
            task_description=task_description,
            examples_text=sample_examples_text,
            text_to_annotate=sample_input,
            prompt_variant=prompt_variant,
        )

        raw_prediction = ""
        for attempt in range(DEFAULT_REQUEST_RETRIES + 1):
            raw_prediction = annotate(prompt, infer_cfg)
            if raw_prediction:
                break
            if attempt < DEFAULT_REQUEST_RETRIES:
                time.sleep(DEFAULT_RETRY_BACKOFF_SECONDS * (attempt + 1))

        prediction = postprocess_prediction(raw_prediction, task_id)

        if task_id == 4 and task4_mode == "self_refine":
            review_cfg = InferenceConfig(
                api_base=infer_cfg.api_base,
                model_name=infer_cfg.model_name,
                timeout=infer_cfg.timeout,
                max_new_tokens=args.task4_review_max_new_tokens,
                temperature=infer_cfg.temperature,
                top_p=infer_cfg.top_p,
                stop=infer_cfg.stop,
            )
            review_prompt = build_task4_review_prompt(
                task_description=task_description,
                text_to_annotate=sample_input,
                draft_answer=prediction,
            )
            review_raw = annotate(review_prompt, review_cfg)
            reviewed_prediction = postprocess_prediction(review_raw, task_id)
            if reviewed_prediction is not None:
                prediction = reviewed_prediction

        if task_id == 4 and task4_mode in {"gated_retry", "prompt_gated_retry"}:
            issues = assess_task4_prediction(sample_input, prediction)
            retry_count = 0
            while issues and retry_count < args.task4_gate_retries:
                retry_prompt = build_task4_retry_prompt(
                    task_description=task_description,
                    examples_text=examples_text,
                    text_to_annotate=sample_input,
                    issues=issues,
                )
                retry_raw = annotate(retry_prompt, infer_cfg)
                retry_prediction = postprocess_prediction(retry_raw, task_id)
                if retry_prediction is not None:
                    prediction = retry_prediction
                issues = assess_task4_prediction(sample_input, prediction)
                retry_count += 1

        if task_id == 8 and task8_mode == "repair_pass":
            issues = assess_task8_prediction(prediction)
            retry_count = 0
            while issues and retry_count < args.task8_repair_retries:
                retry_prompt = build_task8_repair_prompt(
                    text_to_annotate=sample_input,
                    draft_code=prediction,
                    issues=issues,
                )
                retry_raw = annotate(retry_prompt, infer_cfg)
                retry_prediction = postprocess_prediction(retry_raw, task_id)
                if retry_prediction is not None:
                    prediction = retry_prediction
                issues = assess_task8_prediction(prediction)
                retry_count += 1

            rewrite_count = 0
            while issues and rewrite_count < args.task8_rewrite_retries:
                rewrite_prompt = build_task8_rewrite_prompt(
                    text_to_annotate=sample_input,
                    issues=issues,
                )
                rewrite_raw = annotate(rewrite_prompt, infer_cfg)
                rewrite_prediction = postprocess_prediction(rewrite_raw, task_id)
                if rewrite_prediction is not None:
                    prediction = rewrite_prediction
                issues = assess_task8_prediction(prediction)
                rewrite_count += 1

        ok = prediction is not None
        if task_id == 8:
            safe_prediction = prediction if prediction is not None else ""
            row = {
                "test_sample_id": sample_id,
                "prediction": safe_prediction,
                "code": safe_prediction,
            }
        else:
            row = {"test_sample_id": sample_id, "prediction": prediction}
        audit_row: Dict[str, object] = {
            "task_id": task_id,
            "sample_id": str(sample_id),
            "stage": "candidate_generation",
            "mode": {
                2: task2_mode,
                5: task5_mode,
                6: task6_mode,
                8: task8_mode,
            }.get(task_id, "baseline"),
            "target_min_context_tokens": context_target,
            "used_example_tokens": sample_used_tokens,
            "selected_example_count": sample_selected_count,
            "pass_min_context": sample_used_tokens >= context_target,
            "cache_friendly_context": bool(args.cache_friendly_context),
            "cache_prefix_context_tokens": int(args.cache_prefix_context_tokens),
            "cache_prefix_used_tokens": int(cache_prefix_used_tokens if cache_prefix_enabled else 0),
        }
        return sample_index, row, ok, audit_row

    output_file = next_output_file(output_dir, task_id)
    print(f"[Task {task_id}] writing predictions to: {output_file}")

    ordered_rows: List[dict] = [None] * len(test_samples)
    ordered_audit_rows: List[dict] = [None] * len(test_samples)
    failed = 0
    if test_samples:
        with ThreadPoolExecutor(max_workers=client_concurrency) as executor:
            futures = {
                executor.submit(_infer_one, sample_index, sample): sample_index
                for sample_index, sample in enumerate(test_samples)
            }
            with tqdm(total=len(test_samples), desc=f"Task {task_id}", unit="sample") as pbar:
                for future in as_completed(futures):
                    sample_index, row, ok, audit_row = future.result()
                    ordered_rows[sample_index] = row
                    ordered_audit_rows[sample_index] = audit_row
                    if not ok:
                        failed += 1
                    pbar.update(1)
                    pbar.set_postfix_str(f"ok={pbar.n - failed} fail={failed}")

    with output_file.open("w", encoding="utf-8") as writer:
        for row in ordered_rows:
            if row is None:
                continue
            writer.write(json.dumps(row, ensure_ascii=False) + "\n")

    if args.official_min_context:
        audit_dir = Path(args.context_audit_dir).resolve() if args.context_audit_dir else output_dir
        audit_mode = {
            2: task2_mode,
            5: task5_mode,
            6: task6_mode,
            8: task8_mode,
        }.get(task_id, "baseline")
        audit_path = _context_audit_path(audit_dir, task_id, f"{audit_mode}_{output_dir.name}")
        _write_context_audit(
            audit_path,
            [row for row in ordered_audit_rows if row is not None],
        )
        _write_context_audit(audit_dir / f"context_audit_task{task_id}.jsonl", [
            {
                "task_id": task_id,
                "stage": "candidate_generation",
                "mode": audit_mode,
                "target_min_context_tokens": context_target,
                "rows": sum(1 for row in ordered_audit_rows if row is not None),
                "min_used_example_tokens": min(
                    int(row["used_example_tokens"])
                    for row in ordered_audit_rows
                    if row is not None
                ) if any(row is not None for row in ordered_audit_rows) else 0,
                "context_pass_rows": sum(
                    1 for row in ordered_audit_rows if row is not None and row.get("pass_min_context")
                ),
                "mode_audit_file": str(audit_path),
            }
        ])

    print(
        f"[Task {task_id}] examples={selected_count}, example_tokens~{used_tokens}, "
        f"samples={len(test_samples)}, failed={failed}, output={output_file}"
    )
    return output_file


def main() -> None:
    args = parse_args()
    print("[Pipeline] checking API status...")
    ensure_api_ready(args)
    tasks = list(range(1, 9)) if args.all_tasks else [args.task_id]
    print(f"[Pipeline] tasks to run: {tasks}")
    for task_id in tqdm(tasks, desc="All tasks", unit="task"):
        run_one_task(task_id, args)


if __name__ == "__main__":
    main()
