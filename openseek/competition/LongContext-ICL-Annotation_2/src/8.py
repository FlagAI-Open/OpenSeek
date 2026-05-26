import json
import os
import re
import random

import numpy as np
import openai
from tqdm import tqdm

# --- Configuration ---
client = openai.OpenAI(api_key="EMPTY", base_url="http://localhost:2026/v1")

DATA_PATH = "../data/openseek-8_kernel_generation.json"
OUTPUT_PATH = "../outputs/result/openseek-8-v1.jsonl"
TASK_ID = 8

REFERENCE_COUNT = 166
REFERENCE_SELECT_TOP_K = 5
REFERENCE_MIN_LEN = 50000
REFERENCE_MAX_CHARS = 300000
EMBED_DEBUG = True
PREFERRED_REFERENCE_IDS = [
	
]

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DEVICE = "cpu"
EMBED_BATCH_SIZE = 32

_EMBEDDER = None
_EXAMPLE_INDEX = None
_TASK_DATASET = None


def _extract_example_output_text(example: dict) -> str:
	output_value = example.get("output", "")
	if isinstance(output_value, list):
		return str(output_value[0]).strip() if output_value else ""
	return str(output_value).strip()


def _extract_example_input_text(example: dict) -> str:
	return str(example.get("input", "")).strip()


def _get_embedder():
	global _EMBEDDER
	if _EMBEDDER is None:
		from sentence_transformers import SentenceTransformer
		_EMBEDDER = SentenceTransformer(EMBED_MODEL_NAME, device=EMBED_DEVICE)
	return _EMBEDDER


def _load_task_dataset():
	global _TASK_DATASET
	if _TASK_DATASET is None:
		with open(DATA_PATH, "r", encoding="utf-8") as f:
			_TASK_DATASET = json.load(f)
	return _TASK_DATASET


def _build_example_records(examples: list) -> list:
	records = []
	for example in examples:
		input_text = _extract_example_input_text(example)
		text = _extract_example_output_text(example)
		if not text:
			continue
		embed_text = f"{input_text}\n{text}".strip()
		records.append({
			"id": example.get("id", "unknown"),
			"input": input_text,
			"output": text,
			"embed_text": embed_text,
			"length": len(text),
		})
	return records


def _ensure_example_index(examples: list):
	global _EXAMPLE_INDEX
	if _EXAMPLE_INDEX is not None:
		return _EXAMPLE_INDEX

	records = _build_example_records(examples)
	if not records:
		_EXAMPLE_INDEX = (records, None)
		return _EXAMPLE_INDEX

	try:
		model = _get_embedder()
		texts = [r.get("embed_text", r["input"]) for r in records]
		embeddings = model.encode(
			texts,
			batch_size=EMBED_BATCH_SIZE,
			show_progress_bar=False,
			normalize_embeddings=True,
		)
		embeddings = np.asarray(embeddings, dtype="float32")
		if EMBED_DEBUG:
			print(
				"[Embedding] Built example index: "
				f"records={len(records)}, embeddings_shape={embeddings.shape}"
			)
	except Exception as exc:
		print(f"[Embedding] Failed to build example index: {exc}")
		embeddings = None

	_EXAMPLE_INDEX = (records, embeddings)
	return _EXAMPLE_INDEX


def _rank_records_by_embedding(records: list, embeddings, query_text: str) -> list:
	if not query_text or embeddings is None or not records:
		return records
	try:
		model = _get_embedder()
		query_vec = model.encode(
			[query_text],
			batch_size=1,
			show_progress_bar=False,
			normalize_embeddings=True,
		)
		query_vec = np.asarray(query_vec, dtype="float32")[0]
		scores = embeddings @ query_vec
		order = np.argsort(-scores)
		return [records[int(i)] for i in order]
	except Exception as exc:
		print(f"[Embedding] Failed to rank examples: {exc}")
		return records


def _select_preferred_records(records: list) -> list:
	record_by_id = {str(r.get("id", "")): r for r in records}
	return [record_by_id[sid] for sid in PREFERRED_REFERENCE_IDS if sid in record_by_id]


def select_output_references(examples: list, count: int = REFERENCE_COUNT, query_text: str = "") -> list:
	"""Randomly sample candidates, then rank by embedding similarity and keep top-K."""
	records, embeddings = _ensure_example_index(examples)
	if not records or count <= 0:
		return []

	pool_size = min(len(records), count)
	if pool_size == len(records):
		candidate_indices = list(range(len(records)))
	else:
		candidate_indices = random.sample(range(len(records)), k=pool_size)
	candidate = [(idx, records[idx]) for idx in candidate_indices]

	top_k = min(REFERENCE_SELECT_TOP_K, len(candidate))
	if query_text and embeddings is not None:
		try:
			if EMBED_DEBUG:
				print(
					"[Embedding] Using embedding ranking: "
					f"candidates={len(candidate)}, top_k={top_k}"
				)
			model = _get_embedder()
			query_vec = model.encode(
				[query_text],
				batch_size=1,
				show_progress_bar=False,
				normalize_embeddings=True,
			)
			query_vec = np.asarray(query_vec, dtype="float32")[0]
			candidate_embeddings = embeddings[[idx for idx, _ in candidate]]
			scores = candidate_embeddings @ query_vec
			order = np.argsort(-scores)
			if EMBED_DEBUG:
				best_scores = [float(scores[int(i)]) for i in order[:top_k]]
				print(f"[Embedding] Top-{top_k} scores: {best_scores}")
			return [candidate[int(i)][1] for i in order[:top_k]]
		except Exception as exc:
			print(f"[Embedding] Failed to rank examples: {exc}")

	if EMBED_DEBUG:
		reason = "missing query_text" if not query_text else "embeddings unavailable"
		print(f"[Embedding] Falling back to random selection ({reason}).")

	return [r for _, r in candidate[:top_k]]


def format_output_references_for_prompt(references: list) -> str:
	if not references:
		return ""

	blocks = []
	for i, ref in enumerate(references, 1):
		input_text = ref.get("input", "") or ""
		output_text = ref.get("output", "") or ""
		if len(output_text) > REFERENCE_MAX_CHARS:
			output_text = output_text[:REFERENCE_MAX_CHARS].rstrip() + "\n"
		block = (
			#f"Few-shot Example {i} (id={ref.get('id','unknown')}, output_len={ref.get('length', len(output_text))}):\n"
			#f"Input:\n```\n{input_text}\n```\n\n"
			f"```python\n{output_text}\n```"
		)
		blocks.append(block)

	return "\n\n".join(blocks)


TRAILING_NOTE = "After generation, verify if the Triton wrapper aligns with the provided func_inputs. If not, regenerate."


def clean_input_text(input_text: str) -> str:
	if not input_text:
		return ""
	cleaned = input_text.replace(TRAILING_NOTE, "")
	return cleaned.rstrip()


def extract_input_sections(input_text: str) -> dict:
	cleaned_input = clean_input_text(input_text).strip()

	desc_match = re.search(r'Functional Description:\s*(.*?)(?=\nWrapper Entry Information:|$)', cleaned_input, re.DOTALL)
	wrap_match = re.search(r'Wrapper Entry Information:\s*(.*?)(?=\nMath:|$)', cleaned_input, re.DOTALL)
	math_match = re.search(r'Math:\s*(.*?)(?=\nother:|$)', cleaned_input, re.DOTALL | re.IGNORECASE)
	other_match = re.search(r'other:\s*(.*)$', cleaned_input, re.DOTALL | re.IGNORECASE)

	desc_text = desc_match.group(1).strip() if desc_match else ""
	wrap_text = wrap_match.group(1).strip() if wrap_match else ""
	math_text = math_match.group(1).strip() if math_match else ""
	other_text = other_match.group(1).strip() if other_match else ""

	# Fallback: if no explicit sections, treat whole input as functional description
	if not desc_text and not wrap_text and not math_text:
		desc_text = cleaned_input

	return {
		"desc": desc_text,
		"wrap": wrap_text,
		"math": math_text,
		"other": other_text,
	}


def trim_wrapper_entry_information(wrap_text: str) -> str:
	"""Keep only the interface signature line from wrapper entry info."""
	if not wrap_text:
		return ""

	lines = [line.strip() for line in str(wrap_text).splitlines() if line.strip()]
	if not lines:
		return ""

	first_line = lines[0]
	if "Args:" in first_line:
		first_line = first_line.split("Args:", 1)[0].strip()

	# If the signature and Args are on separate lines, stop at the first Args line.
	for line in lines[1:]:
		if line.startswith("Args:"):
			break

	return first_line


def build_icl_prompt(task_file_path: str, input_text: str) -> str:
	"""Build Task 8 prompt using prompt-style template with one-shot example."""
	dataset = _load_task_dataset() if task_file_path == DATA_PATH else None
	if dataset is None:
		with open(task_file_path, "r", encoding="utf-8") as f:
			dataset = json.load(f)

	definition = dataset.get("Definition", [""])[0]
	sections = extract_input_sections(input_text)
	wrap_text = trim_wrapper_entry_information(sections.get("wrap", ""))

	base_prompt = (
		    " "
			"Few-shot Examples:\n"
			#f"{definition}\n\n"
	)

	# Few-shot I/O references
	query_text = "\n".join([
		sections.get("desc", ""),
		wrap_text,
		sections.get("math", ""),
		sections.get("other", ""),
	]).strip()
	references = select_output_references(
		dataset.get("examples", []),
		REFERENCE_COUNT,
		query_text=query_text,
	)
	output_reference_text = format_output_references_for_prompt(references)
	if output_reference_text:
		base_prompt +="" + output_reference_text + "\n\n"
		#base_prompt += "Few-shot Examples:\n"
		#base_prompt +="```python\nimport triton\nimport triton.language as tl\nimport torch\n\n@triton.jit\ndef _fwd_kernel_token_att2(\n    Prob,\n    V,\n    Out,\n    Req_to_tokens,\n    B_req_idx,\n    B_Start_Loc,\n    B_Seqlen,\n    stride_req_to_tokens_b,\n    stride_req_to_tokens_s,\n    stride_ph,\n    stride_pbs,\n    stride_vbs,\n    stride_vh,\n    stride_vd,\n    stride_obs,\n    stride_oh,\n    stride_od,\n    kv_group_num,\n    BLOCK_DMODEL: tl.constexpr,\n    BLOCK_N: tl.constexpr,\n):\n    cur_batch = tl.program_id(0)\n    cur_head = tl.program_id(1)\n\n    cur_kv_head = cur_head // kv_group_num\n\n    offs_n = tl.arange(0, BLOCK_N)\n    offs_d = tl.arange(0, BLOCK_DMODEL)\n    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)\n    cur_batch_start_index = 0\n    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)\n    cur_batch_req_idx = tl.load(B_req_idx + cur_batch)\n\n    v_loc_off = cur_batch_req_idx * stride_req_to_tokens_b + (cur_batch_start_index + offs_n) * stride_req_to_tokens_s\n    p_offs = cur_head * stride_ph + (cur_batch_in_all_start_index + offs_n) * stride_pbs\n    v_offs = cur_kv_head * stride_vh + offs_d[None, :] * stride_vd\n\n    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)\n    for start_n in range(0, cur_batch_seq_len, BLOCK_N):\n        start_n = tl.multiple_of(start_n, BLOCK_N)\n        p_value = tl.load(Prob + p_offs + start_n, mask=(start_n + offs_n) < cur_batch_seq_len, other=0.0)\n        v_loc = tl.load(\n            Req_to_tokens + v_loc_off + start_n * stride_req_to_tokens_s,\n            mask=(start_n + offs_n) < cur_batch_seq_len,\n            other=0.0,\n        )\n        v_value = tl.load(\n            V + v_offs + v_loc[:, None] * stride_vbs, mask=(start_n + offs_n[:, None]) < cur_batch_seq_len, other=0.0\n        )\n        acc += tl.sum(p_value[:, None] * v_value, 0)\n\n    acc = acc.to(Out.dtype.element_ty)\n    off_o = cur_batch * stride_obs + cur_head * stride_oh + offs_d * stride_od\n    out_ptrs = Out + off_o\n    tl.store(out_ptrs, acc)\n    return\n\n\n@torch.no_grad()\ndef token_att_fwd2(prob, v, out, Req_to_tokens, B_req_idx, B_Start_Loc, B_Seqlen):\n    BLOCK = 128\n    batch, head = B_req_idx.shape[0], prob.shape[0]\n    grid = (batch, head)\n    num_warps = 4\n    dim = v.shape[-1]\n\n    kv_group_num = prob.shape[0] // v.shape[1]\n\n    _fwd_kernel_token_att2[grid](\n        prob,\n        v,\n        out,\n        Req_to_tokens,\n        B_req_idx,\n        B_Start_Loc,\n        B_Seqlen,\n        Req_to_tokens.stride(0),\n        Req_to_tokens.stride(1),\n        prob.stride(0),\n        prob.stride(1),\n        v.stride(0),\n        v.stride(1),\n        v.stride(2),\n        out.stride(0),\n        out.stride(1),\n        out.stride(2),\n        kv_group_num=kv_group_num,\n        BLOCK_DMODEL=dim,\n        BLOCK_N=BLOCK,\n        num_warps=num_warps,\n        num_stages=1,\n    )\n    return\n\n\n\n\n```\n"
		#base_prompt +="```python\nimport torch\nimport triton\nimport triton.language as tl\n\n# global quantize and transpose\n@triton.autotune(\n    configs=[\n        triton.Config({\"BLOCK_M\": 128, \"BLOCK_N\": 128, \"GROUP_M\": 8}, num_warps=4),\n        triton.Config({\"BLOCK_M\": 128, \"BLOCK_N\": 128, \"GROUP_M\": 8}, num_warps=4),\n        # ...\n    ],\n    key=[\"M\", \"N\"],\n)\n@triton.jit\ndef _quantize_global_transpose(\n    A,\n    absmax_inv_ptr,\n    B,\n    stride_am,\n    stride_an,\n    stride_bn,\n    stride_bm,\n    M,\n    N,\n    BLOCK_M: tl.constexpr,\n    BLOCK_N: tl.constexpr,\n    GROUP_M: tl.constexpr,\n):\n    pid = tl.program_id(0)\n    grid_m = (M + BLOCK_M - 1) // BLOCK_M\n    grid_n = (N + BLOCK_N - 1) // BLOCK_N\n\n    width = GROUP_M * grid_n\n    group_id = pid // width\n    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)\n    pid_m = group_id * GROUP_M + (pid % group_size)\n    pid_n = (pid % width) // group_size\n\n    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)\n    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)\n    A = A + (rm[:, None] * stride_am + rn[None, :] * stride_an)\n    mask = (rm < M)[:, None] & (rn < N)[None, :]\n    a = tl.load(A, mask=mask)\n    absmax_inv = tl.load(absmax_inv_ptr)\n\n    # rematerialize to save registers\n    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)\n    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)\n    B = B + (rm[:, None] * stride_bm + rn[None, :] * stride_bn)\n    mask = (rm < M)[:, None] & (rn < N)[None, :]\n\n    output = tl.extra.cuda.libdevice.llrint(127.0 * (a * absmax_inv))\n\n    tl.store(B, output, mask=mask)\n\ndef quantize_global_transpose(input):\n    absmax = input.abs().max().unsqueeze(0)\n    absmax_inv = 1.0 / absmax\n    M, N = input.shape\n    out = torch.empty(N, M, device=\"cuda\", dtype=torch.int8)\n\n    assert out.size(0) == N and out.size(1) == M\n    assert input.stride(0) == 1 or input.stride(1) == 1\n    assert out.stride(0) == 1 or out.stride(1) == 1\n\n    grid = lambda META: (triton.cdiv(M, META[\"BLOCK_M\"]) * triton.cdiv(N, META[\"BLOCK_N\"]),)\n    _quantize_global_transpose[grid](\n        input,\n        absmax_inv,\n        out,\n        input.stride(0),\n        input.stride(1),\n        out.stride(0),\n        out.stride(1),\n        M,\n        N,\n    )\n    return out, absmax\n\n\n\n\n\n```\n"
		#base_prompt +="```python\nimport torch\nimport triton\nimport triton.language as tl\n\n@triton.jit\ndef matmul_tma_load_store(\n        a_ptr, b_ptr, c_ptr,\n        M, N, K,\n        stride_am, stride_ak,\n        stride_bk, stride_bn,\n        stride_cm, stride_cn,\n        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,\n        OUTPUT_F16: tl.constexpr\n):\n    # Create block pointers for A, B, and C matrices\n    a_block_ptr = tl.make_block_ptr(base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak), offsets=(0, 0),\n                                    block_shape=(BLOCK_M, BLOCK_K), order=(1, 0))\n    b_block_ptr = tl.make_block_ptr(base=b_ptr, shape=(K, N), strides=(stride_bk, stride_bn), offsets=(0, 0),\n                                    block_shape=(BLOCK_K, BLOCK_N), order=(0, 1))\n    c_block_ptr = tl.make_block_ptr(base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn), offsets=(0, 0),\n                                    block_shape=(BLOCK_M, BLOCK_N), order=(1, 0))\n    # Load A and B blocks\n    a = tl.load(a_block_ptr)\n    b = tl.load(b_block_ptr)\n\n    # Compute matrix product\n    c = tl.dot(a, b)\n    # Optionally convert the result to float16\n    if OUTPUT_F16:\n        c = c.to(tl.float16)\n\n    # Store the result\n    tl.store(c_block_ptr, c)\n\n\ndef warpper_tma_load_store(M, N, K, NUM_CTAS, NUM_WARPS, TRANS_A, TRANS_B, OUTPUT_F16):\n    # Prepare input matrices\n    if (TRANS_A):\n        a = torch.randn((K, M), device='cuda', dtype=torch.float16).T\n    else:\n        a = torch.randn((M, K), device='cuda', dtype=torch.float16)\n    if (TRANS_B):\n        b = torch.randn((N, K), device='cuda', dtype=torch.float16).T\n    else:\n        b = torch.randn((K, N), device='cuda', dtype=torch.float16)\n\n    # Prepare output matrix\n    c = torch.empty((M, N), device=a.device, dtype=torch.float32)\n    if OUTPUT_F16:\n        c = torch.empty((M, N), device=a.device, dtype=torch.float16)\n\n    # Execute Triton kernel\n    matmul_tma_load_store[(1, 1)](\n        a_ptr=a, b_ptr=b, c_ptr=c,\n        M=M, N=N, K=K,\n        stride_am=a.stride(0), stride_ak=a.stride(1),\n        stride_bk=b.stride(0), stride_bn=b.stride(1),\n        stride_cm=c.stride(0), stride_cn=c.stride(1),\n        BLOCK_M=M, BLOCK_N=N, BLOCK_K=K,\n        num_warps=NUM_WARPS, num_ctas=NUM_CTAS,\n        OUTPUT_F16=OUTPUT_F16)\n    return c\n    \n\n\n\n\n```\n"
		#base_prompt +="```python\nimport torch\nimport triton\nimport triton.language as tl\n\n# Define the Triton kernel\n@triton.jit\ndef spinning_lock_kernel(P, C, locks, num_sms, k, M, N, stride_cm, stride_cn, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):\n    pid = tl.program_id(0)\n    pid_m = pid // num_sms\n    pid_n = pid % num_sms\n\n    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)  # Assuming acc initialization\n\n    # Perform reduction for every kth pid\n    for iters in range(1, 10):\n        if (pid % k == 0):\n            next_pid = pid + 1\n\n            while next_pid < pid + k and next_pid < num_sms:\n                while tl.atomic_cas(locks + next_pid, 1, 1) != 1:\n                    pass\n\n                rm1 = tl.arange(0, BLOCK_SIZE_M)\n                rn1 = tl.arange(0, BLOCK_SIZE_N)\n                P_ = P + next_pid * BLOCK_SIZE_M * BLOCK_SIZE_N + rm1[:, None] * BLOCK_SIZE_N + rn1[None, :]\n                acc1 = tl.load(P_)\n                acc += acc1\n\n                next_pid += 1\n              \n        # Store results using temporary storage P for every k-1 pids\n        else:\n            rm1 = tl.arange(0, BLOCK_SIZE_M)\n            rn1 = tl.arange(0, BLOCK_SIZE_N)\n            P_ = P + pid * BLOCK_SIZE_M * BLOCK_SIZE_N + rm1[:, None] * BLOCK_SIZE_N + rn1[None, :]\n            tl.store(P_, acc)\n            tl.atomic_xchg(locks + pid, 1)\n\n        # Store final results in C\n        rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)\n        rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)\n        C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn\n        mask = (rm < M)[:, None] & (rn < N)[None, :]\n        tl.store(C_, acc, mask=mask)\n\n\ndef spinning_lock(P, C, locks, num_sms, k, M, N, stride_cm, stride_cn, BLOCK_SIZE_M, BLOCK_SIZE_N):\n    grid = (num_sms,)\n    spinning_lock_kernel[grid](\n        P, C, locks, num_sms, k, M, N, stride_cm, stride_cn, BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N,)\n\n\n\n\n```\n"
		#base_prompt +="```python\nimport torch\nimport triton\nimport triton.language as tl\n\n@triton.jit\ndef rmsnorm_triton(x_ptr, rms_w_ptr, out_ptr,\n                   stride_x_batch, stride_x_m, stride_x_k,\n                   stride_rms_w,\n                   stride_out_batch, stride_out_m, stride_out_k,\n                   N_SIZE: tl.constexpr, eps: tl.constexpr, BLOCK_N_SIZE: tl.constexpr):\n    pid_batch = tl.program_id(0)\n    pid_m = tl.program_id(1)\n\n    # parallel at m dimension\n    offset_m = pid_batch * stride_x_batch + pid_m * stride_x_m\n    block_n_size = tl.arange(0, BLOCK_N_SIZE)\n    var = tl.zeros((BLOCK_N_SIZE,), tl.float32)\n    # parallel between blocks\n    for block_n_strart_ptr in range(0, N_SIZE, BLOCK_N_SIZE):\n        offset_n = block_n_strart_ptr + block_n_size\n        x_ptr_mask = offset_n < N_SIZE\n        x = tl.load(x_ptr + offset_m + offset_n * stride_x_k, mask=x_ptr_mask, other=0.)  # careful stride_x_k\n        xf = x.to(tl.float32)\n        var += xf*xf\n    var = tl.sum(var, axis=0) / N_SIZE  # reduce between wrap\n    std = tl.sqrt(var + eps)\n\n    for block_n_strart_ptr in range(0, N_SIZE, BLOCK_N_SIZE):\n        offset_n = block_n_strart_ptr + block_n_size\n        x_ptr_mask = offset_n < N_SIZE\n\n        rms_w_offset = tl.load(rms_w_ptr + offset_n * stride_rms_w, mask=x_ptr_mask)\n        x = tl.load(x_ptr + offset_m + offset_n * stride_x_k, mask=x_ptr_mask, other=0.)\n\n        x_new = x / std\n        out = x_new * rms_w_offset\n        out_offset = pid_batch * stride_out_batch + pid_m * stride_out_m + offset_n * stride_out_k\n        tl.store(out_ptr + out_offset, out, mask=x_ptr_mask)\n\n\ndef rmsnorm_wrapper(x, rms_weights, eps=1e-6):\n    batch, M, K = x.shape\n    out = torch.empty_like(x)\n    rmsnorm_triton[(batch, M,)](x, rms_weights, out,\n                                *x.stride(),\n                                *rms_weights.stride(),  # 1\n                                *out.stride(),\n                                N_SIZE=K, eps=eps, BLOCK_N_SIZE=4096,\n                                num_warps=16\n                                )\n    return out\n\n\n\n\n\n```\n"
		
		#"Give the complete structure only, and start the answer with ```python and end with ```."
		
        
		base_prompt += "The needed information are as follows:"        

	# Add functional sections from the input
	if sections.get("desc"):
		base_prompt += "Functional Description:\n" + sections.get("desc") + "\n\n"
	if wrap_text:
		base_prompt += "Wrapper Entry Information:\n" + sections.get("wrap", "") + "\n\n"
	if sections.get("math"):
		base_prompt += "Math:\n" + sections.get("math") + "\n\n"
	if sections.get("other"):
		base_prompt += "Other Information:\n" + sections.get("other") + "\n\n"

	target_prompt = (
		"Now solve the target input.\n"
		#f"--- Target ---\nInput: {input_text}\n"
		"You MUST implement the complete, runnable Triton kernel + Python wrapper in one response.\n"
		#"Strictly follow the description in the Functional Description and provide the runnable code.\n"
		"1.Clarify the mathematical objective.In actual programming, it must be translated into an appropriate computational form.Parse the wrapper entry information (function name, arguments, return type).\n"
		#"2.Parse the wrapper entry information (function name, arguments, return type).\n"
		"2.Design and Implement the Triton kernel:Decide on parallelization strategy (e.g., one thread per element).Define necessary kernel parameters (BLOCK_SIZE, strides, etc.),and using the provided math formula.\n"
		#"4.Implement the Triton kernel using the provided math formula.\n"
		"3.Implement the Python wrapper function that exactly matches the provided interface.\n"        

		"Output python code start with ```python ,end with ```.\n"
		"```python\n"
	)

	return base_prompt + target_prompt
"""
        You MUST implement the complete, runnable Triton kernel + Python wrapper in one response.
        Requirements:
        1) Implementing custom algorithms or functions using Triton, and ensuring correct block masking and stride handling for memory safety.
        2) The wrapper signature MUST exactly match the provided "Wrapper Entry Information".
        3) Use correct strides for all tensors; every tl.load/tl.store must have masks for out-of-bounds.
        Only output runnable code.
        """



def count_answer(text: str) -> str:
	"""Extract task 8 code answer, preferring content after </think>."""
	if not text:
		return "no"

	if "</think>" in text:
		text = text.split("</think>")[-1].strip()

	block_pattern = re.compile(r"```python\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
	matches = block_pattern.findall(text)
	if matches:
		return matches[-1].strip()

	import_pattern = re.compile(r"(import\s+.*?)```", re.DOTALL)
	import_matches = import_pattern.findall(text)
	if import_matches:
		return import_matches[-1].strip()

	lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
	if lines and ("import " in lines[0] or "def " in lines[0] or "class " in lines[0]):
		return "\n".join(lines)

	return "no"


def persist_results_jsonl(results_dict: dict, test_samples: list) -> None:
	"""Write ordered results back to the JSONL file immediately."""
	ordered_results = []
	for sample in test_samples:
		sample_id = sample["id"]
		if sample_id in results_dict:
			ordered_results.append(results_dict[sample_id])

	with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
		for result in ordered_results:
			f.write(json.dumps(result, ensure_ascii=False) + "\n")


def call_llm(prompt_text: str) -> str:
	"""Call the local OpenAI-compatible model endpoint."""
	try:
		response = client.chat.completions.create(
			model="/root/ascend/log/debug/Qwen3-4B",
			messages=[
				{
					"role": "system",
					"content": "You are an expert in writing Triton operators for efficient GPU programming.\n ",
				},
				{"role": "user", "content": prompt_text},
			],
			temperature=0.5,
			top_p=0.85,
			max_tokens=20000,
		)
		whole_result = response.choices[0].message.content
		print(f"\n[Model Response] Task {TASK_ID}:\n{whole_result}\n" + "=" * 50)
		return whole_result
	except Exception as e:
		print(f"Exception occurred for Task {TASK_ID}: {e}")
		return ""


def reflect_and_revise_answer(sample_id: str, input_text: str, candidate_answer: str) -> str:
	"""Run a secondary review and replace the candidate when the model gives a different answer."""
	sections = extract_input_sections(input_text)
	wrap_text = trim_wrapper_entry_information(sections.get("wrap", ""))
	query_parts = [
		sections.get("desc", ""),
		wrap_text,
		sections.get("math", ""),
		sections.get("other", ""),
	]
	if candidate_answer and candidate_answer != "no":
		query_parts.append(f"#sym:candidate_answer\n{candidate_answer}")
	query_text = "\n".join(part for part in query_parts if part).strip()
	dataset = _load_task_dataset()
	references = select_output_references(
		dataset.get("examples", []),
		REFERENCE_COUNT,
		query_text=query_text,
	)
	ref_text = format_output_references_for_prompt(references)
	ref_block = ""
	if ref_text:
		ref_block = "Few-shot Examples:\n" + ref_text + "\n\n"

	review_prompt = (
		ref_block +
        
		"You are performing a strict revision pass for Task.\n"
		"Compare the generated code against the functional description and the wrapper entry information.\n"
		"Focus on memory safety, block masking, stride correctness, and exact wrapper compatibility.\n\n"
		"Checklist:\n"
		"1. Verify the Python wrapper signature, defaults, shape checks, dtype checks, and return behavior,Parameter order is consistent with the definition..\n"
		"2. Verify every Triton pointer expression uses the correct stride for its tensor dimension.\n"
		"3. Verify every partial block has a mask on tl.load and tl.store, with safe fallback values,grid use BLOCK + triton.cdiv.\n"
		"Refer to the official Triton basic rules and correct any existing errors.\n"
		"Functional Description:\n" + input_text + "\n\n"
		"Generated Code:\n" + candidate_answer + "\n\n"
		"Output only a single corrected code block and nothing else."
		"Only output the code ```python\n  ``` ."
	)

	 #"Please review the Functional Description and Code\n"
	
		
	review_raw = call_llm(review_prompt)
	revised_answer = count_answer(review_raw)

	if revised_answer and revised_answer != "no" and revised_answer != candidate_answer:
		print(
			f"[Reflect] Task {TASK_ID}, sample {sample_id}: "
			f"updated answer."
		)
		return revised_answer

	return candidate_answer


def main():
	if not os.path.exists(os.path.dirname(OUTPUT_PATH)):
		os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

	with open(DATA_PATH, "r", encoding="utf-8") as f:
		dataset = json.load(f)
	test_samples = dataset.get("test_samples", [])
	all_sample_ids = {sample["id"] for sample in test_samples}

	results_dict = {}
	if os.path.exists(OUTPUT_PATH):
		with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
			for line in f:
				if not line.strip():
					continue
				try:
					data = json.loads(line)
					results_dict[data["test_sample_id"]] = data
				except Exception:
					continue

	completed_before_run = all_sample_ids.issubset(set(results_dict.keys()))
	run_review_mode = False
	review_start_question = 1
	if completed_before_run:
		user_choice = input(
			f"All test samples are already present in openseek-8-v1.jsonl. Enter review mode now? (y/n): "
		).strip().lower()
		run_review_mode = user_choice == "y"
		if not run_review_mode:
			print("Skip review mode by user choice (n). Exiting.")
			return

		start_choice = input(
			f"Enter review start question number (1-{len(test_samples)}, default 1): "
		).strip()
		if start_choice:
			try:
				review_start_question = max(1, min(len(test_samples), int(start_choice)))
			except ValueError:
				print("Invalid start question number, defaulting to 1.")
				review_start_question = 1

	if not completed_before_run:
		with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
			for sample in tqdm(test_samples):
				sample_id = sample["id"]
				if sample_id in results_dict:
					continue

				input_text = sample.get("input", "")
				prompt_text = build_icl_prompt(DATA_PATH, input_text)

				#print_prompt
				print(f"\n{'='*20} PROMPT SENT TO MODEL (TASK {TASK_ID}) {'='*20}")
				print(prompt_text)
				print(f"{'='*60}\n")

				response_raw = call_llm(prompt_text)
				prediction = count_answer(response_raw)

				if prediction == "no" or not prediction:
					sections = extract_input_sections(input_text)
					wrap_text = trim_wrapper_entry_information(sections.get("wrap", ""))
					prediction = (
						"import torch\nimport triton\nimport triton.language as tl\n@triton.autotune( )\n@triton.jit\n"
						"def kernel(\n    in1, in2, out,\n    stride1_0, stride1_1,\n    N,\n    BLOCK_SIZE: tl.constexpr,\n):\n"
						"    pid = tl.program(0)\n    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)\n"
						"    mask = offs < N\n    x = tl.load(in1 + offs, mask=mask, other=0.0)\n"
						"    y = tl.load(in2 + offs, mask=mask, other=0.0)\n    z = 1+1  #真实计算逻辑\n"
						"    tl.store(out + offs, z, mask=mask)\n"
						f"{wrap_text}\n"
						"    assert input.is_cuda\n    input = input.contiguous()\n"
						"    if out is None:\n        out = torch.empty_like(input)\n"
						"    N = input.numel()\n    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']), )\n"
						"    kernel[grid](input, other, out, input.stride(0), input.stride(1), N)\n"
						"    return out\n"
					)

				result = {
					"test_sample_id": sample_id,
					"prediction": prediction,
				}
				f.write(json.dumps(result, ensure_ascii=False) + "\n")
				f.flush()
				results_dict[sample_id] = result
		print(f"\nAll samples completed. Please run the script again to enter review mode.")
		return

	if run_review_mode:
		for sample_index, sample in enumerate(test_samples, start=1):
			if sample_index < review_start_question:
				continue

			sample_id = sample["id"]
			input_text = sample.get("input", "")
			current_result = results_dict.get(sample_id)
			if not current_result:
				continue

			candidate_answer = current_result.get("prediction", "no")
			revised_answer = reflect_and_revise_answer(sample_id, input_text, candidate_answer)
			
			if revised_answer != candidate_answer:
				current_result["prediction"] = revised_answer
				results_dict[sample_id] = current_result
				
				# 立即将更新写入文件，防止中断导致丢失，循环执行后间隔时间太长
				with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
					for s in test_samples:
						sid = s["id"]
						if sid in results_dict:
							f.write(json.dumps(results_dict[sid], ensure_ascii=False) + "\n")

		persist_results_jsonl(results_dict, test_samples)
		print(f"\nReview pass finished from question {review_start_question}; results are already persisted during updates.")


if __name__ == "__main__":
	main()
