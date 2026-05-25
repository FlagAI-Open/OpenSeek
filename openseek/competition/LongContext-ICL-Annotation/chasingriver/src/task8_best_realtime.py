"""
Task 8 realtime inference / labeled ablation script.

This script now supports:
1. A legacy baseline loop for comparison.
2. A guided iterative loop with four local checks:
   - extract success
   - compile success
   - structure success
   - semantic proxy success
3. Failure classification into format / compile / semantic.
4. Best-of-history candidate retention across rounds.
5. Labeled-set ablation on a sampled subset.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import random
import re
import time
from pathlib import Path

import requests

import postfix_task8
import task8_best_prompt as task8
from task8_best_common import task8_signature_hint


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "openseek-8_kernel_generation.json"
OUTPUT_DIR = BASE_DIR / "outputs"
DEFAULT_SERVICE_URL = "http://127.0.0.1:2027/v1/completions"
SERVICE_URL = os.environ.get("OPENSEEK_TASK8_SERVICE_URL", DEFAULT_SERVICE_URL)

DEFAULT_VARIANT = "pytorch_v1_zero"
DEFAULT_TARGET_LENGTH = 16_000
DEFAULT_LABELED_LIMIT = 40
DEFAULT_SAFE_CODE_LIMIT = 12_000
MODEL_ID_CACHE: dict[str, str] = {}

CONFIGS = {
    # ═══════════════════════════════════════════════════════════════
    #  对照实验组 (Ablation)
    #
    #  V1 (base):         重构 example + v2 prompt + 1-shot
    #  V2 (zeroshot):     0-shot 对照 → 验证 restructured examples 是否有帮助
    #  V3 (conservative): 保守策略 + confidence gating → 利用 S_Exec²/S_Call
    #  V4 (more_rounds):  更多轮次对照 → 验证多轮修复是否值得
    # ═══════════════════════════════════════════════════════════════

    # V1: 新 baseline — 重构 example 格式 + v2 prompt
    "v2_base_fs1_r3": {
        "loop_style": "guided",
        "max_examples": 1,
        "max_rounds": 3,
        "max_tokens": 4096,
        "variant": "v2_structured_fs1",
        "semantic_threshold": 0.72,
        "require_signature_match": True,
        "restructure_eval": True,     # ★ 评估时将 example input 重构为 test 格式
        "confidence_gate": "none",    # 不做置信度门控
        "early_stop": "no_improve",   # ★ 连续 no_improve_patience 轮无进步则提前停（默认 2）
    },

    # V2: zero-shot 对照 — 不给 example，测试 zero-shot 能力
    "v2_zero_r3": {
        "loop_style": "guided",
        "max_examples": 0,
        "max_rounds": 3,
        "max_tokens": 4096,
        "variant": "v2_zero_r3_v2",
        "semantic_threshold": 0.72,
        "require_signature_match": True,
        "restructure_eval": True,
        "confidence_gate": "none",
        "early_stop": "no_improve",
    },

    # V3: 保守策略 — PyTorch fallback 优先 + 置信度门控
    # 利用 S_Consistency = S_Exec²/S_Call：宁可不输出也不输出能跑但错的
    "v2_safe_fs1_r3": {
        "loop_style": "guided",
        "max_examples": 1,
        "max_rounds": 3,
        "max_tokens": 4096,
        "variant": "v2_conservative_fs1",
        "semantic_threshold": 0.72,
        "require_signature_match": True,
        "restructure_eval": True,
        "confidence_gate": "strict",  # ★ 不通过 compile+wrapper+module 检查 → 输出 None
        "early_stop": "compile_stop", # ★ compile+wrapper 通过就停（保守不冒险）
    },

    # V4: 更多轮次 — 5 轮修复，测试多轮是否值得
    "v2_base_fs1_r5": {
        "loop_style": "guided",
        "max_examples": 1,
        "max_rounds": 5,
        "max_tokens": 4096,
        "variant": "v2_structured_fs1",
        "semantic_threshold": 0.72,
        "require_signature_match": True,
        "restructure_eval": True,
        "confidence_gate": "none",
        "early_stop": "no_improve",   # ★ 5轮中无进步就停，避免浪费
    },

    # ─── 额外实验组 ───

    # V3-zero: 保守策略 zero-shot 版
    "v2_safe_zero_r3": {
        "loop_style": "guided",
        "max_examples": 0,
        "max_rounds": 3,
        "max_tokens": 4096,
        "variant": "v2_conservative_fs0",
        "semantic_threshold": 0.72,
        "require_signature_match": True,
        "restructure_eval": True,
        "confidence_gate": "strict",
    },

    # V1-fs2: 2-shot 对照
    "v2_base_fs2_r3": {
        "loop_style": "guided",
        "max_examples": 2,
        "max_rounds": 3,
        "max_tokens": 4096,
        "variant": "v2_structured_fs2",
        "semantic_threshold": 0.72,
        "require_signature_match": True,
        "restructure_eval": True,
        "confidence_gate": "none",
    },

    # ═══════════════════════════════════════════════════════════════
    #  V2 系列改进：智能 Repair + 强化 prompt
    #
    #  核心改进点：
    #  1. smart_repair=True → 根据错误类型分发不同 repair prompt
    #     - placeholder (省略号/pass) → 专用 placeholder prompt
    #     - compile (语法/缩进) → 精确编译错误 + 行号上下文
    #     - import 缺失 → 后处理自动补全（不消耗 LLM 调用）
    #  2. 强化初始 prompt 中禁止省略号的约束
    #  3. 每轮在 repair prompt 中嵌入上一轮代码（见 _trim_text；full_code_prompt 时不截断）
    # ═══════════════════════════════════════════════════════════════

    # V2_v2: 基于最佳方案 v2_zero_r3 + 智能 repair
    "v2_zero_r3_v2": {
        "loop_style": "guided",
        "max_examples": 0,
        "max_rounds": 3,
        "max_tokens": 4096,
        "variant": "v2_zero_r3_v2",
        "semantic_threshold": 0.72,
        "require_signature_match": True,
        "restructure_eval": True,
        "confidence_gate": "none",
        "early_stop": "no_improve",
        "smart_repair": True,          # ★ 启用智能 repair
    },

    # A: 单轮 baseline — 当前默认 prompt，只看首轮质量
    "v2_zero_r1": {
        "loop_style": "guided",
        "max_examples": 0,
        "max_rounds": 1,
        "max_tokens": 4096,
        "variant": "v2_zero_r3_v2",
        "semantic_threshold": 0.72,
        "require_signature_match": True,
        "restructure_eval": True,
        "confidence_gate": "none",
        "early_stop": "none",
        "smart_repair": False,
    },

    # C: 单轮 PyTorch-first — 主逻辑优先 torch，Triton 最小安全参与
    "v5_torch_first_r1": {
        "loop_style": "guided",
        "max_examples": 0,
        "max_rounds": 1,
        "max_tokens": 4096,
        "variant": "v5_torch_first_zero",
        "semantic_threshold": 0.72,
        "require_signature_match": True,
        "restructure_eval": True,
        "confidence_gate": "none",
        "early_stop": "none",
        "smart_repair": False,
        "prefer_compilable_output": True,
    },

    # D: 多轮 PyTorch-first — 针对签名、launch、out、tuple 做 repair
    "v5_torch_first_r3": {
        "loop_style": "guided",
        "max_examples": 0,
        "max_rounds": 3,
        "max_tokens": 4096,
        "variant": "v5_torch_first_zero",
        "semantic_threshold": 0.72,
        "require_signature_match": True,
        "restructure_eval": True,
        "confidence_gate": "none",
        "early_stop": "no_improve",
        "no_improve_patience": 2,
        "smart_repair": True,
        "full_code_prompt": True,
        "repair_repeat_invariants": True,
        "format_prompt_raw_limit": 1200,
        "prefer_compilable_output": True,
    },

    # E: 多轮 API-contract — 强化 out/kw-only/dim/tuple/in-place 语义
    "v5_api_contract_r3": {
        "loop_style": "guided",
        "max_examples": 0,
        "max_rounds": 3,
        "max_tokens": 4096,
        "variant": "v5_api_contract_zero",
        "semantic_threshold": 0.72,
        "require_signature_match": True,
        "restructure_eval": True,
        "confidence_gate": "none",
        "early_stop": "no_improve",
        "no_improve_patience": 2,
        "smart_repair": True,
        "full_code_prompt": True,
        "repair_repeat_invariants": True,
        "format_prompt_raw_limit": 1200,
        "prefer_compilable_output": True,
    },

    # Clean ablation: API contract + no-think/code-only prompt + aggressive sanitization.
    "v5_api_contract_clean_r3": {
        "loop_style": "guided",
        "max_examples": 0,
        "max_rounds": 3,
        "max_tokens": 1024,
        "variant": "v5_api_contract_clean_zero",
        "semantic_threshold": 0.72,
        "require_signature_match": True,
        "restructure_eval": True,
        "confidence_gate": "none",
        "early_stop": "no_improve",
        "no_improve_patience": 2,
        "smart_repair": True,
        "full_code_prompt": True,
        "repair_repeat_invariants": True,
        "format_prompt_raw_limit": 800,
        "placeholder_prompt_code_limit": 1800,
        "compile_prompt_code_limit": 1800,
        "semantic_prompt_code_limit": 1800,
        "prefer_compilable_output": True,
        "no_think": True,
        "clean_repair": True,
        "use_completion_api": True,
    },

    # V6: chat API + hard executable skeleton + long-comment/comment-only rejection.
    "v6_skeleton_chat_r3": {
        "loop_style": "guided",
        "max_examples": 0,
        "max_rounds": 3,
        "max_tokens": 4096,
        "variant": "v6_skeleton_chat_zero",
        "semantic_threshold": 0.72,
        "require_signature_match": True,
        "restructure_eval": True,
        "confidence_gate": "none",
        "early_stop": "no_improve",
        "no_improve_patience": 2,
        "smart_repair": True,
        "full_code_prompt": True,
        "repair_repeat_invariants": True,
        "format_prompt_raw_limit": 1000,
        "placeholder_prompt_code_limit": 2200,
        "compile_prompt_code_limit": 2200,
        "semantic_prompt_code_limit": 2200,
        "prefer_compilable_output": True,
        "v6_skeleton_repair": True,
        "require_kernel_launch": True,
        "reject_placeholder_code": True,
    },

    # V7: fixed identity-touch Triton helper + torch-first wrapper semantics.
    "v7_touch_template_r3": {
        "loop_style": "guided",
        "max_examples": 0,
        "max_rounds": 3,
        "max_tokens": 4096,
        "variant": "v7_touch_template_zero",
        "semantic_threshold": 0.72,
        "require_signature_match": True,
        "restructure_eval": True,
        "confidence_gate": "none",
        "early_stop": "no_improve",
        "no_improve_patience": 2,
        "smart_repair": True,
        "full_code_prompt": True,
        "repair_repeat_invariants": True,
        "format_prompt_raw_limit": 900,
        "placeholder_prompt_code_limit": 2400,
        "compile_prompt_code_limit": 2400,
        "semantic_prompt_code_limit": 2400,
        "prefer_compilable_output": True,
        "v7_touch_template_repair": True,
        "require_kernel_launch": True,
        "require_touch_helper": True,
        "reject_placeholder_code": True,
    },
    "v7b_touch_template_r3": {
        "loop_style": "guided",
        "max_examples": 0,
        "max_rounds": 3,
        "max_tokens": 4096,
        "variant": "v7_touch_template_zero",
        "semantic_threshold": 0.72,
        "require_signature_match": True,
        "restructure_eval": True,
        "confidence_gate": "none",
        "early_stop": "no_improve",
        "no_improve_patience": 2,
        "smart_repair": True,
        "full_code_prompt": True,
        "repair_repeat_invariants": True,
        "format_prompt_raw_limit": 900,
        "placeholder_prompt_code_limit": 2400,
        "compile_prompt_code_limit": 2400,
        "semantic_prompt_code_limit": 2400,
        "prefer_compilable_output": True,
        "v7_touch_template_repair": True,
        "require_kernel_launch": True,
        "require_touch_helper": True,
        "reject_placeholder_code": True,
    },
    "v7b_pytorch_first_r3": {
        "loop_style": "guided",
        "max_examples": 0,
        "max_rounds": 3,
        "max_tokens": 1024,
        "variant": "v7b_pytorch_first_zero",
        "semantic_threshold": 0.72,
        "require_signature_match": True,
        "restructure_eval": True,
        "confidence_gate": "none",
        "early_stop": "no_improve",
        "no_improve_patience": 2,
        "smart_repair": True,
        "full_code_prompt": False,
        "repair_repeat_invariants": True,
        "format_prompt_raw_limit": 900,
        "placeholder_prompt_code_limit": 1200,
        "compile_prompt_code_limit": 1200,
        "semantic_prompt_code_limit": 1200,
        "prefer_compilable_output": True,
        "reject_placeholder_code": True,
        "pure_pytorch": True,
        "use_completion_api": True,
    },
    "v7b_pytorch_first_smoke10": {
        "loop_style": "guided",
        "max_examples": 0,
        "max_rounds": 3,
        "max_tokens": 1024,
        "variant": "v7b_pytorch_first_zero",
        "semantic_threshold": 0.72,
        "require_signature_match": True,
        "restructure_eval": True,
        "confidence_gate": "none",
        "early_stop": "no_improve",
        "no_improve_patience": 2,
        "smart_repair": True,
        "full_code_prompt": False,
        "repair_repeat_invariants": True,
        "format_prompt_raw_limit": 900,
        "placeholder_prompt_code_limit": 900,
        "compile_prompt_code_limit": 900,
        "semantic_prompt_code_limit": 900,
        "prefer_compilable_output": True,
        "reject_placeholder_code": True,
        "pure_pytorch": True,
        "use_completion_api": True,
    },
    "pytorch_v1_r3": {
        "loop_style": "guided",
        "max_examples": 0,
        "max_rounds": 3,
        "max_tokens": 1024,
        "variant": "pytorch_v1_zero",
        "semantic_threshold": 0.72,
        "require_signature_match": True,
        "restructure_eval": True,
        "confidence_gate": "none",
        "early_stop": "no_improve",
        "no_improve_patience": 2,
        "smart_repair": True,
        "full_code_prompt": False,
        "repair_repeat_invariants": True,
        "format_prompt_raw_limit": 900,
        "placeholder_prompt_code_limit": 1200,
        "compile_prompt_code_limit": 1200,
        "semantic_prompt_code_limit": 1200,
        "prefer_compilable_output": True,
        "reject_placeholder_code": True,
        "pure_pytorch": True,
        "use_completion_api": True,
    },
    "pytorch_v1_r3_long15k": {
        "loop_style": "guided",
        "max_examples": 140,
        "max_rounds": 3,
        "max_tokens": 1024,
        "variant": "pytorch_v1_zero",
        "semantic_threshold": 0.72,
        "require_signature_match": True,
        "restructure_eval": True,
        "confidence_gate": "none",
        "early_stop": "no_improve",
        "no_improve_patience": 2,
        "smart_repair": True,
        "full_code_prompt": False,
        "repair_repeat_invariants": True,
        "format_prompt_raw_limit": 900,
        "placeholder_prompt_code_limit": 1200,
        "compile_prompt_code_limit": 1200,
        "semantic_prompt_code_limit": 1200,
        "prefer_compilable_output": True,
        "reject_placeholder_code": True,
        "pure_pytorch": True,
        "use_completion_api": True,
        "long_context_reference": True,
        "long_context_target_chars": 66_000,
        "long_context_item_limit": 140,
        "long_context_item_char_limit": 900,
    },
    "pytorch_v1_r3_long15k_refine": {
        "loop_style": "guided",
        "max_examples": 140,
        "max_rounds": 3,
        "max_tokens": 1024,
        "variant": "pytorch_v1_zero",
        "semantic_threshold": 0.72,
        "require_signature_match": True,
        "restructure_eval": True,
        "confidence_gate": "none",
        "early_stop": "no_improve",
        "no_improve_patience": 2,
        "smart_repair": True,
        "full_code_prompt": False,
        "repair_repeat_invariants": True,
        "format_prompt_raw_limit": 900,
        "placeholder_prompt_code_limit": 1200,
        "compile_prompt_code_limit": 1200,
        "semantic_prompt_code_limit": 1200,
        "prefer_compilable_output": True,
        "reject_placeholder_code": True,
        "pure_pytorch": True,
        "use_completion_api": True,
        "long_context_reference": True,
        "long_context_target_chars": 66_000,
        "long_context_item_limit": 140,
        "long_context_item_char_limit": 900,
        "final_semantic_refine": True,
        "final_refine_mode": "relaxed",
        "final_refine_max_tokens": 1024,
    },
    "pytorch_v1_r3_probe15k_then_r3": {
        "loop_style": "guided",
        "max_examples": 0,
        "max_rounds": 4,
        "max_tokens": 1024,
        "variant": "pytorch_v1_zero",
        "semantic_threshold": 0.72,
        "require_signature_match": True,
        "restructure_eval": True,
        "confidence_gate": "none",
        "early_stop": "no_improve",
        "no_improve_patience": 2,
        "smart_repair": True,
        "full_code_prompt": False,
        "repair_repeat_invariants": True,
        "format_prompt_raw_limit": 900,
        "placeholder_prompt_code_limit": 1200,
        "compile_prompt_code_limit": 1200,
        "semantic_prompt_code_limit": 1200,
        "prefer_compilable_output": True,
        "reject_placeholder_code": True,
        "pure_pytorch": True,
        "use_completion_api": True,
        "long_context_reference": True,
        "long_context_mode": "rules_pad",
        "long_context_target_chars": 66_000,
        "long_context_probe_rounds": 1,
        "restart_short_after_probe": True,
        "discard_failed_probe": True,
        "probe_accept_min_score": 79.0,
        "probe_accept_min_semantic_ratio": 0.95,
    },
    "pytorch_v1_r3_probe15k_then_r4_refine": {
        "loop_style": "guided",
        "max_examples": 0,
        "max_rounds": 4,
        "max_tokens": 1024,
        "variant": "pytorch_v1_zero",
        "semantic_threshold": 0.72,
        "require_signature_match": True,
        "restructure_eval": True,
        "confidence_gate": "none",
        "early_stop": "no_improve",
        "no_improve_patience": 2,
        "smart_repair": True,
        "full_code_prompt": False,
        "repair_repeat_invariants": True,
        "format_prompt_raw_limit": 900,
        "placeholder_prompt_code_limit": 1200,
        "compile_prompt_code_limit": 1600,
        "semantic_prompt_code_limit": 1600,
        "prefer_compilable_output": True,
        "reject_placeholder_code": True,
        "pure_pytorch": True,
        "use_completion_api": True,
        "long_context_reference": True,
        "long_context_mode": "rules_pad",
        "long_context_target_chars": 66_000,
        "long_context_probe_rounds": 1,
        "restart_short_after_probe": True,
        "discard_failed_probe": True,
        "probe_accept_min_score": 79.0,
        "probe_accept_min_semantic_ratio": 0.95,
        "final_semantic_refine": True,
        "final_refine_mode": "relaxed",
        "final_refine_max_tokens": 1024,
    },
    "pytorch_v1_r3_probe15k_then_r5": {
        "loop_style": "guided",
        "max_examples": 0,
        "max_rounds": 5,
        "max_tokens": 1024,
        "variant": "pytorch_v1_zero",
        "semantic_threshold": 0.72,
        "require_signature_match": True,
        "restructure_eval": True,
        "confidence_gate": "none",
        "early_stop": "no_improve",
        "no_improve_patience": 3,
        "smart_repair": True,
        "full_code_prompt": False,
        "repair_repeat_invariants": True,
        "format_prompt_raw_limit": 900,
        "placeholder_prompt_code_limit": 1200,
        "compile_prompt_code_limit": 1800,
        "semantic_prompt_code_limit": 1800,
        "prefer_compilable_output": True,
        "reject_placeholder_code": True,
        "pure_pytorch": True,
        "use_completion_api": True,
        "long_context_reference": True,
        "long_context_mode": "rules_pad",
        "long_context_target_chars": 66_000,
        "long_context_probe_rounds": 1,
        "restart_short_after_probe": True,
        "discard_failed_probe": True,
        "probe_accept_min_score": 79.0,
        "probe_accept_min_semantic_ratio": 0.95,
    },
    "pytorch_v1_r3_short_r4": {
        "loop_style": "guided",
        "max_examples": 0,
        "max_rounds": 4,
        "max_tokens": 1024,
        "variant": "pytorch_v1_zero",
        "semantic_threshold": 0.72,
        "require_signature_match": True,
        "restructure_eval": True,
        "confidence_gate": "none",
        "early_stop": "no_improve",
        "no_improve_patience": 2,
        "smart_repair": True,
        "full_code_prompt": False,
        "repair_repeat_invariants": True,
        "format_prompt_raw_limit": 900,
        "placeholder_prompt_code_limit": 1200,
        "compile_prompt_code_limit": 1600,
        "semantic_prompt_code_limit": 1600,
        "prefer_compilable_output": True,
        "reject_placeholder_code": True,
        "pure_pytorch": True,
        "use_completion_api": True,
    },
    "pytorch_v1_r3_probe15k_then_r4_api_guard": {
        "loop_style": "guided",
        "max_examples": 0,
        "max_rounds": 4,
        "max_tokens": 1024,
        "variant": "pytorch_v1_zero",
        "semantic_threshold": 0.72,
        "require_signature_match": True,
        "restructure_eval": True,
        "confidence_gate": "none",
        "early_stop": "no_improve",
        "no_improve_patience": 2,
        "smart_repair": True,
        "full_code_prompt": False,
        "repair_repeat_invariants": True,
        "format_prompt_raw_limit": 900,
        "placeholder_prompt_code_limit": 1200,
        "compile_prompt_code_limit": 2000,
        "semantic_prompt_code_limit": 2000,
        "prefer_compilable_output": True,
        "reject_placeholder_code": True,
        "pure_pytorch": True,
        "use_completion_api": True,
        "long_context_reference": True,
        "long_context_mode": "rules_pad",
        "long_context_target_chars": 66_000,
        "long_context_probe_rounds": 1,
        "restart_short_after_probe": True,
        "discard_failed_probe": True,
        "probe_accept_min_score": 79.0,
        "probe_accept_min_semantic_ratio": 0.95,
    },
    "pytorch_v1_r3_short_r4_api_refine": {
        "loop_style": "guided",
        "max_examples": 0,
        "max_rounds": 4,
        "max_tokens": 1024,
        "variant": "pytorch_v1_zero",
        "semantic_threshold": 0.72,
        "require_signature_match": True,
        "restructure_eval": True,
        "confidence_gate": "none",
        "early_stop": "no_improve",
        "no_improve_patience": 2,
        "smart_repair": True,
        "full_code_prompt": False,
        "repair_repeat_invariants": True,
        "format_prompt_raw_limit": 900,
        "placeholder_prompt_code_limit": 1200,
        "compile_prompt_code_limit": 2000,
        "semantic_prompt_code_limit": 2000,
        "prefer_compilable_output": True,
        "reject_placeholder_code": True,
        "pure_pytorch": True,
        "use_completion_api": True,
        "final_semantic_refine": True,
        "final_refine_mode": "relaxed",
        "final_refine_max_tokens": 1024,
    },
    "pytorch_v1_r3_api_guard_r5_pat3": {
        "loop_style": "guided",
        "max_examples": 0,
        "max_rounds": 5,
        "max_tokens": 1024,
        "variant": "pytorch_v1_zero",
        "semantic_threshold": 0.72,
        "require_signature_match": True,
        "restructure_eval": True,
        "confidence_gate": "none",
        "early_stop": "no_improve",
        "no_improve_patience": 3,
        "smart_repair": True,
        "full_code_prompt": False,
        "repair_repeat_invariants": True,
        "format_prompt_raw_limit": 900,
        "placeholder_prompt_code_limit": 1200,
        "compile_prompt_code_limit": 2000,
        "semantic_prompt_code_limit": 2000,
        "prefer_compilable_output": True,
        "reject_placeholder_code": True,
        "pure_pytorch": True,
        "use_completion_api": True,
        "long_context_reference": True,
        "long_context_mode": "rules_pad",
        "long_context_target_chars": 66_000,
        "long_context_probe_rounds": 1,
        "restart_short_after_probe": True,
        "discard_failed_probe": True,
        "probe_accept_min_score": 79.0,
        "probe_accept_min_semantic_ratio": 0.95,
    },
    "pytorch_v1_r3_api_guard_strict_final": {
        "loop_style": "guided",
        "max_examples": 0,
        "max_rounds": 4,
        "max_tokens": 1024,
        "variant": "pytorch_v1_zero",
        "semantic_threshold": 0.72,
        "require_signature_match": True,
        "restructure_eval": True,
        "confidence_gate": "none",
        "early_stop": "no_improve",
        "no_improve_patience": 2,
        "smart_repair": True,
        "full_code_prompt": False,
        "repair_repeat_invariants": True,
        "format_prompt_raw_limit": 900,
        "placeholder_prompt_code_limit": 1200,
        "compile_prompt_code_limit": 2000,
        "semantic_prompt_code_limit": 2000,
        "prefer_compilable_output": True,
        "reject_placeholder_code": True,
        "pure_pytorch": True,
        "use_completion_api": True,
        "long_context_reference": True,
        "long_context_mode": "rules_pad",
        "long_context_target_chars": 66_000,
        "long_context_probe_rounds": 1,
        "restart_short_after_probe": True,
        "discard_failed_probe": True,
        "probe_accept_min_score": 79.0,
        "probe_accept_min_semantic_ratio": 0.95,
        "final_semantic_refine": True,
        "final_refine_mode": "strict",
        "final_refine_max_tokens": 1024,
    },
    "pytorch_v1_r3_api_guard_probe96": {
        "loop_style": "guided",
        "max_examples": 0,
        "max_rounds": 4,
        "max_tokens": 1024,
        "variant": "pytorch_v1_zero",
        "semantic_threshold": 0.72,
        "require_signature_match": True,
        "restructure_eval": True,
        "confidence_gate": "none",
        "early_stop": "no_improve",
        "no_improve_patience": 2,
        "smart_repair": True,
        "full_code_prompt": False,
        "repair_repeat_invariants": True,
        "format_prompt_raw_limit": 900,
        "placeholder_prompt_code_limit": 1200,
        "compile_prompt_code_limit": 2000,
        "semantic_prompt_code_limit": 2000,
        "prefer_compilable_output": True,
        "reject_placeholder_code": True,
        "pure_pytorch": True,
        "use_completion_api": True,
        "long_context_reference": True,
        "long_context_mode": "rules_pad",
        "long_context_target_chars": 96_000,
        "long_context_probe_rounds": 1,
        "restart_short_after_probe": True,
        "discard_failed_probe": True,
        "probe_accept_min_score": 79.0,
        "probe_accept_min_semantic_ratio": 0.95,
    },
    "pytorch_v1_r3_api_guard_fullcode": {
        "loop_style": "guided",
        "max_examples": 0,
        "max_rounds": 4,
        "max_tokens": 1024,
        "variant": "pytorch_v1_zero",
        "semantic_threshold": 0.72,
        "require_signature_match": True,
        "restructure_eval": True,
        "confidence_gate": "none",
        "early_stop": "no_improve",
        "no_improve_patience": 2,
        "smart_repair": True,
        "full_code_prompt": True,
        "repair_repeat_invariants": True,
        "format_prompt_raw_limit": 1200,
        "placeholder_prompt_code_limit": 2000,
        "compile_prompt_code_limit": 4000,
        "semantic_prompt_code_limit": 4000,
        "prefer_compilable_output": True,
        "reject_placeholder_code": True,
        "pure_pytorch": True,
        "use_completion_api": True,
        "long_context_reference": True,
        "long_context_mode": "rules_pad",
        "long_context_target_chars": 66_000,
        "long_context_probe_rounds": 1,
        "restart_short_after_probe": True,
        "discard_failed_probe": True,
        "probe_accept_min_score": 79.0,
        "probe_accept_min_semantic_ratio": 0.95,
    },
    "pytorch_v1_r3_api_guard_relaxed_final_r5": {
        "loop_style": "guided",
        "max_examples": 0,
        "max_rounds": 5,
        "max_tokens": 1024,
        "variant": "pytorch_v1_zero",
        "semantic_threshold": 0.72,
        "require_signature_match": True,
        "restructure_eval": True,
        "confidence_gate": "none",
        "early_stop": "no_improve",
        "no_improve_patience": 3,
        "smart_repair": True,
        "full_code_prompt": False,
        "repair_repeat_invariants": True,
        "format_prompt_raw_limit": 900,
        "placeholder_prompt_code_limit": 1200,
        "compile_prompt_code_limit": 2000,
        "semantic_prompt_code_limit": 2000,
        "prefer_compilable_output": True,
        "reject_placeholder_code": True,
        "pure_pytorch": True,
        "use_completion_api": True,
        "long_context_reference": True,
        "long_context_mode": "rules_pad",
        "long_context_target_chars": 66_000,
        "long_context_probe_rounds": 1,
        "restart_short_after_probe": True,
        "discard_failed_probe": True,
        "probe_accept_min_score": 79.0,
        "probe_accept_min_semantic_ratio": 0.95,
        "final_semantic_refine": True,
        "final_refine_mode": "relaxed",
        "final_refine_max_tokens": 1024,
    },
    "pytorch_v1_r3_api_guard_promptfix_r5": {
        "loop_style": "guided",
        "max_examples": 0,
        "max_rounds": 5,
        "max_tokens": 1024,
        "variant": "pytorch_v1_zero",
        "semantic_threshold": 0.72,
        "require_signature_match": True,
        "restructure_eval": True,
        "confidence_gate": "none",
        "early_stop": "no_improve",
        "no_improve_patience": 3,
        "smart_repair": True,
        "full_code_prompt": False,
        "repair_repeat_invariants": True,
        "format_prompt_raw_limit": 900,
        "placeholder_prompt_code_limit": 1200,
        "compile_prompt_code_limit": 2000,
        "semantic_prompt_code_limit": 2000,
        "prefer_compilable_output": True,
        "reject_placeholder_code": True,
        "pure_pytorch": True,
        "use_completion_api": True,
        "long_context_reference": True,
        "long_context_mode": "rules_pad",
        "long_context_target_chars": 66_000,
        "long_context_probe_rounds": 1,
        "restart_short_after_probe": True,
        "discard_failed_probe": True,
        "probe_accept_min_score": 79.0,
        "probe_accept_min_semantic_ratio": 0.95,
    },
    "pytorch_v1_r3_api_guard_promptfix_probe96": {
        "loop_style": "guided",
        "max_examples": 0,
        "max_rounds": 4,
        "max_tokens": 1024,
        "variant": "pytorch_v1_zero",
        "semantic_threshold": 0.72,
        "require_signature_match": True,
        "restructure_eval": True,
        "confidence_gate": "none",
        "early_stop": "no_improve",
        "no_improve_patience": 2,
        "smart_repair": True,
        "full_code_prompt": False,
        "repair_repeat_invariants": True,
        "format_prompt_raw_limit": 900,
        "placeholder_prompt_code_limit": 1200,
        "compile_prompt_code_limit": 2000,
        "semantic_prompt_code_limit": 2000,
        "prefer_compilable_output": True,
        "reject_placeholder_code": True,
        "pure_pytorch": True,
        "use_completion_api": True,
        "long_context_reference": True,
        "long_context_mode": "rules_pad",
        "long_context_target_chars": 96_000,
        "long_context_probe_rounds": 1,
        "restart_short_after_probe": True,
        "discard_failed_probe": True,
        "probe_accept_min_score": 79.0,
        "probe_accept_min_semantic_ratio": 0.95,
    },
    "pytorch_v1_r3_api_guard_staticfix_r5": {
        "loop_style": "guided",
        "max_examples": 0,
        "max_rounds": 5,
        "max_tokens": 1024,
        "variant": "pytorch_v1_zero",
        "semantic_threshold": 0.72,
        "require_signature_match": True,
        "restructure_eval": True,
        "confidence_gate": "none",
        "early_stop": "no_improve",
        "no_improve_patience": 3,
        "smart_repair": True,
        "full_code_prompt": False,
        "repair_repeat_invariants": True,
        "format_prompt_raw_limit": 900,
        "placeholder_prompt_code_limit": 1200,
        "compile_prompt_code_limit": 2000,
        "semantic_prompt_code_limit": 2000,
        "prefer_compilable_output": True,
        "reject_placeholder_code": True,
        "pure_pytorch": True,
        "use_completion_api": True,
        "long_context_reference": True,
        "long_context_mode": "rules_pad",
        "long_context_target_chars": 66_000,
        "long_context_probe_rounds": 1,
        "restart_short_after_probe": True,
        "discard_failed_probe": True,
        "probe_accept_min_score": 79.0,
        "probe_accept_min_semantic_ratio": 0.95,
        "static_api_postfix": True,
    },
    "pytorch_v1_r3_api_guard_staticfix_probe96": {
        "loop_style": "guided",
        "max_examples": 0,
        "max_rounds": 4,
        "max_tokens": 1024,
        "variant": "pytorch_v1_zero",
        "semantic_threshold": 0.72,
        "require_signature_match": True,
        "restructure_eval": True,
        "confidence_gate": "none",
        "early_stop": "no_improve",
        "no_improve_patience": 2,
        "smart_repair": True,
        "full_code_prompt": False,
        "repair_repeat_invariants": True,
        "format_prompt_raw_limit": 900,
        "placeholder_prompt_code_limit": 1200,
        "compile_prompt_code_limit": 2000,
        "semantic_prompt_code_limit": 2000,
        "prefer_compilable_output": True,
        "reject_placeholder_code": True,
        "pure_pytorch": True,
        "use_completion_api": True,
        "long_context_reference": True,
        "long_context_mode": "rules_pad",
        "long_context_target_chars": 96_000,
        "long_context_probe_rounds": 1,
        "restart_short_after_probe": True,
        "discard_failed_probe": True,
        "probe_accept_min_score": 79.0,
        "probe_accept_min_semantic_ratio": 0.95,
        "static_api_postfix": True,
    },
    "pytorch_v1_smoke10": {
        "loop_style": "guided",
        "max_examples": 0,
        "max_rounds": 3,
        "max_tokens": 1024,
        "variant": "pytorch_v1_zero",
        "semantic_threshold": 0.72,
        "require_signature_match": True,
        "restructure_eval": True,
        "confidence_gate": "none",
        "early_stop": "no_improve",
        "no_improve_patience": 2,
        "smart_repair": True,
        "full_code_prompt": False,
        "repair_repeat_invariants": True,
        "format_prompt_raw_limit": 900,
        "placeholder_prompt_code_limit": 900,
        "compile_prompt_code_limit": 900,
        "semantic_prompt_code_limit": 900,
        "prefer_compilable_output": True,
        "reject_placeholder_code": True,
        "pure_pytorch": True,
        "use_completion_api": True,
    },
    "v8_signature_touch_r4": {
        "loop_style": "guided",
        "max_examples": 0,
        "max_rounds": 4,
        "max_tokens": 4096,
        "variant": "v8_signature_touch_zero",
        "semantic_threshold": 0.72,
        "require_signature_match": True,
        "restructure_eval": True,
        "confidence_gate": "none",
        "early_stop": "no_improve",
        "no_improve_patience": 3,
        "smart_repair": True,
        "full_code_prompt": True,
        "repair_repeat_invariants": True,
        "format_prompt_raw_limit": 1000,
        "placeholder_prompt_code_limit": 2600,
        "compile_prompt_code_limit": 2600,
        "semantic_prompt_code_limit": 2600,
        "prefer_compilable_output": True,
        "v8_signature_family_repair": True,
        "require_kernel_launch": True,
        "require_touch_helper": True,
        "reject_placeholder_code": True,
    },
    "v9_family_breadcrumb_r4": {
        "loop_style": "guided",
        "max_examples": 0,
        "max_rounds": 4,
        "max_tokens": 4096,
        "variant": "v9_family_touch_zero",
        "semantic_threshold": 0.72,
        "require_signature_match": True,
        "restructure_eval": True,
        "confidence_gate": "none",
        "early_stop": "no_improve",
        "no_improve_patience": 3,
        "smart_repair": True,
        "full_code_prompt": True,
        "repair_repeat_invariants": True,
        "format_prompt_raw_limit": 1000,
        "placeholder_prompt_code_limit": 2600,
        "compile_prompt_code_limit": 2600,
        "semantic_prompt_code_limit": 2600,
        "prefer_compilable_output": True,
        "v8_signature_family_repair": True,
        "semantic_breadcrumb": True,
        "require_kernel_launch": True,
        "require_touch_helper": True,
        "reject_placeholder_code": True,
    },

    # V2_v2_A: repair 中嵌入完整代码 + 重复不变量 + 连续 3 轮无进步才停
    "v2_zero_r3_v2_A": {
        "loop_style": "guided",
        "max_examples": 0,
        "max_rounds": 3,
        "max_tokens": 4096,
        "variant": "v2_zero_r3_v2",
        "semantic_threshold": 0.72,
        "require_signature_match": True,
        "restructure_eval": True,
        "confidence_gate": "none",
        "early_stop": "no_improve",
        "no_improve_patience": 3,
        "smart_repair": True,
        "full_code_prompt": True,
        "repair_repeat_invariants": True,
        "format_prompt_raw_limit": 1200,
    },
    # New direction 1: signature/API lock first, then multi-round static repair.
    "new_siglock_zero_r4": {
        "loop_style": "guided",
        "max_examples": 0,
        "max_rounds": 4,
        "max_tokens": 4096,
        "variant": "v3_siglock_zero",
        "semantic_threshold": 0.72,
        "require_signature_match": True,
        "restructure_eval": True,
        "confidence_gate": "none",
        "early_stop": "no_improve",
        "no_improve_patience": 3,
        "smart_repair": True,
        "full_code_prompt": True,
        "repair_repeat_invariants": True,
        "format_prompt_raw_limit": 1200,
    },
    # New direction 2: consistency-safe generation with strict gate.
    "new_safe_gate_zero_r3": {
        "loop_style": "guided",
        "max_examples": 0,
        "max_rounds": 3,
        "max_tokens": 4096,
        "variant": "v3_safe_gate_zero",
        "semantic_threshold": 0.72,
        "require_signature_match": True,
        "restructure_eval": True,
        "confidence_gate": "strict",
        "early_stop": "no_improve",
        "no_improve_patience": 2,
        "smart_repair": True,
        "full_code_prompt": True,
        "repair_repeat_invariants": True,
        "format_prompt_raw_limit": 1200,
    },
    # New direction 3: staged draft plus verifier-guided repairs.
    "new_stage_verify_zero_r5": {
        "loop_style": "guided",
        "max_examples": 0,
        "max_rounds": 5,
        "max_tokens": 4096,
        "variant": "v3_stage_verify_zero",
        "semantic_threshold": 0.72,
        "require_signature_match": True,
        "restructure_eval": True,
        "confidence_gate": "none",
        "early_stop": "no_improve",
        "no_improve_patience": 3,
        "smart_repair": True,
        "full_code_prompt": True,
        "repair_repeat_invariants": True,
        "format_prompt_raw_limit": 1200,
        "staged_round_plan": "gemini_three_stage",
        "gemini_repeat_header": True,
        "gemini_forbid_rename": True,
        "final_semantic_refine": True,
        "final_refine_mode": "relaxed",
        "final_refine_max_tokens": 4096,
    },
    "night_codeonly_zero_r3": {
        "loop_style": "guided",
        "max_examples": 0,
        "max_rounds": 3,
        "max_tokens": 4096,
        "variant": "v4_codeonly_zero",
        "semantic_threshold": 0.72,
        "require_signature_match": True,
        "restructure_eval": True,
        "confidence_gate": "none",
        "early_stop": "no_improve",
        "no_improve_patience": 2,
        "smart_repair": True,
        "full_code_prompt": True,
        "repair_repeat_invariants": True,
        "format_prompt_raw_limit": 1000,
        "prefer_compilable_output": True,
        "null_if_no_compilable": True,
    },
    "night_wrapperfirst_zero_r3": {
        "loop_style": "guided",
        "max_examples": 0,
        "max_rounds": 3,
        "max_tokens": 4096,
        "variant": "v4_wrapperfirst_zero",
        "semantic_threshold": 0.72,
        "require_signature_match": True,
        "restructure_eval": True,
        "confidence_gate": "strict",
        "early_stop": "compile_stop",
        "smart_repair": True,
        "full_code_prompt": True,
        "repair_repeat_invariants": True,
        "format_prompt_raw_limit": 1000,
        "prefer_compilable_output": True,
        "null_if_no_compilable": True,
    },
    "night_compactstrict_zero_r4": {
        "loop_style": "guided",
        "max_examples": 0,
        "max_rounds": 4,
        "max_tokens": 4096,
        "variant": "v4_compactstrict_zero",
        "semantic_threshold": 0.72,
        "require_signature_match": True,
        "restructure_eval": True,
        "confidence_gate": "none",
        "early_stop": "no_improve",
        "no_improve_patience": 3,
        "smart_repair": True,
        "full_code_prompt": True,
        "repair_repeat_invariants": True,
        "format_prompt_raw_limit": 1000,
        "final_semantic_refine": True,
        "final_refine_mode": "relaxed",
        "final_refine_max_tokens": 4096,
        "prefer_compilable_output": True,
        "null_if_no_compilable": True,
    },
    "night_apicontract_zero_r3": {
        "loop_style": "guided",
        "max_examples": 0,
        "max_rounds": 3,
        "max_tokens": 4096,
        "variant": "v4_apicontract_zero",
        "semantic_threshold": 0.72,
        "require_signature_match": True,
        "restructure_eval": True,
        "confidence_gate": "strict",
        "early_stop": "compile_stop",
        "smart_repair": True,
        "full_code_prompt": True,
        "repair_repeat_invariants": True,
        "format_prompt_raw_limit": 1000,
        "prefer_compilable_output": True,
        "null_if_no_compilable": True,
    },
    "night_state_machine_zero_r8": {
        "loop_style": "guided",
        "max_examples": 0,
        "max_rounds": 8,
        "max_tokens": 4096,
        "variant": "v4_apicontract_zero",
        "semantic_threshold": 0.72,
        "require_signature_match": True,
        "restructure_eval": True,
        "confidence_gate": "none",
        "early_stop": "no_improve",
        "no_improve_patience": 3,
        "smart_repair": True,
        "full_code_prompt": True,
        "repair_repeat_invariants": True,
        "format_prompt_raw_limit": 1000,
        "prefer_compilable_output": True,
        "null_if_no_compilable": True,
        "staged_round_plan": "cenzihan_v1_state_machine",
        "gemini_repeat_header": True,
        "gemini_forbid_rename": True,
    },
    "night_stage_refine_zero_r5": {
        "loop_style": "guided",
        "max_examples": 0,
        "max_rounds": 5,
        "max_tokens": 4096,
        "variant": "v4_compactstrict_zero",
        "semantic_threshold": 0.72,
        "require_signature_match": True,
        "restructure_eval": True,
        "confidence_gate": "none",
        "early_stop": "no_improve",
        "no_improve_patience": 3,
        "smart_repair": True,
        "full_code_prompt": True,
        "repair_repeat_invariants": True,
        "format_prompt_raw_limit": 1000,
        "prefer_compilable_output": True,
        "null_if_no_compilable": True,
        "staged_round_plan": "gemini_three_stage",
        "gemini_repeat_header": True,
        "gemini_forbid_rename": True,
        "final_semantic_refine": True,
        "final_refine_mode": "relaxed",
        "final_refine_max_tokens": 4096,
    },
    # Gemini 主流程：跨轮次三阶段（R1 数学逻辑 -> R2 kernel -> R3 wrapper），总 5 轮偏 success
    "task8_gemini_r5_main": {
        "loop_style": "guided",
        "max_examples": 0,
        "max_rounds": 5,
        "max_tokens": 4096,
        "variant": "task8_gemini",
        "semantic_threshold": 0.72,
        "require_signature_match": True,
        "restructure_eval": True,
        "confidence_gate": "none",
        "early_stop": "no_improve",
        "no_improve_patience": 3,
        "smart_repair": True,
        "full_code_prompt": True,
        "repair_repeat_invariants": True,
        "format_prompt_raw_limit": 1200,
        "staged_round_plan": "gemini_three_stage",
        "gemini_repeat_header": True,
        "gemini_forbid_rename": True,
    },
    # Gemini ablation-1：不启用跨轮次阶段计划（退化为常规 guided repair）
    "task8_gemini_r5_ab_no_stage": {
        "loop_style": "guided",
        "max_examples": 0,
        "max_rounds": 5,
        "max_tokens": 4096,
        "variant": "task8_gemini_ab_raw_request",
        "semantic_threshold": 0.72,
        "require_signature_match": True,
        "restructure_eval": True,
        "confidence_gate": "none",
        "early_stop": "no_improve",
        "no_improve_patience": 3,
        "smart_repair": True,
        "full_code_prompt": True,
        "repair_repeat_invariants": True,
        "format_prompt_raw_limit": 1200,
    },
    # Gemini ablation-2：启用三阶段，但不重复 wrapper/header 参数锁定
    "task8_gemini_r5_ab_no_lock": {
        "loop_style": "guided",
        "max_examples": 0,
        "max_rounds": 5,
        "max_tokens": 4096,
        "variant": "task8_gemini_ab_raw_example",
        "semantic_threshold": 0.72,
        "require_signature_match": True,
        "restructure_eval": True,
        "confidence_gate": "none",
        "early_stop": "no_improve",
        "no_improve_patience": 3,
        "smart_repair": True,
        "full_code_prompt": True,
        "repair_repeat_invariants": True,
        "format_prompt_raw_limit": 1200,
        "staged_round_plan": "gemini_three_stage",
        "gemini_repeat_header": False,
        "gemini_forbid_rename": False,
    },
    # Gemini ablation-3：与主版相同，但 variant 走 thinking 配置
    "task8_gemini_r5_ab_thinking": {
        "loop_style": "guided",
        "max_examples": 0,
        "max_rounds": 5,
        "max_tokens": 4096,
        "variant": "task8_gemini_thinking",
        "semantic_threshold": 0.72,
        "require_signature_match": True,
        "restructure_eval": True,
        "confidence_gate": "none",
        "early_stop": "no_improve",
        "no_improve_patience": 3,
        "smart_repair": True,
        "full_code_prompt": True,
        "repair_repeat_invariants": True,
        "format_prompt_raw_limit": 1200,
        "staged_round_plan": "gemini_three_stage",
        "gemini_repeat_header": True,
        "gemini_forbid_rename": True,
    },
    # Cenzihan v1：三段式状态机 + 回退重试（每阶段最多 3 次）
    "task8cenzihan_v1": {
        "loop_style": "guided",
        "max_examples": 0,
        "max_rounds": 25,
        "max_tokens": 4096,
        "variant": "task8_gemini",
        "semantic_threshold": 0.72,
        "require_signature_match": True,
        "restructure_eval": True,
        "confidence_gate": "none",
        "early_stop": "no_improve",
        "no_improve_patience": 4,
        "smart_repair": True,
        "full_code_prompt": True,
        "repair_repeat_invariants": True,
        "format_prompt_raw_limit": 1200,
        "staged_round_plan": "cenzihan_v1_state_machine",
        "gemini_repeat_header": True,
        "gemini_forbid_rename": True,
        "return_last_candidate": True,
    },

    # V2_v3: 智能 repair + legacy loop (对比 guided)
    "v2_zero_r3_v3": {
        "loop_style": "legacy",        # ★ 用 legacy loop + smart_repair
        "max_examples": 0,
        "max_rounds": 3,
        "max_tokens": 4096,
        "variant": "v2_zero_r3_v2",
        "smart_repair": True,          # ★ 启用智能 repair
    },

    # V2_v4: 智能 repair + 1-shot (对比 zero-shot)
    "v2_fs1_r3_v2": {
        "loop_style": "guided",
        "max_examples": 1,
        "max_rounds": 3,
        "max_tokens": 4096,
        "variant": "v2_structured_fs1",
        "semantic_threshold": 0.72,
        "require_signature_match": True,
        "restructure_eval": True,
        "confidence_gate": "none",
        "early_stop": "no_improve",
        "smart_repair": True,          # ★ 启用智能 repair
    },

    # V2_r5: 智能 repair + 5轮 (测试更多轮是否值得)
    "v2_zero_r5_v2": {
        "loop_style": "guided",
        "max_examples": 0,
        "max_rounds": 5,
        "max_tokens": 4096,
        "variant": "v2_zero_r3_v2",
        "semantic_threshold": 0.72,
        "require_signature_match": True,
        "restructure_eval": True,
        "confidence_gate": "none",
        "early_stop": "no_improve",
        "smart_repair": True,          # ★ 启用智能 repair
    },

    # V2_v2_AB: A + 最终语义补刀（触发条件：compile/module/wrapper 全通过）
    "v2_zero_r3_v2_AB": {
        "loop_style": "guided",
        "max_examples": 0,
        "max_rounds": 3,
        "max_tokens": 4096,
        "variant": "v2_zero_r3_v2",
        "semantic_threshold": 0.72,
        "require_signature_match": True,
        "restructure_eval": True,
        "confidence_gate": "none",
        "early_stop": "no_improve",
        "no_improve_patience": 3,
        "smart_repair": True,
        "full_code_prompt": True,
        "repair_repeat_invariants": True,
        "format_prompt_raw_limit": 1200,
        # B：最终语义补刀（严格：总分更高才替换）
        "final_semantic_refine": True,
        "final_refine_mode": "strict",
        "final_refine_max_tokens": 4096,
    },

    # V2_v2_AB2: 同 AB，末轮语义补刀放宽（见 final_semantic_refine_prediction）
    "v2_zero_r3_v2_AB2": {
        "loop_style": "guided",
        "max_examples": 0,
        "max_rounds": 3,
        "max_tokens": 4096,
        "variant": "v2_zero_r3_v2",
        "semantic_threshold": 0.72,
        "require_signature_match": True,
        "restructure_eval": True,
        "confidence_gate": "none",
        "early_stop": "no_improve",
        "no_improve_patience": 3,
        "smart_repair": True,
        "full_code_prompt": True,
        "repair_repeat_invariants": True,
        "format_prompt_raw_limit": 1200,
        "final_semantic_refine": True,
        "final_refine_mode": "relaxed",
        "final_refine_max_tokens": 4096,
    },

    # ═══════════════════════════════════════════════════════════════
    #  旧方案保留（用于对比旧 baseline，不重构 eval 输入）
    # ═══════════════════════════════════════════════════════════════
    "old_baseline_fs5_r3": {
        "loop_style": "guided",
        "max_examples": 5,
        "max_rounds": 3,
        "max_tokens": 4096,
        "variant": "strict_chat_fs2_wrapper_compact_module_guard",
        "semantic_threshold": 0.72,
        "require_signature_match": True,
        "restructure_eval": False,    # 旧方案不重构
        "confidence_gate": "none",
    },
}

SEMANTIC_WEIGHTS = {
    "signature_match": 3,
    "family_match": 2,
    "stride_match": 1,
    "mask_match": 1,
    "shape_match": 1,
    "math_match": 1,
    "kernel_launch_match": 1,
}

MASK_HINTS = ("mask", "memory safety", "broadcast", "causal", "padding", "boundary")
STRIDE_HINTS = ("stride", "contiguous", "non-contiguous")
SHAPE_HINTS = ("shape", "dimension", "dim", "dims", "size")
MATH_KEYWORDS = (
    "softmax",
    "argmax",
    "argmin",
    "sum",
    "mean",
    "exp",
    "log",
    "dot",
    "matmul",
    "convolution",
    "conv",
    "gelu",
    "relu",
    "sigmoid",
    "norm",
    "rmsnorm",
    "fft",
    "dequant",
    "attention",
)

CODE_PREFIXES = (
    "import ",
    "from ",
    "def ",
    "class ",
    "@",
    "return",
    "for ",
    "while ",
    "if ",
    "elif ",
    "else:",
    "try:",
    "except",
    "finally:",
    "with ",
    "pass",
    "raise ",
    "assert ",
    "yield ",
    "del ",
)

NARRATIVE_PREFIXES = (
    "But ",
    "But I ",
    "Wait",
    "Now,",
    "Now ",
    "Then,",
    "Then ",
    "First,",
    "First ",
    "Next,",
    "Next ",
    "Also,",
    "Also ",
    "However,",
    "However ",
    "Alternatively,",
    "Alternatively ",
    "So ",
    "In the ",
    "The ",
    "Putting it all together",
    "Let me ",
    "Let's ",
    "To fix",
    "Here ",
    "- ",
    "* ",
)
NARRATIVE_COMMENT_MARKERS = (
    "wait",
    "maybe",
    "how to",
    "placeholder",
    "todo",
    "fixme",
    "let's",
    "i need",
)


def _resolve_model_id() -> str:
    cached = MODEL_ID_CACHE.get(SERVICE_URL)
    if cached:
        return cached

    models_url = SERVICE_URL.replace("/v1/completions", "/v1/models")
    try:
        resp = requests.get(models_url, timeout=30)
        resp.raise_for_status()
        models = resp.json().get("data", [])
        if models and "id" in models[0]:
            model_id = models[0]["id"]
            MODEL_ID_CACHE[SERVICE_URL] = model_id
            return model_id
    except Exception:
        pass
    return str(BASE_DIR / "Qwen3-4B")


def _is_compilable_python(code: str | None) -> bool:
    if not code:
        return False
    try:
        compile(code, "<task8_eval>", "exec")
        return True
    except Exception:
        return False


def _get_compile_error(code: str | None) -> str | None:
    if not code:
        return None
    try:
        compile(code, "<task8_eval>", "exec")
        return None
    except Exception as exc:
        return str(exc)


def _extract_wrapper_name(input_text: str) -> str | None:
    wrapper_entry = task8.extract_wrapper_entry(input_text)
    if not wrapper_entry:
        return None
    if _looks_like_parameter_docs(wrapper_entry):
        return None
    match = re.search(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", wrapper_entry)
    return match.group(1) if match else None


def _looks_like_parameter_docs(value: str) -> bool:
    """Reject parameter documentation that is not a callable wrapper signature."""
    head = value.strip().split(";", 1)[0]
    first = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)\s*[:,]?", head)
    if not first:
        return False
    name, type_hint = first.group(1), first.group(2).lower()
    common_params = {
        "input", "other", "a", "b", "x", "y", "weight", "bias",
        "dim", "index", "src", "tensor", "tensors", "out",
    }
    common_type_words = ("tensor", "int", "float", "bool", "tuple", "number", "dtype")
    return name.lower() in common_params and any(word in type_hint for word in common_type_words)


def _clean_wrapper_signature_text(signature_text: str | None) -> str | None:
    if not signature_text:
        return None
    candidate = " ".join(signature_text.strip().split())
    if _looks_like_parameter_docs(candidate):
        return None
    candidate = re.split(r"\s*->|\s*;|\.\s*Args:|\s+Args:", candidate, maxsplit=1)[0].strip().rstrip(".")
    match = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", candidate)
    if not match:
        return candidate

    start = candidate.find("(", match.start())
    depth = 0
    for idx in range(start, len(candidate)):
        char = candidate[idx]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return candidate[: idx + 1]
    return candidate


def _extract_signature_from_text(signature_text: str | None) -> dict:
    if not signature_text:
        return {"name": None, "arg_names": [], "kwonly_names": [], "has_vararg": False, "has_kwarg": False}

    candidate = _clean_wrapper_signature_text(signature_text) or ""
    try:
        module = ast.parse(f"def {candidate}:\n    pass\n")
        fn = module.body[0]
        assert isinstance(fn, ast.FunctionDef)
        return {
            "name": fn.name,
            "arg_names": [arg.arg for arg in (fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs)],
            "kwonly_names": [arg.arg for arg in fn.args.kwonlyargs],
            "has_vararg": fn.args.vararg is not None,
            "has_kwarg": fn.args.kwarg is not None,
        }
    except Exception:
        match = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)", candidate)
        if not match:
            return {"name": None, "arg_names": [], "kwonly_names": [], "has_vararg": False, "has_kwarg": False}
        name = match.group(1)
        params = []
        raw_args = match.group(2).strip()
        if raw_args:
            for part in raw_args.split(","):
                param = part.strip()
                if not param or param == "*":
                    continue
                param = param.split(":", 1)[0].split("=", 1)[0].strip()
                param = param.lstrip("*")
                if param:
                    params.append(param)
        return {
            "name": name,
            "arg_names": params,
            "kwonly_names": [],
            "has_vararg": "*" in raw_args,
            "has_kwarg": "**" in raw_args,
        }


def _extract_expected_signature(input_text: str) -> dict:
    wrapper_entry = task8.extract_wrapper_entry(input_text)
    return _extract_signature_from_text(wrapper_entry)


def _extract_wrapper_signature_from_code(code: str | None, expected_wrapper: str | None) -> dict | None:
    if not code:
        return None
    try:
        module = ast.parse(code)
    except Exception:
        return None

    defs: list[ast.FunctionDef] = [node for node in module.body if isinstance(node, ast.FunctionDef)]
    if not defs:
        return None

    target: ast.FunctionDef | None = None
    if expected_wrapper:
        for fn in defs:
            if fn.name == expected_wrapper:
                target = fn
                break
    if target is None:
        for fn in defs:
            decorator_names = {
                ast.unparse(deco) if hasattr(ast, "unparse") else ""
                for deco in fn.decorator_list
            }
            if not any("triton.jit" in name for name in decorator_names):
                target = fn
                break
    if target is None:
        target = defs[-1]

    return {
        "name": target.name,
        "arg_names": [arg.arg for arg in (target.args.posonlyargs + target.args.args + target.args.kwonlyargs)],
        "kwonly_names": [arg.arg for arg in target.args.kwonlyargs],
        "has_vararg": target.args.vararg is not None,
        "has_kwarg": target.args.kwarg is not None,
    }


def _check_structure(code: str | None, expected_wrapper: str | None, config: dict | None = None) -> dict:
    if code is None:
        return {
            "has_import_torch": False,
            "has_import_triton": False,
            "has_import_tl": False,
            "has_triton_jit": False,
            "has_wrapper_def": False,
            "wrapper_match": False,
            "module_style_ok": False,
            "struct_ok": False,
        }

    lowered = code.lower()
    has_import_torch = _has_import_torch_namespace(code)
    has_import_triton = _has_import_triton_namespace(code)
    has_import_tl = _has_import_tl_namespace(code)
    has_triton_jit = "@triton.jit" in lowered
    has_wrapper_def = bool(re.search(r"(?m)^def\s+[A-Za-z_][A-Za-z0-9_]*\s*\(", code))
    wrapper_match = True
    if expected_wrapper:
        wrapper_match = bool(re.search(rf"(?m)^def\s+{re.escape(expected_wrapper)}\s*\(", code))

    compile_ok = _is_compilable_python(code)
    if config and config.get("pure_pytorch"):
        module_style_ok = compile_ok and has_import_torch and has_wrapper_def
    else:
        module_style_ok = (
            compile_ok
            and has_import_torch
            and has_import_triton
            and has_import_tl
            and has_triton_jit
            and has_wrapper_def
        )
    struct_ok = module_style_ok and wrapper_match

    return {
        "has_import_torch": has_import_torch,
        "has_import_triton": has_import_triton,
        "has_import_tl": has_import_tl,
        "has_triton_jit": has_triton_jit,
        "has_wrapper_def": has_wrapper_def,
        "wrapper_match": wrapper_match,
        "module_style_ok": module_style_ok,
        "struct_ok": struct_ok,
    }


def _math_keywords_for_query(input_text: str) -> list[str]:
    math_block = None
    if hasattr(task8, "_extract_named_section"):
        math_block = task8._extract_named_section(input_text, "Math")
    lower = (math_block or input_text).lower()
    return [kw for kw in MATH_KEYWORDS if kw in lower]


def _query_requires_any(input_text: str, hints: tuple[str, ...]) -> bool:
    lowered = input_text.lower()
    return any(hint in lowered for hint in hints)


def _build_compact_request_summary(input_text: str) -> str:
    compact_request = (
        task8._structured_request(input_text, compact=True)
        if hasattr(task8, "_structured_request")
        else input_text.strip()
    )
    wrapper_hint = task8.task8_signature_hint(input_text)
    family_tags = sorted(task8._extract_family_tags(input_text)) if hasattr(task8, "_extract_family_tags") else []
    math_keywords = _math_keywords_for_query(input_text)

    lines = [
        "Target wrapper signature:",
        wrapper_hint,
        "",
        "Compact request summary:",
        compact_request,
    ]
    if family_tags:
        lines.extend(["", "Expected operator family:", ", ".join(family_tags)])
    if math_keywords:
        lines.extend(["", "Important math keywords:", ", ".join(math_keywords)])
    return "\n".join(lines).strip()


def _required_wrapper_header(input_text: str) -> str:
    wrapper_entry = task8.extract_wrapper_entry(input_text)
    if wrapper_entry:
        signature = _clean_wrapper_signature_text(wrapper_entry) or " ".join(wrapper_entry.strip().split())
        return "def " + signature + ":"
    wrapper_hint = task8.task8_signature_hint(input_text)
    return wrapper_hint.strip()


def _strip_thinking_sections(text: str | None) -> str | None:
    if not text:
        return text
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<think>.*", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = cleaned.replace("</think>", "")
    return cleaned.strip()


def _has_import_torch_namespace(code: str | None) -> bool:
    if not code:
        return False
    return bool(
        re.search(r"(?m)^\s*import\s+torch\s*(?:#.*)?$", code)
        or re.search(r"(?m)^\s*import\s+torch\s+as\s+\w+\s*(?:#.*)?$", code)
    )


def _has_import_triton_namespace(code: str | None) -> bool:
    if not code:
        return False
    return bool(
        re.search(r"(?m)^\s*import\s+triton\s*(?:#.*)?$", code)
        or re.search(r"(?m)^\s*import\s+triton\s+as\s+\w+\s*(?:#.*)?$", code)
    )


def _has_import_tl_namespace(code: str | None) -> bool:
    if not code:
        return False
    return bool(
        re.search(r"(?m)^\s*import\s+triton\.language\s+as\s+tl\b", code)
        or re.search(r"(?m)^\s*from\s+triton(?:\.language)?\s+import\s+.*\btl\b", code)
    )


def _prefer_last_import_module(text: str | None) -> str | None:
    if not text:
        return text
    matches = list(
        re.finditer(
            r"(?m)^\s*import\s+torch\s*(?:#.*)?$|^\s*import\s+torch\s+as\s+\w+\s*(?:#.*)?$",
            text,
        )
    )
    if not matches:
        return text
    best = text
    for match in reversed(matches):
        candidate = text[match.start():].strip()
        if "def " in candidate or "@triton.jit" in candidate:
            best = candidate
            break
    best = re.split(r"(?im)^\s*(?:therefore|however|now|wait|hmm|maybe|explanation)\b", best)[0].strip()
    return best or text


def _extract_code_fallback(raw_response: str | None) -> str | None:
    if not raw_response:
        return None
    raw_response = _strip_thinking_sections(raw_response) or raw_response

    label_match = re.search(r"<label>\s*(.*?)\s*</label>", raw_response, re.DOTALL | re.IGNORECASE)
    if label_match:
        return _prefer_last_import_module(label_match.group(1).strip())

    python_block = re.search(r"```python\s*(.*?)\s*```", raw_response, re.DOTALL | re.IGNORECASE)
    if python_block:
        return _prefer_last_import_module(python_block.group(1).strip())

    code_block = re.search(r"```\s*(.*?)\s*```", raw_response, re.DOTALL)
    if code_block:
        content = code_block.group(1).strip()
        if "import " in content or "def " in content or "@triton" in content:
            return _prefer_last_import_module(content)

    import_module = _prefer_last_import_module(raw_response)
    if import_module != raw_response and import_module:
        return import_module

    lines = raw_response.splitlines()
    code_lines = []
    in_code = False
    for line in lines:
        if line.strip().startswith(("import ", "from ", "@", "def ", "class ")):
            in_code = True
        if in_code:
            code_lines.append(line)

    if code_lines:
        return _prefer_last_import_module("\n".join(code_lines))
    return None


def _looks_like_python_line(stripped: str) -> bool:
    if not stripped:
        return True
    if stripped.startswith(("#", ")", "]", "}", ">>>")):
        return True
    if stripped.startswith(CODE_PREFIXES):
        return True
    if re.match(r"^[A-Za-z_][A-Za-z0-9_.,\[\] ()]*\s*=", stripped):
        return True
    if re.match(r"^[A-Za-z_][A-Za-z0-9_\.]*\s*\(", stripped):
        return True
    if re.match(r"^[A-Za-z_][A-Za-z0-9_\.]*\[[^\]]+\]\s*\(", stripped):
        return True
    return False


def _looks_like_narrative_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.startswith("#"):
        lowered = stripped.lower()
        return any(marker in lowered for marker in NARRATIVE_COMMENT_MARKERS)
    if _looks_like_python_line(stripped):
        return False
    if stripped in {"<label>", "</label>"}:
        return True
    if stripped.startswith("```"):
        return True
    if re.match(r"^\d+\.\s+[A-Z]", stripped):
        return True
    if any(stripped.startswith(prefix) for prefix in NARRATIVE_PREFIXES):
        return True
    if stripped.endswith(":") and len(stripped.split()) >= 3:
        return True
    if re.fullmatch(r"[A-Za-z0-9_`'\"(),:; \-]+", stripped) and len(stripped.split()) >= 4:
        return True
    return False


def _remove_narrative_lines(code: str | None) -> str | None:
    if not code:
        return code
    cleaned_lines = [line for line in code.splitlines() if not _looks_like_narrative_line(line)]
    cleaned = "\n".join(cleaned_lines).strip()
    return cleaned or code


def _truncate_to_compilable_prefix(code: str | None, min_lines: int = 8) -> str | None:
    if not code:
        return code
    if _is_compilable_python(code):
        return code

    lines = code.splitlines()
    floor = min(min_lines, len(lines))
    for end in range(len(lines) - 1, floor - 1, -1):
        candidate = "\n".join(lines[:end]).rstrip()
        if not candidate:
            continue
        if "def " not in candidate and "@triton.jit" not in candidate:
            continue
        if _is_compilable_python(candidate):
            return candidate
    return code


def _ensure_required_imports(code: str | None) -> str | None:
    if not code:
        return code

    lines = code.splitlines()
    missing_lines: list[str] = []
    if not _has_import_torch_namespace(code):
        missing_lines.append("import torch")
    uses_triton = bool("@triton" in code or "triton." in code or re.search(r"\btl\.", code))
    if uses_triton and not _has_import_triton_namespace(code):
        missing_lines.append("import triton")
    if uses_triton and not _has_import_tl_namespace(code):
        missing_lines.append("import triton.language as tl")

    if not missing_lines:
        return code

    insert_at = 0
    while insert_at < len(lines) and lines[insert_at].startswith(("from __future__", "#!")):
        insert_at += 1
    new_lines = lines[:insert_at] + missing_lines + ([""] if insert_at < len(lines) else []) + lines[insert_at:]
    return "\n".join(new_lines).strip()


def _has_comment_only_function_body(code: str | None) -> bool:
    if not code:
        return False
    lines = code.splitlines()
    for i, line in enumerate(lines):
        match = re.match(r"^(\s*)def\s+[A-Za-z_][A-Za-z0-9_]*\s*\(", line)
        if not match:
            continue
        if ":" in line and line.split(":", 1)[1].strip():
            continue
        base_indent = len(match.group(1))
        has_indented_line = False
        has_executable_line = False
        for body_line in lines[i + 1 :]:
            stripped = body_line.strip()
            if not stripped:
                continue
            indent = len(body_line) - len(body_line.lstrip(" "))
            if indent <= base_indent:
                break
            has_indented_line = True
            if stripped.startswith("#"):
                continue
            if stripped in {'"""', "'''"}:
                continue
            has_executable_line = True
            break
        if not has_indented_line or not has_executable_line:
            return True
    return False


def _has_excessive_or_repeated_comments(code: str | None) -> bool:
    if not code:
        return False
    comments = [line.strip()[1:].strip() for line in code.splitlines() if line.strip().startswith("#")]
    if not comments:
        return False
    if len(comments) > 6:
        return True
    if any(len(comment.split()) > 18 for comment in comments):
        return True
    normalized = [re.sub(r"\s+", " ", comment.lower()) for comment in comments if len(comment) >= 24]
    return len(normalized) != len(set(normalized))


def _has_touch_helper_and_call(code: str | None) -> bool:
    if not code:
        return False
    has_helper = bool(re.search(r"(?m)^def\s+_openseek_touch\s*\(", code))
    has_kernel = bool(re.search(r"(?m)^def\s+_openseek_touch_kernel\s*\(", code))
    has_call = len(re.findall(r"_openseek_touch\s*\(", code)) >= 2
    return has_helper and has_kernel and has_call


def _contains_placeholder_code(code: str | None) -> bool:
    if not code:
        return False
    lowered = code.lower()
    placeholder_phrases = (
        "code here",
        "implementation here",
        "your code here",
        "actual implementation",
        "pseudo-code",
        "pseudocode",
        "hmm",
        "maybe",
        "let me",
    )
    if any(phrase in lowered for phrase in placeholder_phrases):
        return True
    if re.search(r"(?m)^\s*def\s+[A-Za-z_][A-Za-z0-9_]*\s*\(\s*\.\.\.\s*\)", code):
        return True
    if re.search(r"(?m)^\s*\.\.\.\s*$", code):
        return True
    if re.search(r"(?m)^\s*pass\s*(?:#.*)?$", code):
        return True
    if re.search(r"(?m)^\s*#?\s*(?:TODO|FIXME|PLACEHOLDER)\b", code):
        return True
    if _has_comment_only_function_body(code):
        return True
    if _has_excessive_or_repeated_comments(code):
        return True
    return "..." in code and "tl.arange(...)" not in code


def _sanitize_pure_pytorch_code(code: str | None) -> str | None:
    if not code:
        return code
    lines = [
        line for line in code.splitlines()
        if not re.match(r"^\s*(?:import\s+triton\b|import\s+triton\.language\b|from\s+triton\b)", line)
    ]
    candidate = "\n".join(lines)
    candidate = re.sub(r"\s*->\s*Tensor(?=\s*:)", "", candidate)
    for name in (
        "sqrt",
        "rsqrt",
        "tanh",
        "sigmoid",
        "abs",
        "floor",
        "erf",
        "add",
        "sub",
        "mul",
        "div",
        "max",
        "min",
        "argmax",
        "argmin",
        "sum",
        "mean",
        "std",
        "var",
        "index_select",
        "gather",
        "scatter",
        "where",
        "sort",
        "topk",
        "i0",
        "signbit",
    ):
        candidate = re.sub(rf"\bF\.{name}\s*\(", f"torch.{name}(", candidate)
        candidate = re.sub(rf"\btorch\.nn\.functional\.{name}\s*\(", f"torch.{name}(", candidate)
    candidate = re.sub(
        r"def\s+sqrt\s*\(\s*input\s*,\s*\*\s*,\s*out\s*=\s*None\s*\):\s*\n[ \t]*return\s+torch\s*$",
        "def sqrt(input, *, out=None):\n    return torch.sqrt(input, out=out)",
        candidate,
        flags=re.MULTILINE,
    )
    candidate = re.sub(
        r"def\s+rsqrt\s*\(\s*input\s*,\s*\*\s*,\s*out\s*=\s*None\s*\):\s*\n[ \t]*return\s+torch\s*$",
        "def rsqrt(input, *, out=None):\n    return torch.rsqrt(input, out=out)",
        candidate,
        flags=re.MULTILINE,
    )
    candidate = re.sub(
        r"def\s+tanh\s*\(\s*input\s*,\s*\*\s*,\s*out\s*=\s*None\s*\):\s*\n[ \t]*return\s+torch\s*$",
        "def tanh(input, *, out=None):\n    return torch.tanh(input, out=out)",
        candidate,
        flags=re.MULTILINE,
    )
    candidate = re.sub(
        r"torch\.sub\(\s*input\s*,\s*other\s*\*\s*alpha\s*,\s*alpha\s*=\s*alpha\s*,\s*out\s*=\s*out\s*\)",
        "torch.sub(input, other, alpha=alpha, out=out)",
        candidate,
    )
    candidate = re.sub(
        r"def\s+tanh\s*\(\s*input\s*,\s*\*\s*,\s*out\s*=\s*None\s*\):\s*\n(?:[ \t].*\n)*?[ \t]*return\s+out\s+if\s+out\s+is\s+not\s+None\s+else\s+input",
        "def tanh(input, *, out=None):\n    return torch.tanh(input, out=out)",
        candidate,
        flags=re.DOTALL,
    )
    return candidate


def _sanitize_candidate_code(code: str | None, config: dict | None = None) -> str | None:
    if not code:
        return code
    candidate = (_strip_thinking_sections(code) or code).strip()
    if config and config.get("pure_pytorch"):
        candidate = _sanitize_pure_pytorch_code(candidate) or candidate
    if config and config.get("static_api_postfix"):
        candidate = _static_api_postfix(candidate) or candidate
    candidate = _prefer_last_import_module(candidate) or candidate
    candidate = _remove_narrative_lines(candidate) or candidate
    candidate = _prefer_last_import_module(candidate) or candidate
    candidate = _truncate_to_compilable_prefix(candidate) or candidate
    candidate = _ensure_required_imports(candidate) or candidate
    if config and config.get("pure_pytorch"):
        candidate = _sanitize_pure_pytorch_code(candidate) or candidate
    if config and config.get("static_api_postfix"):
        candidate = _static_api_postfix(candidate) or candidate
    candidate = _remove_narrative_lines(candidate) or candidate
    candidate = _prefer_last_import_module(candidate) or candidate
    candidate = _ensure_required_imports(candidate) or candidate
    candidate = _truncate_to_compilable_prefix(candidate) or candidate
    if candidate and (not _is_compilable_python(candidate) or _contains_placeholder_code(candidate)):
        fixed, method = postfix_task8.try_fix(candidate)
        if fixed:
            candidate = fixed.strip()
    return candidate.strip() if candidate else candidate


def _static_api_postfix(code: str | None) -> str | None:
    """Conservative repair for frequent PyTorch API call failures."""
    if not code:
        return code
    fixed = code

    fixed = re.sub(r"def\s+(?:torch|linalg|la)\.([A-Za-z_][A-Za-z0-9_]*)\s*\(", r"def \1(", fixed)
    fixed = re.sub(r"def\s+torch\.linalg\.([A-Za-z_][A-Za-z0-9_]*)\s*\(", r"def \1(", fixed)

    for name in ("Optional", "Union", "Tuple", "List", "Tensor"):
        if re.search(rf"\b{name}\s*\[", fixed) or re.search(rf":\s*{name}\b", fixed):
            if f"from typing import" not in fixed and name != "Tensor":
                fixed = "from typing import Optional, Union, Tuple, List\n" + fixed
                break
    if re.search(r"(?<!torch\.)\bTensor\b", fixed) and "from torch import Tensor" not in fixed:
        fixed = "from torch import Tensor\n" + fixed

    fixed = re.sub(r"\b([A-Za-z_][A-Za-z0-9_]*)\.copy_\(out\)", r"out.copy_(\1)", fixed)
    fixed = re.sub(r"\b([A-Za-z_][A-Za-z0-9_]*)\.copy_\(\1\)(\s*\n\s*return\s+out)", r"out.copy_(\1)\2", fixed)

    fixed = re.sub(r",\s*lower\s*=\s*True", ", upper=False", fixed)
    fixed = re.sub(r",\s*lower\s*=\s*False", ", upper=True", fixed)
    fixed = re.sub(
        r"(solve_triangular\([^)\n]*?),\s*inplace\s*=\s*False",
        r"\1",
        fixed,
    )
    fixed = re.sub(
        r"(solve_triangular\([^)\n]*?),\s*transpose\s*=\s*(?:False|True)",
        r"\1",
        fixed,
    )

    fixed = fixed.replace("torch.is_training()", "training")
    fixed = re.sub(r"F\.layer_norm\(([^)\n]*),\s*out\s*=\s*out\)", r"F.layer_norm(\1)", fixed)
    fixed = fixed.replace("torch.noop()", "torch.enable_grad()")

    fixed = re.sub(r"F\.repeat_interleave\s*\(", "torch.repeat_interleave(", fixed)
    fixed = fixed.replace("torch.degrees(", "torch.rad2deg(")

    return fixed


NULL_COMPILE_FAIL_PREDICTION = "import torch\n\ndef _openseek_null_compile_fail(\n"


def _extract_candidate_code(raw_response: str | None, config: dict | None = None) -> str | None:
    cleaned_raw = _strip_thinking_sections(raw_response)
    code = task8.extract_prediction(cleaned_raw)
    if code is None and cleaned_raw:
        code = _extract_code_fallback(cleaned_raw)
    return _sanitize_candidate_code(code, config=config)


def _prefer_compilable_assessment(current: dict | None, candidate: dict | None) -> dict | None:
    if candidate is None:
        return current
    if current is None:
        return candidate
    if candidate["compile_ok"] and not current["compile_ok"]:
        return candidate
    if current["compile_ok"] and not candidate["compile_ok"]:
        return current
    return candidate if candidate["score"] > current["score"] else current


def _tokenize_code(text: str | None) -> list[str]:
    if not text:
        return []
    return re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\S", text)


def _token_f1(prediction: str | None, gold: str | None) -> float:
    pred_tokens = _tokenize_code(prediction)
    gold_tokens = _tokenize_code(gold)
    if not pred_tokens or not gold_tokens:
        return 0.0
    pred_counts: dict[str, int] = {}
    gold_counts: dict[str, int] = {}
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1
    for token in gold_tokens:
        gold_counts[token] = gold_counts.get(token, 0) + 1
    overlap = 0
    for token, count in pred_counts.items():
        overlap += min(count, gold_counts.get(token, 0))
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _check_semantic_proxy(
    input_text: str,
    code: str | None,
    expected_wrapper: str | None,
    expected_signature: dict,
    config: dict,
) -> dict:
    if code is None:
        return {
            "semantic_ratio": 0.0,
            "semantic_points": 0,
            "semantic_max": 0,
            "semantic_pass": False,
            "signature_match": False,
            "family_match": False,
            "stride_match": False,
            "mask_match": False,
            "shape_match": False,
            "math_match": False,
            "kernel_launch_match": False,
        }

    code_lower = code.lower()
    family_tags = task8._extract_family_tags(input_text) if hasattr(task8, "_extract_family_tags") else set()
    pred_families = task8._extract_family_tags(code) if hasattr(task8, "_extract_family_tags") else set()
    requires_stride = _query_requires_any(input_text, STRIDE_HINTS)
    requires_mask = _query_requires_any(input_text, MASK_HINTS)
    requires_shape = _query_requires_any(input_text, SHAPE_HINTS)
    math_keywords = _math_keywords_for_query(input_text)
    expected_code_sig = _extract_wrapper_signature_from_code(code, expected_wrapper)

    if expected_signature["arg_names"]:
        signature_match = (
            expected_code_sig is not None
            and expected_code_sig["arg_names"] == expected_signature["arg_names"]
        )
    else:
        signature_match = expected_code_sig is not None

    family_match = (not family_tags) or bool(family_tags & pred_families)
    stride_match = (not requires_stride) or any(
        token in code_lower for token in ("stride", ".stride(", "stride_", "strides")
    )
    mask_match = (not requires_mask) or any(
        token in code_lower for token in ("mask", "boundary_check", "other=", "tl.where")
    )
    shape_match = (not requires_shape) or any(
        token in code_lower for token in (".shape", "shape", "size(", "numel", "stride(", ".dim(")
    )
    math_match = (not math_keywords) or any(keyword in code_lower for keyword in math_keywords) or family_match
    pure_pytorch = bool(config.get("pure_pytorch"))
    kernel_launch_match = True if pure_pytorch else bool(re.search(r"(?m)[A-Za-z_][A-Za-z0-9_]*\[[^\n]+\]\s*\(", code))

    checks = {
        "signature_match": (bool(expected_signature["arg_names"]), signature_match),
        "family_match": (bool(family_tags), family_match),
        "stride_match": (requires_stride, stride_match),
        "mask_match": (requires_mask, mask_match),
        "shape_match": (requires_shape, shape_match),
        "math_match": (bool(math_keywords), math_match),
        "kernel_launch_match": (not pure_pytorch, kernel_launch_match),
    }

    points = 0
    max_points = 0
    for name, (applicable, passed) in checks.items():
        weight = SEMANTIC_WEIGHTS[name]
        if applicable:
            max_points += weight
            if passed:
                points += weight

    semantic_ratio = (points / max_points) if max_points else 0.0
    semantic_pass = semantic_ratio >= config.get("semantic_threshold", 0.72)
    if config.get("require_signature_match") and not signature_match:
        semantic_pass = False
    if config.get("require_family_match") and family_tags and not family_match:
        semantic_pass = False
    if config.get("require_math_match") and math_keywords and not math_match:
        semantic_pass = False

    return {
        "semantic_ratio": semantic_ratio,
        "semantic_points": points,
        "semantic_max": max_points,
        "semantic_pass": semantic_pass,
        "signature_match": signature_match,
        "family_match": family_match,
        "stride_match": stride_match,
        "mask_match": mask_match,
        "shape_match": shape_match,
        "math_match": math_match,
        "kernel_launch_match": kernel_launch_match,
    }


def _classify_failure(assessment: dict) -> str:
    if not assessment["extract_ok"]:
        return "format"
    if not assessment["compile_ok"]:
        return "compile"
    return "semantic"


def _score_candidate(assessment: dict) -> float:
    score = 0.0
    score += 5.0 if assessment["extract_ok"] else 0.0
    score += 25.0 if assessment["compile_ok"] else -5.0
    score += 10.0 if assessment["structure"]["module_style_ok"] else 0.0
    score += 15.0 if assessment["structure"]["wrapper_match"] else 0.0
    score += (15.0 if assessment["compile_ok"] else 5.0) * assessment["semantic"]["semantic_ratio"]
    score += 10.0 if assessment["compile_ok"] and assessment["semantic"]["semantic_pass"] else 0.0
    return score


def _assess_candidate(
    input_text: str,
    raw_response: str | None,
    code: str | None,
    expected_wrapper: str | None,
    expected_signature: dict,
    config: dict,
) -> dict:
    extract_ok = code is not None
    compile_error = _get_compile_error(code)
    compile_ok = extract_ok and compile_error is None
    structure = _check_structure(code, expected_wrapper, config=config)
    semantic = _check_semantic_proxy(input_text, code, expected_wrapper, expected_signature, config)
    placeholder_rejected = bool(config.get("reject_placeholder_code") and _contains_placeholder_code(code))
    launch_required_failed = bool(
        config.get("require_kernel_launch") and not semantic.get("kernel_launch_match", False)
    )
    touch_helper_failed = bool(config.get("require_touch_helper") and not _has_touch_helper_and_call(code))
    if not compile_ok:
        semantic["semantic_pass"] = False
    if placeholder_rejected or launch_required_failed or touch_helper_failed:
        semantic["semantic_pass"] = False
    structure_success = structure["module_style_ok"] and structure["wrapper_match"]
    success = compile_ok and structure["wrapper_match"] and structure["module_style_ok"] and semantic["semantic_pass"]

    assessment = {
        "raw_response": raw_response,
        "code": code,
        "extract_ok": extract_ok,
        "compile_ok": compile_ok,
        "compile_error": compile_error,
        "structure": structure,
        "structure_success": structure_success,
        "semantic": semantic,
        "semantic_success": semantic["semantic_pass"],
        "placeholder_rejected": placeholder_rejected,
        "launch_required_failed": launch_required_failed,
        "touch_helper_failed": touch_helper_failed,
        "success": success,
        "pure_pytorch": bool(config.get("pure_pytorch")),
    }
    assessment["failure_type"] = _classify_failure(assessment)
    assessment["score"] = _score_candidate(assessment)
    if placeholder_rejected:
        assessment["score"] -= 15.0
    if launch_required_failed:
        assessment["score"] -= 10.0
    if touch_helper_failed:
        assessment["score"] -= 10.0
    return assessment


def _trim_text(text: str | None, limit: int = 2500) -> str:
    if not text:
        return ""
    text = text.strip()
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]..."


def _effective_limit(config: dict, key: str, default_limit: int) -> int:
    value = config.get(key, config.get("safe_code_limit", default_limit))
    try:
        limit = int(value)
    except (TypeError, ValueError):
        return default_limit
    if limit <= 0:
        return 0
    return max(256, limit)


def _code_prompt_limit(config: dict, key: str, default_limit: int) -> int:
    if config.get("full_code_prompt"):
        return 0
    return _effective_limit(config, key, default_limit)


def _repair_repeat_invariants_block(query: str, pure_pytorch: bool = False) -> str:
    sig = task8_signature_hint(query)
    hdr = _required_wrapper_header(query)
    if pure_pytorch:
        return (
            "REPEAT — non-negotiable invariants:\n"
            f"- Target wrapper / API must match: {sig}\n"
            f"- Wrapper def line must align with: {hdr}\n"
            "- Inside <label>...</label>: raw Python only; no English prose (no lines like 'Hmm', 'Wait', 'I need to').\n"
            "- No `...`, no empty bodies with only `pass`, no `# TODO`; implement with torch / torch.nn.functional.\n"
            "- Pure PyTorch only: allowed imports are `import torch` and optionally `import torch.nn.functional as F`.\n"
            "- Do not add custom CUDA/JIT helpers, placeholder launches, or identity touch helpers.\n\n"
            "- If annotations use Optional/Union/Tuple/List, import them from typing; otherwise omit those annotations.\n"
            "- Do not define invalid names like `def torch.foo`, `def linalg.foo`, or `def la.foo`; define the final wrapper identifier only.\n"
            "- Do not write `def f(*args, *, x=...)`; after `*args` the remaining parameters are already keyword-only.\n"
            "- Use exact PyTorch kwargs only: solve_triangular has `upper=`, not `lower=`, `transpose=`, or `inplace=`.\n"
            "- For out tensors, always write `out.copy_(result)` and return out; never write `result.copy_(out)`.\n\n"
        )
    return (
        "REPEAT — non-negotiable invariants:\n"
        f"- Target wrapper / API must match: {sig}\n"
        f"- Wrapper def line must align with: {hdr}\n"
        "- Inside <label>...</label>: raw Python only; no English prose (no lines like 'Hmm', 'Wait', 'I need to').\n"
        "- No `...`, no empty bodies with only `pass`, no `# TODO`; implement or use torch ops in the wrapper.\n"
        "- Required: `import torch`, `import triton`, `import triton.language as tl`; at least one `@triton.jit`;\n"
        "  wrapper must launch via `kernel[grid](...)`.\n\n"
    )


def _semantic_priority_hint() -> str:
    # 来自 v2_zero_r3_v2 现有离线分析：用于提醒模型优先修最常见失配点
    return (
        "Global semantic-risk priorities from recent runs:\n"
        "- signature_match failures: 76.5%\n"
        "- kernel_launch_match failures: 64.7%\n"
        "- family_match failures: 64.7%\n"
        "- shape_match failures: 29.4%\n"
    )


def _build_semantic_issues(assessment: dict) -> list[str]:
    issues: list[str] = []
    compile_error = assessment["compile_error"]
    structure = assessment["structure"]
    semantic = assessment["semantic"]
    code = assessment.get("code")

    if compile_error:
        issues.append(f"Compilation error: {compile_error[:200]}")
        if "SyntaxError" in compile_error:
            issues.append("Fix syntax errors and unmatched delimiters.")
        elif "IndentationError" in compile_error:
            issues.append("Fix indentation and block structure.")
        elif "NameError" in compile_error:
            issues.append("Fix undefined names, imports, or missing helper definitions.")
    if code and any(prefix in code for prefix in NARRATIVE_PREFIXES):
        issues.append("Remove all prose or narrative sentences from inside the label; keep raw Python code only.")
    if _contains_placeholder_code(code):
        issues.append("Do not use ellipsis, placeholders, TODO markers, or pseudo-code.")
    if _has_comment_only_function_body(code):
        issues.append("A function body is empty or comment-only; every def body needs executable statements.")
    if _has_excessive_or_repeated_comments(code):
        issues.append("Remove long or repeated explanatory comments; keep implementation code, not commentary.")
    if assessment.get("touch_helper_failed"):
        issues.append("Missing the fixed _openseek_touch helper or missing a wrapper call to _openseek_touch(result).")

    if not structure["has_import_torch"]:
        issues.append("Missing required import: import torch")
    pure_pytorch = bool(assessment.get("pure_pytorch"))
    if not pure_pytorch and not structure["has_import_triton"]:
        issues.append("Missing required import: import triton")
    if not pure_pytorch and not structure["has_import_tl"]:
        issues.append("Missing required import: import triton.language as tl")
    if not pure_pytorch and not structure["has_triton_jit"]:
        issues.append("Missing @triton.jit kernel definition.")
    if not structure["has_wrapper_def"]:
        issues.append("Missing top-level wrapper function definition.")
    if not structure["wrapper_match"]:
        issues.append("Wrapper function name does not match the target wrapper signature.")
    if not semantic["signature_match"]:
        issues.append("Wrapper parameter list does not match the requested signature.")
    if not semantic["family_match"]:
        issues.append("The implementation does not look aligned to the requested operator family.")
    if not semantic["stride_match"]:
        issues.append("The code does not show enough stride-aware handling for the request.")
    if not semantic["mask_match"]:
        issues.append("The code does not show enough masking / boundary handling for memory safety.")
    if not semantic["shape_match"]:
        issues.append("The wrapper/kernel does not show enough shape-aware handling.")
    if not semantic["math_match"]:
        issues.append("The implementation does not reflect the requested math behavior strongly enough.")
    if not pure_pytorch and not semantic["kernel_launch_match"]:
        issues.append("The wrapper does not appear to launch a Triton kernel.")
    return issues


def _signature_param_lock_block(query: str) -> str:
    signature = _extract_expected_signature(query)
    required_header = _required_wrapper_header(query)
    arg_names = signature.get("arg_names", [])
    arg_list_text = ", ".join(arg_names) if arg_names else "(unavailable)"
    return (
        "Required wrapper header:\n"
        f"{required_header}\n\n"
        "Required parameter names (order-sensitive):\n"
        f"{arg_list_text}\n\n"
        "Rename policy:\n"
        "- Do NOT rename wrapper function name.\n"
        "- Do NOT rename, delete, or reorder wrapper parameters.\n"
        "- Keep defaults / keyword-only markers compatible with the target signature.\n\n"
    )


def _build_gemini_stage1_prompt(query: str) -> str:
    request_summary = _build_compact_request_summary(query)
    return (
        "Stage 1/3 (math core): draft the operator math logic first.\n\n"
        f"{request_summary}\n\n"
        f"{_signature_param_lock_block(query)}"
        "Stage objective:\n"
        "- Focus on the core tensor math for the requested behavior.\n"
        "- Do not finalize PID/offset/mask/load/store orchestration yet.\n"
        "- Return executable Python that can be refined in later stages.\n\n"
        "Hard constraints:\n"
        "- Output ONLY raw Python code between <label> and </label>.\n"
        "- No markdown, no explanation, no prose lines.\n"
        "- No placeholders: `...`, `pass`, `TODO`, pseudo-code.\n"
        "- Keep code compilable.\n\n"
        "Output exactly:\n<label>\nFULL_CODE\n</label>"
    )


def _build_gemini_stage_round_prompt(
    query: str,
    previous_code: str | None,
    assessment: dict,
    target_round: int,
    config: dict,
) -> str:
    request_summary = _build_compact_request_summary(query)
    code_limit = _code_prompt_limit(config, "semantic_prompt_code_limit", 3000)
    previous_block = _trim_text(previous_code, limit=code_limit)
    issue_list = _build_semantic_issues(assessment)
    issue_text = "\n".join(f"- {x}" for x in issue_list) if issue_list else "- keep improving structural fidelity"
    lock_block = _signature_param_lock_block(query) if config.get("gemini_repeat_header", True) else ""
    rename_line = (
        "- Do NOT rename wrapper function or parameters from the required signature.\n"
        if config.get("gemini_forbid_rename", True)
        else ""
    )

    if target_round == 2:
        stage_goal = (
            "Stage 2/3 (kernel framing): convert the previous draft into a real Triton kernel body.\n"
            "- Add pid / block_start / offsets / mask.\n"
            "- Use tl.load/tl.store safely with mask.\n"
            "- Keep math logic aligned with stage-1 draft.\n"
            "- Keep module compilable and include at least one @triton.jit kernel.\n"
        )
    elif target_round == 3:
        stage_goal = (
            "Stage 3/3 (wrapper integration): finalize wrapper scheduling and launch.\n"
            "- Ensure wrapper computes grid and launches kernel via kernel[grid](...).\n"
            "- Ensure imports are complete and module is executable.\n"
            "- Keep behavior aligned with request; preserve signature exactly.\n"
        )
    else:
        stage_goal = (
            f"Stage {target_round}/5 (repair & align): keep iterating on correctness.\n"
            "- Fix compile/structure/semantic issues listed below.\n"
            "- You may rewrite sections, but preserve the required wrapper signature.\n"
        )

    return (
        f"{stage_goal}\n"
        f"{request_summary}\n\n"
        f"{lock_block}"
        "Current issues to fix:\n"
        f"{issue_text}\n\n"
        "Previous round output:\n"
        f"{previous_block}\n\n"
        "Hard constraints:\n"
        "- Include `import torch`, `import triton`, `import triton.language as tl`.\n"
        "- Include at least one `@triton.jit` kernel and a real `kernel[grid](...)` launch.\n"
        f"{rename_line}"
        "- Output ONLY raw Python code between <label> and </label>.\n"
        "- No markdown, no explanation, no placeholders (`...`, `pass`, `TODO`).\n\n"
        "Output exactly:\n<label>\nFULL_CODE\n</label>"
    )


def _build_cenzihan_v1_prompt(
    query: str,
    stage: str,
    round_num: int,
    previous_code: str | None,
    assessment: dict | None,
    state: dict,
    config: dict,
) -> str:
    request_summary = _build_compact_request_summary(query)
    code_limit = _code_prompt_limit(config, "semantic_prompt_code_limit", 3000)
    previous_block = _trim_text(previous_code, limit=code_limit)
    lock_block = _signature_param_lock_block(query)
    issues = _build_semantic_issues(assessment) if assessment is not None else []
    issue_text = "\n".join(f"- {x}" for x in issues) if issues else "- no explicit issue list yet"
    retry_text = (
        f"(math_retry={state['math_retry']}/3, kernel_retry={state['kernel_retry']}/3, "
        f"wrapper_retry={state['wrapper_retry']}/3, final_retry={state['final_retry']}/3)"
    )

    if stage == "math_generate":
        stage_instruction = (
            "Round objective: Stage 1/3 — write core tensor math logic first.\n"
            "- Focus on math transformations and key intermediate variables.\n"
            "- Keep code compilable and ready for kernel integration in later stages.\n"
            "- Do not rename wrapper function or its parameters.\n"
        )
    elif stage == "math_review":
        stage_instruction = (
            "Round objective: Review Stage-1 math logic.\n"
            "- Validate if math matches requested behavior and signature constraints.\n"
            "- If not correct, rewrite the math/core part and keep interface stable.\n"
            "- If mostly correct, tighten edge cases and remove ambiguity.\n"
        )
    elif stage == "kernel_generate":
        stage_instruction = (
            "Round objective: Stage 2/3 — build Triton kernel framework around math.\n"
            "- Add pid/block_start/offsets/mask and safe tl.load/tl.store patterns.\n"
            "- Keep at least one @triton.jit kernel and preserve requested API surface.\n"
        )
    elif stage == "kernel_review":
        stage_instruction = (
            "Round objective: Review Stage-2 kernel framework.\n"
            "- Audit memory indexing/mask safety and kernel launch readiness.\n"
            "- If weak, rewrite kernel framing while preserving wrapper signature.\n"
        )
    elif stage == "wrapper_generate":
        stage_instruction = (
            "Round objective: Stage 3/3 — finalize wrapper scheduling and launch.\n"
            "- Ensure wrapper computes grid and performs real kernel[grid](...) launch.\n"
            "- Keep module executable and maintain exact wrapper header.\n"
        )
    elif stage == "wrapper_review":
        stage_instruction = (
            "Round objective: Review Stage-3 wrapper dispatch correctness.\n"
            "- Validate wrapper args/order/defaults and launch path.\n"
            "- If mismatch exists, repair wrapper without renaming API.\n"
        )
    else:
        stage_instruction = (
            "Round objective: Final sanitize stage.\n"
            "- Repair remaining compile/structure/semantic issues.\n"
            "- Prioritize success criteria while preserving required wrapper signature.\n"
        )

    return (
        f"Cenzihan-v1 staged planner | round={round_num} | stage={stage} {retry_text}\n\n"
        f"{request_summary}\n\n"
        f"{lock_block}"
        "Current known issues:\n"
        f"{issue_text}\n\n"
        "Previous round output:\n"
        f"{previous_block}\n\n"
        f"{stage_instruction}\n"
        "Hard constraints:\n"
        "- Include `import torch`, `import triton`, `import triton.language as tl`.\n"
        "- Keep at least one `@triton.jit` kernel and one real `kernel[grid](...)` launch.\n"
        "- Do NOT rename wrapper function or wrapper parameter names.\n"
        "- Output ONLY raw Python code between <label> and </label>.\n"
        "- No markdown, no explanation, no placeholders (`...`, `pass`, `TODO`).\n\n"
        "Output exactly:\n<label>\nFULL_CODE\n</label>"
    )


def _cenzihan_math_pass(assessment: dict) -> bool:
    code = assessment.get("code")
    return bool(assessment.get("extract_ok")) and not _contains_placeholder_code(code)


def _cenzihan_kernel_pass(assessment: dict) -> bool:
    structure = assessment.get("structure", {})
    return bool(
        assessment.get("compile_ok")
        and structure.get("has_triton_jit")
        and structure.get("has_import_triton")
        and structure.get("has_import_tl")
    )


def _cenzihan_wrapper_pass(assessment: dict) -> bool:
    structure = assessment.get("structure", {})
    semantic = assessment.get("semantic", {})
    return bool(
        assessment.get("compile_ok")
        and structure.get("wrapper_match")
        and semantic.get("kernel_launch_match")
    )


def _cenzihan_state_transition(state: dict, assessment: dict) -> tuple[dict, bool]:
    stage = state["stage"]
    done = False

    if stage == "math_generate":
        state["stage"] = "math_review"
    elif stage == "math_review":
        if _cenzihan_math_pass(assessment):
            state["stage"] = "kernel_generate"
        elif state["math_retry"] < 3:
            state["math_retry"] += 1
            state["stage"] = "math_generate"
        else:
            state["stage"] = "kernel_generate"
    elif stage == "kernel_generate":
        state["stage"] = "kernel_review"
    elif stage == "kernel_review":
        if _cenzihan_kernel_pass(assessment):
            state["stage"] = "wrapper_generate"
        elif state["kernel_retry"] < 3:
            state["kernel_retry"] += 1
            state["stage"] = "kernel_generate"
        else:
            state["stage"] = "wrapper_generate"
    elif stage == "wrapper_generate":
        state["stage"] = "wrapper_review"
    elif stage == "wrapper_review":
        if _cenzihan_wrapper_pass(assessment):
            state["stage"] = "final_sanitize"
        elif state["wrapper_retry"] < 3:
            state["wrapper_retry"] += 1
            state["stage"] = "wrapper_generate"
        else:
            state["stage"] = "final_sanitize"
    else:
        # final_sanitize
        if assessment.get("success"):
            done = True
        elif state["final_retry"] < 3:
            state["final_retry"] += 1
            state["stage"] = "final_sanitize"
        else:
            done = True

    return state, done


def _build_initial_prompt(task_description: str, query: str, examples: list[dict], variant: str | None, config: dict | None = None) -> str:
    if config and config.get("staged_round_plan") == "cenzihan_v1_state_machine":
        init_state = {"stage": "math_generate", "math_retry": 0, "kernel_retry": 0, "wrapper_retry": 0, "final_retry": 0}
        return _build_cenzihan_v1_prompt(
            query=query,
            stage=init_state["stage"],
            round_num=1,
            previous_code=None,
            assessment=None,
            state=init_state,
            config=config,
        )
    if config and config.get("staged_round_plan") == "gemini_three_stage":
        return _build_gemini_stage1_prompt(query)

    prompt = task8.build_prompt(task_description, query, variant=variant)
    if config and config.get("long_context_reference"):
        prompt = _attach_long_context_reference(prompt, query, examples, config)
    if examples and not (config and config.get("long_context_reference")):
        examples_str = "\n\n".join(task8.format_example(ex, variant=variant) for ex in examples)
        prompt = prompt.replace("[[EXAMPLES]]\n\n", examples_str + "\n\n")
    else:
        prompt = prompt.replace("[[EXAMPLES]]\n\n", "")
    return prompt


def _trim_for_long_context(text: str, limit: int) -> str:
    text = " ".join(text.strip().split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 16)].rstrip() + " ...[trimmed]"


def _long_context_entry(example: dict, item_limit: int) -> str:
    raw = example.get("input", "")
    sample_id = example.get("id", "(unknown)")
    wrapper = task8.extract_wrapper_entry(raw) or task8_signature_hint(raw) or "(see task text)"
    try:
        request = task8._structured_request(raw, compact=True)  # type: ignore[attr-defined]
    except Exception:
        request = raw
    request = _trim_for_long_context(request, item_limit)
    return (
        f"Reference training input id: {sample_id}\n"
        f"Wrapper: {wrapper}\n"
        f"Input summary: {request}"
    )


def _build_long_context_reference(query: str, examples: list[dict], config: dict) -> str:
    mode = config.get("long_context_mode", "training_summary")
    if mode == "rules_pad":
        return _build_rules_long_context(query, config)

    target_chars = int(config.get("long_context_target_chars", 66_000))
    item_limit = int(config.get("long_context_item_char_limit", 900))
    item_count = int(config.get("long_context_item_limit", len(examples)))
    selected = examples[:item_count]

    header = (
        "Long-context reference section. These are official TRAINING INPUT summaries only; "
        "training outputs and hidden test answers are deliberately omitted. Use this as broad "
        "API/style context, not as the current task. The current task after this section is "
        "authoritative.\n\n"
        "Stable PyTorch reminders:\n"
        "- Prefer direct torch.*, torch.nn.functional as F, torch.linalg.*, and torch.special.* calls.\n"
        "- Preserve the exact wrapper signature, output structure, broadcasting, dtype/device, out, "
        "inplace, training, reduction, dim, keepdim, eps, alpha/beta, stride/padding/dilation/groups semantics.\n"
        "- Do not implement Triton/CUDA kernels in this PyTorch configuration.\n"
    )
    parts = [header]
    total = len(header)

    for ex in selected:
        entry = _long_context_entry(ex, item_limit)
        block = f"\n---\n{entry}\n"
        parts.append(block)
        total += len(block)
        if total >= target_chars:
            break

    if total < target_chars and selected:
        # Pad with shorter second-pass reminders from the same training inputs.
        for ex in selected:
            wrapper = task8.extract_wrapper_entry(ex.get("input", "")) or task8_signature_hint(ex.get("input", ""))
            block = f"\nTraining wrapper reminder: {wrapper}\n"
            parts.append(block)
            total += len(block)
            if total >= target_chars:
                break

    current_wrapper = task8.extract_wrapper_entry(query) or task8_signature_hint(query) or "(see current task)"
    parts.append(
        "\nEnd long-context reference section.\n"
        f"The next task is the ONLY task to solve now. Current locked wrapper: {current_wrapper}\n"
    )
    return "".join(parts)


def _build_rules_long_context(query: str, config: dict) -> str:
    target_chars = int(config.get("long_context_target_chars", 66_000))
    current_wrapper = task8.extract_wrapper_entry(query) or task8_signature_hint(query) or "(see current task)"
    header = (
        "Long-context rules digest. This is generic reference material only; it contains no examples, "
        "no training outputs, and no hidden test answers. Solve only the current task after this section.\n"
        "<generic_pytorch_digest>\n"
    )
    unit = (
        "Use direct PyTorch semantics. Preserve the exact wrapper signature. Prefer torch.*, F.*, "
        "torch.linalg.*, torch.special.*. Preserve out/inplace/dim/keepdim/dtype/device/training/"
        "reduction/eps/alpha/beta/broadcasting. Do not import triton. Do not write CUDA kernels. "
        "Return only one complete Python module inside label tags.\n"
    )
    parts = [header]
    total = len(header)
    while total < target_chars:
        parts.append(unit)
        total += len(unit)
    parts.append(
        "</generic_pytorch_digest>\n"
        f"The next task is the ONLY task to solve now. Current locked wrapper: {current_wrapper}\n"
    )
    return "".join(parts)


def _attach_long_context_reference(prompt: str, query: str, examples: list[dict], config: dict) -> str:
    reference = _build_long_context_reference(query, examples, config)
    if prompt.startswith("/no_think\n"):
        return "/no_think\n" + reference + "\n" + prompt[len("/no_think\n") :]
    return reference + "\n" + prompt


def _build_format_correction_prompt(
    query: str,
    request_summary: str,
    previous_raw: str | None,
    round_num: int,
    raw_limit: int = 1200,
    invariant_block: str = "",
) -> str:
    prev = _trim_text(previous_raw, limit=raw_limit)
    return (
        f"Round {round_num}: the previous answer did not contain extractable raw Python code.\n\n"
        f"{request_summary}\n\n"
        f"{invariant_block}"
        "Fix strategy:\n"
        "- Start over from scratch.\n"
        "- Return a complete executable Python module.\n"
        "- Include `import torch`, `import triton`, and `import triton.language as tl`.\n"
        "- Include at least one `@triton.jit` kernel.\n"
        "- Include the exact target wrapper function.\n"
        "- Do not use markdown fences.\n"
        "- The first line must be `<label>` and the last line must be `</label>`.\n\n"
        f"Previous raw answer excerpt:\n{prev}\n\n"
        "Output exactly:\n<label>\nFULL_CODE\n</label>"
    )


def _build_compile_correction_prompt(
    query: str,
    request_summary: str,
    previous_code: str | None,
    issues: list[str],
    round_num: int,
    code_limit: int = 3000,
    invariant_block: str = "",
) -> str:
    issue_text = "\n".join(f"- {issue}" for issue in issues)
    code_block = _trim_text(previous_code, limit=code_limit)
    required_header = _required_wrapper_header(query)
    return (
        f"Round {round_num}: the previous code failed local compilation or module checks.\n\n"
        f"{request_summary}\n\n"
        "Required wrapper header:\n"
        f"{required_header}\n\n"
        "Detected problems:\n"
        f"{issue_text}\n\n"
        "Previous code:\n"
        f"{code_block}\n\n"
        f"{invariant_block}"
        "Repair instructions:\n"
        "- You may rewrite from scratch if the current structure is wrong.\n"
        "- Preserve only the target wrapper signature and requested behavior.\n"
        "- The wrapper header must match the required wrapper header exactly.\n"
        "- Return a complete compilable Python module.\n"
        "- The wrapper must launch a Triton kernel via `kernel[grid](...)`.\n"
        "- Do not include prose such as 'But', 'Wait', 'Now', 'Then', or section headings inside the label.\n"
        "- Do not use `...`, placeholders, omitted sections, or pseudo-code.\n"
        "- No markdown fences and no explanation.\n"
        "- First line `<label>`, last line `</label>`.\n\n"
        "Output exactly:\n<label>\nFULL_CODE\n</label>"
    )


def _build_placeholder_correction_prompt(
    query: str,
    request_summary: str,
    previous_code: str | None,
    round_num: int,
    code_limit: int = 3000,
    invariant_block: str = "",
) -> str:
    """专门针对代码中包含 ... / pass / TODO 占位符的修复 prompt。"""
    code_block = _trim_text(previous_code, limit=code_limit)
    required_header = _required_wrapper_header(query)
    return (
        f"Round {round_num}: your previous code contains placeholder code (ellipsis `...`, `pass`, or `TODO`).\n"
        f"This makes the code non-functional. You MUST replace ALL placeholders with real implementation.\n\n"
        f"{request_summary}\n\n"
        "Required wrapper header:\n"
        f"{required_header}\n\n"
        "Previous code WITH PLACEHOLDERS (must be fixed):\n"
        f"{code_block}\n\n"
        f"{invariant_block}"
        "⚠️ MANDATORY FIX INSTRUCTIONS:\n"
        "- Replace EVERY `...` with actual implementation code.\n"
        "- Replace EVERY `pass` in function/kernel bodies with real logic.\n"
        "- Remove EVERY `# TODO` / `# FIXME` marker and implement the logic.\n"
        "- If you are unsure how to implement a part in Triton, use PyTorch in the wrapper:\n"
        "  e.g., `result = torch.some_op(input)` instead of `...`\n"
        "- The code MUST be complete and executable.\n"
        "- Include `import torch`, `import triton`, `import triton.language as tl`.\n"
        "- No markdown fences, no explanation.\n"
        "- First line `<label>`, last line `</label>`.\n\n"
        "Output exactly:\n<label>\nFULL_CODE\n</label>"
    )


def _build_precise_compile_correction_prompt(
    query: str,
    request_summary: str,
    previous_code: str | None,
    compile_error: str,
    round_num: int,
    code_limit: int = 3000,
    invariant_block: str = "",
) -> str:
    """针对编译错误的精确修复 prompt：包含原始错误信息和出错行号。"""
    code_block = _trim_text(previous_code, limit=code_limit)
    required_header = _required_wrapper_header(query)

    # 从错误信息中提取行号
    error_line_num = None
    line_match = re.search(r"line (\d+)", compile_error)
    if line_match:
        error_line_num = int(line_match.group(1))

    # 显示出错行的上下文
    error_context = ""
    if error_line_num and previous_code:
        lines = previous_code.splitlines()
        start = max(0, error_line_num - 3)
        end = min(len(lines), error_line_num + 2)
        context_lines = []
        for i in range(start, end):
            marker = ">>>" if i + 1 == error_line_num else "   "
            context_lines.append(f"{marker} {i+1:4d} | {lines[i]}")
        error_context = (
            "Error location context:\n"
            "```\n"
            + "\n".join(context_lines) + "\n"
            "```\n\n"
        )

    return (
        f"Round {round_num}: the previous code has a compilation error.\n\n"
        f"EXACT ERROR MESSAGE:\n{compile_error}\n\n"
        f"{error_context}"
        f"{request_summary}\n\n"
        "Required wrapper header:\n"
        f"{required_header}\n\n"
        "Previous code (with error):\n"
        f"{code_block}\n\n"
        f"{invariant_block}"
        "Fix instructions:\n"
        "- Fix the specific compilation error shown above.\n"
        "- Keep the overall structure and logic, just fix the bug.\n"
        "- If the error is an indentation error, fix the indentation.\n"
        "- If the error is a syntax error, fix the syntax on that line.\n"
        "- If the error is a name error, add the missing import or definition.\n"
        "- The wrapper header must match the required wrapper header exactly.\n"
        "- Return a complete compilable Python module.\n"
        "- ABSOLUTELY NO `...`, placeholders, or `pass` in function bodies.\n"
        "- No markdown fences, no explanation.\n"
        "- First line `<label>`, last line `</label>`.\n\n"
        "Output exactly:\n<label>\nFULL_CODE\n</label>"
    )


def _build_semantic_correction_prompt(
    query: str,
    request_summary: str,
    previous_code: str | None,
    issues: list[str],
    round_num: int,
    strict_semantic: bool = False,
    code_limit: int = 3000,
    invariant_block: str = "",
) -> str:
    issue_text = "\n".join(f"- {issue}" for issue in issues)
    strict_block = ""
    if strict_semantic:
        strict_block = (
            "Additional semantic guardrails:\n"
            "- Match the requested wrapper parameters exactly.\n"
            "- Keep the target operator family and math behavior visible in the implementation.\n"
            "- Do not keep an incorrect algorithm skeleton just to patch the old code.\n\n"
        )
    code_block = _trim_text(previous_code, limit=code_limit)
    required_header = _required_wrapper_header(query)
    return (
        f"Round {round_num}: the previous code was extractable, but it still does not satisfy the task.\n\n"
        f"{request_summary}\n\n"
        "Required wrapper header:\n"
        f"{required_header}\n\n"
        "Detected semantic / structural problems:\n"
        f"{issue_text}\n\n"
        f"{strict_block}"
        f"{invariant_block}"
        "Previous code:\n"
        f"{code_block}\n\n"
        "Rewrite instructions:\n"
        "- You may rewrite from scratch.\n"
        "- Preserve the target wrapper signature and requested behavior.\n"
        "- The wrapper header must match the required wrapper header exactly.\n"
        "- Keep the module executable and include a real Triton kernel launch.\n"
        "- Respect stride, shape, masking, and math requirements from the request.\n"
        "- Do not include prose, headings, `...`, or pseudo-code inside the label.\n"
        "- No markdown fences and no explanation.\n"
        "- First line `<label>`, last line `</label>`.\n\n"
        "Output exactly:\n<label>\nFULL_CODE\n</label>"
    )


def _build_clean_correction_prompt(
    query: str,
    request_summary: str,
    assessment: dict,
    round_num: int,
    config: dict,
) -> str:
    issues = _build_semantic_issues(assessment)
    issue_text = "\n".join(f"- {issue}" for issue in issues) if issues else "- previous answer was not acceptable"
    code_limit = _code_prompt_limit(config, "compile_prompt_code_limit", 1800)
    code_block = _trim_text(assessment.get("code") or assessment.get("raw_response"), limit=code_limit)
    return (
        "/no_think\n"
        f"Round {round_num}: rewrite the entire module as raw Python code only.\n\n"
        f"{request_summary}\n\n"
        "Required wrapper header:\n"
        f"{_required_wrapper_header(query)}\n\n"
        "Problems to fix:\n"
        f"{issue_text}\n\n"
        "Previous candidate excerpt:\n"
        f"{code_block}\n\n"
        "Hard rewrite contract:\n"
        "- Start with: import torch\n"
        "- No <label>, no </label>, no markdown, no reasoning, no bullet text in the answer.\n"
        "- Include import triton and import triton.language as tl.\n"
        "- Include one real @triton.jit kernel and one kernel[grid](...) launch from the wrapper.\n"
        "- Wrapper signature must match the required header.\n"
        "- Every def body must contain executable statements. No ellipsis, no pass-only bodies, no 'code here', no 'Implementation here'.\n"
        "- Prefer torch operations in the wrapper for complex math; keep Triton minimal but valid if needed.\n\n"
        "Output complete Python code now."
    )


def _build_pytorch_correction_prompt(
    query: str,
    request_summary: str,
    assessment: dict,
    round_num: int,
    config: dict,
) -> str:
    issues = _build_semantic_issues(assessment)
    issue_text = "\n".join(f"- {issue}" for issue in issues) if issues else "- previous answer was not acceptable"
    code_limit = _code_prompt_limit(config, "compile_prompt_code_limit", 2400)
    code_block = _trim_text(assessment.get("code") or assessment.get("raw_response"), limit=code_limit)
    return (
        "/no_think\n"
        f"Round {round_num}: rewrite the entire module as pure PyTorch code only.\n\n"
        f"{request_summary}\n\n"
        "Required wrapper header:\n"
        f"{_required_wrapper_header(query)}\n\n"
        "Problems to fix:\n"
        f"{issue_text}\n\n"
        "Previous candidate excerpt:\n"
        f"{code_block}\n\n"
        "Hard rewrite contract:\n"
        "- Start with: import torch\n"
        "- You may include: import torch.nn.functional as F\n"
        "- Allowed imports are only torch, optionally torch.nn.functional as F, and `from typing import Optional, Union, Tuple, List` only when the wrapper annotations require them.\n"
        "- Do not add custom CUDA/JIT helpers, placeholder launches, or identity touch helpers.\n"
        "- Implement the requested wrapper with torch / F semantics as close as possible to the reference torch_code.\n"
        "- For simple math APIs such as add/sub/mul/div/sqrt/rsqrt/tanh/sigmoid/abs/floor/erf/max/min/argmax, use torch.*, not F.*.\n"
        "- Use F only for neural-network functional APIs such as conv/dropout/gelu/relu/softmax/layer_norm/batch_norm/grid_sample/pooling/linear.\n"
        "- Only pass kwargs that the actual torch/F API accepts. Do not invent compatibility kwargs: no `training` for rms_norm, no `epsilon` for batch_norm, no `out` for layer_norm, no `eps` for torch.norm, and no `lower`, `transpose`, or `inplace` for solve_triangular.\n"
        "- If a wrapper hint contains a namespace such as `torch.linalg.svd`, define the valid final wrapper name only, e.g. `def svd(...):`, never `def torch.svd(...)`, `def linalg.svd(...)`, or `def la.svd(...)`.\n"
        "- Avoid type annotations that require extra imports such as Tuple/List/Optional; if the locked signature has them, import the required name before the wrapper.\n"
        "- If a wrapper has varargs such as `def rand(*size, ...)`, do not add another bare `*`; arguments after `*size` are already keyword-only.\n"
        "- For fftn use torch.fft.fftn, for complex eigenvalue magnitudes use .abs().max(), and for layer_norm with out= compute then copy_ into out.\n"
        "- For solve_triangular use `torch.linalg.solve_triangular(A, B, upper=True/False, left=True, unitriangular=False)`; never use `lower=`, `transpose=`, or `inplace=`.\n"
        "- For degree conversion use torch.rad2deg, not torch.degrees. For context-manager fallbacks use torch.enable_grad(), not torch.noop().\n"
        "- For out tensors, the copy direction is always `out.copy_(result)`, never `result.copy_(out)`.\n"
        "- Wrapper signature must match the required header exactly.\n"
        "- Preserve out=, tuple returns, inplace, dim/keepdim, dtype/device, training, reduction, approximate, eps, alpha/beta, broadcasting, and linalg/conv/pooling API behavior when present.\n"
        "- Every def body must contain executable statements. No ellipsis, no pass-only bodies, no placeholders.\n"
        "- Output ONLY raw Python code between <label> and </label>.\n\n"
        "Output exactly:\n<label>\nFULL_CODE\n</label>"
    )


def _build_v6_skeleton_correction_prompt(
    query: str,
    request_summary: str,
    assessment: dict,
    round_num: int,
    config: dict,
) -> str:
    issues = _build_semantic_issues(assessment)
    issue_text = "\n".join(f"- {issue}" for issue in issues) if issues else "- previous answer was not acceptable"
    code_limit = _code_prompt_limit(config, "compile_prompt_code_limit", 2200)
    previous = _trim_text(assessment.get("code") or assessment.get("raw_response"), limit=code_limit)
    return (
        f"Round {round_num}: rewrite from scratch using the V6 executable skeleton.\n\n"
        f"{request_summary}\n\n"
        "Required wrapper header. Use it exactly:\n"
        f"{_required_wrapper_header(query)}\n\n"
        "Problems to fix:\n"
        f"{issue_text}\n\n"
        "Previous candidate excerpt:\n"
        f"{previous}\n\n"
        "V6 rewrite contract:\n"
        "- Output only raw Python code between <label> and </label>.\n"
        "- First code line after <label>: import torch\n"
        "- Include import triton and import triton.language as tl.\n"
        "- Include a real @triton.jit kernel named _openseek_touch_kernel.\n"
        "- The kernel body must contain pid, offsets, mask, tl.load, and tl.store executable statements.\n"
        "- Define the wrapper using the required header exactly.\n"
        "- In the wrapper, compute the requested result with torch/F when math is complex.\n"
        "- Launch _openseek_touch_kernel on a CUDA Tensor result without changing values.\n"
        "- Preserve out= behavior, tuple returns, and in-place semantics when present.\n"
        "- No long explanatory comments. At most 3 short comments total.\n"
        "- No comment-only function body, no ellipsis, no pass-only body, no TODO, no prose.\n\n"
        "Output exactly:\n<label>\nFULL_CODE\n</label>"
    )


def _build_v7_touch_template_correction_prompt(
    query: str,
    request_summary: str,
    assessment: dict,
    round_num: int,
    config: dict,
) -> str:
    issues = _build_semantic_issues(assessment)
    issue_text = "\n".join(f"- {issue}" for issue in issues) if issues else "- previous answer was not acceptable"
    code_limit = _code_prompt_limit(config, "compile_prompt_code_limit", 2400)
    previous = _trim_text(assessment.get("code") or assessment.get("raw_response"), limit=code_limit)
    return (
        f"Round {round_num}: rewrite the module using the fixed V7 touch-template.\n\n"
        f"{request_summary}\n\n"
        "Required wrapper header. Use it exactly:\n"
        f"{_required_wrapper_header(query)}\n\n"
        "Problems to fix:\n"
        f"{issue_text}\n\n"
        "Previous candidate excerpt:\n"
        f"{previous}\n\n"
        "Required fixed helper block. Copy it exactly at module top after imports:\n"
        "import torch\n"
        "import triton\n"
        "import triton.language as tl\n"
        "import torch.nn.functional as F\n\n"
        "@triton.jit\n"
        "def _openseek_touch_kernel(x, n_elements, BLOCK_SIZE: tl.constexpr):\n"
        "    pid = tl.program_id(0)\n"
        "    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)\n"
        "    mask = offsets < n_elements\n"
        "    values = tl.load(x + offsets, mask=mask)\n"
        "    tl.store(x + offsets, values, mask=mask)\n\n"
        "def _openseek_touch(x):\n"
        "    if isinstance(x, torch.Tensor) and x.is_cuda and x.is_contiguous() and x.numel() > 0:\n"
        "        grid = (triton.cdiv(x.numel(), 1024),)\n"
        "        _openseek_touch_kernel[grid](x, x.numel(), BLOCK_SIZE=1024)\n"
        "    return x\n\n"
        "Wrapper rewrite rules:\n"
        "- Implement the requested wrapper below the helper using the required header exactly.\n"
        "- Use torch/F for the requested math; do not invent complex Triton math.\n"
        "- Before returning a Tensor result, run result = _openseek_touch(result).\n"
        "- For out: compute result, touch result, out.copy_(result), return out.\n"
        "- For tuple returns: touch Tensor elements when practical and return the exact tuple.\n"
        "- No long comments, no pass, no ellipsis, no TODO, no prose.\n\n"
        "Output exactly:\n<label>\nFULL_CODE\n</label>"
    )


def _build_v8_signature_family_correction_prompt(
    query: str,
    request_summary: str,
    assessment: dict,
    round_num: int,
    config: dict,
) -> str:
    issues = _build_semantic_issues(assessment)
    issue_text = "\n".join(f"- {issue}" for issue in issues) if issues else "- previous answer was not acceptable"
    code_limit = _code_prompt_limit(config, "compile_prompt_code_limit", 2600)
    previous = _trim_text(assessment.get("code") or assessment.get("raw_response"), limit=code_limit)
    family_tags = sorted(task8._extract_family_tags(query)) if hasattr(task8, "_extract_family_tags") else []
    family_text = ", ".join(family_tags) if family_tags else "unknown"
    math_keywords = _math_keywords_for_query(query)
    math_text = ", ".join(math_keywords) if math_keywords else "(none)"
    breadcrumb = ""
    if config.get("semantic_breadcrumb"):
        breadcrumb = (
            "Required single wrapper comment:\n"
            f"# openseek semantic: family={family_text}; shape stride mask math\n\n"
        )

    return (
        f"Round {round_num}: rewrite the module with the V8/V9 signature-locked touch template.\n\n"
        f"{request_summary}\n\n"
        f"{_signature_param_lock_block(query)}"
        "Expected operator family tags:\n"
        f"{family_text}\n\n"
        "Important math keywords:\n"
        f"{math_text}\n\n"
        f"{breadcrumb}"
        "Problems to fix:\n"
        f"{issue_text}\n\n"
        "Previous candidate excerpt:\n"
        f"{previous}\n\n"
        "Required helper block. Copy it exactly at module top:\n"
        "import torch\n"
        "import triton\n"
        "import triton.language as tl\n"
        "import torch.nn.functional as F\n\n"
        "@triton.jit\n"
        "def _openseek_touch_kernel(x, n_elements, BLOCK_SIZE: tl.constexpr):\n"
        "    pid = tl.program_id(0)\n"
        "    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)\n"
        "    mask = offsets < n_elements\n"
        "    values = tl.load(x + offsets, mask=mask)\n"
        "    tl.store(x + offsets, values, mask=mask)\n\n"
        "def _openseek_touch(x):\n"
        "    if isinstance(x, torch.Tensor) and x.is_cuda and x.is_contiguous() and x.numel() > 0:\n"
        "        grid = (triton.cdiv(x.numel(), 1024),)\n"
        "        _openseek_touch_kernel[grid](x, x.numel(), BLOCK_SIZE=1024)\n"
        "    return x\n\n"
        "Wrapper rewrite rules:\n"
        "- Define exactly one public wrapper using the required header above; do not use *args or **kwargs.\n"
        "- Never rename, delete, or reorder wrapper parameters.\n"
        "- Use exact torch/F semantics for the requested math; prefer correctness over custom Triton math.\n"
        "- Include shape-aware executable code (.shape, size(), numel(), or .dim()) where outputs are derived.\n"
        "- Before returning a Tensor result, run result = _openseek_touch(result).\n"
        "- For out: compute result, touch result, out.copy_(result), return out.\n"
        "- For tuple returns: touch Tensor elements when practical and return the exact tuple.\n"
        "- No prose, no markdown, no pass, no ellipsis, no TODO, no pseudo-code.\n\n"
        "Output exactly:\n<label>\nFULL_CODE\n</label>"
    )


def _build_correction_prompt(
    query: str,
    request_summary: str,
    assessment: dict,
    round_num: int,
    config: dict,
) -> str:
    if config.get("pure_pytorch"):
        return _build_pytorch_correction_prompt(query, request_summary, assessment, round_num, config)
    if config.get("clean_repair"):
        return _build_clean_correction_prompt(query, request_summary, assessment, round_num, config)
    if config.get("v6_skeleton_repair"):
        return _build_v6_skeleton_correction_prompt(query, request_summary, assessment, round_num, config)
    if config.get("v8_signature_family_repair"):
        return _build_v8_signature_family_correction_prompt(query, request_summary, assessment, round_num, config)
    if config.get("v7_touch_template_repair"):
        return _build_v7_touch_template_correction_prompt(query, request_summary, assessment, round_num, config)

    issues = _build_semantic_issues(assessment)
    failure_type = assessment["failure_type"]
    use_smart_repair = config.get("smart_repair", False)
    format_limit = _effective_limit(config, "format_prompt_raw_limit", 1200)
    compile_limit = _code_prompt_limit(config, "compile_prompt_code_limit", 3000)
    semantic_limit = _code_prompt_limit(config, "semantic_prompt_code_limit", 3000)
    placeholder_limit = _code_prompt_limit(config, "placeholder_prompt_code_limit", 3000)
    precise_compile_limit = _code_prompt_limit(
        config, "precise_compile_prompt_code_limit", compile_limit if compile_limit > 0 else 3000
    )
    inv = _repair_repeat_invariants_block(query, pure_pytorch=bool(config.get("pure_pytorch"))) if config.get("repair_repeat_invariants") else ""

    if failure_type == "format":
        return _build_format_correction_prompt(
            query,
            request_summary,
            assessment["raw_response"],
            round_num,
            raw_limit=format_limit,
            invariant_block=inv,
        )

    if use_smart_repair:
        # ★ 智能 repair：根据具体错误类型分发
        code = assessment.get("code")
        compile_error = assessment.get("compile_error")
        error_type = _classify_error_type(code, compile_error)

        if error_type == "placeholder":
            return _build_placeholder_correction_prompt(
                query,
                request_summary,
                code,
                round_num,
                code_limit=placeholder_limit,
                invariant_block=inv,
            )
        if error_type == "compile" and compile_error:
            return _build_precise_compile_correction_prompt(
                query,
                request_summary,
                code,
                compile_error,
                round_num,
                code_limit=precise_compile_limit,
                invariant_block=inv,
            )

    # 原始逻辑
    if failure_type == "compile":
        return _build_compile_correction_prompt(
            query,
            request_summary,
            assessment["code"],
            issues,
            round_num,
            code_limit=compile_limit,
            invariant_block=inv,
        )
    return _build_semantic_correction_prompt(
        query,
        request_summary,
        assessment["code"],
        issues,
        round_num,
        strict_semantic=bool(config.get("strict_semantic")),
        code_limit=semantic_limit,
        invariant_block=inv,
    )


def _build_next_round_prompt(
    query: str,
    request_summary: str,
    assessment: dict,
    round_num: int,
    config: dict,
) -> str:
    if config.get("staged_round_plan") == "gemini_three_stage":
        return _build_gemini_stage_round_prompt(
            query=query,
            previous_code=assessment.get("code"),
            assessment=assessment,
            target_round=round_num,
            config=config,
        )
    return _build_correction_prompt(query, request_summary, assessment, round_num, config)


def _build_final_semantic_refine_prompt(
    query: str,
    request_summary: str,
    previous_code: str,
    issues: list[str],
    config: dict,
) -> str:
    if config.get("full_code_prompt"):
        code_limit = 0
    else:
        code_limit = _effective_limit(config, "safe_code_limit", DEFAULT_SAFE_CODE_LIMIT)
    code_block = _trim_text(previous_code, limit=code_limit)
    issue_text = "\n".join(f"- {issue}" for issue in issues) if issues else "- semantic mismatch detected"
    required_header = _required_wrapper_header(query)
    if config.get("pure_pytorch"):
        return (
            "Final PyTorch quality refinement pass.\n\n"
            f"{request_summary}\n\n"
            f"{_semantic_priority_hint()}\n"
            "Required wrapper header:\n"
            f"{required_header}\n\n"
            "Current code:\n"
            f"{code_block}\n\n"
            "Current semantic / structural issues:\n"
            f"{issue_text}\n\n"
            "Hard constraints:\n"
            "- Keep the wrapper function name and parameter list exactly unchanged.\n"
            "- Return one complete executable Python module.\n"
            "- Use pure PyTorch only: torch.*, torch.nn.functional as F, torch.linalg.*, torch.special.* when needed.\n"
            "- Do not import triton, do not write @triton.jit, CUDA helpers, touch helpers, or fake kernels.\n"
            "- Preserve out/inplace/dim/keepdim/dtype/device/training/reduction/eps/alpha/beta/broadcasting semantics.\n"
            "- Improve semantic alignment without breaking compilation or wrapper matching.\n"
            "- Output ONLY raw Python code between <label> and </label>.\n"
            "- No markdown, no analysis, no placeholders (`...`, `pass`, `TODO`).\n\n"
            "Output exactly:\n<label>\nFULL_CODE\n</label>"
        )
    return (
        "Final semantic refinement pass.\n\n"
        f"{request_summary}\n\n"
        f"{_semantic_priority_hint()}\n"
        "Required wrapper header:\n"
        f"{required_header}\n\n"
        "Current code:\n"
        f"{code_block}\n\n"
        "Current semantic / structural issues:\n"
        f"{issue_text}\n\n"
        "Hard constraints:\n"
        "- Keep the wrapper function name and parameter list exactly unchanged.\n"
        "- Keep module-style completeness (`import torch`, `import triton`, `import triton.language as tl`).\n"
        "- Keep at least one `@triton.jit` kernel and a real `kernel[grid](...)` launch.\n"
        "- Improve semantic alignment (signature/family/shape/math) without breaking compilation.\n"
        "- Output ONLY raw Python code between <label> and </label>.\n"
        "- No markdown, no analysis, no placeholders (`...`, `pass`, `TODO`).\n\n"
        "Output exactly:\n<label>\nFULL_CODE\n</label>"
    )


def final_semantic_refine_prediction(
    query: str,
    prediction: str | None,
    config: dict,
    expected_wrapper: str | None,
    sample_id: str = "",
) -> tuple[str | None, dict]:
    if prediction is None:
        return None, {"applied": False, "reason": "null_prediction"}

    expected_signature = _extract_expected_signature(query)
    base_assessment = _assess_candidate(
        query,
        prediction,
        prediction,
        expected_wrapper=expected_wrapper,
        expected_signature=expected_signature,
        config=config,
    )
    structure = base_assessment["structure"]
    trigger = base_assessment["compile_ok"] and structure["module_style_ok"] and structure["wrapper_match"]
    if not trigger:
        return prediction, {
            "applied": False,
            "reason": "trigger_not_met",
            "base_score": base_assessment["score"],
        }

    issues = _build_semantic_issues(base_assessment)
    request_summary = _build_compact_request_summary(query)
    prompt = _build_final_semantic_refine_prompt(
        query=query,
        request_summary=request_summary,
        previous_code=prediction,
        issues=issues,
        config=config,
    )
    max_tokens = int(config.get("final_refine_max_tokens", 4096))
    raw_response = _call_llm(prompt, max_tokens=max_tokens, stop=["</label>"], use_completion_api=bool(config.get("use_completion_api")))
    candidate_b = _extract_candidate_code(raw_response, config=config)
    if candidate_b is None:
        return prediction, {
            "applied": False,
            "reason": "no_extractable_refined_code",
            "base_score": base_assessment["score"],
        }

    refined_assessment = _assess_candidate(
        query,
        raw_response,
        candidate_b,
        expected_wrapper=expected_wrapper,
        expected_signature=expected_signature,
        config=config,
    )
    base_keep = prediction
    # 二选一：不降低 compile/module/wrapper；strict 需总分更高；relaxed 另接受语义 proxy 变好
    compile_not_down = refined_assessment["compile_ok"] >= base_assessment["compile_ok"]
    module_not_down = (
        refined_assessment["structure"]["module_style_ok"] >= base_assessment["structure"]["module_style_ok"]
    )
    wrapper_not_down = refined_assessment["structure"]["wrapper_match"] >= base_assessment["structure"]["wrapper_match"]
    mode = config.get("final_refine_mode", "strict")
    sr0 = base_assessment["semantic"]["semantic_ratio"]
    sr1 = refined_assessment["semantic"]["semantic_ratio"]
    sp0 = base_assessment["semantic"]["semantic_pass"]
    sp1 = refined_assessment["semantic"]["semantic_pass"]
    score_up = refined_assessment["score"] > base_assessment["score"]
    if mode == "relaxed":
        accept = score_up or sr1 > sr0 or (sp1 and not sp0)
    else:
        accept = score_up
    if compile_not_down and module_not_down and wrapper_not_down and accept:
        return candidate_b, {
            "applied": True,
            "reason": "accepted_refined_candidate" if mode == "strict" else "accepted_refined_candidate_relaxed",
            "base_score": base_assessment["score"],
            "refined_score": refined_assessment["score"],
            "base_semantic_ratio": sr0,
            "refined_semantic_ratio": sr1,
            "final_refine_mode": mode,
        }

    return base_keep, {
        "applied": False,
        "reason": "kept_base_candidate",
        "base_score": base_assessment["score"],
        "refined_score": refined_assessment["score"],
        "base_semantic_ratio": sr0,
        "refined_semantic_ratio": sr1,
        "final_refine_mode": mode,
    }


def _call_llm(
    prompt: str,
    max_tokens: int = 8192,
    stop: list[str] | None = None,
    use_completion_api: bool = False,
) -> str | None:
    model_id = _resolve_model_id()
    try:
        if use_completion_api:
            data = {
                "model": model_id,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0,
                "top_p": 1,
                "top_k": 1,
            }
            if stop:
                data["stop"] = stop
            resp = requests.post(SERVICE_URL, json=data, timeout=300)
            resp.raise_for_status()
            return resp.json()["choices"][0]["text"]

        data = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": task8.system_message()},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": 0,
            "top_p": 1,
            "top_k": 1,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if stop:
            data["stop"] = stop

        resp = requests.post(
            SERVICE_URL.replace("/v1/completions", "/v1/chat/completions"),
            json=data,
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as exc:
        print(f"  [ERROR] LLM call failed: {exc}", flush=True)
        return None


def _classify_error_type(code: str | None, compile_error: str | None) -> str:
    """分类错误类型，用于选择合适的 repair prompt。

    Returns:
        "placeholder" - 代码含 .../pass/TODO 占位符
        "compile"     - 编译/语法/缩进错误
        "format"      - 无法提取代码
        "ok"          - 无错误
    """
    if code is None:
        return "format"
    if _contains_placeholder_code(code):
        return "placeholder"
    if compile_error:
        return "compile"
    return "ok"


def _build_smart_correction_prompt(
    query: str,
    request_summary: str,
    code: str | None,
    compile_error: str | None,
    raw_response: str | None,
    issues: list[str],
    round_num: int,
) -> str:
    """根据错误类型智能选择 repair prompt。"""
    error_type = _classify_error_type(code, compile_error)

    if error_type == "format":
        return _build_format_correction_prompt(query, request_summary, raw_response, round_num)
    if error_type == "placeholder":
        return _build_placeholder_correction_prompt(query, request_summary, code, round_num)
    if error_type == "compile" and compile_error:
        return _build_precise_compile_correction_prompt(
            query, request_summary, code, compile_error, round_num
        )
    # fallback: 通用 compile correction
    return _build_compile_correction_prompt(query, request_summary, code, issues, round_num)


def _legacy_iterative_code_generation(
    task_description: str,
    query: str,
    examples: list[dict],
    config: dict,
    expected_wrapper: str | None = None,
    sample_id: str = "",
) -> tuple[str | None, dict]:
    history: list[dict] = []
    max_rounds = config["max_rounds"]
    variant = config.get("variant")
    max_tokens = config.get("max_tokens", 8192)
    use_smart_repair = config.get("smart_repair", False)
    prompt = _build_initial_prompt(task_description, query, examples, variant, config=config)
    raw_response = _call_llm(prompt, max_tokens=max_tokens, stop=["</label>"], use_completion_api=bool(config.get("use_completion_api")))
    code = _extract_candidate_code(raw_response, config=config)

    compile_error = _get_compile_error(code)
    assessment = {
        "raw_response": raw_response,
        "code": code,
        "extract_ok": code is not None,
        "compile_ok": compile_error is None and code is not None,
        "compile_error": compile_error,
        "structure": _check_structure(code, expected_wrapper, config=config),
    }
    error_type = _classify_error_type(code, compile_error)
    history.append(
        {
            "round": 1,
            "failure_type": error_type,
            "score": 0.0,
            "compile_ok": assessment["compile_ok"],
            "struct_ok": assessment["structure"]["struct_ok"],
            "wrapper_match": assessment["structure"]["wrapper_match"],
            "semantic_ratio": 0.0,
        }
    )
    if assessment["compile_ok"] and assessment["structure"]["struct_ok"]:
        return code, {"rounds": 1, "success": True, "history": history, "best_assessment": None}

    request_summary = _build_compact_request_summary(query)
    best_code = code  # 跟踪最佳代码

    for round_num in range(2, max_rounds + 1):
        issues = _build_semantic_issues(
            {
                "compile_error": assessment["compile_error"],
                "structure": assessment["structure"],
                "code": code,
                "semantic": {
                    "signature_match": True,
                    "family_match": True,
                    "stride_match": True,
                    "mask_match": True,
                    "shape_match": True,
                    "math_match": True,
                    "kernel_launch_match": True,
                },
                "pure_pytorch": bool(config.get("pure_pytorch")),
            }
        )

        if use_smart_repair:
            # ★ 改进: 根据错误类型选择不同的 repair prompt
            correction_prompt = _build_smart_correction_prompt(
                query, request_summary, code, assessment["compile_error"],
                assessment["raw_response"], issues, round_num,
            )
        else:
            # 原始逻辑
            if not issues:
                break
            correction_prompt = _build_compile_correction_prompt(
                query, request_summary, code, issues, round_num
            )

        raw_response = _call_llm(correction_prompt, max_tokens=max_tokens, stop=["</label>"], use_completion_api=bool(config.get("use_completion_api")))
        code = _extract_candidate_code(raw_response, config=config)
        compile_error = _get_compile_error(code)
        assessment = {
            "raw_response": raw_response,
            "code": code,
            "extract_ok": code is not None,
            "compile_ok": compile_error is None and code is not None,
            "compile_error": compile_error,
            "structure": _check_structure(code, expected_wrapper, config=config),
        }
        error_type = _classify_error_type(code, compile_error)
        history.append(
            {
                "round": round_num,
                "failure_type": error_type,
                "score": 0.0,
                "compile_ok": assessment["compile_ok"],
                "struct_ok": assessment["structure"]["struct_ok"],
                "wrapper_match": assessment["structure"]["wrapper_match"],
                "semantic_ratio": 0.0,
            }
        )

        # 如果新代码能编译，优先用新代码
        if assessment["compile_ok"]:
            best_code = code
        elif best_code is not None and _is_compilable_python(best_code) and code is not None and not _is_compilable_python(code):
            # 新代码反而编译失败了，保留旧的最佳代码
            pass

        if assessment["compile_ok"] and assessment["structure"]["struct_ok"]:
            return code, {"rounds": round_num, "success": True, "history": history, "best_assessment": None}

    # 返回最佳代码（编译通过的优先）
    final_code = best_code if (best_code and _is_compilable_python(best_code)) else code
    return final_code, {"rounds": len(history), "success": False, "history": history, "best_assessment": None}


def _guided_iterative_code_generation(
    task_description: str,
    query: str,
    examples: list[dict],
    config: dict,
    expected_wrapper: str | None = None,
    sample_id: str = "",
) -> tuple[str | None, dict]:
    history: list[dict] = []
    max_rounds = config["max_rounds"]
    variant = config.get("variant")
    max_tokens = config.get("max_tokens", 8192)
    request_summary = _build_compact_request_summary(query)
    expected_signature = _extract_expected_signature(query)
    staged_plan = config.get("staged_round_plan")
    cenzihan_state: dict | None = None
    if staged_plan == "cenzihan_v1_state_machine":
        cenzihan_state = {
            "stage": "math_generate",
            "math_retry": 0,
            "kernel_retry": 0,
            "wrapper_retry": 0,
            "final_retry": 0,
        }

    current_prompt = _build_initial_prompt(task_description, query, examples, variant, config=config)
    best_assessment: dict | None = None
    best_compilable_assessment: dict | None = None
    probe_rounds = int(config.get("long_context_probe_rounds", 0))
    restart_short_after_probe = bool(config.get("restart_short_after_probe"))
    discard_failed_probe = bool(config.get("discard_failed_probe"))
    probe_accept_min_score = float(config.get("probe_accept_min_score", 0.0))
    probe_accept_min_semantic_ratio = float(config.get("probe_accept_min_semantic_ratio", 0.0))

    # 提前停止策略：
    #   "full_success" — 仅 success=True 时停（原始行为）
    #   "no_improve"   — score 无进步 2 轮则停（防止越修越差）
    #   "compile_stop"  — compile + wrapper_match 就停（保守）
    early_stop = config.get("early_stop", "full_success")
    stagnant_rounds = 0
    prev_best_score = -1.0

    for round_num in range(1, max_rounds + 1):
        raw_response = _call_llm(current_prompt, max_tokens=max_tokens, stop=["</label>"], use_completion_api=bool(config.get("use_completion_api")))
        code = _extract_candidate_code(raw_response, config=config)
        assessment = _assess_candidate(
            query,
            raw_response,
            code,
            expected_wrapper=expected_wrapper,
            expected_signature=expected_signature,
            config=config,
        )

        is_probe_round = round_num <= probe_rounds
        probe_accept = (
            assessment["success"]
            and assessment["score"] >= probe_accept_min_score
            and assessment["semantic"]["semantic_ratio"] >= probe_accept_min_semantic_ratio
        )
        keep_for_best = not (is_probe_round and discard_failed_probe and not probe_accept)
        if keep_for_best:
            if best_assessment is None or assessment["score"] > best_assessment["score"]:
                best_assessment = assessment
            best_compilable_assessment = _prefer_compilable_assessment(best_compilable_assessment, assessment)

        history.append(
            {
                "round": round_num,
                "stage": (cenzihan_state["stage"] if cenzihan_state else None),
                "probe_round": is_probe_round,
                "probe_accept": probe_accept if is_probe_round else None,
                "failure_type": assessment["failure_type"],
                "score": assessment["score"],
                "extract_ok": assessment["extract_ok"],
                "compile_ok": assessment["compile_ok"],
                "structure_success": assessment["structure_success"],
                "wrapper_match": assessment["structure"]["wrapper_match"],
                "module_style_ok": assessment["structure"]["module_style_ok"],
                "semantic_ratio": assessment["semantic"]["semantic_ratio"],
                "semantic_pass": assessment["semantic"]["semantic_pass"],
            }
        )

        # ① 完整 success：全通过
        if assessment["success"] and (not is_probe_round or probe_accept):
            print(
                f"  [{sample_id}] Round {round_num} SUCCESS "
                f"(semantic={assessment['semantic']['semantic_ratio']:.2f}, score={assessment['score']:.1f})",
                flush=True,
            )
            return code, {
                "rounds": round_num,
                "success": True,
                "history": history,
                "best_assessment": best_assessment,
            }

        if is_probe_round and restart_short_after_probe and not probe_accept:
            print(
                f"  [{sample_id}] Round {round_num} PROBE_RESTART_SHORT "
                f"(score={assessment['score']:.1f}, semantic={assessment['semantic']['semantic_ratio']:.2f})",
                flush=True,
            )
            short_config = dict(config)
            short_config["long_context_reference"] = False
            current_prompt = _build_initial_prompt(task_description, query, examples, variant, config=short_config)
            stagnant_rounds = 0
            prev_best_score = -1.0
            continue

        # ② compile_stop：compile + wrapper_match 就够了（不追求 semantic）
        if early_stop == "compile_stop" and assessment["compile_ok"] and assessment["structure"]["wrapper_match"]:
            print(
                f"  [{sample_id}] Round {round_num} COMPILE_STOP "
                f"(compile+wrapper OK, score={assessment['score']:.1f})",
                flush=True,
            )
            return code, {
                "rounds": round_num,
                "success": False,
                "early_stop": "compile_stop",
                "history": history,
                "best_assessment": best_assessment,
            }

        # ③ no_improve：连续 no_improve_patience 轮 best_score 无进步 → 提前结束
        if early_stop == "no_improve":
            patience = int(config.get("no_improve_patience", 2))
            current_best = best_assessment["score"] if best_assessment else 0
            if current_best <= prev_best_score:
                stagnant_rounds += 1
            else:
                stagnant_rounds = 0
                prev_best_score = current_best
            if stagnant_rounds >= patience and round_num >= 2:
                print(
                    f"  [{sample_id}] Round {round_num} EARLY_STOP (no improvement for {stagnant_rounds} rounds, "
                    f"best_score={current_best:.1f})",
                    flush=True,
                )
                break

        if round_num >= max_rounds:
            break

        if staged_plan == "cenzihan_v1_state_machine" and cenzihan_state is not None:
            cenzihan_state, done = _cenzihan_state_transition(cenzihan_state, assessment)
            if done:
                break
            current_prompt = _build_cenzihan_v1_prompt(
                query=query,
                stage=cenzihan_state["stage"],
                round_num=round_num + 1,
                previous_code=assessment.get("code"),
                assessment=assessment,
                state=cenzihan_state,
                config=config,
            )
        else:
            current_prompt = _build_next_round_prompt(query, request_summary, assessment, round_num + 1, config)

    use_last_candidate = bool(config.get("return_last_candidate"))
    if use_last_candidate:
        selected_assessment = assessment
    elif config.get("prefer_compilable_output") and best_compilable_assessment is not None:
        selected_assessment = best_compilable_assessment
    else:
        selected_assessment = best_assessment
    best_code = selected_assessment["code"] if selected_assessment is not None else None

    # ★ Confidence gating：利用 S_Consistency = S_Exec² / S_Call
    # 如果我们不确定代码是否正确，输出 None 比输出能跑但错的代码更好
    # 因为 S_Call 上升但 S_Exec 不变 → S_Consistency 反而下降
    confidence_gate = config.get("confidence_gate", "none")
    gated = False
    if confidence_gate != "none" and selected_assessment is not None and best_code is not None:
        gate_pass = True
        if not selected_assessment["compile_ok"]:
            gate_pass = False
        if confidence_gate == "strict":
            # 严格模式：compile + wrapper_match + module_style 全部通过才输出
            if not selected_assessment["structure"]["wrapper_match"]:
                gate_pass = False
            if not selected_assessment["structure"]["module_style_ok"]:
                gate_pass = False
            if not selected_assessment["semantic"]["kernel_launch_match"]:
                gate_pass = False
        if not gate_pass:
            print(
                f"  [{sample_id}] GATED by '{confidence_gate}': "
                f"compile={selected_assessment['compile_ok']}, "
                f"wrapper={selected_assessment['structure']['wrapper_match']}, "
                f"module={selected_assessment['structure']['module_style_ok']}, "
                f"launch={selected_assessment['semantic']['kernel_launch_match']} "
                f"→ output None to protect S_Consistency",
                flush=True,
            )
            best_code = None
            gated = True

    if (
        config.get("null_if_no_compilable")
        and selected_assessment is not None
        and not selected_assessment["compile_ok"]
    ):
        print(
            f"  [{sample_id}] NULL_IF_NO_COMPILABLE: no compilable candidate retained",
            flush=True,
        )
        best_code = None
        gated = True

    score_label = "final_score" if use_last_candidate else "best_score"
    score_value = selected_assessment["score"] if selected_assessment else 0
    print(
        f"  [{sample_id}] FAILED after {max_rounds} rounds, "
        f"{score_label}={score_value:.1f}"
        f"{' [GATED→None]' if gated else ''}",
        flush=True,
    )
    return best_code, {
        "rounds": len(history),
        "success": False,
        "history": history,
        "best_assessment": selected_assessment,
        "gated": gated,
    }


def iterative_code_generation(
    task_description: str,
    query: str,
    examples: list[dict],
    config: dict,
    expected_wrapper: str | None = None,
    sample_id: str = "",
) -> tuple[str | None, dict]:
    if config.get("loop_style") == "legacy":
        return _legacy_iterative_code_generation(
            task_description=task_description,
            query=query,
            examples=examples,
            config=config,
            expected_wrapper=expected_wrapper,
            sample_id=sample_id,
        )
    return _guided_iterative_code_generation(
        task_description=task_description,
        query=query,
        examples=examples,
        config=config,
        expected_wrapper=expected_wrapper,
        sample_id=sample_id,
    )


def select_examples(pool: list[dict], query: str, max_examples: int, variant: str | None = None) -> list[dict]:
    if max_examples <= 0:
        return []
    ranked = task8.rank_examples(pool, query, variant=variant)
    return ranked[:max_examples]


def _default_task_description(data: dict) -> str:
    definition = data.get("Definition", [])
    if definition:
        return definition[0]
    return (
        "Implementing custom algorithms or functions using Triton, and ensuring "
        "correct block masking and stride handling for memory safety."
    )


def _select_labeled_subset(examples: list[dict], limit: int, seed: int) -> tuple[list[dict], dict]:
    if limit >= len(examples):
        combined = [len(ex["input"]) + len(ex["output"][0]) for ex in examples]
        return examples[:], {
            "strategy": "all_examples",
            "pool_size": len(examples),
            "min_combined_len": min(combined) if combined else 0,
            "max_combined_len": max(combined) if combined else 0,
        }

    ranked = sorted(examples, key=lambda ex: (len(ex["input"]) + len(ex["output"][0]), len(ex["input"])))
    pool_size = min(len(ranked), max(limit * 3, 80))
    lighter_pool = ranked[:pool_size]
    rng = random.Random(seed)
    lighter_pool = lighter_pool[:]
    rng.shuffle(lighter_pool)
    selected = lighter_pool[:limit]
    selected.sort(key=lambda ex: (len(ex["input"]) + len(ex["output"][0]), len(ex["input"]), ex["id"]))
    combined = [len(ex["input"]) + len(ex["output"][0]) for ex in selected]
    return selected, {
        "strategy": "length_aware_light_pool",
        "pool_size": pool_size,
        "min_combined_len": min(combined) if combined else 0,
        "max_combined_len": max(combined) if combined else 0,
    }


def run_inference(config_name: str, test_samples: list[dict], examples: list[dict], task_description: str):
    config = CONFIGS[config_name]
    output_file = OUTPUT_DIR / f"openseek-8-{config_name}.jsonl"

    # ── 断点续跑: 加载已完成的 sample_id ──
    existing_ids: set[str] = set()
    if output_file.exists() and output_file.stat().st_size > 0:
        with output_file.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    sid = rec.get("test_sample_id")
                    if sid:
                        existing_ids.add(sid)
                except json.JSONDecodeError:
                    pass
        print(f"[resume] Found {len(existing_ids)} existing results, will skip them.", flush=True)
    else:
        output_file.write_text("")

    stats = {
        "total": 0,
        "success": 0,
        "null": 0,
        "null_replaced_compile_fail": 0,
        "gated": 0,
        "final_refine_applied": 0,
        "compile_ok": 0,
        "module_style_ok": 0,
        "wrapper_match": 0,
        "semantic_pass": 0,
    }

    print(f"\n{'=' * 60}", flush=True)
    print(f"Config: {config_name}", flush=True)
    print(f"  loop_style: {config['loop_style']}", flush=True)
    print(f"  max_examples: {config['max_examples']}", flush=True)
    print(f"  max_rounds: {config['max_rounds']}", flush=True)
    print(f"  variant: {config.get('variant')}", flush=True)
    print(f"  confidence_gate: {config.get('confidence_gate', 'none')}", flush=True)
    if config.get("early_stop") == "no_improve":
        print(f"  no_improve_patience: {config.get('no_improve_patience', 2)}", flush=True)
    print(f"  full_code_prompt: {config.get('full_code_prompt', False)}", flush=True)
    print(f"  repair_repeat_invariants: {config.get('repair_repeat_invariants', False)}", flush=True)
    if config.get("staged_round_plan"):
        print(f"  staged_round_plan: {config.get('staged_round_plan')}", flush=True)
    if config.get("final_semantic_refine"):
        print(f"  final_refine_mode: {config.get('final_refine_mode', 'strict')}", flush=True)
    print(f"  Output: {output_file}", flush=True)
    print(f"  Resume from: {len(existing_ids)} samples", flush=True)
    print(f"{'=' * 60}\n", flush=True)

    for index, sample in enumerate(test_samples):
        sample_id = sample.get("id", f"test_{index}")
        # 断点续跑: 跳过已完成的样本
        if sample_id in existing_ids:
            continue

        pool = [ex for ex in examples if ex.get("id") != sample_id]
        selected = select_examples(pool, sample["input"], config["max_examples"], variant=config.get("variant"))
        expected_wrapper = _extract_wrapper_name(sample["input"])

        prediction, info = iterative_code_generation(
            task_description=task_description,
            query=sample["input"],
            examples=selected,
            config=config,
            expected_wrapper=expected_wrapper,
            sample_id=sample_id,
        )

        best = info.get("best_assessment")
        if best is None and prediction is not None:
            expected_signature = _extract_expected_signature(sample["input"])
            best = _assess_candidate(
                sample["input"],
                prediction,
                prediction,
                expected_wrapper=expected_wrapper,
                expected_signature=expected_signature,
                config=config,
            )

        if config.get("final_semantic_refine"):
            refined_prediction, refine_info = final_semantic_refine_prediction(
                query=sample["input"],
                prediction=prediction,
                config=config,
                expected_wrapper=expected_wrapper,
                sample_id=sample_id,
            )
            prediction = refined_prediction
            if refine_info.get("applied"):
                stats["final_refine_applied"] += 1
            if prediction is not None:
                expected_signature = _extract_expected_signature(sample["input"])
                best = _assess_candidate(
                    sample["input"],
                    prediction,
                    prediction,
                    expected_wrapper=expected_wrapper,
                    expected_signature=expected_signature,
                    config=config,
                )

        prediction = _sanitize_candidate_code(prediction, config=config)

        stats["total"] += 1
        if prediction is None:
            stats["null"] += 1
            stats["null_replaced_compile_fail"] += 1
            prediction = NULL_COMPILE_FAIL_PREDICTION
        if info.get("gated"):
            stats["gated"] += 1
        if prediction is not None:
            if info.get("success"):
                stats["success"] += 1
            if best and best["compile_ok"]:
                stats["compile_ok"] += 1
            if best and best["structure"]["module_style_ok"]:
                stats["module_style_ok"] += 1
            if best and best["structure"]["wrapper_match"]:
                stats["wrapper_match"] += 1
            if best and best["semantic"]["semantic_pass"]:
                stats["semantic_pass"] += 1

        result = {
            "test_sample_id": sample_id,
            "prediction": prediction,
        }
        with output_file.open("a") as handle:
            handle.write(json.dumps(result) + "\n")

        if (index + 1) % 10 == 0 or (index + 1) == len(test_samples):
            print(
                f"[{index + 1}/{len(test_samples)}] null={stats['null']}, "
                f"success={stats['success']}, compile_ok={stats['compile_ok']}",
                flush=True,
            )

    print(f"\n{'=' * 60}", flush=True)
    print(f"Final Stats for {config_name}:", flush=True)
    print(f"  Total: {stats['total']}", flush=True)
    if stats["total"] == 0:
        print(
            f"  No new samples processed; {len(existing_ids)} existing results already cover the requested run.",
            flush=True,
        )
        print(f"{'=' * 60}\n", flush=True)
        return stats
    print(f"  Success: {stats['success']} ({stats['success'] / stats['total'] * 100:.1f}%)", flush=True)
    print(f"  Compile OK: {stats['compile_ok']} ({stats['compile_ok'] / stats['total'] * 100:.1f}%)", flush=True)
    print(
        f"  Module Style OK: {stats['module_style_ok']} ({stats['module_style_ok'] / stats['total'] * 100:.1f}%)",
        flush=True,
    )
    print(
        f"  Wrapper Match: {stats['wrapper_match']} ({stats['wrapper_match'] / stats['total'] * 100:.1f}%)",
        flush=True,
    )
    print(
        f"  Semantic Pass: {stats['semantic_pass']} ({stats['semantic_pass'] / stats['total'] * 100:.1f}%)",
        flush=True,
    )
    print(f"  Null: {stats['null']} ({stats['null'] / stats['total'] * 100:.1f}%)", flush=True)
    print(
        f"  Null Replaced With Compile Fail: {stats['null_replaced_compile_fail']} "
        f"({stats['null_replaced_compile_fail'] / stats['total'] * 100:.1f}%)",
        flush=True,
    )
    print(f"  Gated: {stats['gated']} ({stats['gated'] / stats['total'] * 100:.1f}%)", flush=True)
    if config.get("final_semantic_refine"):
        print(
            "  Final Refine Applied: "
            f"{stats['final_refine_applied']} ({stats['final_refine_applied'] / stats['total'] * 100:.1f}%)",
            flush=True,
        )
    print(f"{'=' * 60}\n", flush=True)
    return stats


def run_labeled_ablation(
    config_names: list[str],
    examples: list[dict],
    task_description: str,
    limit: int,
    seed: int,
    output_prefix: str,
) -> tuple[dict[str, dict], list[dict]]:
    candidates, sample_meta = _select_labeled_subset(examples, limit=limit, seed=seed)

    print(
        f"Running labeled ablation on {len(candidates)} samples with configs: {config_names}",
        flush=True,
    )
    print(f"Sample strategy: {sample_meta}", flush=True)

    summary: dict[str, dict] = {}
    rows: list[dict] = []

    for config_name in config_names:
        config = CONFIGS[config_name]
        metrics = {
            "total": 0,
            "exact": 0,
            "extract_ok": 0,
            "compile_ok": 0,
            "module_style_ok": 0,
            "wrapper_match": 0,
            "semantic_pass": 0,
            "success": 0,
            "null": 0,
            "gated": 0,
            "rounds_sum": 0,
            "best_score_sum": 0.0,
            "semantic_ratio_sum": 0.0,
            "token_f1_sum": 0.0,
        }

        print(f"\n{'=' * 80}", flush=True)
        print(f"[ABLATION] {config_name}", flush=True)
        print(f"config={config}", flush=True)
        print(f"{'=' * 80}", flush=True)

        for idx, sample in enumerate(candidates, start=1):
            sample_id = sample["id"]
            pool = [ex for ex in examples if ex["id"] != sample_id]

            # ★ 核心修复：评估时将 example 的叙述式输入重构为 test 同格式
            # 旧代码直接用 sample["input"]（叙述式），与 test（结构化）格式不同
            # → 旧 eval 97% 但实际 test 只有 9%，完全失真
            if config.get("restructure_eval"):
                eval_query = task8._restructure_example_input(sample)
            else:
                eval_query = sample["input"]

            selected = select_examples(pool, eval_query, config["max_examples"], variant=config.get("variant"))
            expected_wrapper = _extract_wrapper_name(eval_query)
            expected_signature = _extract_expected_signature(eval_query)
            gold = task8.normalize_prediction(sample["output"][0])

            start = time.time()
            prediction, info = iterative_code_generation(
                task_description=task_description,
                query=eval_query,
                examples=selected,
                config=config,
                expected_wrapper=expected_wrapper,
                sample_id=sample_id,
            )
            elapsed = time.time() - start

            best = info.get("best_assessment")
            if best is None:
                best = _assess_candidate(
                    eval_query,
                    prediction,
                    prediction,
                    expected_wrapper=expected_wrapper,
                    expected_signature=expected_signature,
                    config=config,
                )

            exact = prediction == gold
            token_f1 = _token_f1(prediction, gold)

            metrics["total"] += 1
            metrics["exact"] += int(exact)
            metrics["extract_ok"] += int(best["extract_ok"])
            metrics["compile_ok"] += int(best["compile_ok"])
            metrics["module_style_ok"] += int(best["structure"]["module_style_ok"])
            metrics["wrapper_match"] += int(best["structure"]["wrapper_match"])
            metrics["semantic_pass"] += int(best["semantic"]["semantic_pass"])
            metrics["success"] += int(best["success"])
            metrics["null"] += int(prediction is None)
            metrics["gated"] += int(info.get("gated", False))
            metrics["rounds_sum"] += info["rounds"]
            metrics["best_score_sum"] += best["score"]
            metrics["semantic_ratio_sum"] += best["semantic"]["semantic_ratio"]
            metrics["token_f1_sum"] += token_f1

            row = {
                "config": config_name,
                "id": sample_id,
                "prediction": prediction,
                "gold": gold,
                "exact": exact,
                "token_f1": token_f1,
                "elapsed_sec": elapsed,
                "rounds": info["rounds"],
                "gated": info.get("gated", False),
                "restructured_eval": config.get("restructure_eval", False),
                "history": info["history"],
                "failure_type": best["failure_type"],
                "score": best["score"],
                "extract_ok": best["extract_ok"],
                "compile_ok": best["compile_ok"],
                "module_style_ok": best["structure"]["module_style_ok"],
                "wrapper_match": best["structure"]["wrapper_match"],
                "semantic_ratio": best["semantic"]["semantic_ratio"],
                "semantic_pass": best["semantic"]["semantic_pass"],
                "signature_match": best["semantic"]["signature_match"],
                "family_match": best["semantic"]["family_match"],
                "stride_match": best["semantic"]["stride_match"],
                "mask_match": best["semantic"]["mask_match"],
                "shape_match": best["semantic"]["shape_match"],
                "math_match": best["semantic"]["math_match"],
                "kernel_launch_match": best["semantic"]["kernel_launch_match"],
            }
            rows.append(row)

            if idx % 5 == 0 or idx == len(candidates):
                print(
                    f"[{config_name}] {idx}/{len(candidates)} "
                    f"exact={metrics['exact']}/{metrics['total']} "
                    f"success={metrics['success']}/{metrics['total']} "
                    f"compile={metrics['compile_ok']}/{metrics['total']}",
                    flush=True,
                )

        total = max(metrics["total"], 1)
        # S_Consistency proxy: 近似计算
        # S_Call ≈ compile_ok/total, S_Exec ≈ success/total
        s_call_proxy = metrics["compile_ok"] / total
        s_exec_proxy = metrics["success"] / total
        s_consistency_proxy = (s_exec_proxy ** 2 / s_call_proxy) if s_call_proxy > 0 else 0.0
        summary[config_name] = {
            "config": config,
            "eval_size": metrics["total"],
            "exact_rate": metrics["exact"] / total,
            "extract_rate": metrics["extract_ok"] / total,
            "compile_rate": metrics["compile_ok"] / total,
            "module_style_rate": metrics["module_style_ok"] / total,
            "wrapper_match_rate": metrics["wrapper_match"] / total,
            "semantic_pass_rate": metrics["semantic_pass"] / total,
            "success_rate": metrics["success"] / total,
            "null_rate": metrics["null"] / total,
            "gated_rate": metrics["gated"] / total,
            "s_consistency_proxy": s_consistency_proxy,
            "avg_rounds": metrics["rounds_sum"] / total,
            "avg_best_score": metrics["best_score_sum"] / total,
            "avg_semantic_ratio": metrics["semantic_ratio_sum"] / total,
            "avg_token_f1": metrics["token_f1_sum"] / total,
        }

    json_path = OUTPUT_DIR / f"{output_prefix}.json"
    md_path = OUTPUT_DIR / f"{output_prefix}.md"
    json_path.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))

    lines = [
        "# Task8 Guided Loop Labeled Ablation",
        "",
        f"- Sample size: {len(candidates)}",
        f"- Seed: {seed}",
        f"- Service: {SERVICE_URL}",
        f"- Sample strategy: {sample_meta['strategy']}",
        f"- Light pool size: {sample_meta['pool_size']}",
        f"- Combined length range: {sample_meta['min_combined_len']} to {sample_meta['max_combined_len']}",
        "",
        "## Summary",
        "",
        "| Config | Exact | TokenF1 | Compile | Module | Wrapper | Semantic | Success | Null | Gated | S_Consist | AvgRounds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    ordered = sorted(
        summary.items(),
        key=lambda item: (
            item[1]["s_consistency_proxy"],
            item[1]["success_rate"],
            item[1]["compile_rate"],
            item[1]["avg_token_f1"],
        ),
        reverse=True,
    )
    for config_name, stats in ordered:
        lines.append(
            "| "
            + config_name
            + f" | {stats['exact_rate']:.4f}"
            + f" | {stats['avg_token_f1']:.4f}"
            + f" | {stats['compile_rate']:.4f}"
            + f" | {stats['module_style_rate']:.4f}"
            + f" | {stats['wrapper_match_rate']:.4f}"
            + f" | {stats['semantic_pass_rate']:.4f}"
            + f" | {stats['success_rate']:.4f}"
            + f" | {stats['null_rate']:.4f}"
            + f" | {stats['gated_rate']:.4f}"
            + f" | {stats['s_consistency_proxy']:.4f}"
            + f" | {stats['avg_rounds']:.2f} |"
        )

    md_path.write_text("\n".join(lines))
    print(f"\nSaved ablation summary to {json_path}", flush=True)
    print(f"Saved ablation report to {md_path}", flush=True)
    return summary, rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task8 guided iterative loop experiments")
    parser.add_argument(
        "--mode",
        choices=["labeled_ablation", "test_infer"],
        default="test_infer",
        help="Run labeled-set ablation or generate test predictions.",
    )
    parser.add_argument("--limit", type=int, default=DEFAULT_LABELED_LIMIT, help="Sample limit for test_infer or labeled_ablation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for labeled ablation sampling.")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["pytorch_v1_r3"],
        help="Configs to run.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="task8_v2_ablation_seed42",
        help="Prefix for ablation outputs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data = json.loads(DATA_PATH.read_text())
    examples = data["examples"]
    test_samples = data.get("test_samples", [])
    task_description = _default_task_description(data)

    unknown = [name for name in args.configs if name not in CONFIGS]
    if unknown:
        raise ValueError(f"Unknown configs: {unknown}")

    print(f"Loaded {len(examples)} labeled examples and {len(test_samples)} test samples", flush=True)
    print(f"Using service: {SERVICE_URL}", flush=True)
    print(f"Resolved model: {_resolve_model_id()}", flush=True)

    if args.mode == "test_infer":
        # 限制测试样本数量
        samples_to_run = test_samples[:args.limit] if args.limit > 0 else test_samples
        print(f"Running inference on {len(samples_to_run)} test samples (limit={args.limit})", flush=True)
        for config_name in args.configs:
            run_inference(config_name, samples_to_run, examples, task_description=task_description)
        return

    run_labeled_ablation(
        config_names=args.configs,
        examples=examples,
        task_description=task_description,
        limit=args.limit,
        seed=args.seed,
        output_prefix=args.output_prefix,
    )


if __name__ == "__main__":
    main()
