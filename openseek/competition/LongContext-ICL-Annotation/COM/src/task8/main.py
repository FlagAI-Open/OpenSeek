"""
main.py — Task 8 (kernel_generation) 预测入口

依赖：
- ``method.py``                     : Task 8 核心方法库（BM25 ICL + Prompt + 静态校验 + ReAct 修复 + AST 兜底）
- ``common.llm_client.annotate_nvidia``: 共享 vLLM/OpenAI 调用封装（method.py 内部已复用）
- ``common.paths``                  : 统一路径中心（模型/输出）

ICL 数据：``src/task8/normalized_data/openseek-8_kernel_generation_normalized.json``
（已离线解析 func_name / func_signature / math_formula / func_desc / constraints；
 与 task8 代码同目录，整目录可独立打包提交复现）。

核心 Pipeline (每条样本最多 1 + N + 1 + 1 次 LLM 调用)::

  R1 Draft          (1 次 LLM)
    ↓
  Static Validate   (无 LLM: AST + wrapper 函数名 + 参数签名 + placeholder + exec 预检)
    ↓ 失败
  ReAct Repair      (最多 N 轮 LLM, observation = 静态错误)
    ↓
  R2 Verify JSON    (1 次 LLM, 无论 ReAct 是否修复成功都走)
    ↓ safe + high/medium confidence  → 保留 draft
    ↓ 其他                            → R3
  R3 Safe-Exit      (1 次 LLM, 纯文字规范无 few-shot)
    ↓
  AST Guard         (无 LLM: 为 wrapper 添加异常隔离层, parse 失败时生成最简 kernel 骨架)
    ↓
  Submission Sanitizer (空 prediction 用 SAFE_PLACEHOLDER 兜底)

运行示例::

    conda activate flagscale
    bash run.sh                                    # 默认全量评测
    python main.py --max_samples 5                 # 快速测试 5 条
"""

import json
import os
import sys
import argparse
import time
import logging
from collections import Counter
from tqdm import tqdm
from transformers import AutoTokenizer

# 让 main.py 既能直接 ``python main.py`` 运行，又能 import 同目录的 method 与
# 上级目录的 common.*。
_CUR_DIR = os.path.dirname(os.path.abspath(__file__))           # COM/src/task8
_SRC_DIR = os.path.abspath(os.path.join(_CUR_DIR, '..'))        # COM/src
for _p in (_CUR_DIR, _SRC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from method import (  # noqa: E402  (sys.path 设置后才能 import)
    Task8ExampleSelector,
    build_task8_prompt,
    validate_prediction,
    fix_task8_imports,
    postprocess_task8_wrapper,
    annotate,
    _parse_wrapper_info,
    _reformat_icl_example,
    TASK8_MAX_TOKENS,
    TASK8_TIMEOUT,
    TASK8_REP_PENALTY,
    TASK8_MAX_INPUT_LENGTH,
    TASK8_MIN_INPUT_LENGTH,
    TraceLogger,
    static_validate,
    build_verify_prompt,
    parse_verify_json,
    verify_routing_decision,
    classify_static_errors,
    build_safe_exit_prompt,
    react_repair,
    REACT_MAX_ROUNDS,
    apply_ast_guard,
)
from common.paths import (  # noqa: E402
    FINAL_OUTPUT_DIR,
    MODEL_DIR,
    task_log_dir,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ---- 默认路径与配置（task8 数据与代码同目录, 便于整目录打包复现）----
TASK_ID = 8
TASK_FILE       = os.path.join(_CUR_DIR, 'normalized_data', 'openseek-8_kernel_generation_normalized.json')
OUTPUT_PREFIX   = FINAL_OUTPUT_DIR + '/'
TOKENIZER_PATH  = MODEL_DIR
DEFAULT_LOG_DIR = task_log_dir(TASK_ID)
DEFAULT_TRACE_DIR = os.path.join(DEFAULT_LOG_DIR, 'trace')


def parser_args():
    parser = argparse.ArgumentParser(description='Task 8 (kernel_generation) — Multi-Turn Self-Verify')
    parser.add_argument('--max_input_length', type=int, default=17_000,
                        help='Maximum input length (tokens). Default 17000 (Task 8 专属).')
    parser.add_argument('--log_path_prefix', type=str, default=OUTPUT_PREFIX,
                        help='Output path prefix.')
    parser.add_argument('--tokenizer_path', type=str, default=TOKENIZER_PATH,
                        help='Path to Qwen3-4B tokenizer.')
    parser.add_argument('--strategy', type=str, default='dynamic',
                        choices=['dynamic', 'static'],
                        help='ICL example selection strategy.')
    parser.add_argument('--log_level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING'])
    parser.add_argument('--max_samples', type=int, default=0,
                        help='Max test samples to evaluate (0=all). For quick testing.')
    parser.add_argument('--trace_dir', type=str, default=DEFAULT_TRACE_DIR,
                        help='Per-sample trace log directory.')
    parser.add_argument('--react_rounds', type=int, default=REACT_MAX_ROUNDS,
                        help=f'Max ReAct repair rounds. Default {REACT_MAX_ROUNDS}.')
    parser.add_argument('--skip_r3', action='store_true',
                        help='[Debug] Skip R3 safe-exit stage (keep draft even if routed to safe_exit).')
    return parser.parse_args()


# ---- 字段兼容映射 ---------------------------------------------------------
def _map_normalized_fields(icl_examples: list, test_samples: list) -> None:
    """将 normalized 数据补上下游所需的 input/output 字段。

    selector / reformat 依赖 example['input'] 和 example['output']，
    主流程读 test_sample['input']；这里从 raw_input/raw_output 兜底回填。
    """
    for ex in icl_examples:
        if 'input' not in ex and 'raw_input' in ex:
            ex['input'] = ex['raw_input']
        if 'output' not in ex and 'raw_output' in ex:
            ex['output'] = [ex['raw_output']]
    for ts in test_samples:
        if 'input' not in ts and 'raw_input' in ts:
            ts['input'] = ts['raw_input']


# ---- 单样本 Pipeline -----------------------------------------------------
def _dump_io(logger_, tag: str, payload: str) -> None:
    """将 Prompt 或 raw output 以显眼分隔块写入主日志, 便于直接 tail/grep.

    主日志 (非 trace 文件) 也必须能看到模型的完整输入输出.
    使用 logger.info 一条 message 包住整段内容, 避免 multi-line split 导致混乱.
    """
    payload = payload if isinstance(payload, str) else str(payload)
    sep = '=' * 100
    sub = '-' * 100
    block = (
        f"\n{sep}\n"
        f"[IO DUMP] {tag} (chars={len(payload)})\n"
        f"{sub}\n"
        f"{payload}\n"
        f"{sep}"
    )
    logger_.info(block)


def _run_single_sample(
    *,
    idx: int,
    total: int,
    test_sample: dict,
    task_description: str,
    examples_str: str,
    examples_tokens: int = 0,
    react_rounds: int,
    skip_r3: bool,
    trace: TraceLogger,
    op_type: str = 'unknown',
) -> tuple[str, str, dict]:
    """执行单条样本的 R1→Static→React→R2→R3 五阶段 pipeline。

    Returns: (final_prediction, verdict, meta)
      verdict ∈ {'kept_draft', 'safe_exit', 'safe_exit_fallback'}
      meta 为统计字段。
    """
    sid = test_sample['id']
    text2annotate = test_sample['input']

    # normalized 数据已解析 wrapper 信息; 缺失时用 _parse_wrapper_info 兜底
    func_name = test_sample.get('func_name') or ''
    func_signature = test_sample.get('func_signature') or ''
    math_formula = test_sample.get('math_formula') or ''
    func_desc = test_sample.get('func_desc') or ''
    constraints = test_sample.get('constraints') or ''
    if not func_name or not func_signature:
        parsed = _parse_wrapper_info(text2annotate)
        func_name = func_name or parsed.get('func_name', '')
        func_signature = func_signature or parsed.get('func_signature', '')
        math_formula = math_formula or parsed.get('math_formula', '')
        func_desc = func_desc or parsed.get('func_desc', '')
        constraints = constraints or parsed.get('constraints', '')

    # 组装 wrapper_info 透传给下游, 避免重复解析 raw_input
    wrapper_info = {
        'func_name': func_name,
        'func_signature': func_signature,
        'math_formula': math_formula,
        'func_desc': func_desc,
        'constraints': constraints,
    }

    meta = {
        'sid': sid,
        'func_name': func_name,
        'func_signature_len': len(func_signature),
        'op_type': op_type,
        'r1_prediction_chars': 0,
        'static_passed_after_r1': False,
        'react_used_rounds': 0,
        'react_passed': False,
        'static_hard_errors_final': 0,
        'static_soft_errors_final': 0,
        'r2_parsed': False,
        'r2_risk_level': '',
        'r2_confidence': '',
        'r2_routing': '',
        'r3_used': False,
        'r3_prediction_chars': 0,
        'verdict': '',
    }

    # ---------- R1 Draft ----------
    r1_prompt = build_task8_prompt(
        task_description=task_description,
        text2annotate=text2annotate,
        examples_str=examples_str,
        wrapper_info=wrapper_info,
    )
    trace.log(
        "R1 Draft - PROMPT (FULL)",
        r1_prompt,
        meta={'prompt_chars': len(r1_prompt), 'icl_tokens': examples_tokens,
              'func_name': func_name,
              'func_signature': func_signature[:200]},
    )
    logger.info(f"[{idx+1}/{total}] R1 Draft 开始 (prompt={len(r1_prompt)} chars, "
                f"icl_tokens={examples_tokens})")
    _dump_io(logger, f"[{idx+1}/{total}] R1 Draft - INPUT", r1_prompt)

    t0 = time.time()
    r1_raw = annotate(
        r1_prompt,
        max_tokens=TASK8_MAX_TOKENS,
        timeout=TASK8_TIMEOUT,
        temperature=0.0,
        repetition_penalty=TASK8_REP_PENALTY,
        enable_thinking=True,
    )
    r1_elapsed = time.time() - t0
    trace.log(
        "R1 Draft - RAW OUTPUT",
        r1_raw or "(empty)",
        meta={'output_chars': len(r1_raw or ''), 'duration_s': f'{r1_elapsed:.1f}'},
    )
    _dump_io(logger, f"[{idx+1}/{total}] R1 Draft - RAW OUTPUT", r1_raw or "(empty)")

    draft = validate_prediction(r1_raw) or ''
    if draft:
        draft = fix_task8_imports(draft)
        draft = postprocess_task8_wrapper(draft, text2annotate, wrapper_info=wrapper_info)
    meta['r1_prediction_chars'] = len(draft)
    trace.log(
        "R1 Draft - EXTRACTED",
        draft or "(empty)",
        meta={'chars': len(draft)},
    )
    logger.info(f"[{idx+1}/{total}] R1 Draft 完成 ({r1_elapsed:.1f}s, extracted={len(draft)} chars)")

    # ---------- Static Validate (R1.4) ----------
    if draft:
        ok, errors = static_validate(draft, func_name, func_signature)
    else:
        ok, errors = False, ['empty draft from R1']
    meta['static_passed_after_r1'] = ok
    trace.log(
        "R1.4 Static Validate",
        "errors:\n" + ("\n".join(f'  - {e}' for e in errors) if errors else "  (none)"),
        meta={'passed': ok, 'error_count': len(errors)},
    )
    logger.info(f"[{idx+1}/{total}] Static Validate: passed={ok}, errors={len(errors)}")

    # ---------- React Repair (R1.5) — only when static failed ----------
    if not ok:
        if not draft:
            # 空草稿: 跳过 react (无法喂给修复循环), 直接进 R2 让路由判决
            logger.warning(f"[{idx+1}/{total}] R1 空草稿, 跳过 React Repair, 直接 R2")
            trace.log("R1.5 React Repair", "SKIPPED: empty draft",
                      meta={'reason': 'empty_draft'})
        else:
            logger.info(f"[{idx+1}/{total}] 进入 React Repair (最多 {react_rounds} 轮)")
            trace.log("R1.5 React Repair",
                      f"Entering react_repair, max_rounds={react_rounds}")
            draft, react_passed, used = react_repair(
                initial_code=draft,
                initial_errors=errors,
                task_description=task_description,
                text2annotate=text2annotate,
                func_name=func_name,
                func_signature=func_signature,
                max_rounds=react_rounds,
                trace=trace,
                wrapper_info=wrapper_info,
                examples_str=examples_str,
                examples_tokens=examples_tokens,
            )
            meta['react_used_rounds'] = used
            meta['react_passed'] = react_passed
            trace.log(
                "R1.5 React Repair - FINAL",
                draft or "(empty)",
                meta={'passed': react_passed, 'rounds_used': used,
                      'chars': len(draft)},
            )
            logger.info(f"[{idx+1}/{total}] React Repair 完成: "
                        f"passed={react_passed}, rounds={used}")

    # ---------- R2 Verify JSON ----------
    # 无论 React 是否成功, 都走 R2 (方案要求)
    if not draft:
        logger.warning(f"[{idx+1}/{total}] R2 前草稿仍为空, 直接路由 safe_exit")
        trace.log("R2 Verify", "SKIPPED: empty draft, force safe_exit route",
                  meta={'forced_route': 'safe_exit'})
        verify_info = {
            'risk_level': 'unsafe',
            'numerical_equivalence_confidence': 'low',
            '_parsed': False,
            'rationale': 'empty_draft_before_r2',
        }
    else:
        r2_prompt = build_verify_prompt(
            code=draft,
            task_description=task_description,
            text2annotate=text2annotate,
            func_name=func_name,
            examples_str=examples_str,
            wrapper_info=wrapper_info,
        )
        trace.log(
            "R2 Verify - PROMPT (FULL)",
            r2_prompt,
            meta={'prompt_chars': len(r2_prompt),
                  'icl_tokens': examples_tokens},
        )
        logger.info(f"[{idx+1}/{total}] R2 Verify 开始 (prompt={len(r2_prompt)} chars, "
                    f"icl_tokens={examples_tokens})")
        _dump_io(logger, f"[{idx+1}/{total}] R2 Verify - INPUT", r2_prompt)

        t0 = time.time()
        r2_raw = annotate(
            r2_prompt,
            max_tokens=4000,  # JSON 输出短即可
            timeout=TASK8_TIMEOUT,
            temperature=0.3, top_p=0.9, top_k=20, min_p=0.0,
            repetition_penalty=1.1,
            enable_thinking=False,  # 结构化自检无需 thinking
        )
        r2_elapsed = time.time() - t0
        trace.log(
            "R2 Verify - RAW OUTPUT",
            r2_raw or "(empty)",
            meta={'output_chars': len(r2_raw or ''), 'duration_s': f'{r2_elapsed:.1f}'},
        )
        _dump_io(logger, f"[{idx+1}/{total}] R2 Verify - RAW OUTPUT", r2_raw or "(empty)")
        verify_info = parse_verify_json(r2_raw)
        logger.info(f"[{idx+1}/{total}] R2 Verify 完成 ({r2_elapsed:.1f}s): "
                    f"parsed={verify_info.get('_parsed')}, "
                    f"risk={verify_info.get('risk_level')}, "
                    f"conf={verify_info.get('numerical_equivalence_confidence')}")

    meta['r2_parsed'] = bool(verify_info.get('_parsed'))
    meta['r2_risk_level'] = verify_info.get('risk_level', '')
    meta['r2_confidence'] = verify_info.get('numerical_equivalence_confidence', '')

    # ---------- 路由前：对最终 draft 重做一次 static, 得到 hard/soft 分类 ----------
    if draft:
        _ok_final, final_errors = static_validate(draft, func_name, func_signature)
    else:
        final_errors = ['empty draft from R1']
    hard_errors, soft_errors = classify_static_errors(final_errors)
    meta['static_hard_errors_final'] = len(hard_errors)
    meta['static_soft_errors_final'] = len(soft_errors)
    trace.log(
        "R2 Static Error Classification",
        f"op_type={op_type}\n"
        f"hard({len(hard_errors)}):\n"
        + ("\n".join(f'  - {e}' for e in hard_errors) or "  (none)")
        + "\n"
        f"soft({len(soft_errors)}):\n"
        + ("\n".join(f'  - {e}' for e in soft_errors) or "  (none)"),
        meta={'hard_count': len(hard_errors),
              'soft_count': len(soft_errors),
              'op_type': op_type},
    )

    routing = verify_routing_decision(
        verify_info,
        hard_errors=hard_errors,
        soft_errors=soft_errors,
        op_type=op_type,
    )
    meta['r2_routing'] = routing
    trace.log(
        "R2 Verify - PARSED + ROUTING",
        json.dumps(verify_info, indent=2, ensure_ascii=False),
        meta={'routing': routing, 'op_type': op_type,
              'hard_errors': len(hard_errors),
              'soft_errors': len(soft_errors)},
    )
    logger.info(f"[{idx+1}/{total}] R2 Routing: {routing} "
                f"(op_type={op_type}, hard={len(hard_errors)}, "
                f"soft={len(soft_errors)})")

    # ---------- R3 Safe-Exit (仅当 routing=safe_exit) ----------
    if routing == 'keep_draft':
        meta['verdict'] = 'kept_draft'
        trace.log("FINAL VERDICT", "kept_draft",
                  meta={'prediction_chars': len(draft)})
        return draft, 'kept_draft', meta

    # routing == 'safe_exit'
    if skip_r3:
        # Debug 路径: 不跑 R3, 保留 draft (供对比用)
        meta['verdict'] = 'safe_exit_skipped'
        trace.log("FINAL VERDICT", "safe_exit_skipped (--skip_r3)",
                  meta={'prediction_chars': len(draft)})
        return draft, 'safe_exit_skipped', meta

    logger.info(f"[{idx+1}/{total}] R3 Safe-Exit 开始")
    r3_prompt = build_safe_exit_prompt(
        func_name=func_name,
        func_signature=func_signature,
        task_description=task_description,
        text2annotate=text2annotate,
        current_code=draft,
        examples_str=examples_str,
        wrapper_info=wrapper_info,
    )
    trace.log(
        "R3 Safe-Exit - PROMPT (FULL)",
        r3_prompt,
        meta={'prompt_chars': len(r3_prompt),
              'icl_tokens': examples_tokens},
    )
    logger.info(f"[{idx+1}/{total}] R3 Safe-Exit prompt={len(r3_prompt)} chars, "
                f"icl_tokens={examples_tokens}")
    _dump_io(logger, f"[{idx+1}/{total}] R3 Safe-Exit - INPUT", r3_prompt)

    t0 = time.time()
    r3_raw = annotate(
        r3_prompt,
        max_tokens=6000,
        timeout=TASK8_TIMEOUT,
        temperature=0.0,
        repetition_penalty=1.1,
        enable_thinking=False,
    )
    r3_elapsed = time.time() - t0
    trace.log(
        "R3 Safe-Exit - RAW OUTPUT",
        r3_raw or "(empty)",
        meta={'output_chars': len(r3_raw or ''), 'duration_s': f'{r3_elapsed:.1f}'},
    )
    _dump_io(logger, f"[{idx+1}/{total}] R3 Safe-Exit - RAW OUTPUT", r3_raw or "(empty)")

    r3_code = validate_prediction(r3_raw) or ''
    if r3_code:
        r3_code = fix_task8_imports(r3_code)
        r3_code = postprocess_task8_wrapper(r3_code, text2annotate, wrapper_info=wrapper_info)
    meta['r3_used'] = True
    meta['r3_prediction_chars'] = len(r3_code)
    trace.log(
        "R3 Safe-Exit - EXTRACTED",
        r3_code or "(empty)",
        meta={'chars': len(r3_code)},
    )
    logger.info(f"[{idx+1}/{total}] R3 Safe-Exit 完成 ({r3_elapsed:.1f}s, "
                f"extracted={len(r3_code)} chars)")

    if r3_code:
        meta['verdict'] = 'safe_exit'
        trace.log("FINAL VERDICT", "safe_exit",
                  meta={'prediction_chars': len(r3_code)})
        return r3_code, 'safe_exit', meta

    # R3 也失败: 退化保留 draft (即使非空; 若空则返回空)
    meta['verdict'] = 'safe_exit_fallback'
    trace.log("FINAL VERDICT", "safe_exit_fallback (R3 extraction failed)",
              meta={'prediction_chars': len(draft)})
    logger.warning(f"[{idx+1}/{total}] R3 提取失败, 退化保留 draft ({len(draft)} chars)")
    return draft, 'safe_exit_fallback', meta


def evaluate(qwen_tokenizer: AutoTokenizer,
             max_input_length: int = 17_000,
             log_path_prefix: str = OUTPUT_PREFIX,
             strategy: str = 'dynamic',
             max_samples: int = 0,
             trace_dir: str = DEFAULT_TRACE_DIR,
             react_rounds: int = REACT_MAX_ROUNDS,
             skip_r3: bool = False) -> None:
    """Task 8 主评测流程"""
    task_id = TASK_ID

    # ---- 加载任务数据 (normalized) ----
    with open(TASK_FILE, 'r') as f:
        task_dict = json.load(f)

    task_name = task_dict['task_name']
    task_description = task_dict['Definition'][0]
    icl_examples = task_dict['examples']
    test_samples = task_dict['test_samples']

    # 字段兼容映射: 部分下游代码读 input/output, 数据里只有 raw_input/raw_output
    _map_normalized_fields(icl_examples, test_samples)

    if max_samples > 0:
        test_samples = test_samples[:max_samples]
        logger.info(f"[快速测试] 仅评测前 {max_samples} 个样本")

    logger.info(f"Task {task_id}: {task_name}")
    logger.info(f"ICL examples: {len(icl_examples)}, Test samples: {len(test_samples)}")
    logger.info(f"Strategy: {strategy}, Max input length: {max_input_length}, "
                f"React rounds: {react_rounds}, Skip R3: {skip_r3}")

    # per-task 输入长度覆盖
    effective_max_input = TASK8_MAX_INPUT_LENGTH
    if effective_max_input != max_input_length:
        logger.info(f"Task {task_id} 使用专属输入长度: {effective_max_input}")
    max_input_length = effective_max_input
    example_budget = max_input_length
    example_budget_min = TASK8_MIN_INPUT_LENGTH
    logger.info(f"示例 token 预算: {example_budget} (下限 {example_budget_min})")

    # ---- 初始化示例选择器 (BM25) ----
    selector = Task8ExampleSelector(
        qwen_tokenizer,
        max_context_tokens=example_budget,
        min_context_tokens=example_budget_min,
    )

    logger.info(f"max_tokens={TASK8_MAX_TOKENS}, timeout={TASK8_TIMEOUT}s, "
                f"rep_penalty={TASK8_REP_PENALTY}")

    # ---- 准备输出文件 ----
    output_path = os.path.abspath(log_path_prefix)
    os.makedirs(output_path, exist_ok=True)
    version = 1
    output_file = os.path.join(output_path, f'openseek-{task_id}-v{version}.jsonl')
    while os.path.exists(output_file):
        version += 1
        output_file = os.path.join(output_path, f'openseek-{task_id}-v{version}.jsonl')
    with open(output_file, 'w') as f:
        pass
    logger.info(f"Output: {output_file}")

    # ---- 日志文件 ----
    log_dir = DEFAULT_LOG_DIR
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'task_{task_id}_v{version}.log')
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logging.getLogger().addHandler(file_handler)
    logger.info(f"Log: {log_file}")

    # ---- Trace 目录 (每样本一个文件) ----
    trace_dir_abs = os.path.abspath(trace_dir)
    trace_dir_version = os.path.join(trace_dir_abs, f'v{version}')
    os.makedirs(trace_dir_version, exist_ok=True)
    logger.info(f"Trace dir: {trace_dir_version}")

    # ---- static 模式 ----
    static_examples_str = None
    if strategy == 'static':
        example_lines = []
        for ex in icl_examples:
            # 传整个 ex dict, 使用 normalized 字段
            line = _reformat_icl_example(ex, cot=ex.get('cot', ''))
            example_lines.append(line)
        static_examples_str, _ = selector._truncate_by_tokens(example_lines)
        logger.info("Static 模式：示例已选择")

    # ---- 主循环 ----
    verdict_counts: Counter = Counter()
    verdict_by_op_type: Counter = Counter()
    null_count = 0
    start_time = time.time()
    report_interval = max(1, len(test_samples) // 10)

    for i, test_sample in enumerate(tqdm(test_samples,
                                          desc=f'Task {task_id}: {task_name}')):
        sid = test_sample['id']
        text2annotate = test_sample['input']
        sample_t0 = time.time()

        total = len(test_samples)
        logger.info(f"[{i+1}/{total}] sample={sid} input='{text2annotate[:100]}'")

        # 每样本 trace 文件
        id_short = sid.split('-')[-1][:12] if '-' in sid else sid[:12]
        trace_path = os.path.join(trace_dir_version,
                                  f'sample_{i+1:03d}_{id_short}.log')
        trace = TraceLogger(trace_path)
        trace.header({
            'sample_index': f'{i+1}/{total}',
            'sample_id': sid,
            'func_name': test_sample.get('func_name', ''),
            'func_signature': (test_sample.get('func_signature', '') or '')[:200],
            'op_category': test_sample.get('op_category', ''),
            'strategy': strategy,
            'react_rounds': react_rounds,
        })

        # ---- ICL 选择 ----
        if strategy == 'dynamic':
            examples_str = selector.select(
                icl_examples, task_description, text2annotate
            )
        else:
            examples_str = static_examples_str
        examples_tokens = (
            len(qwen_tokenizer.encode(examples_str)) if examples_str else 0
        )
        # 算子类型分类（供 safe_exit 路由的先验使用）
        op_type = Task8ExampleSelector._classify_operation_type(text2annotate)
        trace.log(
            "ICL Selection",
            f"strategy={strategy}, examples_str_chars={len(examples_str)}, "
            f"examples_tokens={examples_tokens}, op_type={op_type}",
            meta={'strategy': strategy, 'examples_chars': len(examples_str),
                  'examples_tokens': examples_tokens, 'op_type': op_type},
        )
        logger.info(
            f"[{i+1}/{total}] ICL: chars={len(examples_str)}, "
            f"tokens={examples_tokens}, op_type={op_type}"
        )

        # ---- 执行 Pipeline ----
        try:
            prediction, verdict, meta = _run_single_sample(
                idx=i,
                total=total,
                test_sample=test_sample,
                task_description=task_description,
                examples_str=examples_str,
                examples_tokens=examples_tokens,
                react_rounds=react_rounds,
                skip_r3=skip_r3,
                trace=trace,
                op_type=op_type,
            )
        except Exception as e:
            logger.exception(f"[{i+1}/{total}] sample={sid} pipeline 异常: {e}")
            trace.log("PIPELINE EXCEPTION", repr(e))
            prediction, verdict, meta = '', 'exception', {'verdict': 'exception', 'error': repr(e), 'op_type': op_type}

        sample_elapsed = time.time() - sample_t0

        if prediction is None:
            prediction = ""

        # ---- AST 兜底: 强制 wrapper 套 try/except + parse 失败 fallback 到骨架 ----
        # 覆盖所有 verdict (kept_draft / safe_exit / safe_exit_fallback /
        # safe_exit_skipped / exception), 确保 wrapper 始终可调用不抛异常.
        _fn_name = test_sample.get('func_name', '') or ''
        _fn_sig = test_sample.get('func_signature', '') or _fn_name
        if _fn_name:
            try:
                prediction, _ast_action = apply_ast_guard(prediction, _fn_name, _fn_sig)
                meta['ast_guard_action'] = _ast_action
            except Exception as _guard_err:
                logger.exception(f"[{i+1}/{total}] apply_ast_guard 异常: {_guard_err}")
                meta['ast_guard_action'] = 'exception'
                meta['ast_guard_error'] = repr(_guard_err)
        else:
            meta['ast_guard_action'] = 'skipped_no_func_name'

        test_record = {
            'test_sample_id': sid,
            'prediction': prediction,
            'verdict': verdict,
            'meta': meta,
        }
        verdict_counts[verdict] += 1
        verdict_by_op_type[(op_type, verdict)] += 1

        if prediction == "":
            null_count += 1
            logger.warning(f"[{i+1}/{total}] sample={sid} "
                           f"prediction='' verdict={verdict} ({sample_elapsed:.1f}s)")
        else:
            logger.info(f"[{i+1}/{total}] sample={sid} "
                        f"prediction='{prediction[:80]}' verdict={verdict} "
                        f"({sample_elapsed:.1f}s)")

        # 写入结果 (逐条 flush, 防止中途崩溃丢失)
        with open(output_file, 'a') as f:
            f.write(json.dumps(test_record, ensure_ascii=False) + '\n')

        trace.close()

        # 进度报告
        if (i + 1) % report_interval == 0 or i == 0:
            elapsed_so_far = time.time() - start_time
            avg = elapsed_so_far / (i + 1)
            remaining = avg * (len(test_samples) - i - 1)
            verdict_summary = ', '.join(f'{k}={v}' for k, v in verdict_counts.most_common())
            logger.info(
                f"[{i+1}/{total}] "
                f"null={null_count}/{i+1} ({null_count/(i+1)*100:.1f}%) | "
                f"verdicts: {verdict_summary} | "
                f"{elapsed_so_far:.0f}s | {avg:.1f}s/样本 | 剩余 ~{remaining:.0f}s"
            )

    # ---- 统计 ----
    elapsed = time.time() - start_time
    total = len(test_samples)
    logger.info(f"完成! {elapsed:.1f}s, 均速 {elapsed/total:.1f}s/样本")
    logger.info(f"Null: {null_count}/{total} ({null_count/total*100:.1f}%)")
    for k, v in verdict_counts.most_common():
        logger.info(f"Verdict {k}: {v}/{total} ({v/total*100:.1f}%)")

    # ---- verdict × op_type 矩阵（用于回调 safe_exit 阈值） ----
    if verdict_by_op_type:
        logger.info("--- Verdict × op_type 分布 ---")
        # 按 op_type 聚合
        op_types_seen = sorted({op for (op, _v) in verdict_by_op_type.keys()})
        for op in op_types_seen:
            total_op = sum(c for (o, _v), c in verdict_by_op_type.items() if o == op)
            parts = []
            for (o, v), c in verdict_by_op_type.most_common():
                if o == op:
                    parts.append(f"{v}={c}")
            logger.info(f"  op_type={op} (n={total_op}): {', '.join(parts)}")

    # 关闭日志
    logging.getLogger().removeHandler(file_handler)
    file_handler.close()

    # ---- 生成提交文件 ----
    generate_submission(output_file, log_path_prefix, task_id, version)


def generate_submission(results_file: str, log_path_prefix: str,
                        task_id: int, version: int) -> None:
    """从带 verdict 的结果文件生成合规提交文件。

    策略:
      - prediction 非空 → 直接原样提交
      - prediction 空   → 用 SAFE_PLACEHOLDER 兜底
    """
    SAFE_PLACEHOLDER = (
        'import torch\nimport triton\nimport triton.language as tl\n\n'
        '@triton.jit\ndef _placeholder_kernel(x_ptr, BLOCK_SIZE: tl.constexpr):\n'
        '    pass\n'
    )

    results = []
    with open(results_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))

    submission_file = f'{log_path_prefix}openseek-{task_id}-v{version}-submit.jsonl'
    kept_count = 0
    placeholder_count = 0
    verdict_counts: Counter = Counter()

    with open(submission_file, 'w') as f:
        for r in results:
            record = {'test_sample_id': r['test_sample_id']}
            pred = (r.get('prediction') or '').strip()
            verdict = r.get('verdict', '')
            verdict_counts[verdict] += 1

            if pred:
                record['prediction'] = pred
                kept_count += 1
            else:
                record['prediction'] = SAFE_PLACEHOLDER
                placeholder_count += 1

            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    total = len(results)
    logger.info(f"\n{'='*60}")
    logger.info(f"提交文件生成: {submission_file}")
    logger.info(f"保留 prediction: {kept_count}/{total} ({kept_count/total*100:.1f}%)")
    logger.info(f"SAFE_PLACEHOLDER 兜底: {placeholder_count}/{total} "
                f"({placeholder_count/total*100:.1f}%)")
    for k, v in verdict_counts.most_common():
        logger.info(f"  verdict={k}: {v}")
    logger.info(f"{'='*60}")


if __name__ == '__main__':
    args = parser_args()
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.getLogger().setLevel(log_level)
    logger.info(f"日志级别: {args.log_level}")

    qwen_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    logger.info(f"Tokenizer: {args.tokenizer_path}")
    evaluate(
        qwen_tokenizer=qwen_tokenizer,
        max_input_length=args.max_input_length,
        log_path_prefix=args.log_path_prefix,
        strategy=args.strategy,
        max_samples=args.max_samples,
        trace_dir=args.trace_dir,
        react_rounds=args.react_rounds,
        skip_r3=args.skip_r3,
    )
