import json, os, argparse
from pathlib import Path
from tqdm import tqdm

# from method import build_prompt, select_examples, annotate

from method_hyb import build_prompt, select_examples

from method_hyb import annotate_nvidia as annotate # For Nvidia GPU
# from method import annotate_ascend as annotate # For Huawei Ascend

# 项目根目录（与当前工作目录无关，避免输出写到仓库外）
REPO_ROOT = Path(__file__).resolve().parent.parent

TASK_DATA_FILES: dict[int, str] = {
    1: 'openseek-1_closest_integers.json',
    2: 'openseek-2_count_nouns_verbs.json',
    3: 'openseek-3_collatz_conjecture.json',
    4: 'openseek-4_conala_concat_strings.json',
    5: 'openseek-5_semeval_2018_task1_tweet_sadness_detection.json',
    6: 'openseek-6_mnli_same_genre_classification.json',
    7: 'openseek-7_jeopardy_answer_generation_all.json',
    8: 'openseek-8_kernel_generation.json',
}


def _task_json_path(task_id: int) -> Path:
    return REPO_ROOT / 'data' / TASK_DATA_FILES[task_id]


def _resolve_output_dir(log_path_prefix: str | None) -> str:
    """输出目录：默认项目根下 outputs/；相对路径相对项目根解析。"""
    if not log_path_prefix:
        return str((REPO_ROOT / 'outputs').resolve())
    p = Path(log_path_prefix)
    if p.is_absolute():
        return str(p.resolve())
    return str((REPO_ROOT / log_path_prefix).resolve())

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_input_length', type=int, default=10_000,
                        help='Maximum input length for the model.')
    parser.add_argument(
        '--log_path_prefix',
        type=str,
        default=None,
        help='结果 jsonl 保存目录；默认 <项目根>/outputs。相对路径相对于项目根目录。',
    )
    parser.add_argument(
        '--tokenizer_path',
        type=str,
        default=None,
        help='Qwen3 tokenizer 目录或 HuggingFace 模型 ID；默认使用仓库根目录下 Qwen3-4B，不存在则回退为 Qwen/Qwen3-4B。',
    )
    parser.add_argument(
        '--task_start',
        type=int,
        default=8,
        help='起始任务编号（含），与 README 中 openseek-[id] 一致，默认 1。',
    )
    parser.add_argument(
        '--task_end',
        type=int,
        default=8,
        help='结束任务编号（含），提交需 8 个 jsonl 时设为 8，默认 8。',
    )
    args = parser.parse_args()
    return args


def _pick_output_jsonl(log_path_prefix: str, task_id: int) -> str:
    """生成评测脚本要求的命名：openseek-{id}-v1.jsonl（FlagOS judge_track3 按此正则排序）。"""
    os.makedirs(log_path_prefix, exist_ok=True)
    primary = os.path.join(log_path_prefix, f'openseek-{task_id}-v1.jsonl')
    if not os.path.exists(primary):
        return primary
    v = 2
    while True:
        alt = os.path.join(log_path_prefix, f'openseek-{task_id}-v{v}.jsonl')
        if not os.path.exists(alt):
            return alt
        v += 1


def run_tasks(
    max_input_length: int = 10_000,
    log_path_prefix: str | None = None,
    tokenizer_path: str | None = None,
    task_start: int = 8,
    task_end: int = 8,
) -> None:
    """依次对 task_id in [task_start, task_end] 调用 evaluate（README 要求共 8 个 jsonl）。"""
    out_dir = _resolve_output_dir(log_path_prefix)
    print(f'[结果保存] 输出目录（绝对路径）: {out_dir}')
    for task_id in range(task_start, task_end + 1):
        evaluate(task_id, max_input_length, out_dir, tokenizer_path)

def evaluate(
    task_id: int,
    max_input_length: int = 128_000,
    log_path_prefix: str = './outputs/',
    tokenizer_path: str | None = None,
) -> float:
    assert task_id in [i for i in range(1, 9)],\
        f"task_id should be in [1, 8], but got {task_id}."
    
    task_file = _task_json_path(task_id)
    with open(task_file, 'r', encoding='utf-8') as f:
        task_dict = json.load(f)
    
    task_name = task_dict['task_name']
    task_description = task_dict['Definition'][0]
    icl_examples = task_dict['examples'][:100]
    test_samples = task_dict['test_samples']

    # 先跑完本任务全部样本，再一次性写入 jsonl（8 个任务各对应一个文件，写完再落盘）
    rows: list[dict] = []
    for test_sample in tqdm(test_samples, desc=f'Evaluation on Task {task_id}: {task_name}'):
        test_sample_id = test_sample['id']
        text2annotate = test_sample['input']
        prompt = build_prompt(task_description, text2annotate, task_id=task_id)
        # 每条样本独立进行混合检索，避免复用首条样本的 ICL 示例。
        # 固定每次召回 top3，且保留 CoT 形态（explanation + label）。
        examples_str = select_examples(
            icl_examples,
            task_description,
            text2annotate,
            tokenizer_path=tokenizer_path,
            hybrid=True,
            top_k=3,
            use_explanation=True,
            use_bm25_semantic_rerank=True,
        )
        input_prompt = prompt.replace("[[EXAMPLES]]\n\n", examples_str+'\n\n')
        raw = annotate(input_prompt)
        # 评测脚本（如 task8 Consistency_Aware）会对 prediction 做子串判断，null 会触发 TypeError
        prediction = "" if raw is None else raw
        print(f"[逐条结果] task={task_id} test_sample_id={test_sample_id} prediction={prediction!r}")
        rows.append(
            {
                'test_sample_id': test_sample_id,
                'prediction': prediction,
            }
        )

    # 提交：8 个 openseek-{id}-v1.jsonl（zip 根目录）；每行含 test_sample_id、prediction
    output_file = _pick_output_jsonl(log_path_prefix, task_id)
    with open(output_file, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    empty_cnt = sum(1 for row in rows if row.get('prediction') == "")
    ratio = (empty_cnt / len(rows)) if rows else 0.0
    print(
        f'[结果保存] 任务 {task_id} 已完成，共 {len(rows)} 条，prediction 为空串={empty_cnt} ({ratio:.1%}) -> {os.path.abspath(output_file)}'
    )

if __name__ == '__main__':
    args = parser_args()
    _default_tok = REPO_ROOT / 'Qwen3-4B'
    tokenizer_path = args.tokenizer_path
    if tokenizer_path is None:
        tokenizer_path = str(_default_tok) if _default_tok.is_dir() else 'Qwen/Qwen3-4B'
    run_tasks(
        args.max_input_length,
        args.log_path_prefix,
        tokenizer_path,
        task_start=args.task_start,
        task_end=args.task_end,
    )