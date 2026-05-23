import ast
import json
import re
import time
from typing import Any, Callable, Dict, List, Tuple
from tqdm import tqdm
from openai import OpenAI
import os
# -----------------------------
# 数据集加载
# -----------------------------
def load_task(file_path: str) -> Tuple[str, str, List[str], List[Dict[str, Any]], List[Dict[str, Any]]]:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return (
        data.get("task_id", ""),
        data.get("task_name", ""),
        data.get("Definition", []),
        data.get("examples", []),
        data.get("test_samples", []),
    )

# -----------------------------
# 模型 API 调用
# -----------------------------
client = OpenAI(
    api_key="dummy",
    base_url="https://flagos.io/flagos-lab/hw/node/HW-gpu57/port/22653/v1",
)

def qwen_api(messages: List[Dict[str, str]], model: str = "/Qwen3-4B/Qwen/Qwen3-4B", retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            res = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
            )
            return res.choices[0].message.content or ""
        except Exception as e:
            if attempt == retries - 1:
                print(f"\nAPI 调用失败: {e}")
                return ""
            time.sleep(2)
    return ""

# -----------------------------
# 提取 Python 代码块
# -----------------------------
CODE_BLOCK_PATTERN = re.compile(r"```python\s*(.*?)```|```\s*(.*?)```", re.DOTALL | re.IGNORECASE)
def extract_code(text: str) -> str:
    if not text:
        return ""
    match = CODE_BLOCK_PATTERN.search(text)
    if match:
        return (match.group(1) or match.group(2) or "").strip()
    return text.strip()

# -----------------------------
# 保底 solver
# -----------------------------
def fallback_solver_code() -> str:
    return '''
def solve(input_text: str) -> str:
    nums = ast.literal_eval(input_text.strip())
    result = []
    for x in nums:
        if x % 2 == 0:
            result.append(x // 2)
        else:
            result.append(x * 3 + 1)
    return str(result)
'''.strip()

# -----------------------------
# 构建 Prompt
# -----------------------------
def build_code_generation_prompt(task_name: str, definition: List[str], examples: List[Dict[str, Any]]) -> str:
    demo_examples = examples[:400]
    examples_text = "\n".join(f"输入: {ex['input']}\n输出: {ex['output'][0]}" for ex in demo_examples)
    return f"""
你是 Python 算法工程师。
请根据任务定义和样例生成可执行 Python 代码。

任务名：{task_name}
任务定义：
{chr(10).join(definition)}

样例：
{examples_text}

严格要求：
1. 定义函数 solve(input_text: str) -> str
2. input_text 是完整字符串，例如 "[1,2,3]"
3. 可以使用 import 或标准库
4. 用 ast.literal_eval 解析输入
5. 返回值必须是字符串
6. 生成代码必须能被 exec 直接执行
"""

# -----------------------------
# 编译 solver（取消安全限制）
# -----------------------------
def compile_solver(code: str) -> Callable[[str], str]:
    namespace: Dict[str, Any] = {}
    exec(code, namespace)  # 直接执行，不限制 import
    if "solve" not in namespace:
        raise ValueError("生成代码中未定义 solve")
    return namespace["solve"]

# -----------------------------
# 验证 solver
# -----------------------------
def validate_solver(solve_func: Callable[[str], str], examples: List[Dict[str, Any]], limit: int = 100) -> Tuple[float, List[Dict[str, Any]]]:
    total = min(limit, len(examples))
    errors: List[Dict[str, Any]] = []
    correct = 0
    for sample in examples[:total]:
        try:
            pred = str(solve_func(sample["input"])).strip()
        except Exception as e:
            pred = f"<runtime_error: {e}>"
        gt = str(sample["output"][0]).strip()
        if pred == gt:
            correct += 1
        else:
            errors.append({"id": sample.get("id",""), "input": sample["input"], "gt": gt, "pred": pred})
    return correct / total if total else 0.0, errors

# -----------------------------
# 自动生成 solver
# -----------------------------
def generate_valid_solver(task_name: str, definition: List[str], examples: List[Dict[str, Any]], max_attempts: int = 3):
    last_code = ""
    last_errors = []
    for attempt in range(1, max_attempts + 1):
        if attempt == 1:
            prompt = build_code_generation_prompt(task_name, definition, examples)
        else:
            error_text = "\n".join(f"输入: {e['input']} | 正确输出: {e['gt']} | 你的输出: {e['pred']}" for e in last_errors[:10])
            prompt = f"""
你上一次生成的 solve 函数在以下样例上出错：
{error_text}
请重新生成完整 Python 代码，并修正逻辑。
严格要求：
- 定义 solve(input_text: str) -> str
- 可以使用 import 或标准库
- 输出字符串
"""
        messages = [
            {"role": "system", "content": "你是严格输出可执行 Python 代码的助手。"},
            {"role": "user", "content": prompt}
        ]
        raw_code = qwen_api(messages, model="/Qwen3-4B/Qwen/Qwen3-4B")
        code = extract_code(raw_code)
        last_code = code
        try:
            solve_func = compile_solver(code)
            acc, errors = validate_solver(solve_func, examples, limit=min(200, len(examples)))
            print(f"第 {attempt} 次生成，前 {min(200,len(examples))} 条准确率: {acc:.4f}")
            if acc == 1.0:
                return code, solve_func, acc, errors
            last_errors = errors
        except Exception as e:
            print(f"第 {attempt} 次代码编译失败: {e}")
            last_errors = [{"input":"编译失败","gt":"solve(input_text: str) -> str","pred":str(e)}]
    print("回退到保底求解器。")
    last_code = fallback_solver_code()
    solve_func = compile_solver(last_code)
    acc, errors = validate_solver(solve_func, examples, limit=min(200, len(examples)))
    return last_code, solve_func, acc, errors

# -----------------------------
# 对测试集运行预测
# -----------------------------
def run_predictions(solve_func: Callable[[str], str], samples: List[Dict[str, Any]], has_label: bool=False):
    results = []
    correct = 0
    for s in tqdm(samples, desc="Running solver"):
        try:
            pred = str(solve_func(s["input"])).strip()
        except Exception as e:
            pred = f"<runtime_error: {e}>"
        item = {"test_sample_id": s.get("id",""), "prediction": pred}
        if has_label:
            gt = str(s["output"][0]).strip()
            item["gt"] = gt
            item["is_correct"] = pred == gt
            if item["is_correct"]:
                correct += 1
        results.append(item)
    output = {"total": len(samples), "details": results}
    if has_label:
        output["correct"] = correct
        output["accuracy"] = correct / len(samples) if samples else 0.0
    return output

# -----------------------------
# 主流程
# -----------------------------
if __name__ == "__main__":
    file_path = r"D:\WorkSpace\python\flagOS赛题三\LongContext-ICL-Annotation\data\openseek-4_conala_concat_strings.json"
    out_path = r"D:\work_files\python_project\flagOS赛题三\LongContext-ICL-Annotation\rgs_q4\experiment\openseek4_collatz_results.json"
    test_jsonl_path = r"D:\work_files\python_project\flagOS赛题三\LongContext-ICL-Annotation\experiment\openseek-4-v1.jsonl"
    task_id, task_name, definition, examples, test_samples = load_task(file_path)
    print(f"任务: {task_id} / {task_name}")
    print(f"训练样例数: {len(examples)}, 测试样例数: {len(test_samples)}")

    generated_code, solve_func, preview_acc, preview_errors = generate_valid_solver(task_name, definition, examples, max_attempts=3)

    example_eval = run_predictions(solve_func, examples, has_label=True)
    test_eval = run_predictions(solve_func, test_samples, has_label=False)

    output_data = {
        "task_id": task_id,
        "task_name": task_name,
        "evaluate_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "generated_code": generated_code,
        "preview_accuracy_on_examples": preview_acc,
        "preview_errors": preview_errors[:20],
        "examples_eval": example_eval,
        "test_eval": test_eval
    }

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

    print(f"完整样例集准确率: {example_eval['accuracy']:.4f}")
    print(f"预测结果已保存到: {out_path}")

    os.makedirs(os.path.dirname(test_jsonl_path), exist_ok=True)
    with open(test_jsonl_path, 'w', encoding='utf-8') as f:
        for item in test_eval["details"]:
            line = json.dumps(item, ensure_ascii=False)
            f.write(line + '\n')

    print(f"测试集提交文件已保存为 JSONL: {test_jsonl_path}")