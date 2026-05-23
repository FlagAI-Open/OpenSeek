import json
import os
import time
import re
import ast
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from functools import partial


def task1_data_loader(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return (
        data.get("task_id"),
        data.get("task_name"),
        data.get("Definition", []),
        data.get("examples", []),
        data.get("test_samples", [])
    )


client = OpenAI(
    api_key="dummy",  # 你的接口如果不需要密钥，填任意字符串即可
    base_url="https://flagos.io/flagos-lab/hw/node/HW-gpu57/port/22653/v1"
)


def qwen_api(messages, model="/Qwen3-4B/Qwen/Qwen3-4B", retries=3):
    for attempt in range(retries):
        try:
            res = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
            )
            return res.choices[0].message.content
        except Exception as e:
            if attempt == retries - 1:
                print(f"\nAPI 调用失败: {e}")
                return ""
            time.sleep(2)


CODEGEN_SYSTEM_PROMPT = """
你是一个严格的 Python 算法代码生成器。

你的任务：只输出可执行 Python 代码，不要解释，不要 markdown，不要 ``` 代码块，不要任何额外文字。

严格要求：
1. 只定义一个函数：solve(input_text: str) -> str
2. 不要写 import
3. 不要写 main
4. 不要写示例
5. 不要调用 open、eval、exec、compile、__import__
6. 已经预先提供了 ast 对象，你可以直接使用 ast.literal_eval
7. 输入是一个字符串，例如 "[59, 26, -96, -30]"
8. 输出必须是字符串，例如 "33"

任务目标：
给定一个整数列表，返回其中任意两个整数的最小绝对差。

提示：
- 先把字符串解析为整数列表
- 排序后，最小绝对差一定出现在相邻元素之间
- 若存在重复整数，答案就是 0
- 返回字符串类型
""".strip()


def build_codegen_user_prompt(task_id, task_name, definition_list, examples, max_examples=900):
    definition_text = "\n".join(f"- {item}" for item in definition_list)

    example_text_list = []
    for ex in examples[:max_examples]:
        example_text_list.append(
            f"输入: {ex['input']}\n输出: {ex['output'][0]}"
        )
    example_text = "\n\n".join(example_text_list)

    prompt = f"""
任务ID: {task_id}
任务名: {task_name}

任务定义:
{definition_text}

下面是若干样例:
{example_text}

请直接输出 solve(input_text: str) -> str 的完整 Python 代码。
再次强调：
- 不要 import
- 不要解释
- 只输出代码
""".strip()

    return prompt


def extract_code(text: str) -> str:
    if not text:
        return ""

    text = text.strip()

    fenced = re.findall(r"```(?:python)?\s*(.*?)```", text, flags=re.S | re.I)
    if fenced:
        return fenced[0].strip()

    return text


def postprocess_generated_code(code: str) -> str:
    """
    自动清洗模型生成代码：
    1. 删除 import / from 行
    2. 删除 markdown 残留
    3. 若前面有解释，只保留从 def solve 开始的代码
    """
    if not code:
        return ""

    code = code.replace("```python", "").replace("```", "").strip()
    lines = code.splitlines()

    cleaned = []
    for line in lines:
        s = line.strip()
        if s.startswith("import ") or s.startswith("from "):
            continue
        cleaned.append(line)

    code = "\n".join(cleaned).strip()

    match = re.search(r"def\s+solve\s*\(\s*input_text\s*:\s*str\s*\)\s*->\s*str\s*:", code)
    if match:
        code = code[match.start():].strip()

    return code


def is_code_safe(code: str):
    forbidden_patterns = [
        r"\bopen\s*\(",
        r"\beval\s*\(",
        r"\bexec\s*\(",
        r"\bcompile\s*\(",
        r"__import__",
        r"\bos\b",
        r"\bsys\b",
        r"\bsubprocess\b",
        r"\bpathlib\b",
        r"\bshutil\b",
        r"\bpickle\b",
    ]
    for pattern in forbidden_patterns:
        if re.search(pattern, code):
            return False, f"生成代码包含不允许内容: {pattern}"
    return True, ""


def compile_solver(code: str):
    ok, reason = is_code_safe(code)
    if not ok:
        raise ValueError(reason)

    safe_builtins = {
        "len": len,
        "min": min,
        "max": max,
        "sum": sum,
        "abs": abs,
        "str": str,
        "int": int,
        "float": float,
        "list": list,
        "dict": dict,
        "set": set,
        "tuple": tuple,
        "sorted": sorted,
        "range": range,
        "enumerate": enumerate,
        "zip": zip,
        "map": map,
        "filter": filter,
        "any": any,
        "all": all,
    }

    exec_globals = {
        "__builtins__": safe_builtins,
        "ast": ast,
    }
    exec_locals = {}

    exec(code, exec_globals, exec_locals)

    solve_func = exec_locals.get("solve") or exec_globals.get("solve")
    if solve_func is None:
        raise ValueError("生成代码中未找到 solve 函数")

    return solve_func


def fallback_solve(input_text: str) -> str:
    nums = ast.literal_eval(input_text.strip())
    nums = sorted(int(x) for x in nums)

    if len(nums) < 2:
        return "0"

    best = min(nums[i] - nums[i - 1] for i in range(1, len(nums)))
    return str(best)


def validate_solver(solve_func, examples, limit=200):
    eval_data = examples[:min(limit, len(examples))]
    total = len(eval_data)
    correct = 0
    error_cases = []

    for sample in eval_data:
        gt = str(sample["output"][0]).strip()
        try:
            pred = str(solve_func(sample["input"])).strip()
        except Exception as e:
            pred = f"[ERROR] {e}"

        if pred == gt:
            correct += 1
        else:
            error_cases.append({
                "id": sample.get("id", ""),
                "input": sample["input"],
                "gt": gt,
                "pred": pred
            })

    acc = correct / total if total > 0 else 0.0
    return acc, error_cases


def generate_valid_solver(task_id, task_name, definition_list, examples, max_try=3):
    user_prompt = build_codegen_user_prompt(
        task_id=task_id,
        task_name=task_name,
        definition_list=definition_list,
        examples=examples
    )

    best_code = ""
    best_acc = -1.0
    best_errors = []
    best_func = None

    for i in range(max_try):
        messages = [
            {"role": "system", "content": CODEGEN_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        raw_output = qwen_api(messages)
        code = extract_code(raw_output)
        code = postprocess_generated_code(code)

        try:
            solve_func = compile_solver(code)
        except Exception as e:
            print(f"第 {i + 1} 次代码编译失败: {e}")
            continue

        acc, errors = validate_solver(solve_func, examples, limit=200)
        print(f"第 {i + 1} 次代码生成，前 200 条样例准确率: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_code = code
            best_errors = errors[:20]
            best_func = solve_func

        if acc >= 1.0:
            return best_code, best_func, best_acc, best_errors

    if best_func is not None:
        return best_code, best_func, best_acc, best_errors

    return "", fallback_solve, 0.0, [{"error": "模型生成代码均未通过验证，已回退"}]


def process_single_sample(sample, solve_func, task_id):
    """
    只处理测试集单条样本，输出提交所需字段
    """
    sample_id = sample.get("id", "")
    input_text = sample["input"]

    try:
        pred = str(solve_func(input_text)).strip()
    except Exception as e:
        print(f"样本 {sample_id} 处理失败: {e}")
        pred = "0"

    return {
        "task_id": task_id,
        "sample_id": sample_id,
        "prediction": pred
    }


if __name__ == "__main__":
    file_path = r"D:\WorkSpace\python\flagOS赛题三\LongContext-ICL-Annotation\data\openseek-1_closest_integers.json"
    out_path = r"D:\WorkSpace\python\flagOS赛题三\LongContext-ICL-Annotation\rgs_q1\experiment\openseek-1-v1.jsonl"

    task_id, task_name, definition_list, examples, test_samples = task1_data_loader(file_path)

    print(f"任务: {task_id} / {task_name}")
    print(f"训练样例数: {len(examples)}")
    print(f"测试样例数: {len(test_samples)}")

    # 先生成并验证求解器
    generated_code, solve_func, preview_acc, preview_errors = generate_valid_solver(
        task_id=task_id,
        task_name=task_name,
        definition_list=definition_list,
        examples=examples,
        max_try=3
    )

    used_fallback = (solve_func == fallback_solve)

    if used_fallback:
        print("模型生成代码未通过验证，回退到内置保底求解器。")
    else:
        print("模型生成代码可用，继续处理测试集。")

    # 只处理测试集
    eval_data = test_samples
    total_cnt = len(eval_data)

    process_func = partial(process_single_sample, solve_func=solve_func, task_id=task_id)

    max_workers = 200
    results_list = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for result in tqdm(executor.map(process_func, eval_data), total=total_cnt, desc="Predicting"):
            results_list.append(result)

    # 生成指定 jsonl 格式
    output_data = [
        {
            "test_sample_id": item["sample_id"],
            "prediction": item["prediction"]
        }
        for item in results_list
    ]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, 'w', encoding='utf-8') as f:
        for item in output_data:
            line = json.dumps(item, ensure_ascii=False)
            f.write(line + '\n')

    print(f"预验证准确率(前200条examples): {preview_acc:.4f}")
    print(f"测试集结果已成功保存为 JSONL 格式：{out_path}")