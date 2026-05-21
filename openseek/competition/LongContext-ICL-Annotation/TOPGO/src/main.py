"""
FlagOS OpenSeek 赛道三 - TOPGO团队
主程序入口

运行方式:
    python main.py --task_id 1 --max_input_length 10000 --log_path_prefix ../outputs/
"""

import json
import os
import argparse
import time
import shutil
from tqdm import tqdm

# 设置国内镜像源（解决HuggingFace连接问题）
os.environ['HF_ENDPOINT'] = os.environ.get('HF_ENDPOINT', 'https://hf-mirror.com')

from transformers import AutoTokenizer

from method import (build_prompt, select_examples, annotate_ascend, annotate_nvidia, 
                     build_prompt_for_triton,
                     multi_agent_task2, multi_agent_task6, multi_agent_task8)


from typing import Optional as _Optional

def _deterministic_compute(task_id: int, raw_input: str) -> _Optional[str]:
    """
    确定性任务Python兜底：对于数学/规则类任务，直接用Python计算正确答案
    
    覆盖任务：
    - Task1: 最小绝对差（纯数学计算）
    - Task3: 一步Collatz变换（纯数学计算）
    - Task4: 字符串拼接（纯字符串操作）
    
    Returns:
        正确答案字符串，如果无法计算则返回None
    """
    import re as _re
    
    if task_id == 1:
        # Task1: 找列表中两个整数之间的最小绝对差
        try:
            numbers = eval(raw_input.strip())
            if not isinstance(numbers, list) or len(numbers) < 2:
                return None
            min_diff = float('inf')
            for i in range(len(numbers)):
                for j in range(i + 1, len(numbers)):
                    diff = abs(numbers[i] - numbers[j])
                    if diff < min_diff:
                        min_diff = diff
            return str(int(min_diff))
        except Exception:
            return None
    
    elif task_id == 3:
        # Task3: 对列表中每个元素应用一步Collatz变换
        try:
            numbers = eval(raw_input.strip())
            if not isinstance(numbers, list):
                return None
            result = []
            for x in numbers:
                if x % 2 == 0:
                    result.append(x // 2)
                else:
                    result.append(x * 3 + 1)
            return str(result)
        except Exception:
            return None
    
    elif task_id == 4:
        # Task4: 字符串拼接
        try:
            input_list = eval(raw_input.strip())
            if isinstance(input_list, list):
                return ''.join(str(item) for item in input_list)
        except Exception:
            pass
        # 如果eval失败，尝试用正则提取
        try:
            items = _re.findall(r"'([^']*)'", raw_input)
            if items:
                return ''.join(items)
        except Exception:
            pass
        return None
    
    return None


# 任务文件映射
TASK_FILES = {
    1: 'openseek-1_closest_integers.json',
    2: 'openseek-2_count_nouns_verbs.json',
    3: 'openseek-3_collatz_conjecture.json',
    4: 'openseek-4_conala_concat_strings.json',
    5: 'openseek-5_semeval_2018_task1_tweet_sadness_detection.json',
    6: 'openseek-6_mnli_same_genre_classification.json',
    7: 'openseek-7_jeopardy_answer_generation_all.json',
    8: 'openseek-8_kernel_generation.json',
}


def find_data_file(filename: str) -> str:
    """
    查找数据文件，支持多种路径
    
    Args:
        filename: 数据文件名
    
    Returns:
        找到的完整路径
    """
    # 获取项目根目录（main.py所在目录的父目录）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # 可能的数据目录路径
    data_dirs = [
        os.path.join(project_root, 'data'),           # 项目根目录/data
        os.path.join(script_dir, 'data'),             # src/data
        './data',                                      # 当前目录/data
        '../data',                                     # 上级目录/data
        '/home/data',                                  # 容器标准路径
        '/home/TOPGO/data',                           # 容器项目路径
        '/home/topgo-openseek/data',                  # 容器项目路径
    ]
    
    for data_dir in data_dirs:
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            return filepath
    
    # 如果都没找到，返回默认路径并报错
    return None


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='FlagOS OpenSeek 赛道三 - TOPGO团队')
    parser.add_argument('--task_id', type=int, required=True,
                        help='任务ID，范围 [1, 8]')
    parser.add_argument('--max_input_length', type=int, default=10000,
                        help='最大输入长度')
    parser.add_argument('--log_path_prefix', type=str, default='../outputs/',
                        help='输出文件前缀路径')
    parser.add_argument('--tokenizer_path', type=str, default=None,
                        help='Tokenizer路径（可选，支持本地路径或HuggingFace模型名）')
    parser.add_argument('--device', type=str, default='ascend',
                        choices=['ascend', 'nvidia'],
                        help='设备类型: ascend(华为) 或 nvidia(NVIDIA)')
    parser.add_argument('--resume', action='store_true',
                        help='断点续跑模式：跳过已完成的样本，从断点继续（默认开启）')
    parser.add_argument('--force-rerun', action='store_true',
                        help='强制重跑：清空已有结果，从头开始')
    return parser.parse_args()


def evaluate(task_id: int,
             qwen_tokenizer: AutoTokenizer,
             max_input_length: int = 10000,
             log_path_prefix: str = '../outputs/',
             device: str = 'ascend',
             resume: bool = True,
             force_rerun: bool = False):
    """
    执行标注任务

    Args:
        task_id: 任务ID (1-8)
        qwen_tokenizer: Tokenizer
        max_input_length: 最大输入长度
        log_path_prefix: 输出路径前缀
        device: 设备类型
        resume: 是否断点续跑（默认True）
        force_rerun: 是否强制重跑（默认False）
    """
    assert task_id in [i for i in range(1, 9)], \
        f"task_id should be in [1, 8], but got {task_id}."

    # 加载任务数据
    task_filename = TASK_FILES[task_id]
    task_file = find_data_file(task_filename)
    
    if task_file is None:
        print(f"[ERROR] 找不到数据文件: {task_filename}")
        print("[INFO] 请确保数据文件已放置在以下任一目录:")
        print("       - ./data/")
        print("       - ../data/")
        print("       - /home/TOPGO/data/")
        print("\n[INFO] 可以使用以下命令获取数据:")
        print("       git clone https://github.com/FlagAI-Open/OpenSeek.git OpenSeek_temp --depth 1")
        print("       mkdir -p data && cp OpenSeek_temp/openseek/competition/LongContext-ICL-Annotation/data/*.json data/")
        print("       rm -rf OpenSeek_temp")
        return
    
    print(f"[INFO] 加载任务数据: {task_file}")

    with open(task_file, 'r', encoding='utf-8') as f:
        task_dict = json.load(f)

    # 解析任务信息
    task_name = task_dict['task_name']
    task_description = task_dict['Definition'][0]
    icl_examples = task_dict['examples'][:100]  # 使用前100个示例
    test_samples = task_dict['test_samples']

    print(f"[INFO] 任务名称: {task_name}")
    print(f"[INFO] 任务描述: {task_description[:100]}...")
    print(f"[INFO] 示例数量: {len(icl_examples)}")
    print(f"[INFO] 测试样本数量: {len(test_samples)}")

    # 创建输出目录
    version = 1
    # 使用日期命名，避免覆盖旧结果
    date_str = time.strftime("%Y%m%d")
    output_file = f'{log_path_prefix}openseek-{task_id}-v{version}_{date_str}.jsonl'
    output_path = os.path.dirname(output_file)
    os.makedirs(output_path, exist_ok=True)

    # ====== 断点续跑核心逻辑 ======
    completed_ids = set()  # 已完成的sample id集合
    total_test_samples = len(test_samples)  # 记录总样本数
    
    if force_rerun:
        # 强制重跑模式：删除旧文件，从头开始
        if os.path.exists(output_file):
            backup_file = output_file.replace('.jsonl', f'_backup_{time.strftime("%Y%m%d_%H%M%S")}.jsonl')
            shutil.copy(output_file, backup_file)
            os.remove(output_file)
            print(f"[INFO] 强制重跑模式：已备份旧结果到 {backup_file}")
        print(f"[INFO] 输出文件: {output_file} (强制重跑)")
    elif resume and os.path.exists(output_file):
        # 断点续跑模式：读取已完成的结果，跳过已处理的样本
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            record = json.loads(line)
                            sid = record.get('test_sample_id', '')
                            pred = record.get('prediction')
                            # 只记录有效预测为已完成
                            if sid and pred is not None:
                                completed_ids.add(sid)
                        except json.JSONDecodeError:
                            pass
            
            if completed_ids:
                print(f"[INFO] 断点续跑模式: 发现 {len(completed_ids)} 个已完成样本")
                # 检查是否所有样本都已完成
                if len(completed_ids) >= total_test_samples:
                    print(f"[INFO] 所有 {total_test_samples} 个样本已完成，无需重新运行!")
                    print(f"[INFO] 输出文件: {output_file}")
                    # 验证最终行数
                    with open(output_file, 'r', encoding='utf-8') as f:
                        final_lines = sum(1 for _ in f if _.strip())
                    print(f"[INFO] 文件总行数: {final_lines}/{total_test_samples}")
                    return  # 直接返回，不继续处理
                print(f"[INFO] 输出文件: {output_file} (追加模式，不覆盖已有结果)")
            else:
                print(f"[INFO] 输出文件: {output_file} (文件存在但无有效记录，重新运行)")
        except Exception as e:
            print(f"[WARN] 读取已有结果失败: {e}, 将从头开始")
    else:
        print(f"[INFO] 输出文件: {output_file} (新建)")

    # 判断是否为Triton任务（任务8）
    is_triton_task = (task_id == 8)
    if is_triton_task:
        print("[INFO] 检测到Triton代码生成任务，启用优化模式")
    
    # 所有任务都使用专用提示词
    print(f"[INFO] 任务{task_id}使用专用优化提示词")

    # 选择标注函数
    if device == 'ascend':
        annotate = annotate_ascend
    else:
        annotate = annotate_nvidia

    # 预选示例
    print(f"[INFO] 开始处理...")
    examples_str = None

    # 处理每个测试样本（支持断点续跑）
    skipped_count = 0
    processed_count = 0
    
    for test_sample in tqdm(test_samples, desc=f'任务 {task_id}: {task_name}'):
        test_sample_id = test_sample['id']
        
        # 断点续跑：跳过已完成的样本
        if resume and test_sample_id in completed_ids:
            skipped_count += 1
            continue
        
        test_record = dict()
        test_record['test_sample_id'] = test_sample_id

        text2annotate = test_sample['input']
        processed_count += 1

        # 构建提示词（针对不同任务使用专用提示词）
        if is_triton_task:
            prompt = build_prompt_for_triton(task_description, text2annotate)
        else:
            # 使用优化版提示词，传入task_id
            prompt = build_prompt(task_description, text2annotate, task_id=task_id)

        # 选择示例（只执行一次，传入task_id用于优化示例选择）
        if examples_str is None:
            examples_str = select_examples(icl_examples, task_description, text2annotate, qwen_tokenizer, task_id=task_id)

        # 组装完整输入
        input_prompt = prompt.replace("[[EXAMPLES]]\n\n", examples_str + '\n\n')

        # 执行标注
        # V12.3: 只有T8走多智能体(代码生成需要特殊处理)，T2/T6回滚到标准annotate_ascend路径
        # 原因: T2/T6走_call_llm没有ICL示例+没有10次重试，分数反而下降
        try:
            if is_triton_task:
                # Task 8: 多智能体代码生成（V20: 注入ICL示例 + 自动重试 + 伪码检测 + 质量校验）
                print(f"[INFO] Task 8 using multi-agent pipeline (V20 with ICL)...")
                prediction = multi_agent_task8(task_description, text2annotate,
                                                icl_examples=icl_examples)
            else:
                # T2/T6及其他任务: 使用标准annotate_ascend（带ICL + 10次重试 + 600s超时）
                prediction = annotate(input_prompt, task_id=task_id)
            
            # V8确定性兜底：对Task1/3/4，Python直接计算正确答案
            # 不管模型输出什么，确定性任务的答案可以精确计算
            if task_id in [1, 3, 4]:
                correct_answer = _deterministic_compute(task_id, text2annotate)
                if correct_answer is not None:
                    prediction = correct_answer
            
            # V10: Task 5 关键词后处理校正（在main.py中调用以传入原始文本）
            # 这里不直接修改prediction，因为关键词校正在extract_task5_answer内部完成
            # count_answer已经会调用extract_task5_answer，但需要传入original_text
            # 通过重新调用count_answer并传入原始文本来触发校正
            if task_id == 5 and prediction is not None:
                from method import extract_task5_answer as _t5_extract
                # 用原始文本进行关键词校正
                corrected = _t5_extract(prediction if isinstance(prediction, str) else str(prediction), original_text=text2annotate)
                if corrected is not None:
                    prediction = corrected
            
            test_record['prediction'] = prediction
        except Exception as e:
            print(f"[ERROR] 样本 {test_sample_id} 标注异常: {e}")
            test_record['prediction'] = None

        # 写入结果
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(test_record, ensure_ascii=False) + '\n')

    # 打印统计信息
    total_samples = len(test_samples)
    print(f"[INFO] 任务 {task_id} 完成!")
    print(f"[INFO] 总样本: {total_samples} | 跳过(已存在): {skipped_count} | 本次处理: {processed_count}")
    print(f"[INFO] 结果已保存到: {output_file}")
    
    # 验证最终行数
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            final_lines = sum(1 for _ in f if _.strip())
        print(f"[INFO] 文件总行数: {final_lines}/{total_samples}")
        
        # 如果行数超出，清理重复并发出警告
        if final_lines > total_samples:
            print(f"[WARN] 文件行数超出预期，开始清理重复样本...")
            # 读取所有记录，按样本ID去重（保留最后一个，即最新结果）
            records = {}
            with open(output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            record = json.loads(line)
                            sid = record.get('test_sample_id', '')
                            if sid:
                                records[sid] = record
                        except json.JSONDecodeError:
                            pass
            
            # 重新写入
            with open(output_file, 'w', encoding='utf-8') as f:
                for record in records.values():
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
            print(f"[INFO] 清理完成，最终行数: {len(records)}/{total_samples}")


def main():
    """主函数"""
    args = parse_args()

    print("=" * 60)
    print("FlagOS OpenSeek 赛道三 - TOPGO团队")
    print("长上下文自动数据标注系统")
    print("=" * 60)

    # 加载Tokenizer
    qwen_tokenizer = None
    
    # 优先级: 1.命令行指定路径 2.本地模型路径 3.HuggingFace镜像
    tokenizer_paths = []
    
    if args.tokenizer_path:
        tokenizer_paths.append(args.tokenizer_path)
    
    # 添加常见的本地模型路径
    local_paths = [
        '/home/Qwen3-4B',           # 容器环境标准路径
        '/home/models/Qwen3-4B',    # 备用路径
        '/root/Qwen3-4B',           # 备用路径
        '../Qwen3-4B',              # 相对路径
        './Qwen3-4B',               # 当前目录
    ]
    tokenizer_paths.extend(local_paths)

    # 注意：比赛规定禁止使用Qwen3-4B以外的模型
    # 已移除Qwen2.5-3B-Instruct回退选项
    # 如果所有本地路径都失败，将使用字符估算模式（不加载Tokenizer）

    for tokenizer_path in tokenizer_paths:
        try:
            # 检查路径是否存在
            if not os.path.exists(tokenizer_path):
                continue
            
            print(f"[INFO] 尝试加载Tokenizer: {tokenizer_path}")
            
            # 尝试多种加载方式
            # 方式1: 使用local_files_only
            try:
                qwen_tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_path, 
                    trust_remote_code=True,
                    local_files_only=True
                )
                print(f"[INFO] Tokenizer加载成功: {tokenizer_path}")
                break
            except Exception:
                pass
            
            # 方式2: 直接加载tokenizer文件
            try:
                tokenizer_file = os.path.join(tokenizer_path, 'tokenizer.json')
                if os.path.exists(tokenizer_file):
                    from tokenizers import Tokenizer
                    qwen_tokenizer = AutoTokenizer.from_pretrained(
                        tokenizer_path,
                        trust_remote_code=True
                    )
                    print(f"[INFO] Tokenizer加载成功: {tokenizer_path}")
                    break
            except Exception:
                pass
            
        except Exception as e:
            print(f"[WARN] 加载失败: {e}")
            continue
    
    if qwen_tokenizer is None:
        print("[WARN] 所有Tokenizer加载尝试均失败，使用字符估算模式")
        print("[INFO] 这不会影响标注功能，只是token计数不够精确")

    # 执行任务
    evaluate(
        task_id=args.task_id,
        qwen_tokenizer=qwen_tokenizer,
        max_input_length=args.max_input_length,
        log_path_prefix=args.log_path_prefix,
        device=args.device,
        resume=not args.force_rerun,  # 默认启用断点续跑
        force_rerun=args.force_rerun
    )

    print("=" * 60)
    print("运行完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()
