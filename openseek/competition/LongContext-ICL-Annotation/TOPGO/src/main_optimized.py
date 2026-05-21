"""
FlagOS OpenSeek 赛道三 - TOPGO团队
主程序入口（优化版 v2.0）

改进内容:
1. 使用优化版的method模块
2. 传递task_id以使用任务特定的策略
3. 改进错误处理和日志
4. 支持Self-Consistency模式

运行方式:
    python main_optimized.py --task_id 4 --max_input_length 10000 --log_path_prefix ../outputs/
    python main_optimized.py --task_id 7 --use_consistency --consistency_samples 3
"""

import json
import os
import argparse
from tqdm import tqdm

# 设置国内镜像源（解决HuggingFace连接问题）
os.environ['HF_ENDPOINT'] = os.environ.get('HF_ENDPOINT', 'https://hf-mirror.com')

from transformers import AutoTokenizer

# 使用优化版的method模块
try:
    from method_optimized import build_prompt, select_examples, annotate_ascend, annotate_nvidia, build_prompt_for_triton, annotate_with_consistency
    print("[INFO] 使用优化版method模块")
except ImportError:
    from method import build_prompt, select_examples, annotate_ascend, annotate_nvidia, build_prompt_for_triton
    print("[WARN] 优化版method模块加载失败，使用原始模块")
    annotate_with_consistency = None


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
    parser = argparse.ArgumentParser(description='FlagOS OpenSeek 赛道三 - TOPGO团队（优化版）')
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
    parser.add_argument('--use_consistency', action='store_true',
                        help='使用Self-Consistency模式（多次采样取最一致答案）')
    parser.add_argument('--consistency_samples', type=int, default=3,
                        help='Self-Consistency采样次数（默认3）')
    parser.add_argument('--use_optimized', action='store_true', default=True,
                        help='使用优化版方法（默认启用）')
    return parser.parse_args()


def evaluate(task_id: int,
             qwen_tokenizer: AutoTokenizer,
             max_input_length: int = 10000,
             log_path_prefix: str = '../outputs/',
             device: str = 'ascend',
             use_consistency: bool = False,
             consistency_samples: int = 3):
    """
    执行标注任务（优化版）

    Args:
        task_id: 任务ID (1-8)
        qwen_tokenizer: Tokenizer
        max_input_length: 最大输入长度
        log_path_prefix: 输出路径前缀
        device: 设备类型
        use_consistency: 是否使用Self-Consistency
        consistency_samples: Self-Consistency采样次数
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
    print(f"[INFO] 使用Self-Consistency: {use_consistency}")

    # 创建输出目录
    version = 1
    output_file = f'{log_path_prefix}openseek-{task_id}-v{version}-optimized.jsonl'
    output_path = os.path.dirname(output_file)
    os.makedirs(output_path, exist_ok=True)

    # 检查是否已有结果文件
    while os.path.exists(output_file):
        version += 1
        output_file = f'{log_path_prefix}openseek-{task_id}-v{version}-optimized.jsonl'

    # 创建空文件
    with open(output_file, 'w', encoding='utf-8') as f:
        pass

    print(f"[INFO] 输出文件: {output_file}")

    # 判断是否为Triton任务（任务8）
    is_triton_task = (task_id == 8)
    if is_triton_task:
        print("[INFO] 检测到Triton代码生成任务，启用优化模式")

    # 选择标注函数
    if device == 'ascend':
        annotate_func = annotate_ascend
    else:
        annotate_func = annotate_nvidia

    # 预选示例（传入task_id以使用任务特定的筛选）
    print(f"[INFO] 开始选择高质量示例...")
    examples_str = select_examples(icl_examples, task_description, "", qwen_tokenizer, task_id=task_id)
    print(f"[INFO] 选择了 {examples_str.count('输入:') + examples_str.count('问题:') + examples_str.count('# ')} 个示例")

    # 统计信息
    stats = {
        'total': len(test_samples),
        'success': 0,
        'failed': 0,
        'null_predictions': 0
    }

    # 处理每个测试样本
    for test_sample in tqdm(test_samples, desc=f'任务 {task_id}: {task_name}'):
        test_record = dict()

        test_sample_id = test_sample['id']
        test_record['test_sample_id'] = test_sample_id

        text2annotate = test_sample['input']

        # 构建提示词（传入task_id）
        if is_triton_task:
            prompt = build_prompt_for_triton(task_description, text2annotate)
        else:
            prompt = build_prompt(task_description, text2annotate, task_id=task_id)

        # 组装完整输入
        if examples_str:
            input_prompt = prompt.replace("[[EXAMPLES]]\n\n", examples_str + '\n\n')
        else:
            input_prompt = prompt

        # 执行标注
        try:
            if use_consistency and annotate_with_consistency:
                # 使用Self-Consistency模式
                prediction = annotate_with_consistency(input_prompt, task_id=task_id, num_samples=consistency_samples)
            else:
                # 普通模式
                if is_triton_task:
                    prediction = annotate_func(input_prompt, task_id=task_id, is_triton_task=True)
                else:
                    prediction = annotate_func(input_prompt, task_id=task_id)
            
            test_record['prediction'] = prediction
            
            if prediction is None:
                stats['null_predictions'] += 1
            else:
                stats['success'] += 1
                
        except Exception as e:
            print(f"[WARN] 样本 {test_sample_id} 标注失败: {e}")
            test_record['prediction'] = None
            stats['failed'] += 1

        # 写入结果
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(test_record, ensure_ascii=False) + '\n')

    # 打印统计
    print(f"\n[INFO] 任务 {task_id} 完成!")
    print(f"[INFO] 总样本: {stats['total']}")
    print(f"[INFO] 成功: {stats['success']}")
    print(f"[INFO] 失败: {stats['failed']}")
    print(f"[INFO] Null预测: {stats['null_predictions']}")
    print(f"[INFO] 有效率: {(stats['success'] / stats['total'] * 100):.1f}%")
    print(f"[INFO] 结果已保存到: {output_file}")


def main():
    """主函数"""
    args = parse_args()

    print("=" * 60)
    print("FlagOS OpenSeek 赛道三 - TOPGO团队（优化版）")
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
            print(f"[INFO] 尝试加载Tokenizer: {tokenizer_path}")
            qwen_tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path, 
                trust_remote_code=True,
                resume_download=True
            )
            print(f"[INFO] Tokenizer加载成功: {tokenizer_path}")
            break
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
        use_consistency=args.use_consistency,
        consistency_samples=args.consistency_samples
    )

    print("=" * 60)
    print("运行完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()
