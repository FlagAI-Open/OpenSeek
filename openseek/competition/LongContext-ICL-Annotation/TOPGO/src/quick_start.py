#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FlagOS OpenSeek 赛道三 - 一键运行脚本
快速配置和运行标注系统
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def print_banner():
    """打印横幅"""
    banner = """
========================================
FlagOS OpenSeek 赛道三 - 自动标注系统
团队: TOPGO
========================================
"""
    print(banner)


def print_info(msg):
    """打印信息"""
    print(f"[INFO] {msg}")


def print_warn(msg):
    """打印警告"""
    print(f"[WARN] {msg}")


def print_error(msg):
    """打印错误"""
    print(f"[ERROR] {msg}")


def check_python():
    """检查Python版本"""
    print_info(f"Python版本: {sys.version}")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print_error("Python版本过低，需要 >= 3.8")
        return False
    
    print_info("Python版本检查通过")
    return True


def check_dependencies():
    """检查依赖"""
    print_info("检查依赖包...")
    
    required = ['torch', 'transformers', 'sentence_transformers', 'pandas', 'numpy']
    missing = []
    
    for pkg in required:
        try:
            __import__(pkg)
            print_info(f"  ✓ {pkg}")
        except ImportError:
            print_warn(f"  ✗ {pkg} (未安装)")
            missing.append(pkg)
    
    if missing:
        print_warn(f"缺少依赖包: {', '.join(missing)}")
        print_info("运行: pip install -r requirements.txt")
        return False
    
    return True


def install_dependencies():
    """安装依赖"""
    print_info("安装依赖包...")
    
    req_file = Path("requirements.txt")
    if req_file.exists():
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt",
            "-i", "https://pypi.tuna.tsinghua.edu.cn/simple"
        ])
        print_info("依赖安装完成")
    else:
        print_warn("requirements.txt不存在")


def check_data():
    """检查数据"""
    print_info("检查数据目录...")
    
    dirs = ['data/raw', 'data/processed', 'output/results', 'output/logs']
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
        print_info(f"  目录: {d} ✓")
    
    # 检查数据文件
    train_file = Path("data/raw/train.jsonl")
    test_file = Path("data/raw/test.jsonl")
    
    if train_file.exists():
        print_info(f"训练数据: {train_file} ✓")
    else:
        print_warn(f"训练数据: {train_file} 不存在")
        print_info("请从OpenSeek仓库获取数据")
    
    if test_file.exists():
        print_info(f"测试数据: {test_file} ✓")
    else:
        print_warn(f"测试数据: {test_file} 不存在")
        print_info("请从OpenSeek仓库获取数据")
    
    return train_file.exists() and test_file.exists()


def check_api():
    """检查API配置"""
    print_info("检查API配置...")
    
    api_base = os.getenv("QWEN_API_BASE", "")
    api_key = os.getenv("QWEN_API_KEY", "")
    
    if api_base:
        print_info(f"QWEN_API_BASE: {api_base}")
    else:
        print_warn("QWEN_API_BASE未设置")
        print_info("设置方式: export QWEN_API_BASE='您的API端点'")
    
    if api_key:
        print_info("QWEN_API_KEY: ******")
    else:
        print_warn("QWEN_API_KEY未设置")
        print_info("设置方式: export QWEN_API_KEY='您的API密钥'")
    
    return bool(api_base and api_key)


def run_demo():
    """运行演示"""
    print_info("运行演示模式...")
    
    demo_script = Path("scripts/demo.py")
    if demo_script.exists():
        subprocess.run([sys.executable, str(demo_script)])
    else:
        print_error("scripts/demo.py不存在")


def run_pipeline():
    """运行完整流程"""
    print_info("运行完整标注流程...")
    
    # 检查数据
    if not check_data():
        print_error("数据文件不存在，请先准备数据")
        return False
    
    # 检查API
    if not check_api():
        print_warn("API未配置，将使用模拟模式")
    
    # 运行Pipeline
    pipeline_script = Path("scripts/run_annotation.py")
    if pipeline_script.exists():
        cmd = [
            sys.executable, str(pipeline_script),
            "--config", "configs/config.yaml",
            "--train", "data/raw/train.jsonl",
            "--test", "data/raw/test.jsonl",
            "--output", "output/results/predictions.json"
        ]
        subprocess.run(cmd)
        print_info("预测结果已保存到: output/results/predictions.json")
    else:
        print_error("scripts/run_annotation.py不存在")
    
    return True


def interactive_menu():
    """交互式菜单"""
    while True:
        print("\n========================================")
        print("请选择操作:")
        print("========================================")
        print("1. 检查环境")
        print("2. 安装依赖")
        print("3. 配置API")
        print("4. 运行演示")
        print("5. 运行完整流程")
        print("6. 查看帮助")
        print("0. 退出")
        print("========================================")
        
        choice = input("请输入选项 [0-6]: ").strip()
        
        if choice == "1":
            check_python()
            check_dependencies()
            check_data()
            check_api()
        elif choice == "2":
            install_dependencies()
        elif choice == "3":
            api_base = input("请输入API端点: ").strip()
            api_key = input("请输入API密钥: ").strip()
            
            os.environ["QWEN_API_BASE"] = api_base
            os.environ["QWEN_API_KEY"] = api_key
            
            print_info("环境变量已设置（当前会话有效）")
        elif choice == "4":
            run_demo()
        elif choice == "5":
            run_pipeline()
        elif choice == "6":
            print("\n帮助信息:")
            print("  - Python版本要求: >= 3.8")
            print("  - 依赖包: torch, transformers, sentence-transformers")
            print("  - 数据格式: JSONL")
            print("  - 输出格式: JSON")
            print("\n更多信息请查看 README.md 和 环境配置说明.md")
        elif choice == "0":
            print_info("退出程序")
            break
        else:
            print_error("无效选项")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="FlagOS OpenSeek 赛道三 - 快速启动")
    parser.add_argument("command", nargs="?", choices=["demo", "run", "install", "check", "menu"],
                       default="menu", help="执行命令")
    parser.add_argument("--train", type=str, help="训练数据路径")
    parser.add_argument("--test", type=str, help="测试数据路径")
    parser.add_argument("--output", type=str, default="output/results/predictions.json", help="输出文件路径")
    
    args = parser.parse_args()
    
    print_banner()
    
    if args.command == "demo":
        run_demo()
    elif args.command == "run":
        run_pipeline()
    elif args.command == "install":
        install_dependencies()
    elif args.command == "check":
        check_python()
        check_dependencies()
        check_data()
        check_api()
    else:
        interactive_menu()


if __name__ == "__main__":
    main()
