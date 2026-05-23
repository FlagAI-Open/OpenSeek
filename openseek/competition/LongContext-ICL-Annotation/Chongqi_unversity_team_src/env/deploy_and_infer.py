#!/usr/bin/env python3
"""
FlagOS 赛题三: Ascend 910C × 2 服务器部署脚本
================================================
功能:
  - 在华为 Ascend 910C 服务器上部署 FlagScale + vLLM/MindIE 推理服务
  - 模型: Qwen3-4B
  - 容器内端口: 30000
  - 公网映射端口: 22653
  - 外部调用地址: https://flagos.io/flagos-lab/hw/node/HW-gpu57/port/22653/v1

底层框架: FlagScale
运行设备: 华为 Ascend 910C × 2 (NPU)
容器端口: 30000 (映射到公网 22653)
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml


# ============================================================================
# 配置常量
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent          # env/ 目录
PROJECT_ROOT = SCRIPT_DIR.parent                       # LongContext-ICL-Annotation/

# --------------------------------------------------------------------------
# FlagScale 相关文件全部放在 env/ 下
# --------------------------------------------------------------------------
FLAGSCALE_REPO = SCRIPT_DIR / "FlagScale"              # env/FlagScale/
FLAGSCALE_RUN_PY = FLAGSCALE_REPO / "run.py"           # env/FlagScale/run.py

# FlagScale 配置文件放在 env/ 下
CONFIG_PATH = SCRIPT_DIR / "llm_config_ascend.yaml"    # env/llm_config_ascend.yaml

# --------------------------------------------------------------------------
# 服务器与网络配置
# --------------------------------------------------------------------------
# 容器内监听端口 (FlagScale / vLLM 实际监听的端口)
CONTAINER_PORT = 30000
# 公网映射端口 (由 lab 平台自动映射)
PUBLIC_PORT = 22653
# 公网服务地址
PUBLIC_API_URL = f"https://flagos.io/flagos-lab/hw/node/HW-gpu57/port/{PUBLIC_PORT}/v1"
PUBLIC_HEALTH_URL = f"https://flagos.io/flagos-lab/hw/node/HW-gpu57/port/{PUBLIC_PORT}/health"

# --------------------------------------------------------------------------
# 模型配置 (模型权重存放在 env/ 目录下)
# --------------------------------------------------------------------------
MODEL_REPO_ID = "Qwen/Qwen3-4B"
LOCAL_MODEL_DIR = SCRIPT_DIR / "Qwen3-4B"              # env/Qwen3-4B/

# --------------------------------------------------------------------------
# Timeout / 重试配置
# --------------------------------------------------------------------------
STARTUP_TIMEOUT = 300         # 服务启动最大等待时间 (秒)
CHECK_INTERVAL = 5            # 健康检查间隔 (秒)
MAX_RETRIES = 3               # API 重试次数
RETRY_DELAY = 3               # 重试间隔 (秒)


# ============================================================================
# 工具函数
# ============================================================================

def print_banner():
    """打印程序标题横幅"""
    banner = r"""
╔══════════════════════════════════════════════════════════╗
║   FlagOS Long-Context ICL Annotation                     ║
║   Server Deployment: Ascend 910C × 2 + FlagScale        ║
║   Container Port: 30000 → Public Port: 22653            ║
╚══════════════════════════════════════════════════════════╝
"""
    print(banner)


def generate_ascend_config(container_port=None, model_dir=None):
    """
    生成适用于华为 Ascend 910C 的 FlagScale 配置文件。

    Args:
        container_port: 容器内监听端口，默认 30000
        model_dir:      模型本地路径
    """
    if container_port is None:
        container_port = CONTAINER_PORT
    if model_dir is None:
        model_dir = str(LOCAL_MODEL_DIR)

    config = {
        "serve": [
            {
                "serve_id": "ascend_vllm_model",
                "engine": "mindie",          # MindIE Engine (Ascend 推荐)
                # 如果使用的是昇腾适配版 vLLM，改为 "vllm-ascend"
                "engine_args": {
                    "model": model_dir,     # 模型权重路径
                    "host": "0.0.0.0",
                    "port": container_port, # 容器内端口
                    "num_gpus": 2,          # Ascend 910C 卡数
                    "gpu_memory_utilization": 0.9,
                    "trust_remote_code": True,
                    "no_enable_prefix_caching": True,
                    # --- Ascend 特有参数 ---
                    "device_type": "ascend",
                    "npu_device_ids": "0,1",  # Ascend NPU 卡号
                },
            }
        ],
        "experiment": {
            "exp_name": "qwen3_4b_ascend",
            "exp_dir": "outputs/${experiment.exp_name}",
            "task": {"type": "serve"},
            "runner": {
                "hostfile": None,
                "deploy": {"use_fs_serve": False},
            },
            "envs": {
                "ASCEND_RT_VISIBLE_DEVICES": "0,1",  # Ascend 可见 NPU 卡
                "ASCEND_DEVICE_MAX_CONNECTIONS": "1",
            },
        },
        "action": "run",
        "hydra": {
            "run": {"dir": "${experiment.exp_dir}/hydra"},
        },
    }

    # 写入文件
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"[OK] Ascend 配置文件已生成: {CONFIG_PATH}")
    return CONFIG_PATH


def read_original_config():
    """读取原始 llm_config.yaml 作为参考"""
    if not ORIGINAL_CONFIG_PATH.exists():
        return None
    try:
        with open(ORIGINAL_CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"[WARN] 无法读取原始配置: {e}")
        return None


def check_ascend_env():
    """检查 Ascend NPU 环境是否就绪"""
    import platform
    print("\n=== Ascend 环境检测 ===")

    # 检查 CANN 版本
    cann_version = os.environ.get("CANN_VERSION", "")
    if cann_version:
        print(f"  [OK] CANN 版本: {cann_version}")
    else:
        print("  [WARN] 未检测到 CANN_VERSION 环境变量")

    # 检查 Ascend 设备
    try:
        result = subprocess.run(
            ["npu-smi", "info"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            print(f"  [OK] npu-smi 可用")
            # 统计设备数量
            output_lines = result.stdout.strip().split("\n")
            device_count = sum(1 for line in output_lines if "Total Count" in line)
            print(f"       Ascend 设备总数: {device_count}")
            return True
        else:
            print(f"  [ERROR] npu-smi 返回错误: {result.stderr}")
            return False
    except FileNotFoundError:
        print("  [ERROR] npu-smi 命令不存在")
        print("         请确认已安装 Ascend 驱动和 CANN 工具包")
        return False
    except Exception as e:
        print(f"  [ERROR] 检测设备异常: {e}")
        return False


def download_model(repo_id=None, local_dir=None):
    """从 HuggingFace 下载模型权重到本地"""
    if repo_id is None:
        repo_id = MODEL_REPO_ID
    if local_dir is None:
        local_dir = LOCAL_MODEL_DIR

    local_dir = Path(local_dir)
    if local_dir.exists() and (local_dir / "config.json").exists():
        print(f"[INFO] 模型已存在: {local_dir}")
        return True

    print(f"[INFO] 下载模型: {repo_id}")
    print(f"       目标目录: {local_dir}")
    print(f"       大小约: ~8GB")

    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            resume_download=True,
        )
        print(f"[OK] 模型下载完成: {local_dir}")
        return True
    except ImportError:
        print("[ERROR] 缺少 huggingface_hub，请安装:")
        print("  pip install huggingface-hub")
        return False
    except Exception as e:
        print(f"[ERROR] 模型下载失败: {e}")
        return False


def setup_long_context_fix(model_dir=None):
    """
    修改模型 config.json 以支持长上下文 (rope_scaling yarn)
    
    将 rope_type 从 "default" 改为 "yarn"，factor=4.0，
    original_max_position_embeddings=32768
    """
    if model_dir is None:
        model_dir = LOCAL_MODEL_DIR

    config_json = Path(model_dir) / "config.json"
    if not config_json.exists():
        print(f"[ERROR] 找不到模型配置: {config_json}")
        return False

    try:
        with open(config_json, "r", encoding="utf-8") as f:
            model_config = json.load(f)

        old_scaling = model_config.get("rope_scaling")
        new_scaling = {
            "rope_type": "yarn",
            "factor": 4.0,
            "original_max_position_embeddings": 32768,
        }

        if old_scaling == new_scaling:
            print("[INFO] rope_scaling 已经是目标配置，无需修改")
            return True

        model_config["rope_scaling"] = new_scaling
        with open(config_json, "w", encoding="utf-8") as f:
            json.dump(model_config, f, indent=2, ensure_ascii=False)

        print(f"[OK] 已更新 rope_scaling 为 yarn 模式:")
        print(f"       {new_scaling}")
        return True
    except Exception as e:
        print(f"[ERROR] 修改 config.json 失败: {e}")
        return False


def start_service(config_path=None):
    """使用 FlagScale 启动 Ascend 推理服务"""
    if config_path is None:
        config_path = CONFIG_PATH

    # 如果没有 Ascend 专用配置，先生成一份
    if not config_path.exists():
        print("[INFO] 未找到 Ascend 配置文件，自动生成...")
        generate_ascend_config()
        config_path = CONFIG_PATH

    # 验证配置文件
    if not config_path.exists():
        print(f"[ERROR] 配置文件不存在: {config_path}")
        return False

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        print(f"[OK] 成功加载配置文件: {config_path}")
        print(f"       服务ID: {config['serve'][0]['serve_id']}")
        engine = config['serve'][0].get('engine', 'unknown')
        print(f"       引擎:   {engine} (Ascend)")
        print(f"       端口:   {config['serve'][0]['engine_args']['port']} (容器内)")
    except Exception as e:
        print(f"[ERROR] 配置文件解析失败: {e}")
        return False

    # 检查 FlagScale run.py
    if not FLAGSCALE_RUN_PY.exists():
        print(f"[ERROR] FlagScale run.py 不存在: {FLAGSCALE_RUN_PY}")
        print("请确认 FlagScale 仓库已克隆到项目根目录下。")
        return False

    print(f"\n[INFO] 启动 FlagScale 服务 (Ascend 910C × 2)...")
    print(f"       配置文件: {config_path}")
    print(f"       运行脚本: {FLAGSCALE_RUN_PY}")
    print(f"       容器端口: {CONTAINER_PORT}")
    print(f"       公网端口: {PUBLIC_PORT}")
    print(f"       公网地址: {PUBLIC_API_URL}")

    cmd = [
        sys.executable, str(FLAGSCALE_RUN_PY),
        "--config-path", "..",
        "--config-name", "llm_config_ascend",  # 使用 ascend 专属配置名
        "action=run",
    ]

    def on_exit(signum=None, frame=None):
        """退出时清理服务"""
        if process.poll() is None:
            print("\n[INFO] 正在关闭 FlagScale 服务...")
            try:
                stop_command = [
                    sys.executable, str(FLAGSCALE_RUN_PY),
                    "--config-path", ".",
                    "--config-name", "llm_config_ascend",
                    "action=stop",
                ]
                subprocess.run(stop_command, cwd=str(SCRIPT_DIR), timeout=30)
                print("[OK] 服务已关闭")
            except Exception as e:
                print(f"[WARN] 关闭服务失败: {e}")
                process.kill()

    process = subprocess.Popen(
        cmd,
        cwd=str(SCRIPT_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    signal.signal(signal.SIGINT, on_exit)
    signal.signal(signal.SIGTERM, on_exit)

    # 等待服务启动
    print(f"\n[INFO] 等待 FlagScale 服务启动 (超时 {STARTUP_TIMEOUT} 秒)...")
    start_time = time.time()

    while time.time() - start_time < STARTUP_TIMEOUT:
        # 先检查本地容器端口
        try:
            resp = requests.get(
                f"http://0.0.0.0:{CONTAINER_PORT}/health",
                timeout=3
            )
            if resp.status_code == 200:
                print(f"[OK] 容器内服务已就绪 (端口 {CONTAINER_PORT})")
                break
        except (requests.ConnectionError, requests.Timeout):
            pass

        elapsed = int(time.time() - start_time)
        if elapsed % CHECK_INTERVAL == 0:
            print(f"  ...已等待 {elapsed}s，服务尚未就绪")
        time.sleep(CHECK_INTERVAL)
    else:
        print(f"[ERROR] 服务启动超时 ({STARTUP_TIMEOUT} 秒)")
        process.kill()
        return False

    # 输出公网地址信息
    print(f"\n{'='*60}")
    print("[OK] FlagScale 服务已启动!")
    print(f"{'='*60}")
    print(f"容器内地址: http://0.0.0.0:{CONTAINER_PORT}")
    print(f"公网映射地址: {PUBLIC_API_URL}")
    print(f"{"-"*60}")
    print("[调用示例]")
    print(f"  client = OpenAI(")
    print(f"      api_key=\"dummy\",")
    print(f'      base_url="{PUBLIC_API_URL.rstrip("/v1")}"')
    print(f"  )")
    print(f"  response = client.chat.completions.create(")
    print(f'      model="/Qwen3-4B/Qwen/Qwen3-4B",')
    print(f'      messages=[{{"role": "user", "content": "Hello"}}],')
    print(f"      max_tokens=10000,")
    print(f"  )")
    print(f"{'='*60}")

    return True


def stop_service():
    """停止 FlagScale 推理服务"""
    print("[INFO] 正在停止 FlagScale 服务...")

    if not FLAGSCALE_RUN_PY.exists():
        print("[ERROR] FlagScale run.py 不存在")
        return False

    try:
        cmd = [
            sys.executable, str(FLAGSCALE_RUN_PY),
            "--config-path", ".",
            "--config-name", "llm_config_ascend",
            "action=stop",
        ]
        result = subprocess.run(
            cmd,
            cwd=str(SCRIPT_DIR),
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            print("[OK] FlagScale 服务已停止")
            return True
        else:
            print(f"[WARN] 停止服务非零退出码: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("[ERROR] 停止服务超时")
        return False
    except Exception as e:
        print(f"[ERROR] 停止服务异常: {e}")
        return False


def test_api(api_url=None):
    """测试公网 API 是否正常响应"""
    if api_url is None:
        api_url = PUBLIC_API_URL

    clean_url = api_url.rstrip("/v1")
    health_url = f"{clean_url}/health"

    print(f"\n[INFO] 测试 API 连通性...")
    print(f"       地址: {api_url}")

    # 先试 health 端点
    try:
        resp = requests.get(health_url, timeout=10)
        if resp.status_code == 200:
            print(f"[OK] Health 端点正常: {health_url}")
    except requests.RequestException:
        pass

    # 再试 models 端点
    models_url = f"{clean_url}/v1/models"
    try:
        resp = requests.get(models_url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            models = data.get("data", [])
            model_names = [m["id"] for m in models] if models else ["unknown"]
            print(f"[OK] Models 端点正常: {models_url}")
            print(f"       可用模型: {model_names}")
            return True
    except requests.RequestException as e:
        print(f"[WARN] Models 端点不可达: {e}")

    print(f"[WARN] 远程 API 可能未完全就绪，请稍后重试")
    return False


def send_test_request(api_url=None, model_name=None):
    """发送一条测试请求"""
    if api_url is None:
        api_url = PUBLIC_API_URL
    if model_name is None:
        model_name = "/Qwen3-4B/Qwen/Qwen3-4B"

    from openai import OpenAI
    client = OpenAI(
        api_key="dummy",
        base_url=api_url.rstrip("/v1"),
    )

    prompt = "Tell me a one-sentence joke."
    print(f"\n[INFO] 发送测试请求...")
    print(f"       Prompt: {prompt}")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                top_p=0.95,
                max_tokens=128,
                stream=False,
            )
            result = response.choices[0].message.content
            print(f"[OK] 测试通过")
            print(f"       生成内容: {result.strip()[:100]}")
            return True
        except Exception as e:
            print(f"[WARN] 请求失败 (尝试 {attempt}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
            else:
                print("[ERROR] 测试请求最终失败")
                return False


# ============================================================================
# CLI 入口
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Ascend 910C × 2 FlagScale 部署管理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
用法示例:
  # 查看帮助
  python deploy_and_infer.py --help

  # 检测 Ascend 环境
  python deploy_and_infer.py check-env

  # 下载模型权重
  python deploy_and_infer.py download-model

  # 修复长上下文配置
  python deploy_and_infer.py fix-context

  # 生成 Ascend 配置文件
  python deploy_and_infer.py gen-config

  # 一键部署 (下载+配置+启动)
  python deploy_and_infer.py deploy

  # 启动服务 (假设前置工作已完成)
  python deploy_and_infer.py start

  # 测试 API
  python deploy_and_infer.py test-api

  # 停止服务
  python deploy_and_infer.py stop

  # 完整流程 (下载→修复→生成配置→启动→测试)
  python deploy_and_infer.py full
        """,
    )
    parser.add_argument(
        "command",
        choices=[
            "check-env", "download-model", "fix-context", "gen-config",
            "deploy", "start", "test-api", "send-test", "stop", "full",
        ],
        help="要执行的命令",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help=f"模型本地路径 (默认: {LOCAL_MODEL_DIR})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help=f"容器内端口 (默认: {CONTAINER_PORT})，需与 lab 平台一致",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default=None,
        help=f"公网 API 地址 (默认: {PUBLIC_API_URL})",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="/Qwen3-4B/Qwen/Qwen3-4B",
        help="模型名称 (默认: /Qwen3-4B/Qwen/Qwen3-4B)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print_banner()

    # --- check-env: 检测 Ascend 环境 ---
    if args.command == "check-env":
        ok = check_ascend_env()
        sys.exit(0 if ok else 1)

    # --- download-model: 下载模型 ---
    if args.command == "download-model":
        success = download_model()
        sys.exit(0 if success else 1)

    # --- fix-context: 修复长上下文配置 ---
    if args.command == "fix-context":
        success = setup_long_context_fix(args.model_dir)
        sys.exit(0 if success else 1)

    # --- gen-config: 生成 Ascend 配置 ---
    if args.command == "gen-config":
        config_path = generate_ascend_config(
            container_port=args.port,
            model_dir=args.model_dir,
        )
        print(f"\n[OK] 配置文件已生成，可使用以下命令启动服务:")
        print(f"  python deploy_and_infer.py start")

    # --- deploy: 部署 (启动服务) ---
    if args.command == "deploy":
        print("\n=== 第 1 步: 检测 Ascend 环境 ===")
        if not check_ascend_env():
            print("[ERROR] Ascend 环境异常，请先解决")
            sys.exit(1)

        print("\n=== 第 2 步: 下载模型权重 ===")
        if not download_model():
            print("[WARN] 模型下载跳过或失败")

        print("\n=== 第 3 步: 修复长上下文配置 ===")
        if not setup_long_context_fix(args.model_dir):
            print("[WARN] 长上下文配置修复失败，可能需要手动处理")

        print("\n=== 第 4 步: 生成 Ascend 配置文件 ===")
        config_path = generate_ascend_config(
            container_port=args.port,
            model_dir=args.model_dir,
        )

        print("\n=== 第 5 步: 启动 FlagScale 服务 ===")
        success = start_service(config_path)
        if success:
            print("\n\n[OK] 部署完成！服务可通过以下地址访问：")
            print(f"       {PUBLIC_API_URL}")
        sys.exit(0 if success else 1)

    # --- start: 启动服务 ---
    if args.command == "start":
        print("\n=== 步骤: 启动 FlagScale 服务 ===")
        # 确保配置存在
        if not CONFIG_PATH.exists():
            print("[INFO] 未找到配置文件，自动生成...")
            generate_ascend_config(
                container_port=args.port,
                model_dir=args.model_dir,
            )
        success = start_service()
        sys.exit(0 if success else 1)

    # --- stop: 停止服务 ---
    if args.command == "stop":
        print("\n=== 步骤: 停止 FlagScale 服务 ===")
        success = stop_service()
        sys.exit(0 if success else 1)

    # --- test-api: 测试 API (仅检查连通性) ---
    if args.command == "test-api":
        api_url = args.api_url or PUBLIC_API_URL
        ok = test_api(api_url)
        sys.exit(0 if ok else 1)

    # --- send-test: 发送实际测试请求 ---
    if args.command == "send-test":
        api_url = args.api_url or PUBLIC_API_URL
        test_ok = test_api(api_url)
        if test_ok:
            send_test_request(api_url, args.model_name)
        else:
            print("[WARN] 服务可能未就绪，仍尝试发送请求...")
            send_test_request(api_url, args.model_name)

    # --- full: 全流程 ---
    if args.command == "full":
        print("\n========================================")
        print("   FlagScale Ascend 全自动部署")
        print("========================================\n")

        print("=== 第 1 步: 检测 Ascend 环境 ===")
        if not check_ascend_env():
            print("[ERROR] Ascend 环境异常")
            sys.exit(1)

        print("\n=== 第 2 步: 下载模型权重 ===")
        if not download_model():
            print("[WARN] 模型下载跳过或失败")

        print("\n=== 第 3 步: 修复长上下文配置 ===")
        if not setup_long_context_fix(args.model_dir):
            print("[WARN] 长上下文配置修复跳过")

        print("\n=== 第 4 步: 生成 Ascend 配置文件 ===")
        config_path = generate_ascend_config(
            container_port=args.port,
            model_dir=args.model_dir,
        )

        print("\n=== 第 5 步: 启动 FlagScale 服务 ===")
        success = start_service(config_path)

        if success:
            print("\n=== 第 6 步: 测试 API ===")
            if args.api_url:
                test_api(args.api_url)
            else:
                test_api(PUBLIC_API_URL)

            print(f"\n{'='*60}")
            print(f"[OK] 全流程执行完毕！")
            print(f"{'='*60}")
            print(f"容器内地址: http://0.0.0.0:{CONTAINER_PORT}")
            print(f"公网地址:   {PUBLIC_API_URL}")
            print(f"{'='*60}")
        else:
            print("[ERROR] 全流程部分步骤失败")
            sys.exit(1)


if __name__ == "__main__":
    main()
