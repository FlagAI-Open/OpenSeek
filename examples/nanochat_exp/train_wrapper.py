"""
Wrapper script to run nanochat training with OpenSeek data.

This script provides a convenient way to run nanochat training using OpenSeek datasets
without needing to modify nanochat's source code directly.
"""

import os
import sys
import argparse
import subprocess

def find_nanochat():
    """Try to find nanochat installation."""
    # Check common locations
    possible_paths = [
        os.path.join(os.path.expanduser("~"), "nanochat"),
        os.path.join(os.path.dirname(__file__), "../../../nanochat"),
        "/opt/nanochat",
    ]
    
    # Check PYTHONPATH
    pythonpath = os.environ.get("PYTHONPATH", "").split(os.pathsep)
    possible_paths.extend(pythonpath)
    
    for path in possible_paths:
        scripts_path = os.path.join(path, "scripts")
        if os.path.exists(scripts_path):
            return path
    
    return None

def setup_environment():
    """Set up environment variables for OpenSeek nanochat."""
    data_dir = os.environ.get(
        "OPENSEEK_NANOCHAT_DATA_DIR",
        os.path.join(os.path.expanduser("~"), ".cache", "openseek_nanochat")
    )
    os.makedirs(data_dir, exist_ok=True)
    
    # Set nanochat base dir to use our parquet files
    os.environ["NANOCHAT_BASE_DIR"] = data_dir
    
    return data_dir

def check_parquet_files(data_dir):
    """Check if parquet files exist."""
    parquet_dir = os.path.join(data_dir, "parquet_shards")
    if not os.path.exists(parquet_dir):
        return False
    
    parquet_files = [f for f in os.listdir(parquet_dir) if f.endswith(".parquet")]
    return len(parquet_files) > 0

def main():
    parser = argparse.ArgumentParser(
        description="Run nanochat training with OpenSeek datasets"
    )
    parser.add_argument(
        "--nanochat-path",
        type=str,
        default=None,
        help="Path to nanochat directory"
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=20,
        help="Model depth (default: 20)"
    )
    parser.add_argument(
        "--nproc-per-node",
        type=int,
        default=8,
        help="Number of GPUs per node (default: 8)"
    )
    parser.add_argument(
        "--run",
        type=str,
        default="openseek",
        help="Wandb run name (default: openseek)"
    )
    parser.add_argument(
        "--device-batch-size",
        type=int,
        default=32,
        help="Device batch size (default: 32)"
    )
    parser.add_argument(
        "--check-data",
        action="store_true",
        help="Check if data is available before training"
    )
    parser.add_argument(
        "--convert-data",
        action="store_true",
        help="Convert dataset before training if not available"
    )
    
    args, extra_args = parser.parse_known_args()
    
    # Find nanochat
    nanochat_path = args.nanochat_path or find_nanochat()
    if not nanochat_path:
        print("Error: Could not find nanochat installation.")
        print("Please specify --nanochat-path or ensure nanochat is in PYTHONPATH")
        sys.exit(1)
    
    print(f"Using nanochat at: {nanochat_path}")
    
    # Setup environment
    data_dir = setup_environment()
    print(f"Data directory: {data_dir}")
    
    # Check data
    if args.check_data or args.convert_data:
        has_data = check_parquet_files(data_dir)
        if not has_data:
            if args.convert_data:
                print("Converting dataset...")
                from examples.nanochat_exp.dataset import convert_to_parquet, load_openseek_dataset
                dataset = load_openseek_dataset(streaming=True)
                # 对于示例数据集，使用 -1 处理所有数据（不限制 shards 数量）
                convert_to_parquet(dataset, max_shards=-1)
            else:
                print("Error: No parquet files found.")
                print("Please run dataset conversion first:")
                print("  python -m examples.nanochat_exp.dataset --dataset BAAI/OpenSeek-Pretrain-Data-Examples")
                sys.exit(1)
    
    # Modify sys.path to include nanochat
    if nanochat_path not in sys.path:
        sys.path.insert(0, nanochat_path)
    
    # Patch the dataloader import in nanochat scripts
    # This is a workaround - ideally nanochat scripts would be modified
    print("\nNote: You need to modify nanochat's base_train.py to use OpenSeek dataloader.")
    print("Change:")
    print("  from nanochat.dataloader import tokenizing_distributed_data_loader")
    print("to:")
    print("  from examples.nanochat_exp.dataloader import tokenizing_distributed_data_loader")
    print()
    
    # Build command
    script_path = os.path.join(nanochat_path, "scripts", "base_train.py")
    if not os.path.exists(script_path):
        print(f"Error: Could not find {script_path}")
        sys.exit(1)
    
    cmd = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={args.nproc_per_node}",
        "-m", "scripts.base_train",
        "--",
        f"--depth={args.depth}",
        f"--run={args.run}",
        f"--device_batch_size={args.device_batch_size}",
    ] + extra_args
    
    print("Running command:")
    print(" ".join(cmd))
    print()
    
    # Change to nanochat directory
    os.chdir(nanochat_path)
    
    # Run training
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"\nTraining failed with exit code {e.returncode}")
        sys.exit(e.returncode)

if __name__ == "__main__":
    main()

