import subprocess
import os
import sys
import time

def run_task(task_file):
    """运行单个 Python 脚本任务，并实时输出结果。"""
    print(f"\n{'='*30}")
    print(f"🚀 正在启动任务: {task_file}")
    print(f"{'='*30}\n")
    
    start_time = time.time()
    try:
        # 使用 sys.executable 确保使用当前的 Python 环境
        # stderr=subprocess.STDOUT 将错误输出也合并到 stdout 中实时显示
        process = subprocess.Popen(
            [sys.executable, task_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            encoding='utf-8'
        )
        
        # 实时读取并打印输出
        if process.stdout:
            for line in process.stdout:
                print(line, end='')
        
        process.wait()
        duration = time.time() - start_time
        
        if process.returncode == 0:
            print(f"\n✅ [成功] {task_file} 已完成。耗时: {duration:.2f}s")
        else:
            print(f"\n❌ [失败] {task_file} 退出代码: {process.returncode}。耗时: {duration:.2f}s")
            
    except Exception as e:
        print(f"\n🚨 [异常] 无法运行 {task_file}: {e}")

def main():
    # 待执行的任务列表
    task_map = {
        "1": "1.py",
        "2": "2.py",
        "3": "3.py",
        "4": "4.py",
        "5": "5.py",
        "6": "6.py",
        "7": "7.py",
        "8": "8.py"
    }
    
    # 确保在 src 目录下运行
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print("可用任务: " + " ".join(task_map.keys()))
    user_input = input("请输入要执行的任务编号 (例如 '1 2 3' 或 'all'): ").strip().lower()
    
    if user_input == 'all':
        selected_tasks = list(task_map.values())
    else:
        indices = user_input.split()
        selected_tasks = [task_map[i] for i in indices if i in task_map]

    if not selected_tasks:
        print("未选择有效任务，退出。")
        return

    print(f"开始执行任务: {', '.join(selected_tasks)}...")
    total_start = time.time()
    
    for task in selected_tasks:
        if os.path.exists(task):
            run_task(task)
        else:
            print(f"⚠️ 跳过 {task}: 文件不存在。")

    total_duration = time.time() - total_start
    print(f"\n{'='*30}")
    print(f"🏁 选定任务执行完毕。总耗时: {total_duration/60:.2f} 分钟")
    print(f"{'='*30}")

if __name__ == "__main__":
    main()
