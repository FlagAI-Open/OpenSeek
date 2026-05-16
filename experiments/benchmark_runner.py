import subprocess
import logging
import time

logger = logging.getLogger("BenchmarkRunner")
logging.basicConfig(level=logging.INFO, format='%(message)s')

def run_mode(mode: str):
    logger.info(f"\n================================================")
    logger.info(f"STARTING {mode.upper()} RUN")
    logger.info(f"================================================\n")
    
    cmd = ["python", "run_cog6.py", "--mode", mode, "--task", "openseek-5"]
    
    start_time = time.time()
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in iter(process.stdout.readline, ''):
            print(line, end='')
        process.wait()
    except Exception as e:
        logger.error(f"Failed to execute {mode}: {e}")
        
    duration = time.time() - start_time
    logger.info(f"\n{mode.upper()} run completed in {duration:.2f}s")

if __name__ == "__main__":
    logger.info("Initializing Full COG-6 Benchmark Suite...")
    run_mode("baseline")
    run_mode("cog6")
    logger.info("\nBenchmark execution finished. Check 'experiments/run_metrics.json' and predictions in 'outputs/'.")
