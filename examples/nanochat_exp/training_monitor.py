#!/usr/bin/env python3
"""
Training monitor script to parse and display training loss and metrics.

This script reads training output from stdin and formats it nicely,
highlighting loss values and other important metrics.
"""

import sys
import re
import time
import os
from datetime import datetime
from typing import Optional, Dict

# Try to import wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# ANSI color codes
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Patterns to match common loss/metric formats
# Order matters: more specific patterns first
LOSS_PATTERNS = [
    # Step-based formats (most specific first)
    (r'step[:\s]+(\d+).*?loss[:\s=]+([0-9]+\.[0-9]+)', True),  # step: X, loss: Y
    (r'Step[:\s]+(\d+).*?loss[:\s=]+([0-9]+\.[0-9]+)', True),  # Step: X, loss: Y
    (r'step[:\s]+(\d+).*?Loss[:\s=]+([0-9]+\.[0-9]+)', True),  # step: X, Loss: Y
    (r'Step[:\s]+(\d+).*?Loss[:\s=]+([0-9]+\.[0-9]+)', True),  # Step: X, Loss: Y
    # Iteration-based formats
    (r'iter[:\s]+(\d+).*?loss[:\s=]+([0-9]+\.[0-9]+)', True),  # iter: X, loss: Y
    (r'Iter[:\s]+(\d+).*?Loss[:\s=]+([0-9]+\.[0-9]+)', True),  # Iter: X, Loss: Y
    # Epoch-based formats
    (r'epoch[:\s]+(\d+).*?loss[:\s=]+([0-9]+\.[0-9]+)', True),  # epoch: X, loss: Y
    (r'Epoch[:\s]+(\d+).*?Loss[:\s=]+([0-9]+\.[0-9]+)', True),  # Epoch: X, Loss: Y
    # Standard loss formats (no step)
    (r'loss[:\s=]+([0-9]+\.[0-9]+)', False),  # loss: X
    (r'Loss[:\s=]+([0-9]+\.[0-9]+)', False),  # Loss: X
    (r'LOSS[:\s=]+([0-9]+\.[0-9]+)', False),  # LOSS: X
    (r'train_loss[:\s=]+([0-9]+\.[0-9]+)', False),  # train_loss: X
    (r'training_loss[:\s=]+([0-9]+\.[0-9]+)', False),  # training_loss: X
    # More specific patterns
    (r'\[.*?\]\s+loss[:\s=]+([0-9]+\.[0-9]+)', False),  # [tag] loss: X
    (r'loss\s*=\s*([0-9]+\.[0-9]+)', False),  # loss = X
]

# Patterns for other metrics
METRIC_PATTERNS = {
    'lr': [
        r'learning_rate[:\s=]+([0-9]+\.[0-9e\-]+)',
        r'lr[:\s=]+([0-9]+\.[0-9e\-]+)',
        r'LR[:\s=]+([0-9]+\.[0-9e\-]+)',
    ],
    'grad_norm': [
        r'grad_norm[:\s=]+([0-9]+\.[0-9e\-]+)',
        r'grad\s+norm[:\s=]+([0-9]+\.[0-9e\-]+)',
        r'gradient_norm[:\s=]+([0-9]+\.[0-9e\-]+)',
        r'grad\s*=\s*([0-9]+\.[0-9e\-]+)',  # grad = X
        r'\|grad\|[:\s=]+([0-9]+\.[0-9e\-]+)',  # |grad| = X
    ],
    'step': [
        r'step[:\s]+(\d+)',
        r'Step[:\s]+(\d+)',
        r'iteration[:\s]+(\d+)',
        r'Iter[:\s]+(\d+)',
    ],
    'epoch': [
        r'epoch[:\s]+(\d+)',
        r'Epoch[:\s]+(\d+)',
    ],
    'time': [
        r'time[:\s=]+([0-9]+\.[0-9]+)',
        r'Time[:\s=]+([0-9]+\.[0-9]+)',
    ],
    'throughput': [
        r'throughput[:\s=]+([0-9]+\.[0-9e\-]+)',
        r'samples/s[:\s=]+([0-9]+\.[0-9e\-]+)',
        r'tokens/s[:\s=]+([0-9]+\.[0-9e\-]+)',
    ],
    'memory': [
        r'memory[:\s=]+([0-9]+\.[0-9]+)',
        r'GPU\s+memory[:\s=]+([0-9]+\.[0-9]+)',
    ],
}

def extract_loss(line: str) -> Optional[tuple]:
    """Extract loss value and step/epoch from a line."""
    for pattern_info in LOSS_PATTERNS:
        pattern, has_step = pattern_info
        match = re.search(pattern, line, re.IGNORECASE)
        if match:
            groups = match.groups()
            if has_step and len(groups) == 2:
                # Step/epoch and loss
                try:
                    step = int(groups[0])
                    loss = float(groups[1])
                    return (step, loss)
                except (ValueError, IndexError):
                    pass
            elif not has_step and len(groups) == 1:
                # Only loss value
                try:
                    loss = float(groups[0])
                    return (None, loss)
                except (ValueError, IndexError):
                    pass
    return None

def extract_metrics(line: str) -> dict:
    """Extract various metrics from a line."""
    metrics = {}
    for metric_name, patterns in METRIC_PATTERNS.items():
        for pattern in patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                try:
                    value = float(match.group(1))
                    metrics[metric_name] = value
                    break
                except (ValueError, IndexError):
                    pass
    return metrics

def format_loss_line(step: Optional[int], loss: float, metrics: dict = None) -> str:
    """Format a loss line nicely."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    parts = []
    parts.append(f"{Colors.OKCYAN}[{timestamp}]{Colors.ENDC}")
    
    if step is not None:
        parts.append(f"{Colors.BOLD}Step {step:>6}{Colors.ENDC}")
    
    # Format loss with color based on value
    if loss < 1.0:
        loss_color = Colors.OKGREEN
    elif loss < 3.0:
        loss_color = Colors.WARNING
    else:
        loss_color = Colors.FAIL
    
    parts.append(f"{loss_color}Loss: {loss:.6f}{Colors.ENDC}")
    
    # Add other metrics
    if metrics:
        if 'lr' in metrics:
            parts.append(f"{Colors.OKBLUE}LR: {metrics['lr']:.2e}{Colors.ENDC}")
        if 'grad_norm' in metrics:
            parts.append(f"{Colors.OKBLUE}Grad Norm: {metrics['grad_norm']:.4f}{Colors.ENDC}")
        if 'time' in metrics:
            parts.append(f"{Colors.OKBLUE}Time: {metrics['time']:.2f}s{Colors.ENDC}")
        if 'throughput' in metrics:
            parts.append(f"{Colors.OKBLUE}Throughput: {metrics['throughput']:.2f}{Colors.ENDC}")
    
    return " | ".join(parts)

def init_wandb(run_name: Optional[str] = None):
    """Initialize wandb if available."""
    if not WANDB_AVAILABLE:
        return None
    
    # Check if wandb is already initialized (by nanochat)
    if wandb.run is not None:
        print(f"{Colors.OKGREEN}✓ Wandb already initialized by training script{Colors.ENDC}")
        return wandb.run
    
    # Try to get run name from environment or use default
    if run_name is None:
        run_name = os.environ.get("WANDB_RUN", "openseek_training")
    
    project = os.environ.get("WANDB_PROJECT", "openseek-nanochat")
    
    try:
        wandb.init(
            project=project,
            name=run_name,
            resume="allow",
            settings=wandb.Settings(_disable_stats=True)  # Disable system stats to avoid conflicts
        )
        print(f"{Colors.OKGREEN}✓ Wandb initialized: project={project}, run={run_name}{Colors.ENDC}")
        return wandb.run
    except Exception as e:
        print(f"{Colors.WARNING}⚠️  Failed to initialize wandb: {e}{Colors.ENDC}")
        print(f"{Colors.WARNING}   继续运行，但不记录到 wandb{Colors.ENDC}")
        return None

def log_to_wandb(step: Optional[int], metrics: Dict[str, float], wandb_run=None):
    """Log metrics to wandb."""
    if not WANDB_AVAILABLE or wandb_run is None:
        return
    
    try:
        log_dict = {}
        for key, value in metrics.items():
            if value is not None:
                log_dict[key] = value
        
        if log_dict and step is not None:
            wandb.log(log_dict, step=step)
        elif log_dict:
            wandb.log(log_dict)
    except Exception as e:
        # Silently fail to avoid disrupting training
        pass

def main():
    """Main monitoring loop."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}🚀 Training Monitor Started{Colors.ENDC}")
    print(f"{Colors.OKCYAN}监控训练输出，提取并高亮显示 Loss 和关键指标...{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}\n")
    
    # Initialize wandb
    wandb_run = None
    if WANDB_AVAILABLE:
        run_name = os.environ.get("WANDB_RUN")
        wandb_run = init_wandb(run_name)
        if wandb_run:
            print(f"{Colors.OKCYAN}✓ 训练指标将记录到 Wandb{Colors.ENDC}\n")
    else:
        print(f"{Colors.WARNING}⚠️  Wandb 未安装，训练指标不会记录到 Wandb{Colors.ENDC}")
        print(f"{Colors.WARNING}   安装命令: pip install wandb{Colors.ENDC}\n")
    
    last_loss = None
    last_step = None
    loss_history = []
    
    try:
        for line in sys.stdin:
            # Print original line (for debugging or full output)
            sys.stdout.write(line)
            sys.stdout.flush()
            
            # Try to extract loss
            loss_info = extract_loss(line)
            if loss_info:
                step, loss = loss_info
                metrics = extract_metrics(line)
                
                # Update tracking
                if step is not None:
                    last_step = step
                last_loss = loss
                loss_history.append(loss)
                
                # Keep only last 100 losses
                if len(loss_history) > 100:
                    loss_history.pop(0)
                
                # Prepare metrics for wandb logging
                wandb_metrics = {
                    'train/loss': loss,
                }
                if 'lr' in metrics:
                    wandb_metrics['train/learning_rate'] = metrics['lr']
                if 'grad_norm' in metrics:
                    wandb_metrics['train/grad_norm'] = metrics['grad_norm']
                if 'throughput' in metrics:
                    wandb_metrics['train/throughput'] = metrics['throughput']
                if 'memory' in metrics:
                    wandb_metrics['train/memory'] = metrics['memory']
                if 'time' in metrics:
                    wandb_metrics['train/time'] = metrics['time']
                
                # Log to wandb
                log_to_wandb(step or last_step, wandb_metrics, wandb_run)
                
                # Format and print loss summary
                formatted = format_loss_line(step or last_step, loss, metrics)
                print(f"\n{Colors.BOLD}{Colors.OKGREEN}{'═'*80}{Colors.ENDC}")
                print(f"{Colors.BOLD}📊 Training Progress:{Colors.ENDC}")
                print(formatted)
                
                # Add grad norm to display if available
                if 'grad_norm' in metrics:
                    print(f"  {Colors.OKBLUE}Grad Norm: {metrics['grad_norm']:.6f}{Colors.ENDC}")
                
                # Show trend if we have history
                if len(loss_history) >= 2:
                    recent_avg = sum(loss_history[-10:]) / min(10, len(loss_history))
                    if len(loss_history) >= 10:
                        older_avg = sum(loss_history[-20:-10]) / 10
                        trend = recent_avg - older_avg
                        if trend < -0.01:
                            trend_str = f"{Colors.OKGREEN}↓{abs(trend):.4f}{Colors.ENDC}"
                        elif trend > 0.01:
                            trend_str = f"{Colors.FAIL}↑{abs(trend):.4f}{Colors.ENDC}"
                        else:
                            trend_str = f"{Colors.OKBLUE}→{abs(trend):.4f}{Colors.ENDC}"
                        print(f"  {Colors.OKBLUE}📈 Trend (last 10 vs prev 10): {trend_str}{Colors.ENDC}")
                
                print(f"{Colors.BOLD}{Colors.OKGREEN}{'═'*80}{Colors.ENDC}\n")
            
            # Also try to extract grad norm even without loss
            elif 'grad_norm' in extract_metrics(line):
                metrics = extract_metrics(line)
                step_metric = extract_metrics(line).get('step')
                if step_metric:
                    wandb_metrics = {'train/grad_norm': metrics['grad_norm']}
                    log_to_wandb(int(step_metric), wandb_metrics, wandb_run)
            
            # Also look for other important patterns
            if re.search(r'error|Error|ERROR|exception|Exception|EXCEPTION', line):
                print(f"{Colors.FAIL}⚠️  Error detected in output above{Colors.ENDC}\n")
            elif re.search(r'warning|Warning|WARNING', line):
                print(f"{Colors.WARNING}⚠️  Warning detected in output above{Colors.ENDC}\n")
            elif re.search(r'checkpoint|Checkpoint|CHECKPOINT|saved|Saved|SAVED', line):
                print(f"{Colors.OKGREEN}✓ Checkpoint saved{Colors.ENDC}\n")
    
    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}Monitoring interrupted by user{Colors.ENDC}")
        if last_loss is not None:
            print(f"{Colors.OKCYAN}Last loss: {last_loss:.6f}{Colors.ENDC}")
        # Finish wandb run
        if wandb_run is not None and WANDB_AVAILABLE:
            try:
                wandb.finish()
            except:
                pass
        sys.exit(0)
    except Exception as e:
        print(f"{Colors.FAIL}Error in monitor: {e}{Colors.ENDC}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
