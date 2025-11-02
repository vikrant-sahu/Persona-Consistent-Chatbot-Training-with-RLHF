import time
import torch
import psutil
import GPUtil
from typing import Dict, Any

class MetricsTracker:
    """Track training metrics including cost, time, and memory usage"""

    def __init__(self, gpu_hourly_rate: float = None, storage_cost_per_gb: float = None):
        """
        Initialize MetricsTracker

        Args:
            gpu_hourly_rate: Cost per GPU hour (default: 0.35 for Kaggle T4)
            storage_cost_per_gb: Cost per GB of storage (default: 0.10)
        """
        # Default to Kaggle T4 pricing if not specified
        self.gpu_hourly_rate = gpu_hourly_rate if gpu_hourly_rate is not None else 0.35
        self.storage_cost_per_gb = storage_cost_per_gb if storage_cost_per_gb is not None else 0.10
        self.start_time = None
        self.gpu_usage = []
        self.memory_usage = []
    
    def track_cost(self, gpu_hours: float, gpu_count: int = 2) -> float:
        """Calculate training cost"""
        return gpu_hours * gpu_count * self.gpu_hourly_rate
    
    def track_time(self, start_time: float = None, end_time: float = None) -> float:
        """Track time duration"""
        if start_time is None:
            start_time = self.start_time if self.start_time else time.time()
        if end_time is None:
            end_time = time.time()
        
        return end_time - start_time
    
    def track_memory(self) -> Dict[str, float]:
        """Track GPU and system memory usage"""
        memory_info = {}
        
        # System memory
        system_memory = psutil.virtual_memory()
        memory_info['system_used_gb'] = system_memory.used / (1024 ** 3)
        memory_info['system_available_gb'] = system_memory.available / (1024 ** 3)
        memory_info['system_usage_percent'] = system_memory.percent
        
        # GPU memory
        try:
            gpus = GPUtil.getGPUs()
            for i, gpu in enumerate(gpus):
                memory_info[f'gpu_{i}_used_gb'] = gpu.memoryUsed
                memory_info[f'gpu_{i}_total_gb'] = gpu.memoryTotal
                memory_info[f'gpu_{i}_usage_percent'] = gpu.memoryUtil * 100
        except Exception:
            # GPU tracking not available
            pass
        
        self.memory_usage.append(memory_info)
        return memory_info
    
    def calculate_savings(self, baseline_type: str, current_type: str) -> Dict[str, float]:
        """Calculate cost and time savings compared to baseline"""
        # Baseline estimates (full fine-tuning)
        baseline_estimates = {
            'full_finetuning': {
                'time_hours': 35,
                'cost_dollars': 20.30,
                'gpu_memory_gb': 15
            },
            'lora': {
                'time_hours': 19,
                'cost_dollars': 10.50,
                'gpu_memory_gb': 7
            }
        }
        
        baseline = baseline_estimates.get(baseline_type, baseline_estimates['full_finetuning'])
        current = baseline_estimates.get(current_type, baseline_estimates['lora'])
        
        time_savings = baseline['time_hours'] - current['time_hours']
        cost_savings = baseline['cost_dollars'] - current['cost_dollars']
        memory_savings = baseline['gpu_memory_gb'] - current['gpu_memory_gb']
        
        return {
            'time_savings_hours': time_savings,
            'time_savings_percent': (time_savings / baseline['time_hours']) * 100,
            'cost_savings_dollars': cost_savings,
            'cost_savings_percent': (cost_savings / baseline['cost_dollars']) * 100,
            'memory_savings_gb': memory_savings,
            'memory_savings_percent': (memory_savings / baseline['gpu_memory_gb']) * 100
        }
    
    def generate_report(self) -> str:
        """Generate comprehensive metrics report"""
        if not self.memory_usage:
            return "No metrics data available"
        
        # Calculate averages
        avg_system_memory = sum(m['system_usage_percent'] for m in self.memory_usage) / len(self.memory_usage)
        
        report_lines = [
            "=== Training Metrics Report ===",
            f"Average System Memory Usage: {avg_system_memory:.1f}%",
            f"Total Memory Samples: {len(self.memory_usage)}",
            "",
            "Cost Analysis:",
        ]
        
        # Add cost analysis
        savings = self.calculate_savings('full_finetuning', 'lora')
        for key, value in savings.items():
            report_lines.append(f"  {key}: {value:.2f}")
        
        return "\n".join(report_lines)
    
    def start_timing(self):
        """Start timing for an operation"""
        self.start_time = time.time()
    
    def stop_timing(self) -> float:
        """Stop timing and return duration"""
        if self.start_time is None:
            return 0.0
        duration = time.time() - self.start_time
        self.start_time = None
        return duration