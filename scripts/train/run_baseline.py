#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from src.model.base import load_base_model, load_tokenizer
from src.training.sft import SFTrainer
from src.data.loader import DatasetLoader
from src.utils.config import load_config
from src.utils.metrics import MetricsTracker
import time

def main():
    """Run full fine-tuning baseline for comparison"""
    print("=== Full Fine-Tuning Baseline ===")
    
    # Load configurations
    model_config = load_config('config/model.yaml')
    train_config = load_config('config/training.yaml')['sft']
    
    # Modify config for full fine-tuning (no LoRA)
    train_config['output_dir'] = 'models/baseline_full_ft'
    
    # Start tracking metrics
    metrics_tracker = MetricsTracker()
    metrics_tracker.start_timing()
    
    # Load model and tokenizer (without LoRA)
    print("Loading base model for full fine-tuning...")
    model = load_base_model(model_config['base'])
    tokenizer = load_tokenizer(model_config['base'])
    
    # Load data
    print("Loading training data...")
    loader = DatasetLoader()
    train_data = loader.load_processed_data('data/processed/train.jsonl')
    val_data = loader.load_processed_data('data/processed/val.jsonl')
    
    # Tokenize data
    from src.data.processor import DataProcessor
    processor = DataProcessor(model_config['base'])
    train_dataset = processor.tokenize(train_data)
    val_dataset = processor.tokenize(val_data)
    
    # Track memory before training
    initial_memory = metrics_tracker.track_memory()
    
    # Train model (full fine-tuning)
    print("Starting full fine-tuning...")
    trainer = SFTrainer(model, train_dataset, val_dataset, train_config)
    metrics = trainer.train()
    
    # Calculate metrics
    training_time = metrics_tracker.stop_timing()
    final_memory = metrics_tracker.track_memory()
    
    # Generate report
    cost = metrics_tracker.track_cost(training_time / 3600)  # Convert to hours
    savings = metrics_tracker.calculate_savings('full_finetuning', 'full_finetuning')
    
    print(f"\n=== Full Fine-Tuning Results ===")
    print(f"Training time: {training_time / 3600:.2f} hours")
    print(f"Estimated cost: ${cost:.2f}")
    print(f"Final loss: {metrics.get('train_loss', 'N/A')}")
    print(f"Peak memory usage: {max(m['system_used_gb'] for m in metrics_tracker.memory_usage):.1f} GB")
    
    # Save baseline results
    results = {
        'training_time_hours': training_time / 3600,
        'estimated_cost': cost,
        'final_loss': metrics.get('train_loss'),
        'peak_memory_gb': max(m['system_used_gb'] for m in metrics_tracker.memory_usage),
        'config': train_config
    }
    
    os.makedirs('outputs/results', exist_ok=True)
    import json
    with open('outputs/results/baseline_full_ft.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Baseline results saved to: outputs/results/baseline_full_ft.json")

if __name__ == "__main__":
    main()