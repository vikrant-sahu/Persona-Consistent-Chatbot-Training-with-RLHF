#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from src.model.base import load_base_model, load_tokenizer
from src.model.lora import LoRAWrapper
from src.training.sft import SFTrainer
from src.data.loader import DatasetLoader
from src.utils.config import load_config

def main():
    """Run supervised fine-tuning"""
    print("=== Supervised Fine-Tuning ===")
    
    # Load configurations
    model_config = load_config('config/model.yaml')
    train_config = load_config('config/training.yaml')['sft']
    
    # Load model and tokenizer
    print("Loading base model...")
    model = load_base_model(model_config['base'])
    tokenizer = load_tokenizer(model_config['base'])
    
    # Apply LoRA
    print("Applying LoRA...")
    lora_wrapper = LoRAWrapper(model_config['lora'])
    model = lora_wrapper.apply_lora(model)
    lora_wrapper.print_trainable_params(model)
    
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
    
    # Train model
    print("Starting SFT training...")
    trainer = SFTrainer(model, train_dataset, val_dataset, train_config)
    metrics = trainer.train()
    
    print(f"SFT training completed! Final metrics: {metrics}")

if __name__ == "__main__":
    main()