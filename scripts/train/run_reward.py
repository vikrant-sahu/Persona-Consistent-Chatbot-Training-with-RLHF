#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from src.model.reward import RewardModel
from src.training.reward import RewardModelTrainer
from src.data.loader import DatasetLoader
from src.utils.config import load_config

def main():
    """Train reward model on preference pairs"""
    print("=== Reward Model Training ===")
    
    # Load configurations
    model_config = load_config('config/model.yaml')
    train_config = load_config('config/training.yaml')['reward']
    
    # Load SFT model as base for reward model
    print("Loading SFT model as base...")
    from src.model.base import load_model
    sft_model = load_model('models/sft/final')
    
    # Initialize reward model
    print("Initializing reward model...")
    reward_model = RewardModel(sft_model, model_config['reward'])
    
    # Load preference pairs
    print("Loading preference pairs...")
    loader = DatasetLoader()
    pairs_data = loader.load_processed_data('data/processed/preference_pairs.jsonl')
    
    # Convert to training format (simplified)
    # In practice, you'd create a proper dataset with tokenization
    train_data = pairs_data[:int(0.9 * len(pairs_data))]
    val_data = pairs_data[int(0.9 * len(pairs_data)):]
    
    # Train reward model
    print("Starting reward model training...")
    trainer = RewardModelTrainer(reward_model, train_data, val_data, train_config)
    metrics = trainer.train()
    
    print(f"Reward model training completed! Final loss: {metrics.get('final_loss', 'N/A')}")

if __name__ == "__main__":
    main()