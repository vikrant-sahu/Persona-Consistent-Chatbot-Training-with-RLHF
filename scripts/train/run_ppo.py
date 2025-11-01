#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from src.training.ppo import PPOTrainer
from src.model.base import load_model
from src.model.reward import RewardModel
from src.data.loader import DatasetLoader
from src.utils.config import load_config

def main():
    """Run PPO training for RLHF"""
    print("=== PPO RLHF Training ===")
    
    # Load configurations
    model_config = load_config('config/model.yaml')
    train_config = load_config('config/training.yaml')['ppo']
    
    # Load models
    print("Loading policy model (SFT)...")
    policy_model = load_model('models/sft/final')
    
    print("Loading reward model...")
    reward_model = RewardModel('models/reward/final', model_config['reward'])
    
    print("Loading reference model (frozen SFT)...")
    ref_model = load_model('models/sft/final')
    
    # Load prompts for PPO
    print("Loading prompts...")
    loader = DatasetLoader()
    prompts_data = loader.load_processed_data('data/processed/prompts.jsonl')
    prompts = [item['prompt'] for item in prompts_data]
    
    # Initialize PPO trainer
    print("Initializing PPO trainer...")
    trainer = PPOTrainer(policy_model, reward_model, ref_model, train_config)
    
    # Start PPO training
    print("Starting PPO training...")
    metrics = trainer.train(prompts)
    
    print(f"PPO training completed! Final reward: {metrics.get('final_reward', 'N/A')}")

if __name__ == "__main__":
    main()