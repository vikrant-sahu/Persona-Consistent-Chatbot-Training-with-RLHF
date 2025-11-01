import torch
from trl import PPOTrainer, PPOConfig
from transformers import AutoTokenizer
from typing import Dict, List
import os

class PPOTrainer:
    """PPO training for RLHF"""
    
    def __init__(self, policy_model, reward_model, ref_model, config):
        self.policy_model = policy_model
        self.reward_model = reward_model
        self.ref_model = ref_model
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.get('tokenizer_name', 'gpt2-medium'))
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def train(self, prompts: List[str]) -> Dict:
        """Execute PPO training"""
        ppo_config = PPOConfig(
            model_name=self.config.get('model_name', 'gpt2-medium'),
            learning_rate=self.config['learning_rate'],
            batch_size=self.config['batch_size'],
            ppo_epochs=self.config['ppo_epochs'],
            gradient_accumulation_steps=self.config.get('gradient_accumulation_steps', 2),
            init_kl_coef=self.config['kl_coef'],
            target_kl=self.config.get('target_kl', 6.0),
            clip_range=self.config['clip_range'],
            vf_coef=self.config['vf_coef'],
            gamma=self.config['gamma'],
            lam=self.config['lam']
        )
        
        ppo_trainer = PPOTrainer(
            config=ppo_config,
            model=self.policy_model,
            ref_model=self.ref_model,
            tokenizer=self.tokenizer
        )
        
        # Training loop
        for step in range(self.config['total_steps']):
            # Sample batch of prompts
            batch_prompts = prompts[step % len(prompts): (step % len(prompts)) + ppo_config.batch_size]
            
            if len(batch_prompts) < ppo_config.batch_size:
                continue
            
            # Generate responses
            rollout_data = self.rollout(batch_prompts)
            
            # Compute rewards
            rewards = self.compute_rewards(
                rollout_data['prompts'],
                rollout_data['responses']
            )
            
            # Update policy
            stats = ppo_trainer.step(rollout_data['responses_tensors'], rewards)
            
            # Save checkpoint
            if step % self.config.get('save_freq', 500) == 0:
                self.save_checkpoint(step)
        
        # Save final model
        self.save_checkpoint('final')
        
        return {'final_reward': stats.get('objective/reward', 0)}
    
    def rollout(self, prompts: List[str]) -> Dict:
        """Generate responses using current policy"""
        responses = []
        responses_tensors = []
        
        for prompt in prompts:
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            outputs = self.policy_model.generate(
                inputs,
                max_new_tokens=self.config['max_new_tokens'],
                do_sample=True,
                temperature=self.config['temperature'],
                pad_token_id=self.tokenizer.eos_token_id
            )
            response_tensor = outputs[0][len(inputs[0]):]
            response_text = self.tokenizer.decode(response_tensor, skip_special_tokens=True)
            
            responses.append(response_text)
            responses_tensors.append(response_tensor)
        
        return {
            'prompts': prompts,
            'responses': responses,
            'responses_tensors': responses_tensors
        }
    
    def compute_rewards(self, prompts: List[str], responses: List[str]) -> List[float]:
        """Compute rewards for generated responses"""
        rewards = []
        
        for prompt, response in zip(prompts, responses):
            # Extract persona and context from prompt
            # This is simplified - in practice you'd parse the prompt structure
            full_text = prompt + response
            reward = self.reward_model.compute_reward("", [""], response)  # Simplified
            rewards.append(reward)
        
        return rewards
    
    def update_policy(self, rollout_data: Dict) -> Dict:
        """Update policy model (handled by PPOTrainer)"""
        return {}
    
    def save_checkpoint(self, step: int):
        """Save training checkpoint"""
        checkpoint_path = f"{self.config['output_dir']}/checkpoint-{step}"
        self.policy_model.save_pretrained(checkpoint_path)