import torch
from trl import PPOTrainer as TRLPPOTrainer, PPOConfig
from transformers import AutoTokenizer
from typing import Dict, List
import os
import re


class PPOTrainer:
    """PPO training wrapper for RLHF"""

    def __init__(self, policy_model, reward_model, ref_model, config, tokenizer=None):
        self.policy_model = policy_model
        self.reward_model = reward_model
        self.ref_model = ref_model
        self.config = config

        # Use provided tokenizer or load from config
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(config.get('tokenizer_name', 'gpt2-medium'))

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def parse_prompt(self, prompt: str) -> Dict[str, str]:
        """
        Parse structured prompt to extract persona and context

        Industry-standard format: [PERSONA] persona_text [DIALOGUE] dialogue_text [RESPONSE]

        Args:
            prompt: Structured prompt string

        Returns:
            Dictionary with persona and context
        """
        result = {'persona': '', 'context': []}

        # Extract persona
        persona_match = re.search(r'\[PERSONA\](.*?)\[DIALOGUE\]', prompt, re.DOTALL)
        if persona_match:
            result['persona'] = persona_match.group(1).strip()

        # Extract dialogue context
        dialogue_match = re.search(r'\[DIALOGUE\](.*?)\[RESPONSE\]', prompt, re.DOTALL)
        if dialogue_match:
            dialogue_text = dialogue_match.group(1).strip()
            # Split by separator
            context_turns = [turn.strip() for turn in dialogue_text.split('[SEP]') if turn.strip()]
            result['context'] = context_turns

        return result
    
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
        
        ppo_trainer = TRLPPOTrainer(
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
            # Parse prompt to extract persona and context
            parsed = self.parse_prompt(prompt)
            persona = parsed['persona']
            context = parsed['context']

            # Compute reward using reward model
            reward = self.reward_model.compute_reward(persona, context, response)
            rewards.append(reward)

        return rewards
    
    def update_policy(self, rollout_data: Dict) -> Dict:
        """Update policy model (handled by PPOTrainer)"""
        return {}
    
    def save_checkpoint(self, step):
        """Save training checkpoint"""
        output_dir = self.config.get('output_dir', 'models/rlhf')
        checkpoint_path = os.path.join(output_dir, f"checkpoint-{step}")
        os.makedirs(checkpoint_path, exist_ok=True)
        self.policy_model.save_pretrained(checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")