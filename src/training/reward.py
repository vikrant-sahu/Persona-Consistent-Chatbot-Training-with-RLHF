import torch
import torch.nn as nn
from transformers import Trainer, TrainingArguments
from typing import Dict

class RewardModelTrainer:
    """Train reward model on preference pairs"""
    
    def __init__(self, model, train_data, val_data, config):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.config = config
    
    def train(self) -> Dict:
        """Train reward model"""
        training_args = TrainingArguments(
            output_dir=self.config['output_dir'],
            num_train_epochs=self.config['num_epochs'],
            per_device_train_batch_size=self.config['per_device_batch_size'],
            per_device_eval_batch_size=self.config['per_device_batch_size'],
            gradient_accumulation_steps=self.config['gradient_accumulation_steps'],
            learning_rate=self.config['learning_rate'],
            warmup_steps=self.config.get('warmup_steps', 100),
            fp16=self.config.get('fp16', False),
            bf16=self.config.get('bf16', False),
            logging_steps=50,
            eval_steps=500,
            evaluation_strategy="steps",
            save_strategy="steps"
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_data,
            eval_dataset=self.val_data,
            compute_loss=self.compute_ranking_loss
        )
        
        trainer.train()
        trainer.save_model()
        
        return {
            'final_loss': trainer.state.log_history[-1]['loss']
        }
    
    def compute_ranking_loss(self, model, inputs, return_outputs=False):
        """Compute ranking loss for reward model"""
        chosen_inputs = {
            'input_ids': inputs['chosen_input_ids'],
            'attention_mask': inputs['chosen_attention_mask']
        }
        rejected_inputs = {
            'input_ids': inputs['rejected_input_ids'],
            'attention_mask': inputs['rejected_attention_mask']
        }
        
        chosen_rewards = model(**chosen_inputs).logits
        rejected_rewards = model(**rejected_inputs).logits
        
        # Ranking loss
        loss = -torch.nn.functional.logsigmoid(chosen_rewards - rejected_rewards).mean()
        
        if return_outputs:
            return loss, {'chosen_rewards': chosen_rewards, 'rejected_rewards': rejected_rewards}
        return loss
    
    def evaluate_ranking_accuracy(self) -> float:
        """
        Evaluate ranking accuracy on validation set

        Returns:
            Accuracy of choosing the preferred response
        """
        correct = 0
        total = 0

        for batch in self.val_data:
            chosen_inputs = {
                'input_ids': batch['chosen_input_ids'],
                'attention_mask': batch['chosen_attention_mask']
            }
            rejected_inputs = {
                'input_ids': batch['rejected_input_ids'],
                'attention_mask': batch['rejected_attention_mask']
            }

            # Get rewards
            chosen_rewards = self.model(**chosen_inputs).logits
            rejected_rewards = self.model(**rejected_inputs).logits

            # Count correct rankings (chosen > rejected)
            correct += (chosen_rewards > rejected_rewards).sum().item()
            total += len(chosen_rewards)

        return correct / total if total > 0 else 0.0
    
    def save_checkpoint(self, step: int):
        """Save training checkpoint"""
        checkpoint_path = f"{self.config['output_dir']}/checkpoint-{step}"
        self.model.save_pretrained(checkpoint_path)