import torch
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
from typing import Dict, List
import os

class SFTTrainer:
    """Supervised fine-tuning with LoRA"""
    
    def __init__(self, model, train_data, val_data, config):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.config = config
    
    def train(self) -> Dict:
        """Execute SFT training"""
        training_args = TrainingArguments(
            output_dir=self.config.get('output_dir', 'models/sft'),
            num_train_epochs=self.config['num_epochs'],
            per_device_train_batch_size=self.config['per_device_batch_size'],
            per_device_eval_batch_size=self.config['per_device_batch_size'],
            gradient_accumulation_steps=self.config['gradient_accumulation_steps'],
            learning_rate=self.config['learning_rate'],
            warmup_steps=self.config.get('warmup_steps', 500),
            weight_decay=self.config.get('weight_decay', 0.01),
            max_grad_norm=self.config.get('max_grad_norm', 1.0),
            fp16=self.config.get('fp16', True),
            logging_steps=self.config.get('logging_steps', 50),
            eval_steps=self.config.get('eval_steps', 500),
            save_steps=self.config.get('save_steps', 1000),
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            report_to="wandb" if self.config.get('use_wandb', False) else None
        )
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.model.tokenizer if hasattr(self.model, 'tokenizer') else None,
            mlm=False
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_data,
            eval_dataset=self.val_data,
            data_collator=data_collator
        )
        
        # Start training
        trainer.train()
        
        # Save final model
        final_path = os.path.join(training_args.output_dir, "final")
        trainer.save_model(final_path)
        
        return {
            'train_loss': trainer.state.log_history[-1]['loss'],
            'eval_loss': trainer.state.log_history[-2]['eval_loss'] if len(trainer.state.log_history) > 1 else None
        }
    
    def evaluate(self) -> Dict:
        """Evaluate model on validation set"""
        # This would be handled by the HuggingFace Trainer during training
        return {}
    
    def save_checkpoint(self, step: int):
        """Save training checkpoint"""
        checkpoint_path = f"{self.config['output_dir']}/checkpoint-{step}"
        self.model.save_pretrained(checkpoint_path)
    
    def generate_samples(self, prompts: List[str]) -> List[str]:
        """Generate samples from current model"""
        samples = []
        for prompt in prompts:
            inputs = self.model.tokenizer.encode(prompt, return_tensors="pt")
            outputs = self.model.generate(
                inputs,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.9
            )
            sample = self.model.tokenizer.decode(outputs[0], skip_special_tokens=True)
            samples.append(sample)
        return samples