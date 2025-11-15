import torch
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
from typing import Dict, List
import os

class SFTTrainer:
    """Supervised fine-tuning with LoRA"""

    def __init__(self, model, tokenizer, train_data, val_data, config):
        self.model = model
        self.tokenizer = tokenizer
        self.train_data = train_data
        self.val_data = val_data
        self.config = config
    
    def train(self) -> Dict:
        """Execute SFT training

        Supports both regular LoRA and QLoRA training.
        For QLoRA, use bf16=True (NOT fp16) to avoid precision errors.
        """
        training_args = TrainingArguments(
            output_dir=self.config.get('output_dir', 'models/sft'),
            num_train_epochs=self.config['num_epochs'],
            max_steps=self.config.get('max_steps', -1),  # Hard limit on steps (safety)
            per_device_train_batch_size=self.config['per_device_batch_size'],
            per_device_eval_batch_size=self.config['per_device_batch_size'],
            gradient_accumulation_steps=self.config['gradient_accumulation_steps'],
            learning_rate=self.config['learning_rate'],
            warmup_steps=self.config.get('warmup_steps', 500),
            weight_decay=self.config.get('weight_decay', 0.01),
            max_grad_norm=self.config.get('max_grad_norm', 1.0),
            # IMPORTANT: fp16 defaults to False to avoid gradient scaling issues
            # For QLoRA, use bf16=True instead (more stable, no gradient scaling)
            fp16=self.config.get('fp16', False),
            bf16=self.config.get('bf16', False),
            logging_steps=self.config.get('logging_steps', 50),
            eval_steps=self.config.get('eval_steps', 500),
            save_steps=self.config.get('save_steps', 1000),
            save_total_limit=self.config.get('save_total_limit', None),  # Limit checkpoint storage
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            gradient_checkpointing=self.config.get('gradient_checkpointing', False),  # Memory efficiency
            dataloader_num_workers=self.config.get('dataloader_num_workers', 0),  # Parallel data loading
            optim=self.config.get('optim', 'adamw_torch'),  # Optimizer (use 'paged_adamw_8bit' for QLoRA)
            report_to="wandb" if self.config.get('use_wandb', False) else None
        )
        
        # OPTIMIZED: Dynamic padding at batch time (3-4x faster than padding all examples to max_length)
        # Pads only to the longest sequence in each batch, not to global max_length
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8  # Pad to multiple of 8 for optimal GPU performance
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_data,
            eval_dataset=self.val_data,
            data_collator=data_collator
        )
        
        # Start training
        train_result = trainer.train()

        # Save final model
        final_path = os.path.join(training_args.output_dir, "final")
        trainer.save_model(final_path)

        # Extract metrics safely from log history
        metrics = {}
        if hasattr(trainer.state, 'log_history') and len(trainer.state.log_history) > 0:
            # Find last training loss
            for entry in reversed(trainer.state.log_history):
                if 'loss' in entry and 'train_loss' not in metrics:
                    metrics['train_loss'] = entry['loss']
                if 'eval_loss' in entry and 'eval_loss' not in metrics:
                    metrics['eval_loss'] = entry['eval_loss']
                if 'train_loss' in metrics and 'eval_loss' in metrics:
                    break

        # Add training result metrics if available
        if hasattr(train_result, 'metrics'):
            metrics.update(train_result.metrics)

        return metrics
    
    def evaluate(self) -> Dict:
        """Evaluate model on validation set"""
        # This would be handled by the HuggingFace Trainer during training
        return {}
    
    def save_checkpoint(self, step: int):
        """Save training checkpoint"""
        checkpoint_path = f"{self.config['output_dir']}/checkpoint-{step}"
        self.model.save_pretrained(checkpoint_path)
    
    def generate_samples(self, prompts: List[str], max_new_tokens: int = 150, temperature: float = 0.9) -> List[str]:
        """Generate samples from current model"""
        samples = []
        for prompt in prompts:
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            outputs = self.model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=self.tokenizer.eos_token_id
            )
            sample = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            samples.append(sample)
        return samples