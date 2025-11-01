import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, get_peft_model
from typing import Dict, Any, List

class RewardModel(nn.Module):
    """Reward model for RLHF training"""
    
    def __init__(self, base_model: Any, config: Dict):
        super().__init__()
        self.config = config
        
        # Initialize reward model from base model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            base_model,
            num_labels=config['num_labels'],
            torch_dtype=torch.float16
        )
        
        # Apply LoRA to reward model
        lora_config = LoraConfig(
            r=config['lora_r'],
            lora_alpha=config['lora_alpha'],
            target_modules=["c_attn", "c_proj"],
            task_type="SEQ_CLS"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def forward(self, input_ids, attention_mask=None):
        return self.model(input_ids, attention_mask=attention_mask)
    
    def compute_reward(self, persona: str, context: List[str], response: str) -> float:
        """Compute reward score for given persona, context and response"""
        # Format input text
        context_str = " [SEP] ".join(context)
        input_text = f"[PERSONA] {persona} [DIALOGUE] {context_str} [RESPONSE] {response}"
        
        # Tokenize and compute reward
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            reward = outputs.logits.item()
        
        return reward
    
    def save(self, path: str):
        """Save reward model"""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
    
    def load(self, path: str):
        """Load reward model"""
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)