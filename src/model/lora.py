from peft import LoraConfig, get_peft_model, PeftModel
import torch
from typing import Dict, Any

class LoRAWrapper:
    """Apply LoRA to base models for parameter-efficient fine-tuning"""
    
    def __init__(self, config: Dict):
        self.config = config
    
    def apply_lora(self, model: Any, config: Dict = None) -> Any:
        """Apply LoRA to model"""
        if config is None:
            config = self.config
        
        lora_config = LoraConfig(
            r=config['r'],
            lora_alpha=config['alpha'],
            lora_dropout=config['dropout'],
            target_modules=config['target_modules'],
            bias=config['bias'],
            task_type=config['task_type'],
            fan_in_fan_out=config.get('fan_in_fan_out', True)
        )
        
        model = get_peft_model(model, lora_config)
        return model
    
    def print_trainable_params(self, model: Any):
        """Print trainable parameters information"""
        model.print_trainable_parameters()
    
    def merge_and_save(self, model: Any, path: str):
        """Merge LoRA adapters and save full model"""
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(path)
    
    def load_lora_model(self, base_model: Any, adapter_path: str) -> Any:
        """Load LoRA adapters onto base model"""
        return PeftModel.from_pretrained(base_model, adapter_path)