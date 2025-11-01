import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Dict, Any

def load_base_model(config: Dict) -> Any:
    """Load base model from config"""
    model_name = config['name']
    torch_dtype = torch.float16 if config.get('fp16', True) else torch.float32
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=config.get('device_map', 'auto'),
        cache_dir=config.get('cache_dir', './models/base')
    )
    return model

def load_tokenizer(config: Dict) -> Any:
    """Load tokenizer from config"""
    model_name = config['name']
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer

def get_model_info(model: Any) -> Dict:
    """Get model information and statistics"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'trainable_percentage': (trainable_params / total_params) * 100,
        'model_class': model.__class__.__name__
    }

def save_model(model: Any, path: str):
    """Save model to path"""
    model.save_pretrained(path)

def load_model(path: str) -> Any:
    """Load model from path"""
    return AutoModelForCausalLM.from_pretrained(path)