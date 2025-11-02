import yaml
import os
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def get_config(config_name: str) -> Dict[str, Any]:
    """Get configuration by name from configs directory"""
    config_path = f'configs/{config_name}.yaml'
    return load_config(config_path)


def merge_configs(*configs: Dict) -> Dict[str, Any]:
    """
    Merge multiple configuration dictionaries
    Later configs override earlier ones

    Args:
        *configs: Variable number of config dictionaries to merge

    Returns:
        Merged configuration dictionary
    """
    merged = {}
    for config in configs:
        if config:
            merged = _deep_update(merged, config)
    return merged


def _deep_update(base_dict: Dict, update_dict: Dict) -> Dict:
    """Recursively update nested dictionaries"""
    for key, value in update_dict.items():
        if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
            base_dict[key] = _deep_update(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict


def save_config(config: Dict[str, Any], config_path: str):
    """Save configuration to YAML file"""
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False, indent=2)


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration values

    Returns:
        Dictionary with default configuration
    """
    return {
        'model': {
            'name': 'gpt2-medium',
            'max_length': 512,
            'device_map': 'auto'
        },
        'lora': {
            'r': 16,
            'alpha': 16,
            'dropout': 0.1,
            'target_modules': ['c_attn', 'c_proj'],
            'bias': 'none',
            'task_type': 'CAUSAL_LM'
        },
        'training': {
            'num_epochs': 3,
            'per_device_batch_size': 4,
            'gradient_accumulation_steps': 4,
            'learning_rate': 2.0e-4,
            'warmup_steps': 500,
            'weight_decay': 0.01,
            'fp16': True,
            'logging_steps': 50,
            'eval_steps': 500,
            'save_steps': 1000
        },
        'data': {
            'max_length': 512,
            'train_split': 0.9,
            'val_split': 0.05,
            'test_split': 0.05
        }
    }