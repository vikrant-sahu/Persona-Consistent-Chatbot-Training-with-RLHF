import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Dict, Any

def load_base_model(config: Dict) -> Any:
    """Load base model from config

    Note: Always loads in FP32. Use Trainer's fp16/bf16 settings for mixed precision.
    This avoids gradient scaling conflicts during training.
    """
    model_name = config['name']
    # Always use FP32 for model loading - let Trainer handle mixed precision
    torch_dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=config.get('device_map', 'auto'),
        cache_dir=config.get('cache_dir', './models/base')
    )
    return model


def load_base_model_qlora(config: Dict) -> Any:
    """Load base model with QLoRA (4-bit quantization)

    QLoRA provides 3-4x speedup and 75% memory reduction with minimal quality loss.
    Recommended for Kaggle environments with limited GPU time.

    IMPORTANT: Uses BF16 (bfloat16) NOT FP16 to avoid gradient scaling conflicts.
    BF16 is more stable and doesn't require gradient scaling.

    Args:
        config: Dict with keys:
            - name: model name/path
            - device_map: device mapping ('auto' recommended)
            - cache_dir: cache directory

    Returns:
        Quantized model ready for LoRA fine-tuning

    Raises:
        ImportError: If bitsandbytes is not installed
    """
    try:
        from transformers import BitsAndBytesConfig
        from peft import prepare_model_for_kbit_training
    except ImportError as e:
        raise ImportError(
            "QLoRA requires 'bitsandbytes' and 'peft' packages. "
            "Install with: pip install bitsandbytes peft"
        ) from e

    model_name = config['name']

    # QLoRA quantization configuration
    # CRITICAL: Use bfloat16 NOT float16 to avoid gradient scaling issues
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,                       # Enable 4-bit quantization
        bnb_4bit_use_double_quant=True,          # Double quantization for extra compression
        bnb_4bit_quant_type="nf4",               # NormalFloat4 quantization type
        bnb_4bit_compute_dtype=torch.bfloat16    # Use BF16 NOT FP16 (prevents precision errors)
    )

    print(f"Loading {model_name} with QLoRA (4-bit quantization)...")
    print(f"  - Quantization: 4-bit NF4")
    print(f"  - Compute dtype: BF16 (stable, no gradient scaling)")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=config.get('device_map', 'auto'),
        cache_dir=config.get('cache_dir', './models/base'),
        trust_remote_code=True
    )

    # Enable gradient checkpointing for memory efficiency
    # This trades compute for memory, enabling larger batches
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        print("  - Gradient checkpointing: Enabled")

    # Prepare model for k-bit training
    # This configures the model to work properly with quantization
    model = prepare_model_for_kbit_training(model)

    print("✓ Model loaded with QLoRA (ready for training)")
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