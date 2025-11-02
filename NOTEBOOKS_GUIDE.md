# Notebooks Implementation Guide

## Overview
The notebooks (3-6) need to be updated to properly use the fixed `src` modules. This guide provides the correct structure and implementation patterns.

## Common Setup for All Notebooks

### 1. Imports
```python
import sys
import os
sys.path.append('../')

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import time
from tqdm import tqdm

# Import from src modules
from src.data.loader import DatasetLoader
from src.data.processor import DataProcessor
from src.data.generator import PreferenceGenerator
from src.model.base import load_base_model, load_tokenizer, get_model_info
from src.model.lora import LoRAWrapper
from src.model.reward import RewardModel
from src.training.sft import SFTTrainer
from src.training.reward import RewardModelTrainer
from src.training.ppo import PPOTrainer
from src.eval.persona import PersonaEvaluator
from src.eval.engagement import EngagementEvaluator
from src.eval.quality import QualityEvaluator
from src.utils.metrics import MetricsTracker
from src.utils.checkpoint import CheckpointManager
```

### 2. W&B Initialization Template
```python
import wandb

# Initialize W&B
wandb.init(
    project="persona-chatbot-rlhf",
    name="notebook-name",
    config={
        'model_name': 'gpt2-medium',
        'lora_r': 16,
        'learning_rate': 2e-4,
        # ... all hyperparameters
    }
)
```

### 3. Dataset Loading Template
```python
# Use DatasetLoader (not direct load_dataset)
loader = DatasetLoader(cache_dir='../data/raw')

# Load google/Synthetic-Persona-Chat
train_data = loader.load_personachat(split='train', use_synthetic=True)
val_data = loader.load_personachat(split='validation', use_synthetic=True)

# Process data
processor = DataProcessor(config={
    'base_model': 'gpt2-medium',
    'max_length': 512,
    'train_split': 0.9,
    'val_split': 0.05
})

processed_train = processor.preprocess(train_data)
tokenized_train = processor.tokenize(processed_train)
```

---

## Notebook 3: 3_sft_training.ipynb

### Purpose
Supervised fine-tuning with LoRA on PersonaChat dataset.

### Key Sections

#### 1. Configuration (Pass parameters directly, NOT from config files)
```python
# Model configuration
model_config = {
    'name': 'gpt2-medium',
    'cache_dir': '../models/base',
    'device_map': 'auto',
    'fp16': True
}

# LoRA configuration
lora_config = {
    'r': 16,
    'alpha': 32,
    'dropout': 0.1,
    'target_modules': ['c_attn', 'c_proj'],
    'bias': 'none',
    'task_type': 'CAUSAL_LM'
}

# Training configuration
training_config = {
    'output_dir': '../models/sft',
    'num_epochs': 3,
    'per_device_batch_size': 4,
    'gradient_accumulation_steps': 4,
    'learning_rate': 2e-4,
    'warmup_steps': 500,
    'weight_decay': 0.01,
    'max_grad_norm': 1.0,
    'fp16': True,
    'logging_steps': 50,
    'eval_steps': 500,
    'save_steps': 1000,
    'use_wandb': True
}
```

#### 2. Load Model with src modules
```python
# Load base model
model = load_base_model(model_config)
tokenizer = load_tokenizer(model_config)

# Get model info
info = get_model_info(model)
print(f"Total parameters: {info['total_parameters'] / 1e6:.1f}M")

# Apply LoRA
lora_wrapper = LoRAWrapper(lora_config)
model = lora_wrapper.apply_lora(model, lora_config)
lora_wrapper.print_trainable_params(model)
```

#### 3. Prepare Data
```python
# Load and process data
loader = DatasetLoader()
train_data = loader.load_personachat(split='train', use_synthetic=True)
val_data = loader.load_personachat(split='validation', use_synthetic=True)

processor = DataProcessor(config={'base_model': 'gpt2-medium', 'max_length': 512})
processed_train = processor.preprocess(train_data)
processed_val = processor.preprocess(val_data)
tokenized_train = processor.tokenize(processed_train)
tokenized_val = processor.tokenize(processed_val)
```

#### 4. Training with Metrics Tracking
```python
# Initialize metrics tracker
tracker = MetricsTracker(gpu_hourly_rate=0.35, storage_cost_per_gb=0.10)
tracker.start_timing()

# Initialize trainer (PASS TOKENIZER!)
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,  # IMPORTANT: Pass tokenizer explicitly
    train_data=tokenized_train,
    val_data=tokenized_val,
    config=training_config
)

# Train
results = trainer.train()

# Track metrics
training_time = tracker.stop_timing()
cost = tracker.track_cost(training_time / 3600, gpu_count=2)
savings = tracker.calculate_savings('full_finetuning', 'lora')

# Log to W&B
wandb.log({
    'training_time_hours': training_time / 3600,
    'training_cost_dollars': cost,
    'cost_savings_percent': savings['cost_savings_percent'],
    'time_savings_percent': savings['time_savings_percent']
})

print(f"✅ Cost reduction: {savings['cost_savings_percent']:.1f}% (Target: 75-80%)")
print(f"✅ Time reduction: {savings['time_savings_percent']:.1f}% (Target: 60-70%)")
```

#### 5. Evaluation
```python
# Quick persona consistency check
persona_eval = PersonaEvaluator(tokenizer_name='gpt2-medium')
consistency_score = persona_eval.evaluate_consistency(
    model=model,
    data=processed_val[:100],
    persona_field='persona',
    context_field='context',
    response_field='response',
    generate_responses=False  # Use existing responses
)

print(f"Persona consistency: {consistency_score:.3f}")
wandb.log({'persona_consistency': consistency_score})
```

---

## Notebook 4: 4_reward_and_ppo.ipynb

### Purpose
Train reward model and perform PPO optimization for RLHF.

### Key Sections

#### 1. Load SFT Model
```python
# Load SFT model (from notebook 3)
sft_model_path = '../models/sft/final'
model = load_base_model({'name': sft_model_path, 'device_map': 'auto'})
tokenizer = load_tokenizer({'name': sft_model_path})

# Apply LoRA if needed
lora_wrapper = LoRAWrapper(lora_config)
model = lora_wrapper.load_lora_model(model, sft_model_path)
```

#### 2. Prepare Preference Data
```python
# Generate preference pairs for reward model
generator = PreferenceGenerator(config={})

# Load SFT dataset
processor = DataProcessor(config={'base_model': 'gpt2-medium', 'max_length': 512})
sft_data = processor.preprocess(train_data)

# Generate preference pairs
preference_pairs = generator.generate_pairs(sft_data, model=model)

print(f"Generated {len(preference_pairs)} preference pairs")
```

#### 3. Train Reward Model
```python
# Initialize reward model
reward_config = {
    'num_labels': 1,
    'lora_r': 8,
    'lora_alpha': 16
}

reward_model = RewardModel(
    base_model='gpt2-medium',
    config=reward_config
)

# Training config for reward model
reward_training_config = {
    'output_dir': '../models/reward',
    'num_epochs': 1,
    'per_device_batch_size': 2,
    'gradient_accumulation_steps': 8,
    'learning_rate': 1e-5,
    'warmup_steps': 100
}

# Train reward model
reward_trainer = RewardModelTrainer(
    model=reward_model,
    train_data=train_preference_data,
    val_data=val_preference_data,
    config=reward_training_config
)

reward_results = reward_trainer.train()
```

#### 4. PPO Training
```python
# Prepare prompts for PPO
prompts = generator.extract_prompts(sft_data)

# Initialize PPO trainer
ppo_config = {
    'model_name': 'gpt2-medium',
    'tokenizer_name': 'gpt2-medium',
    'output_dir': '../models/rlhf',
    'total_steps': 5000,
    'batch_size': 8,
    'learning_rate': 1.5e-5,
    'ppo_epochs': 4,
    'kl_coef': 0.2,
    'clip_range': 0.2,
    'vf_coef': 0.1,
    'gamma': 1.0,
    'lam': 0.95,
    'max_new_tokens': 150,
    'temperature': 0.9,
    'save_freq': 500
}

# Create reference model (frozen copy of SFT model)
ref_model = load_base_model({'name': sft_model_path, 'device_map': 'auto'})
ref_model.eval()
for param in ref_model.parameters():
    param.requires_grad = False

# Initialize PPO trainer (note: class is now PPOTrainer, not PPOTrainerWrapper)
ppo_trainer = PPOTrainer(
    policy_model=model,
    reward_model=reward_model,
    ref_model=ref_model,
    config=ppo_config,
    tokenizer=tokenizer  # Pass tokenizer explicitly
)

# Train with PPO
ppo_results = ppo_trainer.train(prompts)

print(f"Final reward: {ppo_results['final_reward']:.4f}")
wandb.log({'final_ppo_reward': ppo_results['final_reward']})
```

---

## Notebook 5: 5_evaluation.ipynb

### Purpose
Comprehensive evaluation of trained models on all metrics.

### Key Sections

#### 1. Load Models
```python
# Load baseline, SFT, and RLHF models
models = {
    'baseline': load_base_model({'name': 'gpt2-medium', 'device_map': 'auto'}),
    'sft': load_base_model({'name': '../models/sft/final', 'device_map': 'auto'}),
    'rlhf': load_base_model({'name': '../models/rlhf/checkpoint-final', 'device_map': 'auto'})
}

tokenizer = load_tokenizer({'name': 'gpt2-medium'})
```

#### 2. Persona Consistency Evaluation
```python
# Initialize evaluator
persona_eval = PersonaEvaluator(tokenizer_name='gpt2-medium')

# Load test data
loader = DatasetLoader()
test_data = loader.load_personachat(split='validation', use_synthetic=True)
processor = DataProcessor(config={'base_model': 'gpt2-medium', 'max_length': 512})
processed_test = processor.preprocess(test_data)

# Evaluate each model
results = {}
for model_name, model in models.items():
    print(f"\nEvaluating {model_name}...")

    # Persona consistency
    consistency = persona_eval.evaluate_consistency(
        model=model,
        data=processed_test,
        persona_field='persona',
        context_field='context',
        max_samples=200,
        generate_responses=True
    )

    results[model_name] = {'persona_consistency': consistency}
    print(f"  Persona consistency: {consistency:.3f}")

    # Check if target achieved
    if model_name == 'rlhf':
        target_achieved = consistency >= 0.85
        print(f"  Target (85%): {'✅ ACHIEVED' if target_achieved else '❌ NOT ACHIEVED'}")
        wandb.log({
            f'{model_name}_persona_consistency': consistency,
            'target_85_percent_achieved': target_achieved
        })
```

#### 3. Engagement Evaluation
```python
# Initialize engagement evaluator
engagement_eval = EngagementEvaluator()

for model_name, model in models.items():
    engagement_score = engagement_eval.evaluate_engagement(
        model=model,
        data=processed_test,
        persona_field='persona',
        context_field='context',
        max_samples=200,
        generate_responses=True
    )

    results[model_name]['engagement'] = engagement_score
    print(f"{model_name} engagement: {engagement_score:.3f}")
    wandb.log({f'{model_name}_engagement': engagement_score})
```

#### 4. Quality Metrics Evaluation
```python
# Initialize quality evaluator
quality_eval = QualityEvaluator()

for model_name, model in models.items():
    print(f"\nEvaluating quality for {model_name}...")

    # Perplexity
    perplexity = quality_eval.compute_perplexity(
        model=model,
        data=processed_test,
        text_field='text',
        batch_size=8
    )

    # BLEU
    bleu_scores = quality_eval.compute_bleu(
        model=model,
        data=processed_test,
        context_field='context',
        response_field='response',
        persona_field='persona',
        max_samples=100
    )

    # ROUGE
    rouge_scores = quality_eval.compute_rouge(
        model=model,
        data=processed_test,
        context_field='context',
        response_field='response',
        persona_field='persona',
        max_samples=100
    )

    # Distinct-N
    distinct_1 = quality_eval.compute_distinct_n(model, processed_test, n=1, max_samples=100)
    distinct_2 = quality_eval.compute_distinct_n(model, processed_test, n=2, max_samples=100)

    results[model_name].update({
        'perplexity': perplexity,
        'bleu': bleu_scores['bleu'],
        'rouge1': rouge_scores['rouge1'],
        'rouge2': rouge_scores['rouge2'],
        'rougeL': rouge_scores['rougeL'],
        'distinct_1': distinct_1,
        'distinct_2': distinct_2
    })

    wandb.log({f'{model_name}_perplexity': perplexity})
    wandb.log({f'{model_name}_bleu': bleu_scores['bleu']})
```

#### 5. Comparison and Visualization
```python
# Create comparison dataframe
comparison_df = pd.DataFrame(results).T
comparison_df.to_csv('../outputs/model_comparison.csv')

# Plot comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Persona consistency
comparison_df['persona_consistency'].plot(kind='bar', ax=axes[0, 0])
axes[0, 0].axhline(y=0.85, color='r', linestyle='--', label='Target (85%)')
axes[0, 0].set_title('Persona Consistency')
axes[0, 0].legend()

# Engagement
comparison_df['engagement'].plot(kind='bar', ax=axes[0, 1], color='orange')
axes[0, 1].set_title('Engagement Score')

# BLEU
comparison_df['bleu'].plot(kind='bar', ax=axes[1, 0], color='green')
axes[1, 1].set_title('BLEU Score')

# Perplexity (lower is better)
comparison_df['perplexity'].plot(kind='bar', ax=axes[1, 1], color='red')
axes[1, 1].set_title('Perplexity (Lower is Better)')

plt.tight_layout()
plt.savefig('../outputs/model_comparison.png', dpi=300)
wandb.log({"comparison_chart": wandb.Image(plt)})
plt.show()
```

---

## Notebook 6: 6_analysis_demo.ipynb

### Purpose
Results analysis and interactive demo.

### Key Sections

#### 1. Load Best Model
```python
# Load RLHF model (best performing)
model = load_base_model({'name': '../models/rlhf/checkpoint-final', 'device_map': 'auto'})
tokenizer = load_tokenizer({'name': '../models/rlhf/checkpoint-final'})
```

#### 2. Cost-Benefit Analysis
```python
# Load training summaries
with open('../models/sft/training_summary.json', 'r') as f:
    sft_summary = json.load(f)

# Display cost savings
tracker = MetricsTracker(gpu_hourly_rate=0.35)
savings = tracker.calculate_savings('full_finetuning', 'lora')

print("🎯 Project Goals Achievement:")
print("=" * 60)
print(f"1. Cost Reduction:")
print(f"   Target: 75-80%")
print(f"   Achieved: {savings['cost_savings_percent']:.1f}%")
print(f"   Status: {'✅ ACHIEVED' if savings['cost_savings_percent'] >= 75 else '❌ NOT ACHIEVED'}")

print(f"\n2. Time Reduction:")
print(f"   Target: 60-70%")
print(f"   Achieved: {savings['time_savings_percent']:.1f}%")
print(f"   Status: {'✅ ACHIEVED' if savings['time_savings_percent'] >= 60 else '❌ NOT ACHIEVED'}")

print(f"\n3. Persona Consistency:")
print(f"   Target: 85%+")
print(f"   Achieved: {final_consistency:.1%}")
print(f"   Status: {'✅ ACHIEVED' if final_consistency >= 0.85 else '❌ NOT ACHIEVED'}")
```

#### 3. Interactive Demo
```python
def chat_with_bot(persona_traits, conversation_history=[]):
    """Interactive chatbot demo"""
    # Format persona
    persona_text = " | ".join(persona_traits)

    # Format history
    context_str = " [SEP] ".join(conversation_history[-3:]) if conversation_history else ""

    # Create prompt
    prompt = f"[PERSONA] {persona_text} [DIALOGUE] {context_str} [RESPONSE]"

    # Generate response
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.9,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    return response.strip()

# Example usage
persona = [
    "I am a software engineer",
    "I love hiking",
    "I have two dogs"
]

print("Chatbot Demo:")
print(f"Persona: {', '.join(persona)}")
print("\n" + "=" * 60)

conversation = []
user_inputs = [
    "Hi! What do you like to do in your free time?",
    "That sounds fun! What kind of dogs do you have?",
    "Do you take them hiking with you?"
]

for user_input in user_inputs:
    print(f"\nUser: {user_input}")
    conversation.append(f"User: {user_input}")

    bot_response = chat_with_bot(persona, conversation)
    print(f"Bot: {bot_response}")
    conversation.append(f"Bot: {bot_response}")

    # Evaluate consistency
    persona_eval = PersonaEvaluator()
    consistency = persona_eval.calculate_consistency_score(bot_response, persona)
    print(f"Consistency: {consistency:.2%}")
```

---

## Key Reminders

1. **Always pass tokenizer to training classes**:
   - `SFTTrainer(model, tokenizer, ...)`
   - `PPOTrainer(..., tokenizer=tokenizer)`

2. **Use google/Synthetic-Persona-Chat dataset**:
   - `loader.load_personachat(split='train', use_synthetic=True)`

3. **Pass parameters directly, NOT from config files**:
   - ✅ `config = {'learning_rate': 2e-4, ...}`
   - ❌ `config = load_config('config/training.yaml')`

4. **Initialize W&B in each notebook**:
   - `wandb.init(project="persona-chatbot-rlhf", name="...")`

5. **Track metrics for cost/time reduction**:
   - Use `MetricsTracker` class
   - Log to W&B: `wandb.log({...})`

6. **Verify targets achieved**:
   - 75-80% cost reduction
   - 60-70% time reduction
   - 85%+ persona consistency

7. **Use proper prompt format in PPO**:
   - `[PERSONA] text [DIALOGUE] text [RESPONSE]`
   - Parsed automatically by `PPOTrainer.parse_prompt()`

## Testing Checklist

- [ ] All imports from src modules work
- [ ] No config file reading
- [ ] W&B logging enabled
- [ ] Tokenizer passed to all trainers
- [ ] Dataset uses google/Synthetic-Persona-Chat
- [ ] Metrics tracked and logged
- [ ] Cost reduction >= 75%
- [ ] Time reduction >= 60%
- [ ] Persona consistency >= 85%
- [ ] All plots saved and logged to W&B
