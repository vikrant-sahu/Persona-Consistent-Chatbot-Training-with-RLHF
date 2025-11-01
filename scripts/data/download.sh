#!/bin/bash

# Download all required datasets and models for persona-consistent chatbot

echo "=== Downloading Datasets and Models ==="

# Create directories
mkdir -p data/raw
mkdir -p models/base

echo "1. Downloading PersonaChat dataset..."
python -c "
from datasets import load_dataset
dataset = load_dataset('bavard/personachat_truecased', cache_dir='./data/raw')
print(f'PersonaChat downloaded: {len(dataset[\"train\"])} training examples')
"

echo "2. Downloading Blended Skill Talk dataset..."
python -c "
from datasets import load_dataset
dataset = load_dataset('blended_skill_talk', cache_dir='./data/raw')
print(f'Blended Skill Talk downloaded: {len(dataset[\"train\"])} training examples')
"

echo "3. Downloading base models..."
python -c "
from transformers import AutoModel, AutoTokenizer
import os

models = ['gpt2-medium', 'microsoft/DialoGPT-medium']

for model_name in models:
    print(f'Downloading {model_name}...')
    model = AutoModel.from_pretrained(model_name, cache_dir='./models/base')
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='./models/base')
    print(f'✓ {model_name} downloaded successfully')
"

echo "4. Verifying downloads..."
python -c "
import os
print('Verifying dataset downloads...')
personachat_path = './data/raw/bavard___personachat_truecased'
if os.path.exists(personachat_path):
    print('✓ PersonaChat downloaded successfully')
else:
    print('✗ PersonaChat download failed')

print('Verifying model downloads...')
gpt2_path = './models/base/models--gpt2-medium'
if os.path.exists(gpt2_path):
    print('✓ GPT-2 Medium downloaded successfully')
else:
    print('✗ GPT-2 Medium download failed')
"

echo "=== Download Complete ==="
echo "Datasets are available in: ./data/raw/"
echo "Models are available in: ./models/base/"