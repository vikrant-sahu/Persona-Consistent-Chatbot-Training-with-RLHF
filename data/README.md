# Dataset Documentation

## Overview
This directory contains datasets for training and evaluating the persona-consistent chatbot.

## Datasets

### PersonaChat
- Source: HuggingFace `bavard/personachat_truecased`
- Size: 162,064 utterances
- Format: Persona descriptions + multi-turn dialogues
- Usage: Primary training data

### Blended Skill Talk (BST)
- Source: HuggingFace `blended_skill_talk`
- Size: 27,018 dialogues
- Skills: Empathy, Knowledge, Personality
- Usage: Additional training for robustness

## Directory Structure

- `raw/`: Original datasets (gitignored)
- `processed/`: Preprocessed training data (gitignored)
  - `train.jsonl`: Training split
  - `val.jsonl`: Validation split  
  - `test.jsonl`: Test split
  - `preference_pairs.jsonl`: Preference pairs for reward model
  - `prompts.jsonl`: Prompts for PPO training
- `benchmarks/`: Evaluation test sets
  - `consistency_test.jsonl`: Persona consistency evaluation
  - `engagement_test.jsonl`: Engagement evaluation

## Data Format

### Training Data
```json
{
  "text": "[PERSONA] I love hiking | I have a dog [DIALOGUE] Hello! [SEP] Hi there! [RESPONSE] I love hiking with my dog!",
  "persona": "I love hiking | I have a dog",
  "context": ["Hello!", "Hi there!"],
  "response": "I love hiking with my dog!"
}