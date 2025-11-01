import re
from typing import List, Dict, Any
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

class DataProcessor:
    """Preprocess and tokenize data for training"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.get('base_model', 'gpt2-medium'))
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def preprocess(self, data: Dataset) -> Dataset:
        """Preprocess raw dataset into training format"""
        processed_data = []
        
        for example in data:
            # Extract persona and dialogue
            persona = example.get('personality', [])
            history = example.get('history', [])
            
            if not persona or not history:
                continue
                
            # Format for training
            persona_str = " | ".join(persona)
            
            # Create training examples for each turn
            for i in range(1, len(history)):
                context = history[:i]
                response = history[i]
                
                input_text = f"[PERSONA] {persona_str} [DIALOGUE] {' [SEP] '.join(context)}"
                target_text = response
                full_text = f"{input_text} [RESPONSE] {target_text}"
                
                processed_data.append({
                    'text': full_text,
                    'persona': persona_str,
                    'context': context,
                    'response': target_text,
                    'input_text': input_text
                })
        
        return Dataset.from_list(processed_data)
    
    def tokenize(self, data: Dataset) -> Dataset:
        """Tokenize dataset for training"""
        max_length = self.config.get('max_length', 512)
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                truncation=True,
                padding='max_length',
                max_length=max_length,
                return_tensors="pt"
            )
        
        return data.map(tokenize_function, batched=True)
    
    def create_splits(self, data: Dataset) -> DatasetDict:
        """Split data into train/val/test"""
        train_ratio = self.config.get('train_split', 0.9)
        val_ratio = self.config.get('val_split', 0.05)
        
        data_list = [item for item in data]
        n_total = len(data_list)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_data = data_list[:n_train]
        val_data = data_list[n_train:n_train + n_val]
        test_data = data_list[n_train + n_val:]
        
        return DatasetDict({
            'train': Dataset.from_list(train_data),
            'validation': Dataset.from_list(val_data),
            'test': Dataset.from_list(test_data)
        })
    
    def augment(self, data: Dataset) -> Dataset:
        """Data augmentation for robustness"""
        # Simple augmentation: shuffle persona traits
        augmented_data = []
        
        for example in data:
            persona = example['persona']
            if "|" in persona:
                traits = persona.split("|")
                # Create variations by shuffling traits
                import random
                random.shuffle(traits)
                augmented_persona = " | ".join(traits)
                
                augmented_example = example.copy()
                augmented_example['persona'] = augmented_persona
                augmented_example['text'] = example['text'].replace(persona, augmented_persona)
                augmented_data.append(augmented_example)
        
        return Dataset.from_list(list(data) + augmented_data)
    
    def format_for_training(self, data: Dataset) -> Dataset:
        """Final formatting before training"""
        return data