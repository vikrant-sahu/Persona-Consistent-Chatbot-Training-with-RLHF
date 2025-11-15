import re
import os
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
    
    def _get_persona(self, example: Dict) -> List[str]:
        """Extract persona from example (flexible field names)"""
        # Google Synthetic-Persona-Chat uses 'user_1_persona' and 'user_2_persona' (with underscores)
        for field in ['user_1_persona', 'user_2_persona', 'personality', 'persona', 'personas', 'user_persona', 'persona_info', 'user 1 personas', 'user 2 personas']:
            if field in example and example[field]:
                value = example[field]
                if isinstance(value, list):
                    return value
                elif isinstance(value, str):
                    return [value]
        return []

    def _get_conversation(self, example: Dict) -> List[str]:
        """Extract conversation from example (flexible field names)"""
        # Google Synthetic-Persona-Chat uses 'utterances' field
        # Try all possible field names (ordered by likelihood)
        for field in ['utterances', 'history', 'conversation', 'dialogue', 'messages', 'Best Generated Conversation']:
            if field in example and example[field]:
                value = example[field]
                if isinstance(value, list):
                    return value
                elif isinstance(value, str):
                    # Split by turn markers or newlines
                    turns = [line.strip() for line in value.split('\n') if line.strip()]
                    return turns
        return []

    def preprocess(self, data: Dataset) -> Dataset:
        """
        Preprocess raw dataset into training format
        Supports both old (bavard/personachat_truecased) and new (google/Synthetic-Persona-Chat) formats
        """
        if len(data) == 0:
            print("Warning: Empty dataset passed to preprocess()")
            return Dataset.from_dict({'text': [], 'persona': [], 'context': [], 'response': [], 'input_text': []})

        processed_data = []
        skipped_examples = 0

        for example in data:
            # Extract persona and dialogue using flexible field names
            persona = self._get_persona(example)
            history = self._get_conversation(example)

            if not persona or not history:
                skipped_examples += 1
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

        # Log statistics
        if skipped_examples > 0:
            print(f"Skipped {skipped_examples}/{len(data)} examples due to missing persona or dialogue")

        # Convert list of dicts to dict of lists for Dataset.from_dict()
        if not processed_data:
            print("Warning: No valid examples found after preprocessing!")
            return Dataset.from_dict({'text': [], 'persona': [], 'context': [], 'response': [], 'input_text': []})

        print(f"Preprocessed {len(processed_data)} training examples from {len(data)} raw examples")
        dict_data = {key: [d[key] for d in processed_data] for key in processed_data[0].keys()}
        return Dataset.from_dict(dict_data)
    
    def tokenize(self, data: Dataset) -> Dataset:
        """Tokenize dataset for training

        OPTIMIZED: Uses dynamic padding (padding=False) to avoid wasting compute on padding tokens.
        The DataCollator will handle padding at batch time, which is 3-4x faster.

        Note: Does not use return_tensors="pt" when batched=True as this causes
        "Cannot convert list of tensors" errors. The Trainer handles tensor
        conversion automatically during training.
        """
        if len(data) == 0:
            print("Warning: Empty dataset passed to tokenize()")
            return data

        max_length = self.config.get('max_length', 512)

        def tokenize_function(examples):
            # OPTIMIZED: No padding here - let DataCollator handle it dynamically
            # This avoids padding every example to max_length (massive speedup)
            # Don't use return_tensors="pt" with batched=True - causes tensor shape errors
            # The Trainer will handle tensor conversion during training
            return self.tokenizer(
                examples['text'],
                truncation=True,
                padding=False,  # Dynamic padding by DataCollator (3-4x faster)
                max_length=max_length
            )

        # OPTIMIZED: Parallel tokenization for 4x speedup on multi-core systems
        num_proc = min(4, os.cpu_count() or 1)  # Use up to 4 cores
        print(f"Tokenizing with {num_proc} parallel processes...")

        return data.map(tokenize_function, batched=True, num_proc=num_proc)
    
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

        # Convert lists to dict format
        def list_to_dict(data_list):
            if not data_list:
                return {}
            return {key: [d[key] for d in data_list] for key in data_list[0].keys()}

        return DatasetDict({
            'train': Dataset.from_dict(list_to_dict(train_data)),
            'validation': Dataset.from_dict(list_to_dict(val_data)),
            'test': Dataset.from_dict(list_to_dict(test_data))
        })
    
    def augment(self, data: Dataset) -> Dataset:
        """Data augmentation for robustness"""
        if len(data) == 0:
            print("Warning: Empty dataset passed to augment()")
            return data

        # Simple augmentation: shuffle persona traits
        import random
        augmented_data = []

        for example in data:
            if 'persona' not in example:
                continue

            persona = example['persona']
            if "|" in persona:
                traits = [t.strip() for t in persona.split("|")]
                if len(traits) > 1:
                    # Create variations by shuffling traits
                    shuffled_traits = traits.copy()
                    random.shuffle(shuffled_traits)
                    augmented_persona = " | ".join(shuffled_traits)

                    augmented_example = example.copy()
                    augmented_example['persona'] = augmented_persona
                    augmented_example['text'] = example['text'].replace(persona, augmented_persona)
                    augmented_data.append(augmented_example)

        # Combine original and augmented data
        combined_data = list(data) + augmented_data
        if not combined_data:
            return data

        print(f"Augmented dataset: {len(data)} → {len(combined_data)} examples")
        dict_data = {key: [d[key] for d in combined_data] for key in combined_data[0].keys()}
        return Dataset.from_dict(dict_data)
    
    def format_for_training(self, data: Dataset) -> Dataset:
        """Final formatting before training"""
        return data