import json
from datasets import load_dataset, Dataset
from typing import List, Dict, Any
import os

class DatasetLoader:
    """Load and cache datasets from HuggingFace"""

    def __init__(self, cache_dir: str = "./data/raw"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def load_personachat(self, split: str = 'train', use_synthetic: bool = True) -> Dataset:
        """
        Load PersonaChat dataset

        Args:
            split: Dataset split ('train', 'validation', 'test')
            use_synthetic: If True, use google/Synthetic-Persona-Chat (recommended)
                          If False, use old bavard/personachat_truecased

        Returns:
            Dataset with persona conversations
        """
        if use_synthetic:
            # Use new Google Synthetic-Persona-Chat dataset (better quality, more diverse)
            dataset_name = "google/Synthetic-Persona-Chat"
        else:
            # Legacy dataset (kept for backward compatibility)
            dataset_name = "bavard/personachat_truecased"

        return load_dataset(dataset_name, split=split, cache_dir=self.cache_dir)

    def load_blended_skill_talk(self, split: str = 'train') -> Dataset:
        """Load Blended Skill Talk dataset"""
        return load_dataset("blended_skill_talk", split=split, cache_dir=self.cache_dir)

    def load_custom(self, path: str) -> List[Dict]:
        """Load custom persona dataset"""
        if os.path.exists(path):
            with open(path, 'r') as f:
                return [json.loads(line) for line in f]
        return []

    def get_persona_field(self, example: Dict) -> List[str]:
        """
        Get persona traits from an example, handling different field names

        Returns:
            List of persona trait strings
        """
        # Try different possible field names (ordered by likelihood)
        # Google Synthetic-Persona-Chat uses 'user_1_persona' and 'user_2_persona' (with underscores)
        for field in ['user_1_persona', 'user_2_persona', 'personality', 'persona', 'personas', 'user_persona', 'persona_info', 'user 1 personas', 'user 2 personas']:
            if field in example and example[field]:
                value = example[field]
                if isinstance(value, list):
                    return value
                elif isinstance(value, str):
                    return [value]
        return []

    def get_conversation_field(self, example: Dict) -> List[str]:
        """
        Get conversation/dialogue from an example, handling different field names

        Returns:
            List of conversation turns
        """
        # Google Synthetic-Persona-Chat uses 'utterances' field
        # Try all possible field names (ordered by likelihood)
        for field in ['utterances', 'history', 'conversation', 'dialogue', 'messages', 'Best Generated Conversation']:
            if field in example and example[field]:
                value = example[field]
                if isinstance(value, list):
                    return value
                elif isinstance(value, str):
                    # Split by turn markers or newlines
                    # Assuming format like "User: ...\nAssistant: ...\n"
                    turns = [line.strip() for line in value.split('\n') if line.strip()]
                    return turns
        return []

    def extract_personas(self, data: Dataset) -> List[str]:
        """
        Extract persona traits from dataset
        Supports multiple field names for compatibility with different datasets
        """
        personas = set()
        for example in data:
            persona_traits = self.get_persona_field(example)
            personas.update(persona_traits)
        return list(personas)

    def save_processed_data(self, data: List[Dict], file_path: str):
        """Save processed data to JSONL"""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')

    def load_processed_data(self, file_path: str) -> List[Dict]:
        """Load processed data from JSONL"""
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        return data
