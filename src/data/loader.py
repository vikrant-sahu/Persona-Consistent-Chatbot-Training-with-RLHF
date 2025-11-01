import json
from datasets import load_dataset, Dataset
from typing import List, Dict, Any
import os

class DatasetLoader:
    """Load and cache datasets from HuggingFace"""
    
    def __init__(self, cache_dir: str = "./data/raw"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def load_personachat(self, split: str = 'train') -> Dataset:
        """Load PersonaChat dataset"""
        return load_dataset("bavard/personachat_truecased", split=split, cache_dir=self.cache_dir)
    
    def load_blended_skill_talk(self, split: str = 'train') -> Dataset:
        """Load Blended Skill Talk dataset"""
        return load_dataset("blended_skill_talk", split=split, cache_dir=self.cache_dir)
    
    def load_custom(self, path: str) -> List[Dict]:
        """Load custom persona dataset"""
        if os.path.exists(path):
            with open(path, 'r') as f:
                return [json.loads(line) for line in f]
        return []
    
    def extract_personas(self, data: Dataset) -> List[str]:
        """Extract persona traits from dataset"""
        personas = set()
        for example in data:
            if 'personality' in example:
                personas.update(example['personality'])
            elif 'persona' in example:
                if isinstance(example['persona'], list):
                    personas.update(example['persona'])
                else:
                    personas.add(example['persona'])
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