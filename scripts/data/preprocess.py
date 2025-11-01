#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from src.data.loader import DatasetLoader
from src.data.processor import DataProcessor
from src.data.generator import PreferenceGenerator
from src.utils.config import load_config

def main():
    """Complete data preprocessing pipeline"""
    print("=== Data Preprocessing Pipeline ===")
    
    # Load configuration
    config = load_config('config/data.yaml')
    
    # Load datasets
    loader = DatasetLoader()
    print("Loading PersonaChat...")
    personachat = loader.load_personachat()
    print("Loading Blended Skill Talk...")
    bst = loader.load_blended_skill_talk()
    
    # Process data
    print("Processing datasets...")
    processor = DataProcessor(config['preprocessing'])
    processed_personachat = processor.preprocess(personachat)
    processed_bst = processor.preprocess(bst)
    
    # Combine datasets
    combined_data = processed_personachat + [item for item in processed_bst]
    
    # Create splits
    dataset_dict = processor.create_splits(combined_data)
    
    # Generate preference pairs and prompts
    print("Generating preference pairs...")
    generator = PreferenceGenerator(config)
    pairs = generator.generate_pairs(dataset_dict['train'])
    prompts = generator.extract_prompts(dataset_dict['train'])
    
    # Create test sets
    print("Creating test sets...")
    test_sets = generator.create_test_sets(dataset_dict['test'])
    
    # Save everything
    print("Saving processed data...")
    loader.save_processed_data([item for item in dataset_dict['train']], config['output']['train'])
    loader.save_processed_data([item for item in dataset_dict['validation']], config['output']['val'])
    loader.save_processed_data([item for item in dataset_dict['test']], config['output']['test'])
    loader.save_processed_data(pairs, config['output']['preference'])
    loader.save_processed_data([{'prompt': p} for p in prompts], config['output']['prompts'])
    
    print("Data preprocessing completed!")

if __name__ == "__main__":
    main()