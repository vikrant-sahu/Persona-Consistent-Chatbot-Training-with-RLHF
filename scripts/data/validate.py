#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from src.data.loader import DatasetLoader
import json

class DataValidator:
    """Validate processed data quality"""
    
    def check_lengths(self, data):
        """Check data length distribution"""
        lengths = [len(item['text'].split()) for item in data]
        return {
            'min_length': min(lengths),
            'max_length': max(lengths),
            'avg_length': sum(lengths) / len(lengths),
            'valid': min(lengths) > 10 and max(lengths) < 1000
        }
    
    def check_personas(self, data):
        """Check persona coverage and quality"""
        personas = set()
        for item in data:
            personas.add(item['persona'])
        
        return {
            'unique_personas': len(personas),
            'avg_traits_per_persona': sum(len(p.split('|')) for p in personas) / len(personas),
            'valid': len(personas) > 100
        }
    
    def check_formats(self, data):
        """Check data format consistency"""
        valid_count = 0
        for item in data:
            if all(key in item for key in ['text', 'persona', 'context', 'response']):
                valid_count += 1
        
        return {
            'valid_formats': valid_count,
            'total_samples': len(data),
            'valid_ratio': valid_count / len(data),
            'valid': valid_count / len(data) > 0.95
        }
    
    def check_duplicates(self, data):
        """Check for duplicate entries"""
        texts = set()
        duplicates = 0
        
        for item in data:
            if item['text'] in texts:
                duplicates += 1
            texts.add(item['text'])
        
        return {
            'duplicates': duplicates,
            'duplicate_rate': duplicates / len(data),
            'valid': duplicates / len(data) < 0.05
        }

def main():
    """Validate processed data quality"""
    print("=== Data Validation ===")
    
    loader = DatasetLoader()
    validator = DataValidator()
    
    # Load processed data
    datasets = {
        'train': 'data/processed/train.jsonl',
        'val': 'data/processed/val.jsonl', 
        'test': 'data/processed/test.jsonl',
        'preference_pairs': 'data/processed/preference_pairs.jsonl'
    }
    
    all_checks_passed = True
    
    for name, path in datasets.items():
        if os.path.exists(path):
            print(f"\nValidating {name}...")
            data = loader.load_processed_data(path)
            
            checks = {
                "length_distribution": validator.check_lengths(data),
                "persona_coverage": validator.check_personas(data),
                "format_validity": validator.check_formats(data),
                "duplicate_rate": validator.check_duplicates(data)
            }
            
            # Print results
            for check_name, result in checks.items():
                status = "✓ PASS" if result.get('valid', False) else "✗ FAIL"
                print(f"  {check_name}: {status}")
                for k, v in result.items():
                    if k != 'valid':
                        print(f"    {k}: {v}")
            
            # Check if all valid
            dataset_valid = all(result.get('valid', False) for result in checks.values())
            if not dataset_valid:
                all_checks_passed = False
    
    print(f"\n=== Overall Validation: {'✓ PASS' if all_checks_passed else '✗ FAIL'} ===")
    
    if not all_checks_passed:
        sys.exit(1)

if __name__ == "__main__":
    main()