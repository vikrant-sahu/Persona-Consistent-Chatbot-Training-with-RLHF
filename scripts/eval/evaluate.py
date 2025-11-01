#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from src.model.base import load_model
from src.eval.persona import PersonaEvaluator
from src.eval.engagement import EngagementEvaluator
from src.eval.quality import QualityEvaluator
from src.data.loader import DatasetLoader
from src.utils.config import load_config
import json

def main():
    """Run all evaluations"""
    print("=== Comprehensive Model Evaluation ===")
    
    # Load configuration
    config = load_config('config/eval.yaml')
    
    # Load model
    print("Loading model...")
    model = load_model('models/rlhf/final')
    
    # Load test data
    print("Loading test data...")
    loader = DatasetLoader()
    test_data = loader.load_processed_data('data/processed/test.jsonl')
    
    # Run evaluations
    results = {}
    
    if config['metrics']['persona_consistency']:
        print("Evaluating persona consistency...")
        persona_eval = PersonaEvaluator()
        results['persona_consistency'] = persona_eval.evaluate_consistency(model, test_data)
    
    if config['metrics']['engagement']:
        print("Evaluating engagement...")
        engagement_eval = EngagementEvaluator()
        results['engagement'] = engagement_eval.calculate_engagement_score(model, test_data)
    
    if any([config['metrics']['perplexity'], config['metrics']['bleu'], config['metrics']['rouge']]):
        print("Evaluating language quality...")
        quality_eval = QualityEvaluator()
        results['quality'] = {}
        
        if config['metrics']['perplexity']:
            results['quality']['perplexity'] = quality_eval.compute_perplexity(model, test_data)
        
        if config['metrics']['bleu']:
            results['quality']['bleu'] = quality_eval.compute_bleu(model, test_data)
        
        if config['metrics']['rouge']:
            results['quality']['rouge'] = quality_eval.compute_rouge(model, test_data)
        
        if config['metrics']['distinct_n']:
            results['quality']['distinct_n'] = quality_eval.compute_distinct_n(model, test_data)
    
    # Save results
    os.makedirs(config['output_dir'], exist_ok=True)
    results_path = os.path.join(config['output_dir'], 'evaluation_results.json')
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Evaluation completed! Results saved to {results_path}")
    print(f"Results: {json.dumps(results, indent=2)}")

if __name__ == "__main__":
    main()