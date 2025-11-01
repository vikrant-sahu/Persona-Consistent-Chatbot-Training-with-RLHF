import random
from typing import List, Dict
from datasets import Dataset

class PreferenceGenerator:
    """Generate preference pairs and prompts for training"""
    
    def __init__(self, config: Dict):
        self.config = config
    
    def generate_pairs(self, data: Dataset, model=None) -> List[Dict]:
        """Generate preference pairs for reward model training"""
        pairs = []
        
        for example in data:
            persona = example['persona']
            context = example['context']
            chosen_response = example['response']
            
            # Generate rejected response (simplified - in practice use model generations)
            rejected_response = self._degrade_response(chosen_response, persona)
            
            pairs.append({
                'persona': persona,
                'context': context,
                'chosen': chosen_response,
                'rejected': rejected_response
            })
        
        return pairs
    
    def extract_prompts(self, data: Dataset) -> List[str]:
        """Extract prompts for PPO training"""
        prompts = []
        
        for example in data:
            persona = example['persona']
            context = example['context']
            
            # Format: persona + context (without response)
            prompt = f"[PERSONA] {persona} [DIALOGUE] {' [SEP] '.join(context)} [RESPONSE]"
            prompts.append(prompt)
        
        return prompts
    
    def create_test_sets(self, data: Dataset) -> Dict[str, Dataset]:
        """Create test sets for evaluation"""
        # Create persona consistency test
        consistency_test = []
        
        for example in data[:500]:  # Use first 500 examples for testing
            consistency_test.append({
                'persona': example['persona'],
                'context': example['context'],
                'expected_response': example['response']
            })
        
        # Create engagement test
        engagement_test = []
        for example in data[500:800]:
            engagement_test.append({
                'context': example['context'],
                'expected_engagement_markers': ['?', '!', '...']  # Simplified
            })
        
        return {
            'consistency_test': Dataset.from_list(consistency_test),
            'engagement_test': Dataset.from_list(engagement_test)
        }
    
    def _degrade_response(self, response: str, persona: str) -> str:
        """Create degraded version of response for negative examples"""
        degradation_strategies = [
            self._make_generic,
            self._shorten_response,
            self._contradict_persona
        ]
        
        strategy = random.choice(degradation_strategies)
        return strategy(response, persona)
    
    def _make_generic(self, response: str, persona: str) -> str:
        generic_responses = [
            "I don't know.",
            "That's interesting.",
            "I see.",
            "Okay.",
            "Thanks for sharing."
        ]
        return random.choice(generic_responses)
    
    def _shorten_response(self, response: str, persona: str) -> str:
        words = response.split()
        if len(words) > 5:
            return ' '.join(words[:3]) + "..."
        return response
    
    def _contradict_persona(self, response: str, persona: str) -> str:
        if "love" in response.lower():
            return response.replace("love", "don't like")
        return self._make_generic(response, persona)