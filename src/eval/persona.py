"""
Persona consistency evaluation for chatbot responses.
"""

import json
import numpy as np
from typing import List, Dict, Any
from datasets import Dataset
from transformers import AutoTokenizer


class PersonaEvaluator:
    """Evaluate persona consistency in chatbot responses"""

    def __init__(self, tokenizer_name: str = 'gpt2-medium'):
        """
        Initialize PersonaEvaluator

        Args:
            tokenizer_name: Name of tokenizer to use for text processing
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def calculate_consistency_score(self, response: str, persona_traits: List[str],
                                   min_word_length: int = 3) -> float:
        """
        Calculate consistency score between response and persona traits

        Args:
            response: Generated response text
            persona_traits: List of persona trait strings
            min_word_length: Minimum word length to consider (filters common words)

        Returns:
            Consistency score between 0 and 1
        """
        if not persona_traits:
            return 0.0

        response_lower = response.lower()
        matches = 0

        for trait in persona_traits:
            # Extract key concepts from trait
            trait_words = set(trait.lower().split())

            # Remove common words
            common_words = {'i', 'am', 'have', 'has', 'had', 'like', 'love', 'enjoy',
                          'my', 'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be',
                          'to', 'of', 'in', 'on', 'at', 'for', 'with', 'and', 'or'}
            trait_words = trait_words - common_words

            # Check for word overlaps (with minimum length filter)
            for word in trait_words:
                if len(word) > min_word_length and word in response_lower:
                    matches += 1
                    break  # Count each trait only once

        score = matches / len(persona_traits) if persona_traits else 0.0
        return score

    def evaluate_consistency(self, model: Any, data: Dataset,
                           persona_field: str = 'persona',
                           context_field: str = 'context',
                           response_field: str = 'response',
                           max_samples: int = None,
                           generate_responses: bool = True) -> float:
        """
        Evaluate persona consistency on a dataset

        Args:
            model: Model to generate responses (if generate_responses=True)
            data: Dataset to evaluate
            persona_field: Field name containing persona traits
            context_field: Field name containing dialogue context
            response_field: Field name containing response (reference or to evaluate)
            max_samples: Maximum number of samples to evaluate (None = all)
            generate_responses: If True, generate responses from model.
                              If False, use existing responses in data.

        Returns:
            Average consistency score
        """
        scores = []
        n_samples = min(len(data), max_samples) if max_samples else len(data)

        for i in range(n_samples):
            example = data[i]

            # Extract persona
            persona = example.get(persona_field, '')
            if isinstance(persona, str) and '|' in persona:
                persona_traits = [t.strip() for t in persona.split('|')]
            elif isinstance(persona, list):
                persona_traits = persona
            else:
                persona_traits = [str(persona)] if persona else []

            # Get or generate response
            if generate_responses and model is not None:
                # Generate response from model
                context = example.get(context_field, [])
                if isinstance(context, list):
                    context_str = ' '.join(context[-3:])  # Last 3 turns
                else:
                    context_str = str(context)

                prompt = f"[PERSONA] {persona} [DIALOGUE] {context_str} [RESPONSE]"

                # Tokenize and generate
                if hasattr(model, 'tokenizer'):
                    tokenizer = model.tokenizer
                else:
                    tokenizer = self.tokenizer

                inputs = tokenizer(prompt, return_tensors='pt', max_length=512, truncation=True)
                input_ids = inputs.input_ids.to(model.device)

                with torch.no_grad():
                    outputs = model.generate(
                        input_ids,
                        max_new_tokens=50,
                        do_sample=True,
                        temperature=0.9,
                        top_p=0.9,
                        pad_token_id=tokenizer.eos_token_id
                    )

                response = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
            else:
                # Use existing response from data
                response = example.get(response_field, '')
                if isinstance(response, list):
                    response = ' '.join(response)

            # Calculate consistency score
            score = self.calculate_consistency_score(response, persona_traits)
            scores.append(score)

        average_score = np.mean(scores) if scores else 0.0
        return average_score

    def evaluate_batch_responses(self, responses: List[str], personas: List[List[str]]) -> List[float]:
        """
        Evaluate consistency for a batch of responses

        Args:
            responses: List of response texts
            personas: List of persona trait lists

        Returns:
            List of consistency scores
        """
        scores = []
        for response, persona_traits in zip(responses, personas):
            score = self.calculate_consistency_score(response, persona_traits)
            scores.append(score)
        return scores

    def evaluate_persona_file(self, responses: List[Dict], persona_file: str) -> float:
        """
        Evaluate consistency using persona data from file

        Args:
            responses: List of response dictionaries with 'persona_id' and 'text'
            persona_file: Path to JSON file containing persona data

        Returns:
            Average consistency score
        """
        with open(persona_file, 'r') as f:
            persona_data = json.load(f)

        scores = []
        for response in responses:
            persona_id = response.get('persona_id')
            response_text = response.get('text', '')

            persona_traits = persona_data.get(persona_id, [])
            if isinstance(persona_traits, str):
                persona_traits = [persona_traits]

            score = self.calculate_consistency_score(response_text, persona_traits)
            scores.append(score)

        average_score = np.mean(scores) if scores else 0.0
        return average_score

    def analyze_persona_adherence(self, response: str, persona_traits: List[str]) -> Dict[str, Any]:
        """
        Detailed analysis of persona adherence

        Args:
            response: Generated response
            persona_traits: List of persona traits

        Returns:
            Dictionary with detailed analysis
        """
        response_lower = response.lower()
        trait_matches = {}

        for trait in persona_traits:
            trait_words = set(trait.lower().split())
            common_words = {'i', 'am', 'have', 'like', 'love', 'my', 'a', 'an', 'the'}
            trait_words = trait_words - common_words

            matched_words = []
            for word in trait_words:
                if len(word) > 3 and word in response_lower:
                    matched_words.append(word)

            trait_matches[trait] = {
                'matched': len(matched_words) > 0,
                'matched_words': matched_words
            }

        n_matched = sum(1 for v in trait_matches.values() if v['matched'])
        consistency_score = n_matched / len(persona_traits) if persona_traits else 0.0

        return {
            'consistency_score': consistency_score,
            'total_traits': len(persona_traits),
            'matched_traits': n_matched,
            'trait_details': trait_matches
        }

    def compute_multi_turn_consistency(self, model: Any, data: Dataset,
                                     persona_field: str = 'persona',
                                     max_turns: int = 5) -> Dict[str, float]:
        """
        Evaluate persona consistency across multiple conversation turns

        Args:
            model: Model to generate multi-turn conversation
            data: Dataset with conversation examples
            persona_field: Field containing persona
            max_turns: Maximum number of turns to evaluate

        Returns:
            Dictionary with per-turn and average consistency scores
        """
        import torch

        turn_scores = {f'turn_{i+1}': [] for i in range(max_turns)}
        n_samples = min(len(data), 100)  # Limit for multi-turn evaluation

        for i in range(n_samples):
            example = data[i]

            # Extract persona
            persona = example.get(persona_field, '')
            if isinstance(persona, str) and '|' in persona:
                persona_traits = [t.strip() for t in persona.split('|')]
            elif isinstance(persona, list):
                persona_traits = persona
            else:
                persona_traits = [str(persona)] if persona else []

            # Simulate multi-turn conversation
            conversation_history = []

            for turn in range(max_turns):
                # Format prompt
                context_str = ' [SEP] '.join(conversation_history[-3:]) if conversation_history else ''
                prompt = f"[PERSONA] {persona} [DIALOGUE] {context_str} [RESPONSE]"

                # Generate response
                if hasattr(model, 'tokenizer'):
                    tokenizer = model.tokenizer
                else:
                    tokenizer = self.tokenizer

                inputs = tokenizer(prompt, return_tensors='pt', max_length=512, truncation=True)
                input_ids = inputs.input_ids.to(model.device)

                with torch.no_grad():
                    outputs = model.generate(
                        input_ids,
                        max_new_tokens=50,
                        do_sample=True,
                        temperature=0.9,
                        top_p=0.9,
                        pad_token_id=tokenizer.eos_token_id
                    )

                response = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)

                # Calculate consistency
                score = self.calculate_consistency_score(response, persona_traits)
                turn_scores[f'turn_{turn+1}'].append(score)

                # Add to conversation history
                conversation_history.append(f"User: How are you?")  # Simplified
                conversation_history.append(f"Bot: {response}")

        # Calculate average scores per turn
        results = {}
        for turn, scores in turn_scores.items():
            results[turn] = np.mean(scores) if scores else 0.0

        results['average'] = np.mean([v for v in results.values()])

        return results
