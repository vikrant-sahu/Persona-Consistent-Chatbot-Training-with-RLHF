"""
Engagement evaluation metrics for chatbot conversations.
"""

import numpy as np
from typing import List, Dict, Any
from datasets import Dataset
import torch


class EngagementEvaluator:
    """Evaluate engagement quality in chatbot conversations"""

    def __init__(self):
        """Initialize EngagementEvaluator"""
        # Engagement markers that indicate active conversation
        self.engagement_markers = {
            'questions': ['?', 'what', 'why', 'how', 'when', 'where', 'who'],
            'exclamations': ['!'],
            'continuation': ['...', 'and', 'also', 'additionally'],
            'empathy': ['understand', 'feel', 'sorry', 'glad', 'happy', 'sad'],
            'acknowledgment': ['yes', 'no', 'right', 'exactly', 'agree', 'see']
        }

    def calculate_engagement_score(self, response: str) -> Dict[str, float]:
        """
        Calculate engagement score for a single response

        Args:
            response: Generated response text

        Returns:
            Dictionary with engagement metrics
        """
        response_lower = response.lower()
        words = response_lower.split()

        scores = {}

        # Check for questions
        has_question_mark = '?' in response
        has_question_words = any(word in response_lower for word in self.engagement_markers['questions'])
        scores['asks_questions'] = 1.0 if (has_question_mark or has_question_words) else 0.0

        # Check for exclamations
        scores['shows_enthusiasm'] = 1.0 if '!' in response else 0.0

        # Check for continuation markers
        scores['encourages_continuation'] = 1.0 if any(
            marker in response_lower for marker in self.engagement_markers['continuation']
        ) else 0.0

        # Check for empathy markers
        scores['shows_empathy'] = 1.0 if any(
            marker in response_lower for marker in self.engagement_markers['empathy']
        ) else 0.0

        # Check for acknowledgment
        scores['acknowledges_user'] = 1.0 if any(
            marker in response_lower for marker in self.engagement_markers['acknowledgment']
        ) else 0.0

        # Response length (normalized)
        word_count = len(words)
        scores['response_length'] = min(word_count / 30.0, 1.0)  # Normalize to 30 words

        # Calculate overall engagement score (weighted average)
        weights = {
            'asks_questions': 0.25,
            'shows_enthusiasm': 0.15,
            'encourages_continuation': 0.15,
            'shows_empathy': 0.20,
            'acknowledges_user': 0.15,
            'response_length': 0.10
        }

        overall_score = sum(scores[k] * weights[k] for k in weights.keys())
        scores['overall_engagement'] = overall_score

        return scores

    def calculate_engagement_metrics(self, conversations: List[Dict]) -> Dict[str, Any]:
        """
        Calculate engagement metrics for a list of conversations

        Args:
            conversations: List of conversation dictionaries with 'user_message' and 'bot_response'

        Returns:
            Dictionary containing engagement metrics
        """
        if not conversations:
            return {
                'total_messages': 0,
                'total_user_messages': 0,
                'total_bot_responses': 0,
                'user_engagement_ratio': 0.0,
                'bot_engagement_ratio': 0.0,
                'avg_bot_engagement': 0.0
            }

        total_messages = len(conversations)
        total_user_messages = sum(1 for conv in conversations if conv.get('user_message'))
        total_bot_responses = sum(1 for conv in conversations if conv.get('bot_response'))

        # Calculate bot engagement scores
        bot_scores = []
        for conv in conversations:
            bot_response = conv.get('bot_response', '')
            if bot_response:
                score = self.calculate_engagement_score(bot_response)
                bot_scores.append(score['overall_engagement'])

        metrics = {
            'total_messages': total_messages,
            'total_user_messages': total_user_messages,
            'total_bot_responses': total_bot_responses,
            'user_engagement_ratio': total_user_messages / total_messages if total_messages > 0 else 0.0,
            'bot_engagement_ratio': total_bot_responses / total_messages if total_messages > 0 else 0.0,
            'avg_bot_engagement': np.mean(bot_scores) if bot_scores else 0.0,
            'median_bot_engagement': np.median(bot_scores) if bot_scores else 0.0,
            'min_bot_engagement': np.min(bot_scores) if bot_scores else 0.0,
            'max_bot_engagement': np.max(bot_scores) if bot_scores else 0.0
        }

        return metrics

    def evaluate_engagement(self, model: Any, data: Dataset,
                          persona_field: str = 'persona',
                          context_field: str = 'context',
                          response_field: str = 'response',
                          max_samples: int = None,
                          generate_responses: bool = True) -> float:
        """
        Evaluate engagement score on dataset

        Args:
            model: Model to generate responses (if generate_responses=True)
            data: Dataset to evaluate
            persona_field: Field name containing persona
            context_field: Field name containing dialogue context
            response_field: Field name containing response
            max_samples: Maximum number of samples (None = all)
            generate_responses: Whether to generate responses or use existing ones

        Returns:
            Average engagement score
        """
        engagement_scores = []
        n_samples = min(len(data), max_samples) if max_samples else len(data)

        for i in range(n_samples):
            example = data[i]

            # Get or generate response
            if generate_responses and model is not None:
                # Generate response from model
                persona = example.get(persona_field, '')
                context = example.get(context_field, [])

                if isinstance(context, list):
                    context_str = ' '.join(context[-3:])
                else:
                    context_str = str(context)

                prompt = f"[PERSONA] {persona} [DIALOGUE] {context_str} [RESPONSE]"

                # Get tokenizer
                if hasattr(model, 'tokenizer'):
                    tokenizer = model.tokenizer
                else:
                    from transformers import AutoTokenizer
                    tokenizer = AutoTokenizer.from_pretrained('gpt2-medium')
                    if tokenizer.pad_token is None:
                        tokenizer.pad_token = tokenizer.eos_token

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
                # Use existing response
                response = example.get(response_field, '')
                if isinstance(response, list):
                    response = ' '.join(response)

            # Calculate engagement
            score_dict = self.calculate_engagement_score(response)
            engagement_scores.append(score_dict['overall_engagement'])

        return np.mean(engagement_scores) if engagement_scores else 0.0

    def evaluate_multi_turn_engagement(self, conversations: List[List[str]]) -> Dict[str, float]:
        """
        Evaluate engagement across multi-turn conversations

        Args:
            conversations: List of conversation lists (each conversation is list of turns)

        Returns:
            Dictionary with engagement metrics
        """
        all_scores = []
        turn_scores = {}

        for conversation in conversations:
            for turn_idx, turn in enumerate(conversation):
                if turn_idx not in turn_scores:
                    turn_scores[turn_idx] = []

                score = self.calculate_engagement_score(turn)
                turn_scores[turn_idx].append(score['overall_engagement'])
                all_scores.append(score['overall_engagement'])

        results = {
            'overall_engagement': np.mean(all_scores) if all_scores else 0.0,
            'engagement_std': np.std(all_scores) if all_scores else 0.0,
            'turn_wise_engagement': {
                f'turn_{i}': np.mean(scores) if scores else 0.0
                for i, scores in turn_scores.items()
            }
        }

        return results

    def analyze_engagement_patterns(self, response: str) -> Dict[str, Any]:
        """
        Detailed engagement pattern analysis

        Args:
            response: Response text to analyze

        Returns:
            Dictionary with detailed engagement patterns
        """
        scores = self.calculate_engagement_score(response)

        # Additional analysis
        words = response.split()
        sentences = response.split('.')

        analysis = {
            'engagement_scores': scores,
            'word_count': len(words),
            'sentence_count': len([s for s in sentences if s.strip()]),
            'avg_sentence_length': len(words) / max(len(sentences), 1),
            'unique_words': len(set(words)),
            'lexical_diversity': len(set(words)) / len(words) if words else 0.0,
            'question_count': response.count('?'),
            'exclamation_count': response.count('!'),
            'has_personal_reference': any(word in response.lower() for word in ['i', 'me', 'my', 'mine']),
            'has_user_reference': any(word in response.lower() for word in ['you', 'your', 'yours'])
        }

        return analysis

    def compare_engagement_levels(self, responses1: List[str], responses2: List[str],
                                 labels: List[str] = None) -> Dict[str, Any]:
        """
        Compare engagement levels between two sets of responses

        Args:
            responses1: First set of responses
            responses2: Second set of responses
            labels: Optional labels for the two sets

        Returns:
            Comparison results
        """
        if labels is None:
            labels = ['Set 1', 'Set 2']

        scores1 = [self.calculate_engagement_score(r)['overall_engagement'] for r in responses1]
        scores2 = [self.calculate_engagement_score(r)['overall_engagement'] for r in responses2]

        comparison = {
            labels[0]: {
                'mean': np.mean(scores1) if scores1 else 0.0,
                'std': np.std(scores1) if scores1 else 0.0,
                'median': np.median(scores1) if scores1 else 0.0,
                'min': np.min(scores1) if scores1 else 0.0,
                'max': np.max(scores1) if scores1 else 0.0
            },
            labels[1]: {
                'mean': np.mean(scores2) if scores2 else 0.0,
                'std': np.std(scores2) if scores2 else 0.0,
                'median': np.median(scores2) if scores2 else 0.0,
                'min': np.min(scores2) if scores2 else 0.0,
                'max': np.max(scores2) if scores2 else 0.0
            },
            'difference': {
                'mean_diff': np.mean(scores1) - np.mean(scores2) if scores1 and scores2 else 0.0,
                'improvement_percent': ((np.mean(scores1) - np.mean(scores2)) / np.mean(scores2) * 100)
                                      if scores1 and scores2 and np.mean(scores2) > 0 else 0.0
            }
        }

        return comparison
