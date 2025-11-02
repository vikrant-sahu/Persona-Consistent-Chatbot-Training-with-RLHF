"""
Quality evaluation metrics for dialogue generation.
"""

import torch
import numpy as np
from typing import Dict, List, Any
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import Counter
import evaluate


class QualityEvaluator:
    """Evaluate dialogue quality using perplexity, BLEU, and ROUGE metrics"""

    def __init__(self, model_name_or_path: str = None, device: str = None):
        """
        Initialize QualityEvaluator

        Args:
            model_name_or_path: Path to model for perplexity calculation (optional)
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        self.model = None
        self.tokenizer = None
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

        # Load model if provided
        if model_name_or_path:
            self.load_model(model_name_or_path)

        # Load evaluation metrics
        try:
            self.bleu_metric = evaluate.load('bleu')
            self.rouge_metric = evaluate.load('rouge')
        except Exception as e:
            print(f"Warning: Could not load evaluation metrics: {e}")
            self.bleu_metric = None
            self.rouge_metric = None

    def load_model(self, model_name_or_path: str):
        """Load model and tokenizer for perplexity calculation"""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16,
            device_map='auto'
        )
        self.model.eval()

    def compute_perplexity(self, model: Any, data: Dataset, text_field: str = 'text',
                          batch_size: int = 8, max_length: int = 512) -> float:
        """
        Compute perplexity on dataset

        Args:
            model: Model to evaluate (can be path or model object)
            data: Dataset to evaluate on
            text_field: Field name containing text
            batch_size: Batch size for evaluation
            max_length: Maximum sequence length

        Returns:
            Average perplexity score (lower is better)
        """
        # Load model if string path provided
        if isinstance(model, str):
            self.load_model(model)
            eval_model = self.model
            tokenizer = self.tokenizer
        else:
            eval_model = model
            tokenizer = self.tokenizer if self.tokenizer else AutoTokenizer.from_pretrained('gpt2-medium')
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

        eval_model.eval()
        total_loss = 0
        total_tokens = 0

        with torch.no_grad():
            for i in range(0, len(data), batch_size):
                batch = data[i:i+batch_size]
                texts = [item[text_field] if isinstance(item, dict) else item for item in batch]

                # Tokenize
                encodings = tokenizer(
                    texts,
                    max_length=max_length,
                    truncation=True,
                    padding=True,
                    return_tensors='pt'
                )

                input_ids = encodings.input_ids.to(eval_model.device)
                attention_mask = encodings.attention_mask.to(eval_model.device)

                # Compute loss
                outputs = eval_model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                loss = outputs.loss

                # Accumulate
                total_loss += loss.item() * input_ids.size(0)
                total_tokens += input_ids.size(0)

        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        perplexity = np.exp(avg_loss)

        return perplexity

    def compute_bleu(self, model: Any, data: Dataset, context_field: str = 'context',
                    response_field: str = 'response', persona_field: str = 'persona',
                    max_samples: int = None) -> Dict[str, float]:
        """
        Compute BLEU scores

        Args:
            model: Model to generate responses
            data: Dataset with reference responses
            context_field: Field containing dialogue context
            response_field: Field containing reference response
            persona_field: Field containing persona information
            max_samples: Maximum number of samples to evaluate

        Returns:
            Dictionary with BLEU scores
        """
        if self.bleu_metric is None:
            return {'bleu': 0.0, 'bleu_1': 0.0, 'bleu_2': 0.0, 'bleu_3': 0.0, 'bleu_4': 0.0}

        predictions = []
        references = []

        # Load model if needed
        if isinstance(model, str):
            self.load_model(model)
            eval_model = self.model
            tokenizer = self.tokenizer
        else:
            eval_model = model
            tokenizer = self.tokenizer if self.tokenizer else AutoTokenizer.from_pretrained('gpt2-medium')

        eval_model.eval()
        n_samples = min(len(data), max_samples) if max_samples else len(data)

        with torch.no_grad():
            for i in range(n_samples):
                example = data[i]

                # Format input
                persona = example.get(persona_field, '')
                context = example.get(context_field, [])
                if isinstance(context, list):
                    context_str = ' '.join(context[-3:])  # Last 3 turns
                else:
                    context_str = str(context)

                prompt = f"[PERSONA] {persona} [DIALOGUE] {context_str} [RESPONSE]"

                # Generate response
                inputs = tokenizer(prompt, return_tensors='pt', max_length=512, truncation=True)
                input_ids = inputs.input_ids.to(eval_model.device)

                outputs = eval_model.generate(
                    input_ids,
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=0.9,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id
                )

                generated = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
                reference = example.get(response_field, '')

                predictions.append(generated)
                references.append([reference])  # BLEU expects list of references

        # Compute BLEU
        results = self.bleu_metric.compute(predictions=predictions, references=references)

        return {
            'bleu': results.get('bleu', 0.0),
            'bleu_1': results.get('precisions', [0])[0] if results.get('precisions') else 0.0,
            'bleu_2': results.get('precisions', [0, 0])[1] if len(results.get('precisions', [])) > 1 else 0.0,
            'bleu_3': results.get('precisions', [0, 0, 0])[2] if len(results.get('precisions', [])) > 2 else 0.0,
            'bleu_4': results.get('precisions', [0, 0, 0, 0])[3] if len(results.get('precisions', [])) > 3 else 0.0,
        }

    def compute_rouge(self, model: Any, data: Dataset, context_field: str = 'context',
                     response_field: str = 'response', persona_field: str = 'persona',
                     max_samples: int = None) -> Dict[str, float]:
        """
        Compute ROUGE scores

        Args:
            model: Model to generate responses
            data: Dataset with reference responses
            context_field: Field containing dialogue context
            response_field: Field containing reference response
            persona_field: Field containing persona information
            max_samples: Maximum number of samples to evaluate

        Returns:
            Dictionary with ROUGE scores
        """
        if self.rouge_metric is None:
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}

        predictions = []
        references = []

        # Load model if needed
        if isinstance(model, str):
            self.load_model(model)
            eval_model = self.model
            tokenizer = self.tokenizer
        else:
            eval_model = model
            tokenizer = self.tokenizer if self.tokenizer else AutoTokenizer.from_pretrained('gpt2-medium')

        eval_model.eval()
        n_samples = min(len(data), max_samples) if max_samples else len(data)

        with torch.no_grad():
            for i in range(n_samples):
                example = data[i]

                # Format input
                persona = example.get(persona_field, '')
                context = example.get(context_field, [])
                if isinstance(context, list):
                    context_str = ' '.join(context[-3:])  # Last 3 turns
                else:
                    context_str = str(context)

                prompt = f"[PERSONA] {persona} [DIALOGUE] {context_str} [RESPONSE]"

                # Generate response
                inputs = tokenizer(prompt, return_tensors='pt', max_length=512, truncation=True)
                input_ids = inputs.input_ids.to(eval_model.device)

                outputs = eval_model.generate(
                    input_ids,
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=0.9,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id
                )

                generated = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
                reference = example.get(response_field, '')

                predictions.append(generated)
                references.append(reference)

        # Compute ROUGE
        results = self.rouge_metric.compute(predictions=predictions, references=references)

        return {
            'rouge1': results.get('rouge1', 0.0),
            'rouge2': results.get('rouge2', 0.0),
            'rougeL': results.get('rougeL', 0.0),
            'rougeLsum': results.get('rougeLsum', 0.0)
        }

    def compute_distinct_n(self, model: Any, data: Dataset, n: int = 2,
                          context_field: str = 'context', persona_field: str = 'persona',
                          max_samples: int = None) -> float:
        """
        Compute Distinct-N metric (measures diversity)

        Args:
            model: Model to generate responses
            data: Dataset to evaluate on
            n: N-gram size (1 for unigrams, 2 for bigrams)
            context_field: Field containing dialogue context
            persona_field: Field containing persona information
            max_samples: Maximum number of samples to evaluate

        Returns:
            Distinct-N score (ratio of unique n-grams to total n-grams)
        """
        # Load model if needed
        if isinstance(model, str):
            self.load_model(model)
            eval_model = self.model
            tokenizer = self.tokenizer
        else:
            eval_model = model
            tokenizer = self.tokenizer if self.tokenizer else AutoTokenizer.from_pretrained('gpt2-medium')

        eval_model.eval()
        all_ngrams = []
        n_samples = min(len(data), max_samples) if max_samples else len(data)

        with torch.no_grad():
            for i in range(n_samples):
                example = data[i]

                # Format input
                persona = example.get(persona_field, '')
                context = example.get(context_field, [])
                if isinstance(context, list):
                    context_str = ' '.join(context[-3:])
                else:
                    context_str = str(context)

                prompt = f"[PERSONA] {persona} [DIALOGUE] {context_str} [RESPONSE]"

                # Generate response
                inputs = tokenizer(prompt, return_tensors='pt', max_length=512, truncation=True)
                input_ids = inputs.input_ids.to(eval_model.device)

                outputs = eval_model.generate(
                    input_ids,
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=0.9,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id
                )

                generated = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)

                # Extract n-grams
                words = generated.lower().split()
                for j in range(len(words) - n + 1):
                    ngram = tuple(words[j:j+n])
                    all_ngrams.append(ngram)

        if not all_ngrams:
            return 0.0

        unique_ngrams = len(set(all_ngrams))
        total_ngrams = len(all_ngrams)

        return unique_ngrams / total_ngrams if total_ngrams > 0 else 0.0

    def evaluate_all(self, model: Any, data: Dataset, max_samples: int = None) -> Dict[str, float]:
        """
        Compute all quality metrics

        Args:
            model: Model to evaluate
            data: Dataset to evaluate on
            max_samples: Maximum samples for BLEU/ROUGE (None = all)

        Returns:
            Dictionary with all metrics
        """
        results = {}

        # Perplexity
        try:
            results['perplexity'] = self.compute_perplexity(model, data)
        except Exception as e:
            print(f"Error computing perplexity: {e}")
            results['perplexity'] = float('inf')

        # BLEU
        try:
            bleu_scores = self.compute_bleu(model, data, max_samples=max_samples)
            results.update(bleu_scores)
        except Exception as e:
            print(f"Error computing BLEU: {e}")
            results['bleu'] = 0.0

        # ROUGE
        try:
            rouge_scores = self.compute_rouge(model, data, max_samples=max_samples)
            results.update(rouge_scores)
        except Exception as e:
            print(f"Error computing ROUGE: {e}")
            results['rouge1'] = 0.0

        # Distinct-1 and Distinct-2
        try:
            results['distinct_1'] = self.compute_distinct_n(model, data, n=1, max_samples=max_samples)
            results['distinct_2'] = self.compute_distinct_n(model, data, n=2, max_samples=max_samples)
        except Exception as e:
            print(f"Error computing Distinct-N: {e}")
            results['distinct_1'] = 0.0
            results['distinct_2'] = 0.0

        return results
