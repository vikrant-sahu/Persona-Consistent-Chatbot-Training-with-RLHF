import unittest
from src.evaluation.metrics import calculate_bleu, calculate_rouge, calculate_f1

class TestEvaluationMetrics(unittest.TestCase):

    def setUp(self):
        # Sample predictions and references for testing
        self.predictions = [
            "The cat sat on the mat.",
            "Dogs are great companions."
        ]
        self.references = [
            ["The cat is on the mat.", "The mat has a cat."],
            ["Dogs make wonderful pets.", "Dogs are good friends."]
        ]

    def test_bleu(self):
        # Test BLEU score calculation
        bleu_score = calculate_bleu(self.predictions, self.references)
        self.assertIsInstance(bleu_score, float)
        self.assertGreaterEqual(bleu_score, 0.0)

    def test_rouge(self):
        # Test ROUGE score calculation
        rouge_score = calculate_rouge(self.predictions, self.references)
        self.assertIsInstance(rouge_score, dict)
        self.assertIn('rouge-1', rouge_score)
        self.assertIn('rouge-2', rouge_score)

    def test_f1(self):
        # Test F1 score calculation
        f1_score = calculate_f1(self.predictions, self.references)
        self.assertIsInstance(f1_score, float)
        self.assertGreaterEqual(f1_score, 0.0)

if __name__ == '__main__':
    unittest.main()