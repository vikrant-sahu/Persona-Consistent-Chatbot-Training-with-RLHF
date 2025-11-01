import unittest
from src.evaluation.persona_consistency import evaluate_persona_consistency

class TestPersonaConsistency(unittest.TestCase):

    def setUp(self):
        # Setup code to initialize any required variables or states
        self.test_data = [
            {"input": "Hello, I'm a friendly chatbot.", "expected_persona": "friendly"},
            {"input": "I love to help people.", "expected_persona": "helpful"},
            {"input": "I'm not sure what to say.", "expected_persona": "neutral"},
        ]

    def test_persona_consistency(self):
        for data in self.test_data:
            with self.subTest(data=data):
                result = evaluate_persona_consistency(data["input"])
                self.assertEqual(result, data["expected_persona"], 
                                 f"Expected persona '{data['expected_persona']}' but got '{result}' for input: {data['input']}")

if __name__ == '__main__':
    unittest.main()