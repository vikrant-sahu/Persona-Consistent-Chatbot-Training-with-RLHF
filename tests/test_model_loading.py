import unittest
from src.model.base_model import BaseModel

class TestModelLoading(unittest.TestCase):
    def setUp(self):
        self.model_name = "gpt2-medium"
        self.model = BaseModel(self.model_name)

    def test_model_initialization(self):
        self.assertIsNotNone(self.model)
        self.assertEqual(self.model.name, self.model_name)

    def test_model_loading(self):
        self.model.load()
        self.assertTrue(self.model.is_loaded)

    def test_model_forward_pass(self):
        input_data = "Hello, how are you?"
        output = self.model.forward(input_data)
        self.assertIsInstance(output, str)
        self.assertNotEqual(output, input_data)

if __name__ == "__main__":
    unittest.main()