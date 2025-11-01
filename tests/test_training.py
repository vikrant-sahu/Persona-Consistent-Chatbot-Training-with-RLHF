import unittest
from src.training.sft_trainer import SupervisedFineTuner
from src.model.base_model import BaseModel

class TestTraining(unittest.TestCase):

    def setUp(self):
        self.model = BaseModel()
        self.trainer = SupervisedFineTuner(model=self.model)

    def test_training_initialization(self):
        self.assertIsNotNone(self.trainer)
        self.assertEqual(self.trainer.model, self.model)

    def test_training_step(self):
        # Assuming the trainer has a method called `train_step`
        loss = self.trainer.train_step()
        self.assertIsInstance(loss, float)
        self.assertGreater(loss, 0)

    def test_save_checkpoint(self):
        checkpoint_path = "test_checkpoint.pth"
        self.trainer.save_checkpoint(checkpoint_path)
        self.assertTrue(os.path.exists(checkpoint_path))

    def tearDown(self):
        # Clean up any created files or resources
        if os.path.exists("test_checkpoint.pth"):
            os.remove("test_checkpoint.pth")

if __name__ == '__main__':
    unittest.main()