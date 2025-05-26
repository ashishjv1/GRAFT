import unittest
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from GRAFT import TrainingConfig, ModelTrainer

class TestGRAFT(unittest.TestCase):
    def setUp(self):
        self.config = TrainingConfig(
            numEpochs=2,
            batch_size=32,
            device="cpu",
            net=None,
            model_name="resnet18",
            dataset_name="cifar10",
            trainloader=None,
            valloader=None,
            trainset=None,
            data3=None,
            optimizer_name="sgd",
            lr=0.01,
            weight_decay=1e-4,
            grad_clip=None,
            fraction=0.5,
            selection_iter=1,
            warm_start=False,
            imgntselloader=None,
            use_wandb=False  # Disable wandb for tests
        )

    def test_training_config_initialization(self):
        self.assertEqual(self.config.numEpochs, 2)
        self.assertEqual(self.config.batch_size, 32)
        self.assertEqual(self.config.device, "cpu")
        self.assertEqual(self.config.optimizer_name, "sgd")

    def test_config_from_args(self):
        class Args:
            def __init__(self):
                self.numEpochs = 5
                self.batch_size = 64
                self.device = "cuda"
                self.model = "resnet50"
                self.dataset = "cifar100"
                self.optimizer = "adam"
                self.lr = 0.001
                self.weight_decay = 1e-5
                self.grad_clip = 1.0
                self.fraction = 0.7
                self.select_iter = 2
                self.warm_start = True

        args = Args()
        config = TrainingConfig.from_args(args)
        
        self.assertEqual(config.numEpochs, 5)
        self.assertEqual(config.batch_size, 64)
        self.assertEqual(config.model_name, "resnet50")
        self.assertEqual(config.dataset_name, "cifar100")

    def test_invalid_optimizer(self):
        self.config.optimizer_name = "invalid_optimizer"
        with self.assertRaises(Exception):
            trainer = ModelTrainer(self.config, 
                                 torch.nn.Linear(10, 2), 
                                 None, None, None, None)
            trainer._setup()

    def test_subset_selection(self):
        # Create a small synthetic dataset
        X = torch.randn(100, 3, 32, 32)  # 100 images, 3 channels, 32x32 pixels
        y = torch.randint(0, 10, (100,))  # 10 classes
        from torch.utils.data import TensorDataset, DataLoader
        dataset = TensorDataset(X, y)
        trainloader = DataLoader(dataset, batch_size=32)
        
        self.config.fraction = 0.5
        self.config.selection_iter = 1
        self.config.warm_start = False  # Ensure warm start is off
        trainer = ModelTrainer(self.config, 
                             torch.nn.Conv2d(3, 10, 3), 
                             trainloader, trainloader, dataset, None)
        
        # Initial size should be same as dataset
        initial_size = len(trainer.trainloader.dataset)
        self.assertEqual(initial_size, len(dataset))
        
        # After training one epoch, size should change
        trainer.train()
        final_size = len(trainer.trainloader.dataset)
        self.assertLessEqual(final_size, initial_size)

    def test_training_step(self):
        # Create a small synthetic dataset
        X = torch.randn(100, 3, 32, 32)
        y = torch.randint(0, 10, (100,))
        from torch.utils.data import TensorDataset, DataLoader
        dataset = TensorDataset(X, y)
        trainloader = DataLoader(dataset, batch_size=32)
        
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(16, 10)
        )
        
        self.config.selection_iter = self.config.numEpochs + 1  # Disable selection during test
        trainer = ModelTrainer(self.config, model, trainloader, trainloader, dataset, None)
        
        # Save initial model state
        initial_params = [p.clone() for p in model.parameters()]
        
        # Run one training epoch
        trainer.train()
        
        # Check if parameters were updated
        current_params = [p for p in model.parameters()]
        params_changed = any(not torch.equal(i, c) for i, c in zip(initial_params, current_params))
        self.assertTrue(params_changed)

    def test_warm_start(self):
        X = torch.randn(100, 3, 32, 32)
        y = torch.randint(0, 10, (100,))
        from torch.utils.data import TensorDataset, DataLoader
        dataset = TensorDataset(X, y)
        trainloader = DataLoader(dataset, batch_size=32)
        
        self.config.warm_start = True
        trainer = ModelTrainer(self.config, 
                             torch.nn.Conv2d(3, 10, 3), 
                             trainloader, trainloader, dataset, None)
        
        # Check if first iteration uses full dataset
        self.assertEqual(len(trainer.trainloader.dataset), len(dataset))

if __name__ == '__main__':
    unittest.main()
