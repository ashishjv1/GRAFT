import unittest
import torch
import sys
import os
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from genindices import sample_selection
from decompositions import feature_sel

class TestModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten()
        )
        self.classifier = torch.nn.Linear(16, 10)
    
    def forward(self, x, last=False, freeze=False):
        features = self.features(x)
        if last:
            return features, None
        return self.classifier(features)

class TestGenIndices(unittest.TestCase):
    def setUp(self):
        # Set seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)

        # Create synthetic dataset and model for testing
        self.X = torch.randn(100, 3, 32, 32)
        self.y = torch.randint(0, 10, (100,))
        self.dataset = TensorDataset(self.X, self.y)
        self.dataloader = DataLoader(self.dataset, batch_size=32)
        
        # Use actual decomposition instead of synthetic data
        self.data3 = feature_sel(self.dataloader, 32, device="cpu", decomp_type="numpy")
        
        # Use custom test model that supports last/freeze arguments
        self.model = TestModel()
        self.model_state = self.model.state_dict()
        
    def test_sample_selection_size(self):
        # Test if correct number of samples are selected
        fraction = 0.5
        expected_size = int(len(self.dataset) * fraction)
        indices = sample_selection(
            self.dataloader, self.data3, self.model,
            self.model_state, 32, fraction,
            1, 10, "cpu", "cifar10"
        )
        print(f"Expected size: {expected_size}, Got: {len(indices)}")
        print(f"Indices shape: {indices.shape}, Unique values: {len(np.unique(indices))}")
        self.assertEqual(len(indices), expected_size)

    def test_sample_selection_uniqueness(self):
        # Test if selected indices are unique
        fraction = 0.3
        indices = sample_selection(
            self.dataloader, self.data3, self.model,
            self.model_state, 32, fraction,
            1, 10, "cpu", "cifar10"
        )
        unique_indices = np.unique(indices)
        print(f"Total indices: {len(indices)}, Unique indices: {len(unique_indices)}")
        self.assertEqual(len(indices), len(unique_indices))

    def test_sample_selection_range(self):
        # Test if selected indices are within valid range
        fraction = 0.4
        indices = sample_selection(
            self.dataloader, self.data3, self.model,
            self.model_state, 32, fraction,
            1, 10, "cpu", "cifar10"
        )
        self.assertTrue(all(0 <= idx < len(self.dataset) for idx in indices))

    def test_sample_selection_deterministic(self):
        # Test if selection is deterministic with same seed
        torch.manual_seed(42)
        indices1 = sample_selection(
            self.dataloader, self.data3, self.model,
            self.model_state, 32, 0.5,
            1, 10, "cpu", "cifar10"
        )
        
        torch.manual_seed(42)
        indices2 = sample_selection(
            self.dataloader, self.data3, self.model,
            self.model_state, 32, 0.5,
            1, 10, "cpu", "cifar10"
        )
        
        self.assertTrue(np.array_equal(indices1, indices2))

if __name__ == '__main__':
    unittest.main()
