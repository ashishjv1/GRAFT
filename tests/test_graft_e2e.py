import unittest
import sys
import os
import torch
import tempfile

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from GRAFT import TrainingConfig, ModelTrainer, get_model, prepare_data
from utils.loader import loader

class TestGRAFTEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Setup that runs once for all tests
        cls.pickle_dir = "cifar10_pickle"
        if not os.path.exists(cls.pickle_dir):
            os.makedirs(cls.pickle_dir)
    
    @classmethod
    def tearDownClass(cls):
        # Cleanup that runs once after all tests
        import shutil
        if os.path.exists(cls.pickle_dir):
            shutil.rmtree(cls.pickle_dir)

    def setUp(self):
        self.test_args = type('TestArgs', (), {
            'batch_size': 32,
            'numEpochs': 2,
            'numClasses': 10,
            'lr': 0.001,
            'device': 'cpu',
            'model': 'resnet18',
            'dataset': 'cifar10',
            'dataset_dir': './data',
            'pretrained': False,
            'weight_decay': 0.0001,
            'inp_channels': 3,
            'save_pickle': True,  # Enable pickle saving for tests
            'decomp': 'numpy',
            'optimizer': 'sgd',
            'select_iter': 1,
            'fraction': 0.5,
            'grad_clip': 0.0,
            'warm_start': False
        })()

    def tearDown(self):
        # Clean up any test-specific resources
        pass

    def test_full_training_pipeline(self):
        # Load data
        trainloader, valloader, trainset, _ = loader(
            dataset=self.test_args.dataset,
            dirs=self.test_args.dataset_dir,
            trn_batch_size=self.test_args.batch_size,
            val_batch_size=self.test_args.batch_size,
            tst_batch_size=self.test_args.batch_size
        )

        # Initialize config
        config = TrainingConfig.from_args(self.test_args)

        # Get model
        model = get_model(self.test_args)
        
        # Prepare data
        data3 = prepare_data(self.test_args, trainloader)

        # Create trainer
        trainer = ModelTrainer(config, model, trainloader, valloader, trainset, data3)

        # Train for one epoch and verify results
        trainer.train()

        # Verify training produced results
        self.assertTrue(len(trainer.trn_losses) > 0)
        self.assertTrue(len(trainer.val_losses) > 0)
        self.assertTrue(len(trainer.trn_acc) > 0)
        self.assertTrue(len(trainer.val_acc) > 0)

    def test_model_saving(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.test_args.dataset_dir = tmpdir
            trainloader, valloader, trainset, _ = loader(
                dataset=self.test_args.dataset,
                dirs=self.test_args.dataset_dir,
                trn_batch_size=self.test_args.batch_size,
                val_batch_size=self.test_args.batch_size,
                tst_batch_size=self.test_args.batch_size
            )

            config = TrainingConfig.from_args(self.test_args)
            model = get_model(self.test_args)
            data3 = prepare_data(self.test_args, trainloader)
            trainer = ModelTrainer(config, model, trainloader, valloader, trainset, data3)
            
            # Train and verify model checkpoint was saved
            trainer.train()
            model_dir = f"saved_models/{self.test_args.model}"
            self.assertTrue(os.path.exists(model_dir))

if __name__ == '__main__':
    unittest.main()
