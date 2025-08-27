#!/usr/bin/env python3
"""
Command-line interface for GRAFT training.
"""

import sys
from .trainer import *

def main():
    """Main entry point for the CLI."""
    # This preserves the original main function behavior
    if __name__ == '__main__':
        # Import the original main code from trainer
        from .trainer import parser, loader, TrainingConfig, get_model, prepare_data, ModelTrainer, setup_tracker
        
        args = parser.parse_args()

        trainloader, valloader, trainset, valset = loader(
            dataset=args.dataset, 
            dirs=args.dataset_dir, 
            trn_batch_size=args.batch_size, 
            val_batch_size=args.batch_size, 
            tst_batch_size=1000
        )

        config = TrainingConfig.from_args(args)
        model = get_model(args)
        data3 = prepare_data(args, trainloader)
        
        trainer = ModelTrainer(config, model, trainloader, valloader, trainset, data3)
        
        tracker = setup_tracker(args)
        if tracker:
            tracker.start()
        
        train_stats, val_stats = trainer.train()
        
        if tracker:
            tracker.stop()

if __name__ == '__main__':
    main()