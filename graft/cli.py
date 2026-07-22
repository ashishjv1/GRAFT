#!/usr/bin/env python3
"""
Command-line interface for GRAFT training.
"""

import sys
from .trainer import *

def main():
    """Main entry point for the CLI."""
    import argparse
    from .trainer import TrainingConfig, get_model, prepare_data, ModelTrainer, setup_tracker
    from .utils.loader import loader
    
    # Create argument parser (moved from trainer.py)
    parser = argparse.ArgumentParser(description="Model Training with smart Sampling")
    parser.add_argument('--batch_size', default='128', type=int, required=True, help='(default=%(default)s)')
    parser.add_argument('--numEpochs', default='5', type=int, required=True, help='(default=%(default)s)')
    parser.add_argument('--numClasses', default='10', type=int, required=True, help='(default=%(default)s)')
    parser.add_argument('--lr', default='0.001', type=float, required=False, help='learning rate')
    parser.add_argument('--device', default='cuda', type=str, required=False, help='device to use for decompositions')
    parser.add_argument('--model', default='resnet50', type=str, required=False, help='model to train')
    parser.add_argument('--dataset', default="cifar10", type=str, required=False, help='Indicate the dataset')
    parser.add_argument('--dataset_dir', default="./cifar10", type=str, required=False, help='Imagenet folder')
    parser.add_argument('--pretrained', default=False, action='store_true', help='use pretrained or not')
    parser.add_argument('--weight_decay', default=0.0001, type=float, required=False, help='Weight Decay to be used')
    parser.add_argument('--inp_channels', default="3", type=int, required=False, help='Number of input channels')
    parser.add_argument('--save_pickle', default=False,  action='store_true', help='to save or not to save U, S, V components')
    parser.add_argument('--decomp', default="numpy", type=str, required=False, help='To perform SVD using torch or numpy')
    parser.add_argument('--optimizer', default="sgd", type=str, required=True, help='Choice for optimizer')
    parser.add_argument('--select_iter', default="50", type=int, required=True, help='Data Selection Iteration')
    parser.add_argument('--fraction', default="0.50", type=float, required=True, help='fraction of data')
    parser.add_argument('--grad_clip', default=0.00, type=float, required=False, help='Gradient Clipping Value')
    parser.add_argument('--warm_start', default=False, action='store_true', help='Train with a warm-start')
    
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