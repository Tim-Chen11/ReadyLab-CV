#!/usr/bin/env python
"""
Main training script using modular components
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import torch
from datetime import datetime

# Import modular components
from src.models.model_factory import ModelFactory, create_optimizer, create_scheduler
from src.data.url_dataset import URLDataset, CachedDataset
from src.data.transforms import get_transforms_for_model
from src.training.trainer import Trainer
from src.training.metrics import calculate_class_weights
from src.utils.logger import ExperimentLogger
from src.utils.helpers import (
    set_seed, get_device, count_parameters,
    create_experiment_structure, backup_code, get_experiment_name
)
from src.utils.visualization import plot_training_curves, plot_confusion_matrix
from src.data.data_utils import create_data_loaders

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train decade classifier')

    # Model arguments
    parser.add_argument('--model_name', type=str, required=True,
                        choices=ModelFactory.list_available_models(),
                        help='Model architecture to use')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained weights')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (default: model-specific)')
    parser.add_argument('--learning_rate', type=float, default=None,
                        help='Learning rate (default: model-specific)')
    parser.add_argument('--weight_decay', type=float, default=None,
                        help='Weight decay (default: model-specific)')

    # Data arguments
    parser.add_argument('--data_dir', type=str, default='../data',
                        help='Data directory')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--use_cached', action='store_true',
                        help='Use pre-downloaded cached images')

    # Optimization arguments
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['adamw', 'adam', 'sgd'],
                        help='Optimizer to use')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'step', 'exponential', 'reduce_on_plateau'],
                        help='Learning rate scheduler')
    parser.add_argument('--use_amp', action='store_true', default=True,
                        help='Use automatic mixed precision')
    parser.add_argument('--gradient_clip', type=float, default=0.0,
                        help='Gradient clipping value')

    # Loss arguments
    parser.add_argument('--loss', type=str, default='cross_entropy',
                        choices=['cross_entropy', 'label_smoothing', 'focal', 'weighted_ce'],
                        help='Loss function')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                        help='Label smoothing factor')
    parser.add_argument('--class_weights', action='store_true',
                        help='Use class weights for imbalanced data')

    # Experiment arguments
    parser.add_argument('--exp_name', type=str, default=None,
                        help='Experiment name')
    parser.add_argument('--exp_dir', type=str, default='../experiments',
                        help='Experiments directory')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use Weights & Biases logging')

    # Other arguments
    parser.add_argument('--early_stopping', type=int, default=0,
                        help='Early stopping patience (0 to disable)')
    parser.add_argument('--save_every', type=int, default=0,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU ID to use')

    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_arguments()

    # Set random seed
    set_seed(args.seed)

    # Get device
    device = get_device(args.gpu)

    # Create config
    config = ModelFactory.get_model_config(args.model_name)

    # Override with command line arguments
    config.update({
        'model_name': args.model_name,
        'pretrained': args.pretrained,
        'epochs': args.epochs,
        'num_workers': args.num_workers,
        'use_cached': args.use_cached,
        'optimizer': args.optimizer,
        'scheduler': args.scheduler,
        'use_amp': args.use_amp,
        'gradient_clip_val': args.gradient_clip,
        'num_classes': 5,  # 5 decades
        'early_stopping': args.early_stopping,
        'save_every': args.save_every,
        'seed': args.seed,
        'data_dir': args.data_dir,
    })

    # Override specific parameters if provided
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.learning_rate:
        config['learning_rate'] = args.learning_rate
    if args.weight_decay:
        config['weight_decay'] = args.weight_decay

    # Loss configuration
    loss_config = {'name': args.loss, 'params': {}}
    if args.loss == 'label_smoothing':
        loss_config['params']['smoothing'] = args.label_smoothing
    config['loss'] = loss_config

    # Create experiment name and structure
    exp_name = args.exp_name or get_experiment_name(args.model_name)
    exp_dirs = create_experiment_structure(Path(args.exp_dir), exp_name)

    # Initialize logger
    logger = ExperimentLogger(
        experiment_name=exp_name,
        project_name='decade-classifier',
        log_dir=exp_dirs['logs'],
        config=config,
        use_wandb=args.use_wandb,
        use_tensorboard=True
    )

    logger.logger.info(f"Starting experiment: {exp_name}")
    logger.logger.info(f"Using device: {device}")

    # Backup code
    backup_code(
        src_dir=Path(__file__).parent.parent,
        backup_dir=exp_dirs['configs'] / 'code_backup'
    )

    # Create model
    model = ModelFactory.create_model(
        config['model_name'],
        num_classes=config['num_classes'],
        pretrained=config['pretrained']
    )
    model = model.to(device)

    # Log model info
    param_count = count_parameters(model)
    logger.logger.info(f"Model parameters: {param_count['total']:,} "
                       f"(Trainable: {param_count['trainable']:,})")

    # Create data loaders
    train_loader, val_loader, class_weights, class_names = create_data_loaders(config)

    logger.logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.logger.info(f"Val samples: {len(val_loader.dataset)}")

    # Update loss config with class weights
    if args.class_weights and class_weights is not None:
        config['loss']['params']['class_weights'] = class_weights

    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)

    # Create trainer
    trainer = Trainer(
        model=model,
        config=config,
        device=device,
        experiment_dir=exp_dirs['root'],
        logger=logger.logger
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        checkpoint = trainer.load_checkpoint(Path(args.resume))
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']

    # Train model
    results = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        start_epoch=start_epoch
    )

    # Save final results
    logger.log_metrics(
        {
            'final/best_accuracy': results['best_metric'],
            'final/best_epoch': results['best_epoch'],
            'final/total_time_minutes': results['total_time'] / 60
        },
        step=config['epochs']
    )

    # Create visualizations
    logger.logger.info("Creating visualizations...")

    # Plot training curves
    plot_training_curves(
        results['metrics_history'],
        save_path=exp_dirs['visualizations'] / 'training_curves.png',
        show=False
    )

    # Log best model
    best_checkpoint_path = exp_dirs['checkpoints'] / 'best_checkpoint.pth'
    logger.log_model(best_checkpoint_path, aliases=['best', f"acc_{results['best_metric']:.2f}"])

    # Finish logging
    logger.finish()

    print(f"\nTraining complete!")
    print(f"Best accuracy: {results['best_metric']:.2f}% at epoch {results['best_epoch']}")
    print(f"Results saved to: {exp_dirs['root']}")


if __name__ == '__main__':
    main()