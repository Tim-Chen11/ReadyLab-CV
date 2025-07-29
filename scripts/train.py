#!/usr/bin/env python
"""
Main training script using modular components with improved robustness
"""
import argparse
import sys
from pathlib import Path
from datetime import datetime
import logging

# FIXED: Ensure proper path handling
sys.path.append(str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader  # ADDED: Missing import

# Import modular components
from src.models.model_factory import ModelFactory, create_optimizer, create_scheduler
from src.data.url_dataset import URLDataset, CachedDataset
from src.data.transforms import get_transforms_for_model
from src.training.trainer import Trainer, collate_multitask_fn
from src.training.metrics import calculate_class_weights
from src.utils.logger import ExperimentLogger
from src.utils.helpers import (
    set_seed, get_device, count_parameters,
    create_experiment_structure, backup_code
)
from src.utils.visualization import plot_training_curves
from src.data.data_utils import create_data_loaders

def parse_arguments():
    """Parse command line arguments with additional options"""
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

    # Data arguments - FIXED: Correct default path
    parser.add_argument('--data_dir', type=str, default='data',  # FIXED: Removed ../
                        help='Data directory')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--use_cached', action='store_true',
                        help='Use pre-downloaded cached images')
    # REMOVED: refresh_cache (not implemented in data_utils)
    parser.add_argument('--use_subset', action='store_true',
                        help='Use a subset of data for quick testing')
    parser.add_argument('--subset_fraction', type=float, default=0.1,
                        help='Fraction of data to use if use_subset is True')

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
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Gamma parameter for focal loss')
    parser.add_argument('--focal_alpha', type=float, default=0.25,
                        help='Alpha parameter for focal loss')
    parser.add_argument('--class_weights', action='store_true',
                        help='Use class weights for imbalanced data')

    # Experiment arguments
    parser.add_argument('--exp_name', type=str, default=None,
                        help='Experiment name')
    parser.add_argument('--exp_dir', type=str, default='experiments',  # FIXED: Removed ../
                        help='Experiments directory')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use Weights & Biases logging')

    # Other arguments
    parser.add_argument('--early_stopping', type=int, default=5,
                        help='Early stopping patience (0 to disable)')
    # REMOVED: save_every (implement in Trainer if needed)
    parser.add_argument('--save_final', action='store_true', default=True,
                        help='Save final model state')
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU ID to use')
    
    parser.add_argument('--multi_task', action='store_true', 
                        help='Enable multi-task learning for decade and cluster classification')
    parser.add_argument('--decade_weight', type=float, default=1.0, 
                        help='Weight for decade loss in multi-task learning')
    parser.add_argument('--cluster_weight', type=float, default=1.0, 
                        help='Weight for cluster loss in multi-task learning')

    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_arguments()

    # Set random seed
    set_seed(args.seed)

    # Get device with improved handling
    device = get_device(args.gpu)
    if args.gpu is not None and not torch.cuda.is_available():
        print(f"Warning: GPU {args.gpu} requested but not available, falling back to CPU")
        device = torch.device("cpu")
    elif args.gpu is not None and args.gpu >= torch.cuda.device_count():
        print(f"Warning: GPU {args.gpu} invalid, falling back to CPU")
        device = torch.device("cpu")
    
    # Log GPU details
    if torch.cuda.is_available() and device.type == 'cuda':
        gpu_id = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(gpu_id)
        print(f"Using GPU {gpu_id}: {gpu_name}")

    # Initialize basic logging to ensure create_data_loaders can log
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Create config - FIXED: Better error handling
    try:
        config = ModelFactory.get_model_config(args.model_name)
    except KeyError:
        print(f"Error: Unknown model '{args.model_name}'. Available models:")
        for model in ModelFactory.list_available_models():
            print(f"  - {model}")
        sys.exit(1)
    except Exception as e:
        print(f"Failed to load model config: {e}")
        sys.exit(1)

    # Override with command line arguments - FIXED: Match data_utils expectations
    config.update({
        'model_name': args.model_name,
        'pretrained': args.pretrained,
        'epochs': args.epochs,
        'num_workers': args.num_workers,
        'use_cached': args.use_cached,
        'use_subset': args.use_subset,
        'subset_fraction': args.subset_fraction,
        'optimizer': args.optimizer,
        'scheduler': args.scheduler,
        'use_amp': args.use_amp,
        'gradient_clip_val': args.gradient_clip,
        'num_classes': 5,  # 5 decades
        'early_stopping': args.early_stopping,
        'seed': args.seed,
        'data_dir': args.data_dir,
        'use_class_weights': args.class_weights,
        # ADDED: Missing parameters that data_utils expects
        'use_weighted_sampling': False,  # Default to False
        'augmentation_level': 'medium',  # Default augmentation
        'max_download_retries': 3,      # Default retries
        'download_timeout': 10,         # Default timeout
        'multi_task': args.multi_task,
        'decade_weight': args.decade_weight,
        'cluster_weight': args.cluster_weight,
        'num_decade_classes': 5,  # Number of decade classes
        'num_cluster_classes': 5,  # Replace with actual number of cluster classes
    })

    # Override specific parameters if provided
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.learning_rate:
        config['learning_rate'] = args.learning_rate
    if args.weight_decay:
        config['weight_decay'] = args.weight_decay

    # IMPROVED: Better validation with specific error messages
    try:
        if config['batch_size'] > 128:
            print(f"Warning: Large batch size {config['batch_size']} may cause memory issues")
        if not (1e-6 <= config['learning_rate'] <= 1):
            print(f"Warning: Learning rate {config['learning_rate']} outside typical range [1e-6, 1]")
        if not (0 <= config['weight_decay'] <= 1):
            print(f"Warning: Weight decay {config['weight_decay']} outside typical range [0, 1]")
        if args.use_subset and not (0 < args.subset_fraction <= 1):
            print(f"Error: subset_fraction {args.subset_fraction} must be in (0, 1]")
            sys.exit(1)
    except KeyError as e:
        print(f"Error: Missing required config parameter: {e}")
        sys.exit(1)

    # Loss configuration
    loss_config = {'name': args.loss, 'params': {}}
    if args.loss == 'label_smoothing':
        loss_config['params']['smoothing'] = args.label_smoothing
    elif args.loss == 'focal':
        loss_config['params']['gamma'] = args.focal_gamma
        loss_config['params']['alpha'] = args.focal_alpha
    config['loss'] = loss_config

    # Create experiment name with timestamp for uniqueness
    if args.exp_name:
        exp_name = args.exp_name
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_name = f"{args.model_name}_{timestamp}"

    # Create experiment structure with error handling
    try:
        exp_dirs = create_experiment_structure(Path(args.exp_dir), exp_name)
    except Exception as e:
        print(f"Failed to create experiment structure: {e}")
        sys.exit(1)

    # Initialize experiment logger
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
    logger.logger.info(f"Command-line arguments: {vars(args)}")

    # Backup code
    try:
        backup_code(
            src_dir=Path(__file__).parent.parent,
            backup_dir=exp_dirs['configs'] / 'code_backup'
        )
    except Exception as e:
        logger.logger.warning(f"Failed to backup code: {e}")

    # Create model
    try:
        if args.multi_task:
            model = ModelFactory.create_model(
                config['model_name'],
                multi_task=True,  # Enable multi-task mode
                num_decade_classes=config['num_decade_classes'],
                num_cluster_classes=config['num_cluster_classes'],
                pretrained=config['pretrained']
            )
        else:
            model = ModelFactory.create_model(
                config['model_name'],
                num_classes=config['num_classes'],
                pretrained=config['pretrained']
            )
        model = model.to(device)
    except Exception as e:
        logger.logger.error(f"Failed to create model: {e}")
        sys.exit(1)

    # Log model info
    param_count = count_parameters(model)
    logger.logger.info(f"Model parameters: {param_count['total']:,} "
                      f"(Trainable: {param_count['trainable']:,})")

    # FIXED: Create data loaders with consistent parameters
    try:
        # Convert string path to Path object as expected by data_utils
        data_dir_path = Path(args.data_dir)
        
        train_loader, val_loader, class_weights, class_names = create_data_loaders(
            config,
            data_dir=data_dir_path,
            use_subset=args.use_subset,
            subset_fraction=args.subset_fraction,
            multi_task=args.multi_task
        )

        # Log class names
        if args.multi_task:
            logger.info(f"Decade class names: {class_names['decade']}")
            logger.info(f"Cluster class names: {class_names['cluster']}")
        else:
            logger.info(f"Class names: {class_names}")
    except FileNotFoundError as e:
        logger.logger.error(f"Data files not found: {e}")
        logger.logger.error("Please ensure data/splits/ directory contains train.json, val.json, test.json")
        sys.exit(1)
    except Exception as e:
        logger.logger.error(f"Failed to create data loaders: {e}")
        sys.exit(1)

    logger.logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.logger.info(f"Val samples: {len(val_loader.dataset)}")
    logger.logger.info(f"Class names: {class_names}")

    # Update loss config with class weights
    if args.class_weights and class_weights is not None:
        config['loss']['params']['class_weights'] = class_weights
        logger.logger.info(f"Using class weights: {class_weights.numpy()}")

    # Create optimizer and scheduler
    try:
        optimizer = create_optimizer(model, config)
        scheduler = create_scheduler(optimizer, config)
    except Exception as e:
        logger.logger.error(f"Failed to create optimizer or scheduler: {e}")
        sys.exit(1)

    # Create trainer
    try:
        trainer = Trainer(
            model=model,
            config=config,
            device=device,
            experiment_dir=exp_dirs['root'],
            logger=logger.logger,
            multi_task=args.multi_task  # Pass multi_task flag
        )
    except Exception as e:
        logger.logger.error(f"Failed to create trainer: {e}")
        sys.exit(1)

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        try:
            checkpoint = trainer.load_checkpoint(Path(args.resume))
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if scheduler and checkpoint.get('scheduler_state_dict'):
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch']
            logger.logger.info(f"Resumed from checkpoint: {args.resume} (epoch {start_epoch})")
        except FileNotFoundError:
            logger.logger.error(f"Checkpoint file not found: {args.resume}")
            sys.exit(1)
        except Exception as e:
            logger.logger.error(f"Failed to load checkpoint: {e}")
            sys.exit(1)

    # Train model
    try:
        logger.logger.info("Starting training...")
        results = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            start_epoch=start_epoch
        )
    except KeyboardInterrupt:
        logger.logger.info("Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.logger.error(f"Training failed: {e}")
        sys.exit(1)

    # Save final model state if specified
    if args.save_final:
        try:
            final_path = exp_dirs['checkpoints'] / 'final_checkpoint.pth'
            trainer.save_checkpoint(
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=config['epochs'],
                val_metrics=results['metrics_history']['val'][-1] if results.get('metrics_history') and results['metrics_history']['val'] else {},
                is_best=False
            )
            logger.logger.info(f"Final checkpoint saved to: {final_path}")
        except Exception as e:
            logger.logger.error(f"Failed to save final checkpoint: {e}")

    # Save final results
    logger.log_metrics(
        {
            'final/best_accuracy': results['best_metric'],
            'final/best_epoch': results['best_epoch'],
            'final/total_time_minutes': results['total_time'] / 60
        },
        step=config['epochs']
    )

    # Create visualizations with safety check
    if results.get('metrics_history'):
        try:
            plot_training_curves(
                results['metrics_history'],
                save_path=exp_dirs['visualizations'] / 'training_curves.png',
                show=False
            )
            logger.logger.info("Training curves saved")
        except Exception as e:
            logger.logger.warning(f"Failed to plot training curves: {e}")
    else:
        logger.logger.warning("No metrics history available to plot")

    # Log best model
    try:
        best_checkpoint_path = exp_dirs['checkpoints'] / 'best_checkpoint.pth'
        if best_checkpoint_path.exists():
            logger.log_model(best_checkpoint_path, aliases=['best', f"acc_{results['best_metric']:.2f}"])
    except Exception as e:
        logger.logger.warning(f"Failed to log model: {e}")

    # Finish logging
    logger.finish()

    print(f"\nTraining complete!")
    print(f"Best accuracy: {results['best_metric']*100:.2f}% at epoch {results['best_epoch']}")
    print(f"Results saved to: {exp_dirs['root']}")

if __name__ == '__main__':
    main()