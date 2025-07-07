import argparse
import os
import sys
from pathlib import Path
import json
import time
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

from src.models.model_factory import ModelFactory, create_optimizer, create_scheduler
from src.data.url_dataset import URLDataset, CachedDataset
from src.data.transforms import get_transforms_for_model

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
    """Main trainer class"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Create experiment directory
        self.exp_dir = Path(f"../experiments/{config['model_name']}_{config['exp_name']}")
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        with open(self.exp_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        # Initialize best metrics
        self.best_val_acc = 0
        self.best_epoch = 0
        
        # Setup logging
        self.setup_logging()
        
    def setup_logging(self):
        """Setup file and tensorboard logging"""
        # File logging
        log_file = self.exp_dir / 'training.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        
        # Metrics file
        self.metrics_file = self.exp_dir / 'metrics.json'
        self.metrics = {'train': [], 'val': []}
    
    def create_model(self):
        """Create and setup model"""
        model = ModelFactory.create_model(
            self.config['model_name'],
            num_classes=self.config['num_classes'],
            pretrained=self.config.get('pretrained', True)
        )
        
        model = model.to(self.device)
        
        # Optionally freeze backbone
        if self.config.get('freeze_backbone', False):
            from src.models.model_factory import freeze_backbone
            freeze_backbone(model, self.config.get('freeze_ratio', 0.5))
        
        return model
    
    def create_dataloaders(self):
        """Create train and validation dataloaders"""
        # Get transforms
        train_transform = get_transforms_for_model(self.config['model_name'], is_training=True)
        val_transform = get_transforms_for_model(self.config['model_name'], is_training=False)
        
        # Choose dataset type
        dataset_class = CachedDataset if self.config.get('use_cached', False) else URLDataset
        
        # Create datasets
        if dataset_class == URLDataset:
            train_dataset = URLDataset(
                split_file='../data/splits/train.json',
                transform=train_transform,
                cache_dir=self.config.get('cache_dir', '../data/cache/images')
            )
            val_dataset = URLDataset(
                split_file='../data/splits/val.json',
                transform=val_transform,
                cache_dir=self.config.get('cache_dir', '../data/cache/images')
            )
        else:
            train_dataset = CachedDataset(
                split_file='../data/splits/train.json',
                images_dir='../data/cache/images',
                transform=train_transform
            )
            val_dataset = CachedDataset(
                split_file='../data/splits/val.json',
                images_dir='../data/cache/images',
                transform=val_transform
            )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True
        )
        
        logger.info(f"Train dataset: {len(train_dataset)} images")
        logger.info(f"Val dataset: {len(val_dataset)} images")
        
        return train_loader, val_loader, train_dataset.decades
    
    def train_epoch(self, model, dataloader, criterion, optimizer, scaler):
        """Train for one epoch"""
        model.train()
        
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(dataloader, desc='Training')
        for batch_idx, (images, labels, _) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            
            # Mixed precision training
            with autocast(enabled=self.config.get('use_amp', True)):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            # Backward pass
            if self.config.get('use_amp', True):
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        return total_loss / len(dataloader), 100. * correct / total
    
    def validate(self, model, dataloader, criterion):
        """Validate the model"""
        model.eval()
        
        total_loss = 0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels, _ in tqdm(dataloader, desc='Validation'):
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(dataloader)
        
        return avg_loss, accuracy, all_predictions, all_labels
    
    def save_checkpoint(self, model, optimizer, scheduler, epoch, val_acc, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'val_acc': val_acc,
            'config': self.config
        }
        
        # Save last checkpoint
        torch.save(checkpoint, self.exp_dir / 'last_checkpoint.pth')
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, self.exp_dir / 'best_checkpoint.pth')
            logger.info(f"Saved best checkpoint with accuracy: {val_acc:.2f}%")
    
    def train(self):
        """Main training loop"""
        # Create model
        model = self.create_model()
        
        # Create dataloaders
        train_loader, val_loader, decade_names = self.create_dataloaders()
        
        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = create_optimizer(model, self.config)
        scheduler = create_scheduler(optimizer, self.config)
        scaler = GradScaler() if self.config.get('use_amp', True) else None
        
        # Training loop
        logger.info("Starting training...")
        start_time = time.time()
        
        for epoch in range(self.config['epochs']):
            logger.info(f"\nEpoch {epoch+1}/{self.config['epochs']}")
            
            # Train
            train_loss, train_acc = self.train_epoch(
                model, train_loader, criterion, optimizer, scaler
            )
            
            # Validate
            val_loss, val_acc, predictions, labels = self.validate(
                model, val_loader, criterion
            )
            
            # Update scheduler
            if scheduler:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
            
            # Log metrics
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save metrics
            self.metrics['train'].append({
                'epoch': epoch + 1,
                'loss': train_loss,
                'acc': train_acc
            })
            self.metrics['val'].append({
                'epoch': epoch + 1,
                'loss': val_loss,
                'acc': val_acc
            })
            
            # Save checkpoint
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_epoch = epoch + 1
            
            self.save_checkpoint(model, optimizer, scheduler, epoch + 1, val_acc, is_best)
            
            # Save metrics after each epoch
            with open(self.metrics_file, 'w') as f:
                json.dump(self.metrics, f, indent=2)
        
        # Training complete
        total_time = time.time() - start_time
        logger.info(f"\nTraining complete in {total_time/60:.2f} minutes")
        logger.info(f"Best validation accuracy: {self.best_val_acc:.2f}% at epoch {self.best_epoch}")
        
        # Final evaluation on validation set
        logger.info("\nFinal evaluation...")
        val_loss, val_acc, predictions, labels = self.validate(model, val_loader, criterion)
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        report = classification_report(labels, predictions, target_names=decade_names)
        
        logger.info("\nClassification Report:")
        logger.info(report)
        
        # Save final results
        final_results = {
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch,
            'total_time_minutes': total_time / 60,
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'decade_names': decade_names
        }
        
        with open(self.exp_dir / 'final_results.json', 'w') as f:
            json.dump(final_results, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Train decade classifier')
    
    # Model arguments
    parser.add_argument('--model_name', type=str, required=True,
                        choices=ModelFactory.list_available_models(),
                        help='Model architecture to use')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained weights')
    
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
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--cache_dir', type=str, default='../data/cache/images',
                        help='Directory to cache downloaded images')
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
    
    # Other arguments
    parser.add_argument('--exp_name', type=str, default=None,
                        help='Experiment name (default: timestamp)')
    parser.add_argument('--freeze_backbone', action='store_true',
                        help='Freeze early layers of the model')
    parser.add_argument('--freeze_ratio', type=float, default=0.5,
                        help='Ratio of layers to freeze')
    
    args = parser.parse_args()
    
    # Create config
    config = ModelFactory.get_model_config(args.model_name)
    config.update({
        'model_name': args.model_name,
        'pretrained': args.pretrained,
        'epochs': args.epochs,
        'num_workers': args.num_workers,
        'cache_dir': args.cache_dir,
        'use_cached': args.use_cached,
        'optimizer': args.optimizer,
        'scheduler': args.scheduler,
        'use_amp': args.use_amp,
        'freeze_backbone': args.freeze_backbone,
        'freeze_ratio': args.freeze_ratio,
        'num_classes': 5,  # 5 decades
        'exp_name': args.exp_name or datetime.now().strftime('%Y%m%d_%H%M%S')
    })
    
    # Override with command line arguments if provided
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.learning_rate:
        config['learning_rate'] = args.learning_rate
    if args.weight_decay:
        config['weight_decay'] = args.weight_decay
    
    # Create trainer and start training
    trainer = Trainer(config)
    trainer.train()


if __name__ == '__main__':
    main()