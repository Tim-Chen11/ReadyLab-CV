import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np
from pathlib import Path
import json
import time
import logging
from typing import Dict, Tuple, Optional, List
from sklearn.metrics import confusion_matrix, classification_report

from .losses import get_loss_function
from .metrics import MetricTracker

logger = logging.getLogger(__name__)


class Trainer:
    """Main trainer class for model training"""

    def __init__(
            self,
            model: nn.Module,
            config: Dict,
            device: torch.device,
            experiment_dir: Path,
            logger: Optional[logging.Logger] = None
    ):
        self.model = model
        self.config = config
        self.device = device
        self.exp_dir = experiment_dir
        self.logger = logger or logging.getLogger(__name__)

        # Initialize tracking
        self.current_epoch = 0
        self.best_val_metric = 0
        self.best_epoch = 0
        self.metrics_history = {'train': [], 'val': []}

        # Setup components
        self._setup_loss()
        self._setup_amp()

    def _setup_loss(self):
        """Setup loss function"""
        loss_config = self.config.get('loss', {})
        loss_name = loss_config.get('name', 'cross_entropy')
        loss_params = loss_config.get('params', {})

        self.criterion = get_loss_function(loss_name, **loss_params)

    def _setup_amp(self):
        """Setup automatic mixed precision"""
        self.use_amp = self.config.get('use_amp', True) and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None

    def train_epoch(
            self,
            dataloader: DataLoader,
            optimizer: torch.optim.Optimizer,
            scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
            epoch: int = 0
    ) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        metric_tracker = MetricTracker()

        pbar = tqdm(dataloader, desc=f'Epoch {epoch} - Training')

        for batch_idx, (images, labels, _) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)

            # Forward pass
            optimizer.zero_grad()

            with autocast(enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()

                # Gradient clipping
                if self.config.get('gradient_clip_val', 0) > 0:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['gradient_clip_val']
                    )

                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss.backward()

                # Gradient clipping
                if self.config.get('gradient_clip_val', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['gradient_clip_val']
                    )

                optimizer.step()

            # Update metrics
            metric_tracker.update('loss', loss.item())
            _, predicted = outputs.max(1)
            metric_tracker.update('accuracy', predicted.eq(labels).float().mean().item())

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{metric_tracker.avg("loss"):.4f}',
                'acc': f'{metric_tracker.avg("accuracy") * 100:.2f}%'
            })

            # Step scheduler if it's batch-wise
            if scheduler and self.config.get('scheduler_step', 'epoch') == 'batch':
                scheduler.step()

        return metric_tracker.get_averages()

    def validate(
            self,
            dataloader: DataLoader,
            epoch: int = 0
    ) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
        """Validate the model"""
        self.model.eval()
        metric_tracker = MetricTracker()

        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for images, labels, _ in tqdm(dataloader, desc=f'Epoch {epoch} - Validation'):
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                # Update metrics
                metric_tracker.update('loss', loss.item())
                _, predicted = outputs.max(1)
                metric_tracker.update('accuracy', predicted.eq(labels).float().mean().item())

                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        metrics = metric_tracker.get_averages()

        return metrics, np.array(all_predictions), np.array(all_labels)

    def train(
            self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            optimizer: torch.optim.Optimizer,
            scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
            start_epoch: int = 0
    ) -> Dict:
        """Main training loop"""
        self.logger.info("Starting training...")
        start_time = time.time()

        num_epochs = self.config.get('epochs', 30)

        for epoch in range(start_epoch, num_epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch(
                train_loader, optimizer, scheduler, epoch
            )

            # Validate
            val_metrics, predictions, labels = self.validate(
                val_loader, epoch
            )

            # Step scheduler if it's epoch-wise
            if scheduler and self.config.get('scheduler_step', 'epoch') == 'epoch':
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_metrics['loss'])
                else:
                    scheduler.step()

            # Log metrics
            current_lr = optimizer.param_groups[0]['lr']
            self.logger.info(
                f"Epoch {epoch}/{num_epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train Acc: {train_metrics['accuracy'] * 100:.2f}%, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Acc: {val_metrics['accuracy'] * 100:.2f}%, "
                f"LR: {current_lr:.6f}"
            )

            # Save metrics history
            self.metrics_history['train'].append({
                'epoch': epoch,
                **train_metrics,
                'lr': current_lr
            })
            self.metrics_history['val'].append({
                'epoch': epoch,
                **val_metrics
            })

            # Check if best model
            val_metric = val_metrics[self.config.get('monitor_metric', 'accuracy')]
            is_best = val_metric > self.best_val_metric

            if is_best:
                self.best_val_metric = val_metric
                self.best_epoch = epoch
                self.logger.info(f"New best model! {self.config.get('monitor_metric', 'accuracy')}: {val_metric:.4f}")

            # Save checkpoint
            self.save_checkpoint(
                optimizer, scheduler, epoch, val_metrics, is_best
            )

            # Early stopping
            if self.config.get('early_stopping', 0) > 0:
                epochs_without_improvement = epoch - self.best_epoch
                if epochs_without_improvement >= self.config['early_stopping']:
                    self.logger.info(f"Early stopping triggered after {epochs_without_improvement} epochs")
                    break

        # Training complete
        total_time = time.time() - start_time
        self.logger.info(f"Training complete in {total_time / 60:.2f} minutes")
        self.logger.info(
            f"Best {self.config.get('monitor_metric', 'accuracy')}: {self.best_val_metric:.4f} at epoch {self.best_epoch}")

        return {
            'best_metric': self.best_val_metric,
            'best_epoch': self.best_epoch,
            'total_time': total_time,
            'metrics_history': self.metrics_history
        }

    def save_checkpoint(
            self,
            optimizer: torch.optim.Optimizer,
            scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
            epoch: int,
            val_metrics: Dict[str, float],
            is_best: bool = False
    ):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'val_metrics': val_metrics,
            'best_metric': self.best_val_metric,
            'config': self.config,
            'metrics_history': self.metrics_history
        }

        # Save last checkpoint
        checkpoint_path = self.exp_dir / 'checkpoints' / 'last_checkpoint.pth'
        checkpoint_path.parent.mkdir(exist_ok=True)
        torch.save(checkpoint, checkpoint_path)

        # Save best checkpoint
        if is_best:
            best_path = self.exp_dir / 'checkpoints' / 'best_checkpoint.pth'
            torch.save(checkpoint, best_path)

        # Save periodic checkpoint
        if self.config.get('save_every', 0) > 0 and epoch % self.config['save_every'] == 0:
            periodic_path = self.exp_dir / 'checkpoints' / f'checkpoint_epoch_{epoch}.pth'
            torch.save(checkpoint, periodic_path)

    def load_checkpoint(self, checkpoint_path: Path) -> Dict:
        """Load checkpoint and restore training state"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.metrics_history = checkpoint.get('metrics_history', {'train': [], 'val': []})
        self.best_val_metric = checkpoint.get('best_metric', 0)
        self.current_epoch = checkpoint['epoch']

        self.logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")

        return checkpoint


class DistributedTrainer(Trainer):
    """Trainer for distributed training across multiple GPUs"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        self.rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

    def train_epoch(self, dataloader, optimizer, scheduler=None, epoch=0):
        """Override to handle distributed sampling"""
        if hasattr(dataloader.sampler, 'set_epoch'):
            dataloader.sampler.set_epoch(epoch)

        return super().train_epoch(dataloader, optimizer, scheduler, epoch)

    def save_checkpoint(self, *args, **kwargs):
        """Only save checkpoint on main process"""
        if self.rank == 0:
            super().save_checkpoint(*args, **kwargs)