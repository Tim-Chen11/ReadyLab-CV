import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np
from pathlib import Path
import json
import time
import logging
from typing import Dict, Tuple, Optional, List, Union
from sklearn.metrics import confusion_matrix, classification_report

from .losses import get_loss_function
from .metrics import MetricTracker
from ..models.model_factory import MultiTaskLoss

logger = logging.getLogger(__name__)


class Trainer:
    """Main trainer class for model training with multi-task support"""

    def __init__(
            self,
            model: nn.Module,
            config: Dict,
            device: torch.device,
            experiment_dir: Path,
            logger: Optional[logging.Logger] = None,
            multi_task: bool = False
    ):
        self.model = model
        self.config = config
        self.device = device
        self.exp_dir = experiment_dir
        self.logger = logger or logging.getLogger(__name__)
        self.multi_task = multi_task

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
        if self.multi_task:
            # Multi-task loss setup
            decade_weight = self.config.get('decade_weight', 1.0)
            cluster_weight = self.config.get('cluster_weight', 1.0)
            loss_type = self.config.get('loss_type', 'cross_entropy')
            
            self.criterion = MultiTaskLoss(
                decade_weight=decade_weight,
                cluster_weight=cluster_weight,
                loss_type=loss_type
            )
            self.logger.info(f"Multi-task loss: decade_weight={decade_weight}, cluster_weight={cluster_weight}")
        else:
            # Single-task loss setup (original)
            loss_config = self.config.get('loss', {})
            loss_name = loss_config.get('name', 'cross_entropy')
            loss_params = loss_config.get('params', {})
            self.criterion = get_loss_function(loss_name, **loss_params)

    def _setup_amp(self):
        """Setup automatic mixed precision"""
        self.use_amp = self.config.get('use_amp', True) and torch.cuda.is_available()
        self.scaler = GradScaler('cuda') if self.use_amp else None

    def _calculate_accuracy(self, outputs: Union[torch.Tensor, Dict], labels: Union[torch.Tensor, Dict]) -> Dict[str, float]:
        """Calculate accuracy for single-task or multi-task"""
        if self.multi_task:
            # Multi-task accuracy calculation
            accuracies = {}
            
            # Decade accuracy
            _, decade_pred = outputs['decade'].max(1)
            decade_acc = decade_pred.eq(labels['decade']).float().mean().item()
            accuracies['decade_accuracy'] = decade_acc
            
            # Cluster accuracy
            _, cluster_pred = outputs['cluster'].max(1)
            cluster_acc = cluster_pred.eq(labels['cluster']).float().mean().item()
            accuracies['cluster_accuracy'] = cluster_acc
            
            # Overall accuracy (average of both tasks)
            accuracies['accuracy'] = (decade_acc + cluster_acc) / 2
            
            return accuracies
        else:
            # Single-task accuracy calculation
            _, predicted = outputs.max(1)
            accuracy = predicted.eq(labels).float().mean().item()
            return {'accuracy': accuracy}

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
            images = images.to(self.device)
            
            # Handle labels for multi-task vs single-task
            if self.multi_task:
                # labels is a dict with 'decade' and 'cluster' keys
                if isinstance(labels, dict):
                    labels = {k: v.to(self.device) for k, v in labels.items()}
                else:
                    raise ValueError("Multi-task mode requires labels to be a dictionary")
            else:
                # labels is a tensor
                labels = labels.to(self.device)

            # Forward pass
            optimizer.zero_grad()

            with autocast('cuda', enabled=self.use_amp):
                outputs = self.model(images)
                
                if self.multi_task:
                    # Multi-task loss calculation
                    losses = self.criterion(outputs, labels)
                    loss = losses['total_loss']
                else:
                    # Single-task loss calculation
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
            if self.multi_task:
                # Track individual task losses
                metric_tracker.update('total_loss', losses['total_loss'].item())
                metric_tracker.update('decade_loss', losses['decade_loss'].item())
                metric_tracker.update('cluster_loss', losses['cluster_loss'].item())
                
                # Track accuracies
                accuracies = self._calculate_accuracy(outputs, labels)
                for acc_name, acc_value in accuracies.items():
                    metric_tracker.update(acc_name, acc_value)
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{metric_tracker.avg("total_loss"):.4f}',
                    'dec_acc': f'{metric_tracker.avg("decade_accuracy") * 100:.1f}%',
                    'cls_acc': f'{metric_tracker.avg("cluster_accuracy") * 100:.1f}%'
                })
            else:
                # Single-task metrics
                metric_tracker.update('loss', loss.item())
                accuracies = self._calculate_accuracy(outputs, labels)
                metric_tracker.update('accuracy', accuracies['accuracy'])
                
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
    ) -> Tuple[Dict[str, float], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Validate the model"""
        self.model.eval()
        metric_tracker = MetricTracker()

        if self.multi_task:
            all_predictions = {'decade': [], 'cluster': []}
            all_labels = {'decade': [], 'cluster': []}
        else:
            all_predictions = []
            all_labels = []

        with torch.no_grad():
            for images, labels, _ in tqdm(dataloader, desc=f'Epoch {epoch} - Validation'):
                images = images.to(self.device)
                
                # Handle labels
                if self.multi_task:
                    labels = {k: v.to(self.device) for k, v in labels.items()}
                else:
                    labels = labels.to(self.device)

                outputs = self.model(images)
                
                # Calculate loss
                if self.multi_task:
                    losses = self.criterion(outputs, labels)
                    loss = losses['total_loss']
                    
                    # Update loss metrics
                    metric_tracker.update('total_loss', losses['total_loss'].item())
                    metric_tracker.update('decade_loss', losses['decade_loss'].item())
                    metric_tracker.update('cluster_loss', losses['cluster_loss'].item())
                    
                    # Update accuracy metrics
                    accuracies = self._calculate_accuracy(outputs, labels)
                    for acc_name, acc_value in accuracies.items():
                        metric_tracker.update(acc_name, acc_value)
                    
                    # Collect predictions and labels
                    _, decade_pred = outputs['decade'].max(1)
                    _, cluster_pred = outputs['cluster'].max(1)
                    
                    all_predictions['decade'].extend(decade_pred.cpu().numpy())
                    all_predictions['cluster'].extend(cluster_pred.cpu().numpy())
                    all_labels['decade'].extend(labels['decade'].cpu().numpy())
                    all_labels['cluster'].extend(labels['cluster'].cpu().numpy())
                    
                else:
                    loss = self.criterion(outputs, labels)
                    metric_tracker.update('loss', loss.item())
                    
                    accuracies = self._calculate_accuracy(outputs, labels)
                    metric_tracker.update('accuracy', accuracies['accuracy'])
                    
                    # Collect predictions and labels
                    _, predicted = outputs.max(1)
                    all_predictions.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

        # Convert to numpy arrays
        if self.multi_task:
            predictions = {k: np.array(v) for k, v in all_predictions.items()}
            labels_np = {k: np.array(v) for k, v in all_labels.items()}
        else:
            predictions = np.array(all_predictions)
            labels_np = np.array(all_labels)

        metrics = metric_tracker.get_averages()
        
        return metrics, predictions, labels_np

    def _log_classification_reports(
        self, 
        predictions: Union[np.ndarray, Dict[str, np.ndarray]], 
        labels: Union[np.ndarray, Dict[str, np.ndarray]],
        class_names: Optional[Dict[str, List[str]]] = None
    ):
        """Log detailed classification reports"""
        if self.multi_task:
            # Multi-task classification reports
            for task in ['decade', 'cluster']:
                self.logger.info(f"\n{task.capitalize()} Classification Report:")
                task_class_names = class_names.get(task) if class_names else None
                
                report = classification_report(
                    labels[task], 
                    predictions[task],
                    target_names=task_class_names,
                    output_dict=False
                )
                self.logger.info(f"\n{report}")
        else:
            # Single-task classification report
            self.logger.info("\nClassification Report:")
            task_class_names = class_names if isinstance(class_names, list) else None
            
            report = classification_report(
                labels, 
                predictions,
                target_names=task_class_names,
                output_dict=False
            )
            self.logger.info(f"\n{report}")

    def train(
            self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            optimizer: torch.optim.Optimizer,
            scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
            start_epoch: int = 0,
            class_names: Optional[Dict[str, List[str]]] = None
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
                    # Use appropriate metric for scheduler
                    monitor_metric = 'total_loss' if self.multi_task else 'loss'
                    scheduler.step(val_metrics[monitor_metric])
                else:
                    scheduler.step()

            # Log metrics
            current_lr = optimizer.param_groups[0]['lr']
            
            if self.multi_task:
                self.logger.info(
                    f"Epoch {epoch}/{num_epochs} - "
                    f"Train Loss: {train_metrics['total_loss']:.4f} "
                    f"(Dec: {train_metrics['decade_loss']:.4f}, Cls: {train_metrics['cluster_loss']:.4f}), "
                    f"Train Acc: {train_metrics['accuracy'] * 100:.2f}% "
                    f"(Dec: {train_metrics['decade_accuracy'] * 100:.2f}%, Cls: {train_metrics['cluster_accuracy'] * 100:.2f}%), "
                    f"Val Loss: {val_metrics['total_loss']:.4f}, "
                    f"Val Acc: {val_metrics['accuracy'] * 100:.2f}% "
                    f"(Dec: {val_metrics['decade_accuracy'] * 100:.2f}%, Cls: {val_metrics['cluster_accuracy'] * 100:.2f}%), "
                    f"LR: {current_lr:.6f}"
                )
            else:
                self.logger.info(
                    f"Epoch {epoch}/{num_epochs} - "
                    f"Train Loss: {train_metrics['loss']:.4f}, "
                    f"Train Acc: {train_metrics['accuracy'] * 100:.2f}%, "
                    f"Val Loss: {val_metrics['loss']:.4f}, "
                    f"Val Acc: {val_metrics['accuracy'] * 100:.2f}%, "
                    f"LR: {current_lr:.6f}"
                )

            # Log detailed classification report every few epochs
            if epoch % self.config.get('log_report_every', 5) == 0:
                self._log_classification_reports(predictions, labels, class_names)

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
            monitor_metric = self.config.get('monitor_metric', 'accuracy')
            val_metric = val_metrics[monitor_metric]
            is_best = val_metric > self.best_val_metric

            if is_best:
                self.best_val_metric = val_metric
                self.best_epoch = epoch
                self.logger.info(f"New best model! {monitor_metric}: {val_metric:.4f}")

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
            'metrics_history': self.metrics_history,
            'multi_task': self.multi_task  # Save multi-task flag
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
        self.multi_task = checkpoint.get('multi_task', False)  # Load multi-task flag

        self.logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
        self.logger.info(f"Multi-task mode: {self.multi_task}")

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


def collate_multitask_fn(batch):
    """Custom collate function for multi-task learning"""
    images, labels, metadata = zip(*batch)
    
    # Stack images
    images = torch.stack(images)
    
    # Handle labels - check if multi-task or single task
    if isinstance(labels[0], dict):
        # Multi-task: separate decade and cluster labels
        decade_labels = torch.tensor([label['decade'] for label in labels])
        cluster_labels = torch.tensor([label['cluster'] for label in labels])
        labels = {
            'decade': decade_labels,
            'cluster': cluster_labels
        }
    else:
        # Single task: just decade labels
        labels = torch.tensor(labels)
    
    return images, labels, metadata