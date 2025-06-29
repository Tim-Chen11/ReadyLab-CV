import logging
import sys
from pathlib import Path
from datetime import datetime
import json
import wandb
from typing import Dict, Optional, Any


def setup_logger(
        name: str,
        log_dir: Optional[Path] = None,
        log_file: Optional[str] = None,
        level: int = logging.INFO,
        format_str: Optional[str] = None
) -> logging.Logger:
    """
    Setup logger with file and console handlers

    Args:
        name: Logger name
        log_dir: Directory for log files
        log_file: Log file name
        level: Logging level
        format_str: Custom format string

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers
    logger.handlers = []

    # Default format
    if format_str is None:
        format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    formatter = logging.Formatter(format_str)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_dir and log_file:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_dir / log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class ExperimentLogger:
    """Logger for ML experiments with multiple backends"""

    def __init__(
            self,
            experiment_name: str,
            project_name: str,
            log_dir: Path,
            config: Dict,
            use_wandb: bool = True,
            use_tensorboard: bool = True
    ):
        self.experiment_name = experiment_name
        self.project_name = project_name
        self.log_dir = Path(log_dir)
        self.config = config

        # Create directories
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Setup file logger
        self.logger = setup_logger(
            name=experiment_name,
            log_dir=self.log_dir,
            log_file='experiment.log'
        )

        # Setup experiment tracking
        self.use_wandb = use_wandb and self._init_wandb()
        self.use_tensorboard = use_tensorboard and self._init_tensorboard()

        # Log initial config
        self.log_config(config)

    def _init_wandb(self) -> bool:
        """Initialize Weights & Biases"""
        try:
            wandb.init(
                project=self.project_name,
                name=self.experiment_name,
                config=self.config,
                dir=self.log_dir
            )
            self.logger.info("Initialized W&B logging")
            return True
        except Exception as e:
            self.logger.warning(f"Failed to initialize W&B: {e}")
            return False

    def _init_tensorboard(self) -> bool:
        """Initialize TensorBoard"""
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.tb_writer = SummaryWriter(
                log_dir=self.log_dir / 'tensorboard'
            )
            self.logger.info("Initialized TensorBoard logging")
            return True
        except Exception as e:
            self.logger.warning(f"Failed to initialize TensorBoard: {e}")
            return False

    def log_config(self, config: Dict):
        """Log configuration"""
        # Save to file
        config_path = self.log_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        self.logger.info(f"Configuration saved to {config_path}")

    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = ''):
        """Log metrics to all backends"""
        # Add prefix if specified
        if prefix:
            metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}

        # Log to console
        metrics_str = ', '.join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Step {step} - {metrics_str}")

        # Log to W&B
        if self.use_wandb:
            wandb.log(metrics, step=step)

        # Log to TensorBoard
        if self.use_tensorboard:
            for key, value in metrics.items():
                self.tb_writer.add_scalar(key, value, step)

    def log_model(self, model_path: Path, aliases: Optional[list] = None):
        """Log model artifact"""
        if self.use_wandb:
            artifact = wandb.Artifact(
                name=f"{self.experiment_name}_model",
                type='model'
            )
            artifact.add_file(str(model_path))
            wandb.log_artifact(artifact, aliases=aliases or [])

    def log_image(self, tag: str, image: Any, step: int):
        """Log image to tensorboard"""
        if self.use_tensorboard:
            self.tb_writer.add_image(tag, image, step)

    def log_text(self, tag: str, text: str, step: int):
        """Log text"""
        if self.use_tensorboard:
            self.tb_writer.add_text(tag, text, step)

        if self.use_wandb:
            wandb.log({tag: wandb.Html(text)}, step=step)

    def finish(self):
        """Cleanup logging"""
        if self.use_wandb:
            wandb.finish()

        if self.use_tensorboard:
            self.tb_writer.close()

        self.logger.info("Experiment logging finished")


class MetricsLogger:
    """Simple metrics logger to JSON file"""

    def __init__(self, log_path: Path):
        self.log_path = Path(log_path)
        self.metrics = []

    def log(self, epoch: int, metrics: Dict[str, float]):
        """Log metrics for an epoch"""
        entry = {
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        self.metrics.append(entry)

        # Save to file
        with open(self.log_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)

    def load(self) -> list:
        """Load metrics from file"""
        if self.log_path.exists():
            with open(self.log_path, 'r') as f:
                self.metrics = json.load(f)
        return self.metrics