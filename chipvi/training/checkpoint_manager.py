"""Flexible checkpointing system for model training."""

import os
import logging
import torch
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages model checkpointing based on multiple metrics and strategies."""
    
    def __init__(self, output_dir: str, checkpoint_configs: List[Dict[str, Any]]):
        """Initialize the checkpoint manager.
        
        Args:
            output_dir: Directory to save checkpoint files
            checkpoint_configs: List of checkpoint configurations, each containing:
                - metric_name: Name of metric to monitor (supports nested with dots)
                - mode: 'min' or 'max' for improvement direction
                - filename: Checkpoint filename (e.g., 'best_loss.pt')
                - overwrite: Boolean for overwrite behavior
        
        Raises:
            ValueError: If configuration is invalid
        """
        # Validate configuration before proceeding
        self._validate_checkpoint_config(checkpoint_configs)
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_configs = checkpoint_configs
        
        # Track best values and filenames for each strategy
        self.best_values: Dict[str, Optional[float]] = {}
        self.filenames: Dict[str, str] = {}
        
        for config in checkpoint_configs:
            metric_name = config['metric_name']
            self.best_values[metric_name] = None
            self.filenames[metric_name] = config['filename']
    
    def update(self, metrics_dict: Dict[str, Any], model_state_dict: Dict[str, torch.Tensor], epoch: int) -> None:
        """Update checkpoint manager with new metrics and save if improved.
        
        Args:
            metrics_dict: Dictionary containing metric values (supports nested dicts)
            model_state_dict: Model's state_dict to save
            epoch: Current epoch number
        """
        for config in self.checkpoint_configs:
            metric_name = config['metric_name']
            mode = config['mode']
            filename = config['filename']
            overwrite = config['overwrite']
            
            # Extract metric value (supports nested access with dots)
            metric_value = self._get_nested_metric(metrics_dict, metric_name)
            
            if metric_value is None:
                logger.warning(f"Metric '{metric_name}' not found in metrics dictionary")
                continue
            
            # Check if this is an improvement
            if self._is_improvement(metric_value, metric_name, mode):
                # Save the checkpoint
                filepath = self.output_dir / filename
                metadata = {
                    'epoch': epoch,
                    'metric_value': metric_value,
                    'timestamp': datetime.now().isoformat(),
                    'config': config
                }
                
                self.save_checkpoint(model_state_dict, filepath, overwrite, metadata)
                
                # Update best value
                self.best_values[metric_name] = metric_value
                
                logger.info(f"Saved checkpoint for {metric_name}={metric_value:.6f} at epoch {epoch}")
    
    def _get_nested_metric(self, metrics_dict: Dict[str, Any], metric_name: str) -> Optional[float]:
        """Extract metric value from potentially nested dictionary using dot notation.
        
        Args:
            metrics_dict: Dictionary containing metrics
            metric_name: Metric name (can use dots for nested access)
            
        Returns:
            Metric value if found, None otherwise
        """
        keys = metric_name.split('.')
        current = metrics_dict
        
        try:
            for key in keys:
                current = current[key]
            return float(current)
        except (KeyError, TypeError, ValueError):
            return None
    
    def _is_improvement(self, metric_value: float, metric_name: str, mode: str) -> bool:
        """Check if the current metric value is an improvement.
        
        Args:
            metric_value: Current metric value
            metric_name: Name of the metric
            mode: 'min' or 'max' for improvement direction
            
        Returns:
            True if this is an improvement, False otherwise
        """
        current_best = self.best_values[metric_name]
        
        # First time seeing this metric
        if current_best is None:
            return True
        
        # Check improvement based on mode
        if mode == 'min':
            return metric_value < current_best
        elif mode == 'max':
            return metric_value > current_best
        else:
            raise ValueError(f"Invalid mode '{mode}'. Must be 'min' or 'max'")
    
    def save_checkpoint(
        self, 
        state_dict: Dict[str, torch.Tensor], 
        filepath: Path, 
        overwrite: bool, 
        metadata: Dict[str, Any]
    ) -> None:
        """Save checkpoint to disk.
        
        Args:
            state_dict: Model state dictionary to save
            filepath: Path where to save the checkpoint
            overwrite: Whether to overwrite existing files
            metadata: Additional metadata to save with checkpoint
        """
        # Handle overwrite behavior
        if not overwrite and filepath.exists():
            # Append timestamp to create unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            stem = filepath.stem
            suffix = filepath.suffix
            filepath = filepath.parent / f"{stem}_{timestamp}{suffix}"
        
        # Prepare checkpoint data
        checkpoint_data = {
            'state_dict': state_dict,
            **metadata
        }
        
        # Save checkpoint
        torch.save(checkpoint_data, filepath)
        logger.info(f"Checkpoint saved to {filepath}")
    
    @staticmethod
    def load_checkpoint(filepath: Union[str, Path]) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """Load checkpoint from disk.
        
        Args:
            filepath: Path to checkpoint file
            
        Returns:
            Tuple of (state_dict, metadata)
            
        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
        
        # Load checkpoint
        checkpoint_data = torch.load(filepath, map_location='cpu')
        
        # Extract state_dict and metadata
        state_dict = checkpoint_data.pop('state_dict')
        metadata = checkpoint_data  # Everything else is metadata
        
        return state_dict, metadata
    
    @staticmethod
    def _validate_checkpoint_config(checkpoint_configs: List[Dict[str, Any]]) -> None:
        """Validate checkpoint configuration.
        
        Args:
            checkpoint_configs: List of checkpoint configurations to validate
            
        Raises:
            ValueError: If configuration is invalid
        """
        filenames = set()
        for config in checkpoint_configs:
            # Check required fields
            required_fields = ['metric_name', 'mode', 'filename']
            for field in required_fields:
                if field not in config:
                    raise ValueError(f"Missing required field '{field}' in checkpoint configuration")
            
            # Check mode is valid
            if config['mode'] not in ['min', 'max']:
                raise ValueError(f"Invalid mode '{config['mode']}'. Must be 'min' or 'max'")
            
            # Check filename uniqueness
            filename = config['filename']
            if filename in filenames:
                raise ValueError(f"Duplicate filename '{filename}' in checkpoint configurations")
            filenames.add(filename)
            
            # Set default overwrite if not specified
            if 'overwrite' not in config:
                config['overwrite'] = True