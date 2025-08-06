"""Configuration validation utilities for ChipVI."""

import logging
from typing import List, Dict, Any, Optional
from omegaconf import DictConfig


class ConfigValidationError(Exception):
    """Exception raised when configuration validation fails."""
    pass


class ConfigValidator:
    """Validates ChipVI configuration structures."""
    
    # Valid scheduler types
    VALID_SCHEDULER_TYPES = ['cosine', 'linear', 'exponential', 'step']
    
    # Valid checkpoint metrics
    VALID_CHECKPOINT_METRICS = [
        'val_loss', 'val_residual_spearman', 'val_residual_pearson',
        'train_loss', 'val_nll_loss', 'val_concordance_loss', 
        'val_pearson_loss', 'val_quantile_loss'
    ]
    
    # Valid checkpoint modes
    VALID_CHECKPOINT_MODES = ['min', 'max']
    
    def __init__(self, strict: bool = True):
        """Initialize validator.
        
        Args:
            strict: If True, raise exceptions on validation failures.
                   If False, only log warnings.
        """
        self.strict = strict
        self.logger = logging.getLogger(__name__)
    
    def _validate_or_warn(self, condition: bool, message: str) -> bool:
        """Validate condition and either raise exception or warn."""
        if not condition:
            if self.strict:
                raise ConfigValidationError(message)
            else:
                self.logger.warning(message)
                return False
        return True
    
    def validate_loss_config(self, loss_config: DictConfig) -> bool:
        """Validate loss configuration.
        
        Args:
            loss_config: Loss configuration to validate
            
        Returns:
            True if valid, False if warnings issued (non-strict mode)
            
        Raises:
            ConfigValidationError: If validation fails in strict mode
        """
        if not loss_config:
            return self._validate_or_warn(
                False, "Loss configuration is required but not provided"
            )
        
        # Handle Hydra instantiation format
        if '_target_' in loss_config:
            # Validate that target exists
            target = loss_config['_target_']
            if 'CompositeLoss' in target and 'losses' in loss_config:
                return self._validate_composite_loss_weights(loss_config)
        
        # Handle old format with explicit losses and weights
        if 'losses' in loss_config:
            return self._validate_composite_loss_weights(loss_config)
            
        return True
    
    def _validate_composite_loss_weights(self, loss_config: DictConfig) -> bool:
        """Validate composite loss weights."""
        if 'losses' not in loss_config:
            return True
            
        losses = loss_config['losses']
        
        # Extract weights from different formats
        weights = []
        if isinstance(losses, dict):
            # Hydra format: losses contains sub-configs with weights
            for loss_name, loss_cfg in losses.items():
                if isinstance(loss_cfg, dict) and 'weight' in loss_cfg:
                    weights.append(float(loss_cfg['weight']))
                else:
                    weights.append(1.0)  # Default weight
        elif isinstance(losses, list) and 'weights' in loss_config:
            # Old format: separate losses list and weights list
            weights = [float(w) for w in loss_config['weights']]
        
        if weights:
            total_weight = sum(weights)
            
            # Warning if weights don't sum to reasonable value
            if abs(total_weight - 1.0) > 0.1:
                self.logger.warning(
                    f"Loss weights sum to {total_weight:.3f}, which is not close to 1.0. "
                    f"Consider normalizing weights: {weights}"
                )
            
            # Check for negative weights
            negative_weights = [w for w in weights if w < 0]
            if negative_weights:
                return self._validate_or_warn(
                    False, 
                    f"Loss weights must be non-negative. Found negative weights: {negative_weights}"
                )
        
        return True
    
    def validate_scheduler_config(self, scheduler_config: DictConfig) -> bool:
        """Validate scheduler configuration.
        
        Args:
            scheduler_config: Scheduler configuration to validate
            
        Returns:
            True if valid, False if warnings issued (non-strict mode)
            
        Raises:
            ConfigValidationError: If validation fails in strict mode
        """
        if not scheduler_config:
            return True
            
        # Validate scheduler type
        if 'scheduler_type' in scheduler_config:
            scheduler_type = scheduler_config['scheduler_type']
            if scheduler_type not in self.VALID_SCHEDULER_TYPES:
                return self._validate_or_warn(
                    False,
                    f"Invalid scheduler type '{scheduler_type}'. "
                    f"Valid types are: {self.VALID_SCHEDULER_TYPES}"
                )
        
        # Validate warmup vs total epochs
        warmup_epochs = scheduler_config.get('warmup_epochs', 0)
        total_epochs = scheduler_config.get('total_epochs', 100)
        
        # Handle OmegaConf interpolation syntax
        if isinstance(total_epochs, str) and '${' in total_epochs:
            # Skip validation for interpolated values
            return True
            
        try:
            warmup_epochs = int(warmup_epochs)
            total_epochs = int(total_epochs)
            
            if warmup_epochs >= total_epochs:
                return self._validate_or_warn(
                    False,
                    f"Scheduler warmup epochs ({warmup_epochs}) must be less than "
                    f"total epochs ({total_epochs})"
                )
        except (ValueError, TypeError):
            # Skip validation if values can't be converted to int
            pass
            
        return True
    
    def validate_checkpoint_config(self, checkpoint_config: DictConfig) -> bool:
        """Validate checkpoint configuration.
        
        Args:
            checkpoint_config: Checkpoint configuration to validate
            
        Returns:
            True if valid, False if warnings issued (non-strict mode)
            
        Raises:
            ConfigValidationError: If validation fails in strict mode
        """
        if not checkpoint_config or 'strategies' not in checkpoint_config:
            return True
            
        strategies = checkpoint_config['strategies']
        if not strategies:
            return True
            
        for i, strategy in enumerate(strategies):
            # Check required fields
            required_fields = ['metric', 'mode', 'filename']
            for field in required_fields:
                if field not in strategy:
                    return self._validate_or_warn(
                        False,
                        f"Checkpoint strategy {i} missing required field '{field}'. "
                        f"Required fields: {required_fields}"
                    )
            
            # Validate metric name
            metric = strategy.get('metric', '')
            if metric not in self.VALID_CHECKPOINT_METRICS:
                self.logger.warning(
                    f"Checkpoint metric '{metric}' not in known metrics list: "
                    f"{self.VALID_CHECKPOINT_METRICS}. This may still be valid if it's "
                    f"a custom metric."
                )
            
            # Validate mode
            mode = strategy.get('mode', '')
            if mode not in self.VALID_CHECKPOINT_MODES:
                return self._validate_or_warn(
                    False,
                    f"Checkpoint mode '{mode}' must be one of: {self.VALID_CHECKPOINT_MODES}"
                )
            
            # Validate filename
            filename = strategy.get('filename', '')
            if not filename or not filename.strip():
                return self._validate_or_warn(
                    False,
                    f"Checkpoint strategy {i} has empty filename"
                )
        
        return True
    
    def validate_wandb_config(self, wandb_config: DictConfig) -> bool:
        """Validate Weights & Biases configuration.
        
        Args:
            wandb_config: W&B configuration to validate
            
        Returns:
            True if valid, False if warnings issued (non-strict mode)
        """
        if not wandb_config:
            return True
            
        # Check that enabled is boolean
        if 'enabled' in wandb_config:
            enabled = wandb_config['enabled']
            if not isinstance(enabled, bool):
                return self._validate_or_warn(
                    False,
                    f"W&B 'enabled' field must be boolean, got {type(enabled).__name__}: {enabled}"
                )
        
        # Check that project is provided if enabled
        enabled = wandb_config.get('enabled', True)
        if enabled and 'project' not in wandb_config:
            return self._validate_or_warn(
                False,
                "W&B project name is required when W&B is enabled"
            )
        
        return True
    
    def validate_preprocessing_config(self, preprocessing_config: DictConfig) -> bool:
        """Validate preprocessing configuration.
        
        Args:
            preprocessing_config: Preprocessing configuration to validate
            
        Returns:
            True if valid, False if warnings issued (non-strict mode)
        """
        if not preprocessing_config:
            return True
            
        # Validate log transform configuration
        if 'log_transform' in preprocessing_config:
            log_config = preprocessing_config['log_transform']
            
            # Check enabled is boolean
            if 'enabled' in log_config:
                enabled = log_config['enabled']
                if not isinstance(enabled, bool):
                    return self._validate_or_warn(
                        False,
                        f"Log transform 'enabled' field must be boolean, "
                        f"got {type(enabled).__name__}: {enabled}"
                    )
            
            # Check columns is list of integers
            if 'columns' in log_config:
                columns = log_config['columns']
                if not isinstance(columns, (list, tuple)):
                    return self._validate_or_warn(
                        False,
                        f"Log transform 'columns' must be a list, got {type(columns).__name__}"
                    )
                
                for col in columns:
                    if not isinstance(col, int) or col < 0:
                        return self._validate_or_warn(
                            False,
                            f"Log transform columns must be non-negative integers, "
                            f"got {col} in {columns}"
                        )
        
        return True
    
    def validate_full_config(self, cfg: DictConfig) -> bool:
        """Validate complete configuration.
        
        Args:
            cfg: Full configuration to validate
            
        Returns:
            True if all validations pass, False if any warnings issued (non-strict mode)
            
        Raises:
            ConfigValidationError: If any validation fails in strict mode
        """
        all_valid = True
        
        # Validate each section
        if 'loss' in cfg or 'loss_fn' in cfg:
            loss_config = cfg.get('loss', cfg.get('loss_fn'))
            all_valid &= self.validate_loss_config(loss_config)
        
        if 'scheduler' in cfg:
            all_valid &= self.validate_scheduler_config(cfg.scheduler)
        
        if 'checkpointing' in cfg:
            all_valid &= self.validate_checkpoint_config(cfg.checkpointing)
        
        if 'wandb' in cfg:
            all_valid &= self.validate_wandb_config(cfg.wandb)
        
        if 'preprocessing' in cfg:
            all_valid &= self.validate_preprocessing_config(cfg.preprocessing)
        
        return all_valid


def validate_config(cfg: DictConfig, strict: bool = False) -> bool:
    """Convenience function to validate configuration.
    
    Args:
        cfg: Configuration to validate
        strict: If True, raise exceptions on validation failures
        
    Returns:
        True if valid, False if warnings were issued
        
    Raises:
        ConfigValidationError: If validation fails and strict=True
    """
    validator = ConfigValidator(strict=strict)
    return validator.validate_full_config(cfg)