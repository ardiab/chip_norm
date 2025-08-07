"""Mock infrastructure for Weights & Biases (wandb) to prevent network calls during testing."""

from typing import Any, Dict, List, Optional, Union
from contextlib import contextmanager


class MockRun:
    """Mock wandb run object."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize mock run."""
        self.config = config


class MockWandB:
    """Mock wandb module that intercepts all wandb API calls."""
    
    def __init__(self):
        """Initialize the mock wandb instance."""
        self.config: Dict[str, Any] = {}
        self.logged_metrics: List[Dict[str, Any]] = []
        self._initialized = False
        self._finished = False
        self.run: Optional[MockRun] = None
        
    def init(
        self,
        project: Optional[str] = None,
        entity: Optional[str] = None,
        tags: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> MockRun:
        """Mock wandb.init() call."""
        self._initialized = True
        self._finished = False
        
        # Store configuration parameters
        self.config = {
            'project': project,
            'entity': entity,
            'tags': tags,
            'config': config or {},
            **kwargs
        }
        
        self.run = MockRun(self.config)
        return self.run
        
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Mock wandb.log() call."""
        if not self._initialized:
            raise RuntimeError("wandb.init() must be called before logging")
            
        # Add step to metrics if provided
        log_entry = dict(metrics)
        if step is not None:
            log_entry['step'] = step
            
        self.logged_metrics.append(log_entry)
        
    def finish(self) -> None:
        """Mock wandb.finish() call."""
        self._finished = True
        self.run = None
        
    def is_finished(self) -> bool:
        """Check if wandb.finish() has been called."""
        return self._finished
        
    def get_logged_metrics(self) -> List[Dict[str, Any]]:
        """Get all logged metrics."""
        return self.logged_metrics.copy()
        
    def get_config(self) -> Dict[str, Any]:
        """Get the configuration dict passed to init."""
        return self.config.get('config', {})
        
    def get_project(self) -> Optional[str]:
        """Get the project name."""
        return self.config.get('project')
        
    def get_entity(self) -> Optional[str]:
        """Get the entity name."""
        return self.config.get('entity')
        
    def get_tags(self) -> Optional[List[str]]:
        """Get the tags list."""
        return self.config.get('tags')
        
    def get_metric_values(self, metric_name: str) -> List[Any]:
        """Get all values for a specific metric."""
        values = []
        for metrics in self.logged_metrics:
            if metric_name in metrics:
                values.append(metrics[metric_name])
        return values
        
    def get_latest_metric(self, metric_name: str) -> Any:
        """Get the latest value for a specific metric."""
        values = self.get_metric_values(metric_name)
        return values[-1] if values else None
        
    def reset(self) -> None:
        """Reset the mock to initial state."""
        self.config = {}
        self.logged_metrics = []
        self._initialized = False
        self._finished = False
        self.run = None


@contextmanager
def wandb_mock_context():
    """Context manager for using wandb mock in tests."""
    mock_wandb = MockWandB()
    try:
        yield mock_wandb
    finally:
        mock_wandb.finish()