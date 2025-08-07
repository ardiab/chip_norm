# Changelog

## 2025-08-07

- **Feature:** Create mock infrastructure for Weights & Biases (wandb) that intercepts all wandb API calls during testing. (Task: `create_wandb_mock`)
- **Feature:** Implement a minimal, lightweight model class for CPU-based testing that uses simple linear layers instead of complex MLPs while maintaining the same interface as TechNB_mu_r. (Task: `create_minimal_model`)
- **Feature:** Build utility functions to generate small synthetic datasets in the expected .npy file format for testing purposes. (Task: `create_synthetic_data_generator`)
- **Feature:** Design and create dedicated Hydra configuration files for testing that include multi-component loss setup, minimal training parameters, disabled wandb, and paths configured for synthetic test data. (Task: `create_test_configs`)
- **Feature:** Write the main end-to-end test that orchestrates the entire training pipeline by calling scripts/run.py with test configuration and verifies that the complete system works correctly from configuration parsing through training completion. (Task: `implement_e2e_test`)

## 2025-08-06

- **Feature:** Add three advanced loss functions and composite loss system to enable multi-objective training with biological replicate concordance constraints. (Task: `implement_advanced_losses`)
- **Feature:** Upgrade the Trainer class with professional training capabilities including learning rate scheduling, early stopping, gradient management, and experiment tracking. (Task: `enhance_trainer`)
- **Feature:** Add advanced validation metrics and visualizations for model evaluation including Spearman correlations, Probability Integral Transform (PIT) analysis, and comprehensive plotting. (Task: `implement_validation_metrics`)
- **Feature:** Implement a configurable multi-strategy checkpointing system that can save models based on various metrics and loss components. (Task: `add_checkpointing`)
- **Feature:** Extend the Hydra configuration system to support composite losses, learning rate scheduling, W&B integration, flexible checkpointing strategies, and preprocessing options with clean, composable YAML configs. (Task: `extend_configuration`)