# Task Document: Extend Configuration System

**Task ID:** extend_configuration

**Description:**
Update the Hydra configuration system to support all new features with clean, composable YAML configs. This includes configurations for composite losses, learning rate scheduling, Weights & Biases integration, flexible checkpointing strategies, and preprocessing options. The configuration system should maintain clarity while supporting complex experimental setups.

**Blocked By:**
- `implement_advanced_losses`
- `enhance_trainer`
- `implement_validation_metrics`
- `add_checkpointing`

**Acceptance Criteria:**
- Composite losses configurable with multiple loss functions and associated weights
- Learning rate scheduler parameters configurable (warmup epochs, scheduler type)
- W&B integration configurable with default enabled state
- Multiple checkpoint strategies can be specified simultaneously
- Preprocessing options including log transformation are configurable
- Command-line overrides work correctly with nested configuration structure

**Test Scenarios (for the agent to implement):**
1. **Composite Loss Configuration:** Test that YAML configs with multiple losses and weights are parsed correctly into CompositeLoss.
2. **Scheduler Configuration Validation:** Verify scheduler parameters are correctly loaded and applied to training.
3. **W&B Toggle Functionality:** Test W&B can be enabled/disabled via configuration and defaults to enabled.
4. **Checkpointing Strategy Parsing:** Ensure multiple checkpointing strategies are correctly interpreted from config.
5. **Log Transform Configuration:** Test preprocessing options like log transformation are applied to correct covariate dimensions.
6. **Hydra Override Compatibility:** Verify command-line overrides work with nested structure (e.g., ++loss.concordance.weight=0.2).
7. **Config Validation:** Test that invalid configurations are caught with helpful error messages.

**Implementation Todos:**
1. **Implement tests for configuration system:**
   a. Create `tests/test_config_extension.py`
   b. Test composite loss config parsing
   c. Test scheduler config extraction
   d. Test checkpoint strategy list parsing
   e. Test Hydra CLI overrides with nested parameters
   f. Test config validation catches errors

2. **Ensure tests fail:** Run tests before implementation

3. **Create loss configuration schemas in `configs/loss/`:**
   a. Create `nll_only.yaml` for baseline negative log-likelihood
   b. Create `composite_infonce.yaml`:
      ```yaml
      _target_: chipvi.training.losses.CompositeLoss
      losses:
        nll:
          _target_: chipvi.training.losses.nll_loss
          weight: 1.0
        concordance:
          _target_: chipvi.training.losses.concordance_loss_nce
          weight: 0.1
          tau: 0.1
      ```
   c. Create `composite_pearson.yaml` and `composite_quantile.yaml` similarly
   d. Ensure configs use proper Hydra instantiation syntax

4. **Extend trainer configuration in experiment configs:**
   a. Add to `configs/experiment/exp_001_concordance.yaml`:
      ```yaml
      scheduler:
        warmup_epochs: 2
        scheduler_type: cosine
        total_epochs: ${training.num_epochs}
      
      early_stopping:
        patience: 10
        monitor: val_loss
        mode: min
      
      gradient_clipping:
        max_norm: 1.0
      ```

5. **Add W&B configuration:**
   a. Create `configs/wandb/` directory
   b. Create `default.yaml`:
      ```yaml
      enabled: true
      project: chipvi
      entity: null  # Use default entity
      tags: []
      notes: null
      ```
   c. Create `disabled.yaml` with `enabled: false`
   d. Reference in experiment config with defaults

6. **Configure checkpointing strategies:**
   a. Add to experiment config:
      ```yaml
      checkpointing:
        strategies:
          - metric: val_loss
            mode: min
            filename: best_loss.pt
            overwrite: true
          - metric: val_residual_spearman
            mode: max
            filename: best_corr.pt
            overwrite: true
      ```

7. **Add preprocessing configuration:**
   a. Extend data configs with:
      ```yaml
      preprocessing:
        log_transform:
          enabled: false
          columns: [0, 5]  # Which covariate columns to transform
      ```
   b. Update DataModule or dataset to apply transforms

8. **Update scripts/run.py to handle new configs:**
   a. Parse composite loss configs and instantiate CompositeLoss
   b. Extract scheduler config and pass to Trainer
   c. Parse checkpoint strategies list
   d. Handle W&B config with defaults
   e. Apply preprocessing based on config

9. **Add config validation:**
   a. Check loss weights sum to reasonable value (warning if not 1.0)
   b. Validate scheduler warmup < total epochs
   c. Check checkpoint metric names are valid
   d. Ensure required fields present
   e. Provide helpful error messages for common mistakes

10. **Create example experiment configs:**
    a. `exp_002_infonce.yaml` using InfoNCE loss
    b. `exp_003_multiobjective.yaml` with multiple losses
    c. Document configuration options in comments

11. **Ensure all tests pass:** Verify configuration system complete and usable