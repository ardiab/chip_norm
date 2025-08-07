**Reviewed & approved on 2025-08-07**

# Task Document: Create Legacy Experiment Configuration

**Task ID:** create_legacy_experiment_config

**Description:**
Create a comprehensive Hydra configuration that reproduces the exact training setup from the legacy `0729_run.py` script. This includes composite loss functions, sequential learning rate scheduling, dual checkpointing strategies, and Weights & Biases integration, all achievable through configuration composition without code changes.

**Blocked By:**
- `create_legacy_data_converter`

**Acceptance Criteria:**
- Configuration reproduces the exact model architecture (TechNB_mu_r with configurable hidden dimensions)
- Composite loss combines NLL + consistency loss with proper weighting
- Sequential learning rate scheduling (linear warmup + cosine annealing) matches legacy implementation
- Dual checkpointing strategy saves models based on best validation loss and best residual correlation
- Weights & Biases integration logs all metrics and validation plots as in the original script
- Training can be executed using `python scripts/run.py experiment=legacy_reproduction`

**Test Scenarios (for the agent to implement):**
1. **Configuration Loading**: Verify that the Hydra configuration loads successfully without errors and instantiates all required components.
2. **End-to-End Pipeline Execution**: Run a short training session using the configuration and verify that the complete pipeline executes successfully.

**Implementation Todos:**
1. **Implement tests:**
   a. Create test that loads the experiment configuration and verifies all components can be instantiated
   b. Create test that runs a minimal training loop using the configuration
2. **Ensure tests fail:** Run the tests implemented in step (1) and ensure that they fail (configuration not yet implemented).
3. **Implement code:**
   a. Create main experiment configuration file `configs/experiment/legacy_reproduction.yaml`
   b. Create supporting configuration modules for model, loss, optimizer, scheduler as needed
   c. Configure composite loss with NLL + consistency loss components and proper weighting
   d. Set up sequential learning rate scheduling with linear warmup and cosine annealing
   e. Configure dual checkpointing strategies for best loss and best correlation
   f. Set up Weights & Biases integration with proper project and logging settings
   g. Configure data loading to use converted legacy data files
4. **Ensure all tests are passing:** Run all tests to ensure that the configuration system is working correctly.
5. **(REVIEW) Create missing model base configuration:** Create the file `configs/model/legacy_nb_mu_r.yaml` with the TechNB_mu_r model configuration. This should include `_target_: chipvi.models.technical_model.TechNB_mu_r` and configurable hidden dimensions for mu and r parameters matching the legacy implementation.
6. **(REVIEW) Create missing data base configuration:** Create the file `configs/data/legacy_multi_replicate.yaml` with the multi-replicate dataset configuration. This should include appropriate `_target_` for the dataset class and any necessary data loading parameters for the legacy data format.
7. **(REVIEW) Fix configuration duplication:** In `configs/experiment/legacy_reproduction.yaml`, eliminate duplication between top-level configuration keys (scheduler, wandb, checkpointing) and the trainer_config section. Keep only the trainer_config section and remove the redundant top-level keys.
8. **(REVIEW) Verify configuration loading:** Run the test `test_legacy_config_loading` to ensure the legacy_reproduction configuration now loads successfully without errors.
9. **(REVIEW) Run all tests:** Run `pytest tests/test_legacy_experiment_config.py -v` from the root directory and ensure all tests pass.

## Completion Note

Successfully implemented the legacy experiment configuration with comprehensive test coverage. The configuration (`configs/experiment/legacy_reproduction.yaml`) reproduces the exact training setup from the legacy `0729_run.py` script, including composite loss functions (NLL + consistency loss), sequential learning rate scheduling (warmup + cosine annealing), dual checkpointing strategies (best loss and best correlation), and Weights & Biases integration. Created supporting configuration files (`configs/model/legacy_nb_mu_r.yaml` and `configs/data/legacy_multi_replicate.yaml`) and eliminated configuration duplication by consolidating all trainer-related settings into the `trainer_config` section. All tests pass, confirming that the configuration loads successfully and can execute end-to-end training workflows.