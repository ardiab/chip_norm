# Task Document: Enhance Trainer with Advanced Training Features

**Task ID:** enhance_trainer

**Description:**
Upgrade the Trainer class with professional training capabilities including learning rate scheduling, early stopping, gradient management, and experiment tracking. These enhancements will enable more efficient training, prevent overfitting, and provide comprehensive monitoring of the training process.

**Blocked By:**
- `implement_advanced_losses`

**Acceptance Criteria:**
- Learning rate follows warmup linear schedule then cosine annealing schedule
- Training stops on validation plateau with best model state preserved
- Gradients clipped to specified maximum norm to prevent explosion
- All metrics logged to Weights & Biases when enabled
- All features configurable via Hydra configuration system

**Test Scenarios (for the agent to implement):**
1. **Learning Rate Schedule Verification:** Test that learning rate follows correct warmup pattern (linear increase) and then cosine annealing pattern, checking values at key epochs.
2. **Early Stopping Functionality:** Verify training stops after specified patience epochs without improvement and that the best model state is preserved.
3. **Gradient Clipping Effect:** Test that gradient norms are properly limited to max_norm value when specified.
4. **W&B Logging Completeness:** Verify all metrics (loss, learning rate, gradient norms) are logged to W&B when enabled.
5. **W&B Disable Option:** Ensure training works correctly when W&B is disabled in configuration.
6. **Scheduler Milestone Transitions:** Test that the transition from warmup to cosine scheduler happens at the correct epoch.

**Implementation Todos:**
1. **Implement tests for enhanced trainer features:**
   a. Create `tests/test_trainer_enhancements.py`
   b. Test LR scheduler: mock trainer and verify LR values at epochs 0, warmup_end, mid-training, end
   c. Test early stopping: simulate validation losses and verify stopping at correct epoch
   d. Test gradient clipping: create gradients exceeding max_norm and verify clipping
   e. Test W&B integration: mock wandb.init and wandb.log, verify correct calls

2. **Ensure tests fail:** Run tests to confirm they fail before implementation

3. **Add learning rate scheduling to `chipvi/training/trainer.py`:**
   a. Import LinearLR and CosineAnnealingLR from torch.optim.lr_scheduler
   b. Add scheduler parameters to Trainer.__init__ (warmup_epochs, scheduler_type)
   c. Create warmup scheduler: LinearLR with start_factor=1e-3, end_factor=1.0
   d. Create main scheduler: CosineAnnealingLR with T_max=total_epochs-warmup_epochs
   e. Combine using SequentialLR with milestone at warmup_epochs
   f. Call scheduler.step() after each epoch in fit() method

4. **Implement early stopping mechanism:**
   a. Add early_stopping parameters: patience, monitor_metric (default='val_loss')
   b. Track best_metric_value and epochs_without_improvement
   c. After each validation epoch, check if metric improved
   d. If improved: reset counter, update best_value, save checkpoint
   e. If not improved: increment counter, check if patience exceeded
   f. Stop training loop if patience exceeded

5. **Add gradient clipping:**
   a. Add max_grad_norm parameter (default=1.0, None to disable)
   b. After loss.backward() in training loop
   c. Call torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
   d. Optionally log gradient norm before and after clipping

6. **Integrate Weights & Biases logging:**
   a. Add wandb_config parameter with project, entity, name, enabled flag
   b. If enabled, call wandb.init() with config at training start
   c. Log per-batch: training loss, gradient norm, learning rate
   d. Log per-epoch: validation loss, validation metrics, epoch time
   e. Call wandb.finish() at training end
   f. Handle wandb disabled case gracefully (no-op logging)

7. **Make all features configurable:**
   a. Accept config dict/DictConfig in Trainer.__init__
   b. Extract scheduler_config, early_stopping_config, wandb_config
   c. Set defaults for backward compatibility
   d. Validate configuration parameters

8. **Ensure all tests pass:** Run full test suite