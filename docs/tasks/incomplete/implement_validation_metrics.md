# Task Document: Implement Comprehensive Validation Metrics

**Task ID:** implement_validation_metrics

**Description:**
Add advanced validation metrics and visualizations for model evaluation including Spearman correlations, Probability Integral Transform (PIT) analysis, and comprehensive plotting. These metrics provide deeper insights into model performance beyond simple loss values, particularly for understanding the concordance between biological replicates.

**Blocked By:**
- `implement_advanced_losses`
- `enhance_trainer`

**Acceptance Criteria:**
- Spearman correlations computed for residuals and quantiles between replicates
- PIT analysis produces uniform distributions for well-calibrated models
- 9-panel validation plots generated correctly showing predictions, residuals, and quantiles
- All metrics integrated into the validation loop
- Plots automatically uploaded to Weights & Biases when enabled

**Test Scenarios (for the agent to implement):**
1. **Spearman Correlation Accuracy:** Test correlation computation returns correct values for known patterns (perfect correlation, no correlation, anti-correlation).
2. **PIT Uniformity Test:** Verify PIT produces uniform distributions for correctly specified models and deviations for misspecified models.
3. **Plot Generation Integrity:** Test that all 9 panels of the validation figure are generated with correct content and labels.
4. **Metric Computation with Missing Data:** Verify metrics handle edge cases like NaN values or empty batches gracefully without crashes.
5. **2D Histogram Accuracy:** Test that hist2d utility correctly generates density plots with appropriate binning for both discrete and continuous data.

**Implementation Todos:**
1. **Implement tests for validation metrics:**
   a. Create `tests/test_validation_metrics.py`
   b. Test Spearman correlation against scipy.stats.spearmanr
   c. Test PIT computation with known distributions
   d. Test plot generation creates figure with correct subplots
   e. Test metric computation with edge cases (empty batches, NaN values)

2. **Ensure tests fail:** Run tests before implementation

3. **Add validation metric collection to Trainer._validate_one_epoch():**
   a. Initialize collectors: residuals_r1, residuals_r2, quantiles_r1, quantiles_r2, predictions, observations
   b. For each batch, compute and store:
      - Residuals: y_r1 - model_r1.mean, y_r2 - model_r2.mean
      - Scale r2 residuals by sd_ratio from batch['sd_ratio_r1_r2']
      - Quantiles using compute_numeric_cdf(dist, observation)
      - Raw predictions and observations for plotting

4. **Compute Spearman correlations:**
   a. Concatenate all batch results into numpy arrays
   b. Create pandas DataFrame with all metrics
   c. Use df.corr(method='spearman') to compute correlation matrix
   d. Extract residual correlation: corr.loc['r1_res', 'r2_res_scaled']
   e. Extract quantile correlation: corr.loc['r1_quant', 'r2_quant']
   f. Return both correlations as validation metrics

5. **Create validation plotting function:**
   a. Create `create_validation_figure(metrics_dict)` that returns matplotlib figure
   b. Create 3x3 subplot grid using plt.subplots(3, 3, figsize=(18, 18))
   c. Row 1: Predictions vs Observations
      - (0,0): R1 predicted mean vs observed using hist2d
      - (0,1): R2 predicted mean vs observed using hist2d
      - (0,2): R1 vs R2 predicted means consistency
   d. Row 2: Residual Analysis
      - (1,0): R1 residuals histogram, range(-25, 25)
      - (1,1): R2 scaled residuals histogram, range(-25, 25)
      - (1,2): R1 vs R2 residual scatter plot with correlation
   e. Row 3: Quantile Analysis (PIT)
      - (2,0): R1 quantiles histogram, should be uniform [0,1]
      - (2,1): R2 quantiles histogram, should be uniform [0,1]
      - (2,2): R1 vs R2 quantile consistency plot

6. **Implement hist2d utility function:**
   a. Create `chipvi/utils/plots.py` if not exists
   b. Implement `hist2d(df, x_col, y_col, ax, discrete_bins=False, bins=50)`
   c. Use numpy.histogram2d for binning
   d. Plot using ax.imshow or ax.pcolormesh
   e. Add diagonal reference line for identity
   f. Compute and display Spearman correlation in title

7. **Integrate with W&B logging:**
   a. After creating validation figure, convert to wandb.Image
   b. Log with wandb.log({"validation_plots": wandb.Image(fig)})
   c. Log correlation metrics: {"val_residual_spearman": res_corr, "val_quantile_spearman": quant_corr}
   d. Close matplotlib figure to free memory

8. **Ensure all tests pass:** Verify metrics and visualizations work correctly