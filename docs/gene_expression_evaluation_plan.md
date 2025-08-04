# Gene Expression Prediction Evaluation Plan

### **Objective**

To quantify the improvement of ChipVI's signal normalization by testing the hypothesis that normalized ChIP-seq signals are better predictors of gene expression levels than raw signals.

### **Methodology**

The core idea is to train a simple regression model (e.g., Ridge regression) to predict the expression level of a gene based on the ChIP-seq signal in the vicinity of its Transcription Start Site (TSS). We will perform this evaluation using both the raw signal and the ChipVI-normalized signal and compare the model's performance (e.g., R² score). This process will support using a single ChIP-seq mark or multiple marks as predictors.

---

### **Implementation Plan**

The plan is broken down into four main stages: Data Preparation, Feature Engineering, Model Training & Evaluation, and Reporting.

#### **Stage 1: Prerequisite Data Preparation**

This stage involves gathering and pre-processing all the necessary data files.

1.  **Gene Annotations**:
    *   **Input**: `gencode.v29.annotation.gtf.gz` (already in the project).
    *   **Task**: Parse this GTF file to create a simplified BED file or DataFrame containing the coordinates of each gene's TSS. The output should have columns like `chr`, `tss_start`, `tss_end`, `gene_id`, `gene_name`, `strand`.
    *   **Tool**: A Python library like `gtfparse` can be used.

2.  **Gene Expression Data**:
    *   **Input**: A user-provided gene expression matrix (e.g., `rna_seq.tsv`). This file should contain gene IDs (matching the GTF) and their expression values (e.g., TPM, FPKM, or raw counts).
    *   **Task**:
        *   Load the expression data into a pandas DataFrame.
        *   Apply a log-transform (e.g., `log2(TPM + 1)`) to the expression values to stabilize variance and make the distribution more suitable for linear modeling. This will be our prediction target (`y`).

3.  **ChIP-seq Signal Tracks**:
    *   **Input**: The raw ChIP-seq signal (from existing BED files) and a trained ChipVI model.
    *   **Task**: Create a new script, `runs/generate_normalized_track.py`, that:
        *   Loads a trained ChipVI model.
        *   Processes a given experiment's data (control and experiment reads/covariates).
        *   For each genomic bin, calculates the normalized signal. A simple and effective normalized signal is `y - mu_tech`, where `y` is the observed count and `mu_tech` is the mean of the technical noise predicted by ChipVI.
        *   Outputs two signal tracks in `bedGraph` format: one for the raw signal (`y`) and one for the normalized signal (`y - mu_tech`).
        *   (Optional but recommended) Convert these `bedGraph` files to `bigWig` format for efficient querying, using a tool like `bedGraphToBigWig`.

#### **Stage 2: Feature Engineering**

This stage involves creating the feature matrix (`X`) for the regression model.

1.  **Create a new analysis script**: `analysis/evaluate_gene_expression.py`. This script will contain the logic for the rest of the plan.

2.  **Aggregate ChIP-seq Signal per Gene**:
    *   **Input**: The TSS locations from Stage 1.1 and the signal tracks (raw and normalized) from Stage 1.3.
    *   **Task**: For each gene, define a window around its TSS (e.g., +/- 5kb, this should be a configurable parameter). Then, for both the raw and normalized signal tracks, calculate the average signal within this window.
    *   **Tool**: A library like `pyBigWig` can efficiently query the average signal in genomic regions from a `bigWig` file.
    *   **Output**: Two DataFrames, one for raw features and one for normalized features. Each DataFrame will be indexed by `gene_id` and have columns for each ChIP-seq mark (e.g., `CTCF_signal`, `H3K27me3_signal`).

#### **Stage 3: Model Training and Evaluation**

This stage involves training the regression model and evaluating its performance using a robust method.

1.  **Align Data**:
    *   **Task**: In the `evaluate_gene_expression.py` script, merge the feature DataFrames (from Stage 2.2) with the log-transformed expression data (from Stage 1.2) on `gene_id`. Genes without both signal and expression data will be dropped.

2.  **Train and Evaluate the Model**:
    *   **Model**: Use `sklearn.linear_model.Ridge` for regularized linear regression, which is robust to collinearity when using multiple ChIP-seq marks.
    *   **Evaluation**: Use k-fold cross-validation (e.g., 5-fold) to get a reliable estimate of the model's performance.
    *   **Metric**: The primary metric will be the R-squared (`R²`) score. Pearson correlation can be reported as a secondary, more intuitive metric.
    *   **Task**:
        *   Define the feature matrix `X` (from the signal DataFrame) and the target vector `y` (from the expression DataFrame).
        *   Use `sklearn.model_selection.cross_val_score` to compute the `R²` scores for each fold.
        *   Calculate the mean and standard deviation of the cross-validation scores.

3.  **Run Comparison**:
    *   **Task**: Execute the entire training and evaluation pipeline (Stage 3.2) twice:
        1.  Once using the features derived from the **raw** signal.
        2.  Once using the features derived from the **ChipVI-normalized** signal.

#### **Stage 4: Reporting**

This stage involves presenting the results in a clear and interpretable way.

1.  **Generate a Report**:
    *   **Task**: The `evaluate_gene_expression.py` script should print a summary to the console, for example:
        ```
        Gene Expression Prediction Evaluation
        -------------------------------------
        ChIP-seq Marks Used: CTCF, H3K27me3
        TSS Window: +/- 5kb

        Raw Signal Model:
          - Cross-Validated R^2: 0.45 (+/- 0.03)
          - Cross-Validated Pearson r: 0.67 (+/- 0.02)

        ChipVI-Normalized Signal Model:
          - Cross-Validated R^2: 0.58 (+/- 0.02)
          - Cross-Validated Pearson r: 0.76 (+/- 0.01)

        Conclusion: ChipVI normalization resulted in a 28.8% improvement in R^2 score.
        ```

2.  **Visualize Results**:
    *   **Task**: Create a bar plot that visually compares the mean `R²` scores of the raw vs. normalized models. The plot should include error bars representing the standard deviation from the cross-validation. This plot should be saved to a file.
