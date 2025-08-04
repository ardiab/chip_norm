# Future Work and Experiment Ideas

This document expands on potential new experiments and approaches that can be built upon the existing ChipVI framework.

---

### 1. Model and Architecture Enhancements

The current MLP-based architecture is a strong baseline. The following enhancements could lead to improved performance and a more sophisticated understanding of the data.

*   **Hyperparameter Optimization**:
    *   **Why**: The performance of deep learning models is highly sensitive to hyperparameters. The current settings for learning rate, hidden layer dimensions, and regularization are based on common practices but may not be optimal for this specific task.
    *   **How**: Implement a systematic hyperparameter search using a framework like [Optuna](https://optuna.org/) or [Ray Tune](https://docs.ray.io/en/latest/tune/index.html). Define a search space for key parameters (e.g., `learning_rate`, `weight_decay`, `hidden_dims_mean`, `hidden_dims_r`, loss weights) and run an optimization study where the objective is to maximize the validation log-likelihood or another relevant metric.
    *   **Tasks for Implementation**:
        1.  Create a new script `runs/optimize_hyperparameters.py`.
        2.  In this script, define an `objective` function that takes an `optuna.trial` object as input.
        3.  Inside the `objective` function, use `trial.suggest_` methods to define the search space for learning rate, weight decay, hidden layer dimensions, and other relevant parameters.
        4.  Instantiate and train the model using the suggested hyperparameters. The training loop should be encapsulated within this function.
        5.  Return the best validation log-likelihood from the training run.
        6.  Use `optuna.create_study` and `study.optimize` to run the hyperparameter search.
        7.  Log the results of each trial and the best parameters found.

*   **Advanced Architectures**:
    *   **Attention Mechanisms**:
        *   **Why**: Not all technical covariates are equally important for every genomic bin. An attention mechanism would allow the model to dynamically learn the importance of each covariate for a given input, potentially improving its ability to model complex noise profiles.
        *   **How**: Add a self-attention layer to the beginning of the `TechNB_mu_r` or `TechPoisson` models. This layer would take the input covariates and produce a set of attention weights, which would then be used to create a weighted sum of the covariates. This weighted representation would then be fed into the existing MLP.
        *   **Tasks for Implementation**:
            1.  Implement a simple `Attention` module in `chipvi/models/common.py`.
            2.  Modify the `TechNB_mu_r` model in `chipvi/models/technical_model.py` to optionally include this attention layer.
            3.  The `forward` pass of the model should first pass the covariates through the attention layer and then use the output as input to the MLP.
            4.  Add a command-line argument to the training scripts to enable or disable the attention mechanism.

    *   **Variational Autoencoders (VAEs)**:
        *   **Why**: A VAE is a generative model that can learn a probabilistic latent representation of the data. This is a natural extension of the current framework and could provide a more robust separation of the biological signal from the technical noise. The latent space of the VAE would represent the "true" biological signal, while the decoder would learn to reconstruct the observed signal, incorporating the technical noise.
        *   **How**: Restructure the model into a VAE framework. The encoder would map the observed counts and technical covariates to a latent distribution (the biological signal `z_i`). The decoder would then take a sample from this latent distribution and the technical covariates to reconstruct the observed counts. The VAE loss function (a combination of reconstruction loss and the KL divergence between the latent distribution and a prior) would naturally regularize the latent space.
        *   **Tasks for Implementation**:
            1.  Create a new model file `chipvi/models/vae_model.py`.
            2.  Define an `Encoder` network that maps the input data to the parameters of the latent distribution (mean and log-variance).
            3.  Define a `Decoder` network that reconstructs the observed data from a latent variable and the technical covariates.
            4.  Implement the main `VAE` model that combines the encoder and decoder and includes the reparameterization trick for sampling.
            5.  Create a new training script `chipvi/training/vae_trainer.py` with the appropriate VAE loss function (reconstruction loss + KL divergence).

    *   **Spatial Dependencies (CNNs/RNNs)**:
        *   **Why**: ChIP-seq data is inherently spatial; the signal in one genomic bin is often correlated with the signal in its neighbors. The current model treats each bin independently. A CNN or RNN could capture these spatial relationships, potentially improving the model's ability to distinguish between noise and broad domains of enrichment.
        *   **How**: Instead of processing single bins, the model would take as input a sequence of adjacent bins. A 1D CNN could be used to learn local patterns, or an LSTM/GRU could be used to model longer-range dependencies. The output of the convolutional/recurrent layers would then be fed into the existing MLP to predict the distribution parameters.
        *   **Tasks for Implementation**:
            1.  Modify the `Dataset` classes in `chipvi/data/datasets.py` to return sequences of adjacent bins (e.g., of length 11, with the target bin in the center).
            2.  Create a new model, e.g., `TechNB_CNN`, that includes a 1D CNN layer before the MLP.
            3.  Update the training scripts to handle the sequential input data.

---

### 2. Data-Centric Experiments

The model's performance is fundamentally tied to the data it's trained on. These experiments focus on leveraging data to improve the model.

*   **Expanded Covariate Set**:
    *   **Why**: The current model uses a limited set of covariates. Adding more informative features could improve its ability to model technical noise.
    *   **How**: The `path_helper.py` already includes paths to GC content and mappability tracks. Pre-process these tracks to get values for each genomic bin and add them as input covariates to the model. You could also include other features like distance to the nearest TSS or gene density.
    *   **Tasks for Implementation**:
        1.  Create a script to pre-process the GC content and mappability BigWig files. This script should calculate the average value for each genomic bin and save the results.
        2.  Modify the `load_and_process_bed` function in `chipvi/data/datasets.py` to load these new features.
        3.  Update the `Dataset` classes to include the new covariates in the `covariates` tensor.
        4.  Adjust the `dim_x` parameter in the models and training scripts to reflect the new number of covariates.

*   **Cross-Cell-Type Generalization**:
    *   **Why**: A truly robust technical noise model should be able to generalize across different biological contexts. This experiment would test how well the model learns a universal representation of technical noise.
    *   **How**: Train a model on data from one cell type (e.g., liver) and evaluate its performance on data from a completely different cell type (e.g., brain). A successful result would indicate that the model has learned a generalizable representation of technical noise, independent of the specific biological context.
    *   **Tasks for Implementation**:
        1.  Use `experiment_collections.py` to define two disjoint sets of experiments from different cell types.
        2.  Train a model on the first cell type and save the trained model.
        3.  Write an evaluation script that loads the pre-trained model and evaluates its performance on the second cell type.

*   **Transfer Learning**:
    *   **Why**: Training deep learning models requires large amounts of data. If you have a limited amount of data for your target experiment, you can leverage a larger, more diverse dataset to pre-train a model and then fine-tune it on your specific data.
    *   **How**: Pre-train a technical noise model on a large, public dataset (e.g., all H3K27me3 experiments from ENCODE). Then, freeze the early layers of the model and fine-tune the later layers on your smaller, specific dataset. This could lead to better performance than training from scratch on the small dataset alone.
    *   **Tasks for Implementation**:
        1.  Train a base model on a large, diverse dataset and save the model weights.
        2.  Add a `--finetune` option to the training script.
        3.  When this option is used, the script should load the pre-trained weights, freeze the early layers of the model, and train the remaining layers on the new dataset, likely with a lower learning rate.

---

### 3. Training and Loss Function Strategies

The way the model is trained can have a significant impact on its performance. These experiments explore alternative training strategies.

*   **Refined Loss Functions**:
    *   **Why**: The current loss function in `tech_biol_rep_model_trainer.py` uses hardcoded weights for its different components. The optimal balance between these components is likely not fixed and may even change during training.
    *   **How**:
        *   Make the loss weights configurable parameters and tune them as part of the hyperparameter optimization process.
        *   Implement a loss annealing schedule, where the weights of the different components change over the course of training. For example, you could start by focusing on the likelihood component and then gradually increase the weight of the MSE and correlation components.
        *   Treat the loss weights as learnable parameters, as proposed in some multi-task learning papers.
    *   **Tasks for Implementation**:
        1.  Add command-line arguments to `tech_biol_rep_model_trainer.py` for the weights of the different loss components.
        2.  Replace the hardcoded weights with these arguments.
        3.  (Optional) Implement a loss annealing schedule where the weights are updated at each epoch.

*   **Adversarial Training**:
    *   **Why**: The goal is to learn a biological signal representation that is free of technical artifacts. An adversarial training setup can explicitly enforce this.
    *   **How**: Introduce a second "adversary" network that tries to predict the technical covariates from the inferred biological signal (`y_i - mu_tech`). The main model is then trained with a composite loss that includes not only its primary objective but also a term that penalizes the adversary's success. This forces the main model to produce a biological signal that is "uninformative" to the adversary, effectively removing the technical noise.
    *   **Tasks for Implementation**:
        1.  Implement an `Adversary` model (a simple MLP).
        2.  Modify the training loop to include the adversarial training logic: train the main model with a loss that penalizes the adversary's success, then freeze the main model and train the adversary.

*   **Curriculum Learning**:
    *   **Why**: Starting with "easy" examples and gradually introducing more "difficult" ones can help the model converge to a better solution.
    *   **How**: Design a curriculum based on the properties of the genomic bins. For example, you could start by training the model on bins with high signal-to-noise ratios and then gradually introduce bins with lower signal-to-noise ratios. This could be implemented by creating a custom data sampler that selects batches based on this curriculum.
    *   **Tasks for Implementation**:
        1.  Write a script to pre-calculate a "difficulty score" for each genomic bin.
        2.  Implement a custom PyTorch `Sampler` that yields batches of indices according to the curriculum.
        3.  Integrate the custom sampler into the training script.

---

### 4. Evaluation and Downstream Applications

The ultimate test of ChipVI is whether it improves downstream biological analysis.

*   **Differential Peak Calling**:
    *   **Why**: A key application of ChIP-seq is identifying regions of differential binding between conditions. A successful normalization method should improve the accuracy of this analysis.
    *   **How**:
        1.  Use ChipVI to generate a normalized signal track (e.g., by subtracting the predicted technical mean `mu_tech` from the observed counts `y_i`).
        2.  Run a differential peak caller like MACS2 or DESeq2 on this normalized track.
        3.  Compare the resulting differential peaks to those obtained from the raw data or data normalized with standard methods. The evaluation could be based on the number of called peaks, their p-values, and their overlap with known regulatory elements.
    *   **Tasks for Implementation**:
        1.  Create a script `runs/generate_normalized_track.py` that loads a trained model, processes a dataset, and writes the normalized signal to a `bedGraph` file.
        2.  Use an external tool like `bedGraphToBigWig` to convert the `bedGraph` file to a `bigWig` file for visualization.
        3.  Write a wrapper script to run a peak caller on the normalized data and compare the results to a baseline.

*   **Benchmarking**:
    *   **Why**: To demonstrate the value of ChipVI, it's important to compare it to existing methods.
    *   **How**: Conduct a formal benchmarking study where you compare ChipVI to other ChIP-seq normalization methods. This could include simple methods like library size scaling and quantile normalization, as well as more sophisticated methods like [csaw](https://bioconductor.org/packages/release/bioc/html/csaw.html). The comparison should be based on a range of metrics, including the effect on downstream analysis.
    *   **Tasks for Implementation**:
        1.  Implement several baseline normalization methods in a new script.
        2.  Apply ChipVI and the baseline methods to the same dataset.
        3.  Evaluate the results using a consistent set of metrics, such as replicate correlation and performance on a downstream task like differential peak calling.

*   **Interpretability**:
    *   **Why**: Understanding *why* the model makes certain predictions can provide valuable insights into the sources of technical noise and build trust in the model.
    *   **How**: Use model interpretability techniques like [SHAP](https://shap.readthedocs.io/en/latest/index.html) or Integrated Gradients to determine the contribution of each input covariate to the model's output. This could reveal, for example, that GC content is a major driver of technical noise in certain regions, or that sequencing depth has a non-linear effect on the signal.
    *   **Tasks for Implementation**:
        1.  Create a script `analysis/interpret_model.py`.
        2.  Use a library like `shap` to calculate the SHAP values for the input covariates.
        3.  Generate summary plots to visualize the global importance of each feature.
        4.  Generate force plots to explain individual predictions.

---

### 5. Variational Inference for Direct Biological Signal Estimation

This approach reframes the problem to directly model the biological signal as a latent variable, using the existing technical model as a pre-trained component.

*   **Why**: Instead of treating the biological signal as a residual (`y - mu_tech`), this method provides a principled, probabilistic estimate of the biological signal, `p(z_i | y_i, x_i)`, which can capture uncertainty.
*   **How**: A Variational Autoencoder (VAE) framework will be used. An "encoder" network will learn to approximate the posterior distribution of the latent biological signal. The generative model will assume the observed signal `y_i` comes from a distribution (e.g., Negative Binomial) whose mean is a sum of the biological signal (derived from the latent variable `z_i`) and the technical signal (from the pre-trained model).
*   **Tasks for Implementation**:
    1.  **Phase 1: Pre-train Technical Model**:
        *   Use the existing `technical_model_trainer.py` to train a `TechNB_mu_r` model on the desired dataset. Save the trained model's state dictionary. This model will serve as a fixed, pre-trained component.
    2.  **Phase 2: Implement the VI Model**:
        *   Create a new model file: `chipvi/models/vi_model.py`.
        *   **Encoder**: Define an `Encoder` class (e.g., an MLP) that takes the observed counts `y_i` and covariates `x_i` as input and outputs the parameters (`mu_z`, `log_var_z`) of the Gaussian variational distribution `q(z_i)`.
        *   **VIBiolModel**: Define the main model class.
            *   In its `__init__`, it should instantiate the `Encoder` and the pre-trained `TechNB_mu_r` model, loading its saved weights and freezing its parameters (`model.eval()` and `param.requires_grad = False`).
            *   The `forward` pass will execute the VAE logic:
                a. Get `mu_z`, `log_var_z` from the encoder.
                b. Use the reparameterization trick to sample `z_i`.
                c. Transform the latent variable to the biological signal: `mu_biol = torch.exp(z_i)`.
                d. Get `mu_tech` from the frozen technical model.
                e. The total mean is `mu_total = mu_biol + mu_tech`.
                f. Return `mu_total`, `mu_z`, and `log_var_z` for the loss calculation.
    3.  **Phase 3: Implement the VI Trainer**:
        *   Create a new training script: `chipvi/training/vi_trainer.py`.
        *   It should accept arguments for the path to the pre-trained technical model.
        *   **Loss Function (ELBO)**: Implement the loss function, which is the sum of reconstruction loss and KL divergence.
            *   `reconstruction_loss = -nb_log_prob(y, mu_total, r_total)`. Note: `r_total` also needs to be defined, perhaps from the technical model or as a separate learned parameter.
            *   `kl_loss = -0.5 * torch.sum(1 + log_var_z - mu_z.pow(2) - log_var_z.exp())`.
            *   `loss = reconstruction_loss + beta * kl_loss`. `beta` is a hyperparameter to control the strength of the KL regularization (related to Beta-VAE).
    4.  **Phase 4: Inference and Evaluation**:
        *   Create a script to generate the normalized signal. For a given bin, the biological signal is the mean of the learned latent distribution, `exp(mu_z)`. This can be written to a `bedGraph` file for downstream analysis.

---

### 6. Upsampling Coarse Signals with a Sliding Window Filter

This approach allows for modeling at a computationally efficient coarse resolution (e.g., 200bp) while still producing high-resolution (e.g., 25bp) output tracks.

*   **Why**: Modeling at high resolution can be memory and time-intensive. This post-processing step provides a practical way to bridge the resolution gap.
*   **How**: A normalized signal track is first generated at a coarse resolution. Then, for each bin in the target high-resolution grid, a new signal is calculated by applying a weighted average (e.g., with a Gaussian filter) to the signals of all nearby coarse-resolution bins.
*   **Tasks for Implementation**:
    1.  **Step 1: Generate Coarse-Grained Normalized Signal**:
        *   Run the standard ChipVI pipeline (either the current version or the new VI model) with data binned at the coarse resolution (e.g., 200bp).
        *   Use a script like `runs/generate_normalized_track.py` to save the output signal as a `bedGraph` or `bigWig` file.
    2.  **Step 2: Implement the Upsampling Script**:
        *   Create a new analysis script: `analysis/upsample_signal.py`.
        *   It should take command-line arguments for the input coarse-signal file, the output high-resolution file, the target resolution (e.g., 25), the filter type (`gaussian` or `uniform`), and the filter window size (e.g., 1000bp).
    3.  **Step 3: Implement Filtering Logic**:
        *   Use a library like `pyBigWig` or `numpy` for efficient data handling.
        *   The script will iterate through each chromosome. For each chromosome, it will:
            a. Create an array representing the high-resolution signal, initialized to zeros.
            b. Load the coarse-resolution signal for that chromosome.
            c. For each coarse-resolution bin, calculate its contribution to all the high-resolution bins that fall within its "filter window".
            d. **Gaussian Filter Logic**: The contribution is weighted by `exp(-d^2 / (2 * sigma^2))`, where `d` is the distance between the centers of the high-res and coarse-res bins.
            e. Keep track of both the weighted signal sum and the weight sum for each high-res bin.
        *   After iterating through all coarse bins, normalize the signal in the high-res array by dividing the signal sum by the weight sum.
        *   Write the final high-resolution signal array to the output `bedGraph` file.
    4.  **Step 4 (Advanced - Learned Filter)**:
        *   This is a research-level extension. It would involve training a 1D convolutional neural network to learn the optimal upsampling filter. This would require a "ground truth" high-resolution dataset for supervision, which may be difficult to obtain. For the initial implementation, fixed filters are more practical.
