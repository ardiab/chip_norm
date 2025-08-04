# Training

The `chipvi.training` module contains the scripts for training the ChipVI models.

## `phase_a.py`

This script is for the initial phase of training, which focuses on learning the technical noise model. It can train either a Poisson or a Negative Binomial model. The training process is a standard PyTorch loop with the following features:

- **Validation**: The model is evaluated on a validation set at the end of each epoch.
- **Early Stopping**: The training is stopped if the validation loss does not improve for a certain number of epochs.
- **Model Checkpointing**: The best model is saved to a file.
- **Logging**: The training progress is logged to the console and to Weights & Biases.

## `technical_model_trainer.py`

This script is an extension of `phase_a.py` that can handle both single-end and replicate data. When training on replicate data, it adds a regularization term to the loss function to encourage consistency between the biological signals of the two replicates.

## `tech_biol_rep_model_trainer.py`

This script is a more advanced training script that jointly trains the technical and biological models. It uses a composite loss function that includes:

- The likelihood of the technical model.
- The mean squared error between the inferred biological signals of the replicates.
- The Pearson correlation between the inferred biological signals of the replicates.

This script is more experimental and is used for detailed analysis and debugging. It saves a lot of intermediate data during validation, which can be used to analyze the training process in detail.
