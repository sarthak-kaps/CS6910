### Codebase2

- This directory contains our implementation for Neural Machine translation model with attention.
- It contains 4 files.

#### data_process.py
- Contains data processing part.
- It does various functions like creating dataset, tokenizing it and finally adding '<start>' and '<end>' tokens for Neural network

#### model_create.py
- This file contains our implementation for Encoder-Decoder model and custom Attention layer.
- We are using Bahadanau Attention for our use case.

#### train .py
- This file uses the previous two files and trains our model on the dakshina_dataset.
- It uses wandb for hyperparam tuning.
- Finally, validation is also done and the corresponding metrics are logged in wandb.
- It can be run directly using the command `python train.py`.
- For running sweeps, we need to used sweep_configs which are available in the main directory.

#### test .py
- This file is used to get the final results based on best model obtained using hyperparameter tuning.
- It logs various imporatant images like attention heatmap, as well as the prediction tables.
- It can be run using the command `python test.py`.
