## Project Layout

`coadaptive_vqvae/config`

Centralized experiment configuration for paths, data settings, model hyperparameters, loss weights, and training schedules.

`coadaptive_vqvae/data`

Dataset loading and split management.

`coadaptive_vqvae/models`

VQ-VAE modules and the co-adaptive encoder with frequency-feature fusion.

`coadaptive_vqvae/utils`

Metric helpers.

`coadaptive_vqvae/training`

Training logic plus shared runtime helpers.

`scripts`

Command-line experiment entrypoints and configuration inspection tools.

`scripts/train_vqvae.py`

Train the VQ-VAE model with runtime overrides.

`scripts/train_coadaptive_encoder.py`

Train the co-adaptive encoder with runtime overrides.

`scripts/show_config.py`

Print the resolved experiment configuration.
