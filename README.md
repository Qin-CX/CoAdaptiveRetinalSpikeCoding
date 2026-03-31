## Project Layout

`coadaptive_vqvae/config`

Centralized experiment configuration for paths, data settings, model hyperparameters, loss weights, and training schedules.

`coadaptive_vqvae/data`

Dataset loading and split management.

`coadaptive_vqvae/models`

VQ-VAE modules, paper-aligned components, and the co-adaptive encoder.

`coadaptive_vqvae/models/paper_modules.py`

Paper-aligned building blocks including Spike Pattern Attention Module, Latent Space Projection, Topological Feature Extractor, Spectral Feature Extractor, and Spike Fusion Unit.

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
