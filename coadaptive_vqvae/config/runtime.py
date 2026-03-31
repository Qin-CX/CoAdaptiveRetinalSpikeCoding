from dataclasses import asdict, replace

from .defaults import CoAdaptiveConfig, VQVAEConfig, get_coadaptive_config, get_vqvae_config


def config_to_dict(config):
    return asdict(config)


def update_vqvae_config(
    *,
    dataset_root: str | None = None,
    log_dir: str | None = None,
    model_dir: str | None = None,
    batch_size: int | None = None,
    num_epochs: int | None = None,
    learning_rate: float | None = None,
    num_workers: int | None = None,
) -> VQVAEConfig:
    config = get_vqvae_config()
    data_config = replace(
        config.data,
        dataset_root=dataset_root or config.data.dataset_root,
        num_workers=num_workers if num_workers is not None else config.data.num_workers,
    )
    training_config = replace(
        config.training,
        log_dir=log_dir or config.training.log_dir,
        model_dir=model_dir or config.training.model_dir,
        batch_size=batch_size if batch_size is not None else config.training.batch_size,
        num_epochs=num_epochs if num_epochs is not None else config.training.num_epochs,
        learning_rate=learning_rate if learning_rate is not None else config.training.learning_rate,
    )
    return replace(config, data=data_config, training=training_config)


def update_coadaptive_config(
    *,
    dataset_root: str | None = None,
    pretrained_vqvae_path: str | None = None,
    encoder_model_dir: str | None = None,
    batch_size: int | None = None,
    num_epochs: int | None = None,
    learning_rate: float | None = None,
    num_workers: int | None = None,
) -> CoAdaptiveConfig:
    config = get_coadaptive_config()
    data_config = replace(
        config.data,
        dataset_root=dataset_root or config.data.dataset_root,
        num_workers=num_workers if num_workers is not None else config.data.num_workers,
    )
    training_config = replace(
        config.training,
        pretrained_vqvae_path=pretrained_vqvae_path or config.training.pretrained_vqvae_path,
        encoder_model_dir=encoder_model_dir or config.training.encoder_model_dir,
        batch_size=batch_size if batch_size is not None else config.training.batch_size,
        num_epochs=num_epochs if num_epochs is not None else config.training.num_epochs,
        learning_rate=learning_rate if learning_rate is not None else config.training.learning_rate,
    )
    return replace(config, data=data_config, training=training_config)
