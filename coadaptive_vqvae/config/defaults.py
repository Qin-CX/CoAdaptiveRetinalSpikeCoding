from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class DataConfig:
    dataset_root: str
    train_split: str = "train"
    val_split: str = "val"
    image_size: tuple[int, int] = (32, 32)
    grayscale_output_channels: Optional[int] = None
    num_neurons: int = 150
    spike_times: int = 60
    num_workers: int = 0
    pin_memory: bool = False

    def build_transform(self):
        transform_steps = [transforms.Resize(self.image_size)]
        if self.grayscale_output_channels is not None:
            transform_steps.append(transforms.Grayscale(num_output_channels=self.grayscale_output_channels))
        transform_steps.append(transforms.ToTensor())
        return transforms.Compose(transform_steps)


@dataclass(frozen=True)
class VQVAEModelConfig:
    num_hiddens: int = 256
    num_residual_hiddens: int = 64
    num_residual_layers: int = 2
    embedding_dim: int = 128
    num_embeddings: int = 256
    commitment_cost: float = 0.05
    decay: float = 0.99
    in_channels: int = 1
    in_scale: tuple[int, int] = (150, 60)


@dataclass(frozen=True)
class VQVAETrainingConfig:
    learning_rate: float = 1e-4
    num_epochs: int = 3000
    batch_size: int = 12
    data_variance: float = 1.0
    early_stop_patience: int = 250
    log_dir: str = str(PROJECT_ROOT / "logs" / "cifar10" / "log_vqvae_res_big102432")
    model_dir: str = str(PROJECT_ROOT / "models" / "cifar1126_cbam_group_norm_projection_60ms_150channel_no_vq")
    latest_checkpoint_name: str = "checkpoint_vqvae_big102432_latest.pth"
    best_checkpoint_name: str = "checkpoint_vqvae_big102432_best.pth"
    cuda_launch_blocking: str = "1"


@dataclass(frozen=True)
class VQVAEConfig:
    data: DataConfig
    model: VQVAEModelConfig
    training: VQVAETrainingConfig


@dataclass(frozen=True)
class CoAdaptiveModelConfig:
    num_hiddens: int = 256
    num_residual_hiddens: int = 64
    num_residual_layers: int = 2
    embedding_dim: int = 128
    num_embeddings: int = 1024
    commitment_cost: float = 0.25
    decay: float = 0.99
    in_channels: int = 1
    in_scale: tuple[int, int] = (150, 60)
    out_shape: tuple[int, int, int] = (1, 150, 60)


@dataclass(frozen=True)
class CoAdaptiveLossConfig:
    lambda_l2: float = 1.0
    lambda_perceptual: float = 2.0
    lambda_ssim: float = 1.5
    lambda_sparsity: float = 1e-3


@dataclass(frozen=True)
class CoAdaptiveScheduleConfig:
    tau_start: float = 1.0
    tau_end: float = 0.1
    tau_anneal_epochs: int = 1000
    warmup_epochs: int = 20
    eta_min: float = 1e-6


@dataclass(frozen=True)
class CoAdaptiveTrainingConfig:
    learning_rate: float = 1e-4
    num_epochs: int = 2000
    batch_size: int = 32
    pretrained_vqvae_path: str = str(
        PROJECT_ROOT / "models" / "cifar0814_cbam_group_norm_projection_60ms_150channel" / "checkpoint_vqvae_big102432_best.pth"
    )
    encoder_model_dir: str = str(PROJECT_ROOT / "models" / "coadaptive_encoder_frequency_fusion")
    checkpoint_name: str = "coadaptive_encoder_best.pth"
    training_curve_name: str = "training_curves.png"


@dataclass(frozen=True)
class CoAdaptiveConfig:
    data: DataConfig
    model: CoAdaptiveModelConfig
    training: CoAdaptiveTrainingConfig
    loss: CoAdaptiveLossConfig
    schedule: CoAdaptiveScheduleConfig


def get_vqvae_config() -> VQVAEConfig:
    return VQVAEConfig(
        data=DataConfig(
            dataset_root=str(PROJECT_ROOT / "dataset" / "cifar10_2_0718"),
            image_size=(32, 32),
            grayscale_output_channels=None,
            num_workers=0,
            pin_memory=False,
        ),
        model=VQVAEModelConfig(),
        training=VQVAETrainingConfig(),
    )


def get_coadaptive_config() -> CoAdaptiveConfig:
    return CoAdaptiveConfig(
        data=DataConfig(
            dataset_root=str(PROJECT_ROOT / "dataset" / "cifar10_2_0704"),
            image_size=(32, 32),
            grayscale_output_channels=1,
            num_workers=4,
            pin_memory=True,
        ),
        model=CoAdaptiveModelConfig(),
        training=CoAdaptiveTrainingConfig(),
        loss=CoAdaptiveLossConfig(),
        schedule=CoAdaptiveScheduleConfig(),
    )
