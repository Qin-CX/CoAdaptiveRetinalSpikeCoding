import os

import lpips
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_msssim import SSIM
from torch.utils.data import DataLoader

from coadaptive_vqvae.config.defaults import CoAdaptiveConfig, get_coadaptive_config
from coadaptive_vqvae.data.datasets import SpikeDataset1
from coadaptive_vqvae.models.coadaptive import CoAdaptiveEncoder, CoAdaptiveFramework
from coadaptive_vqvae.models.vqvae import Model
from coadaptive_vqvae.training.common import print_experiment_banner, resolve_device
from coadaptive_vqvae.utils.metrics import AverageMeter


def train(config: CoAdaptiveConfig, device: torch.device | None = None) -> None:
    device = device or resolve_device()
    print_experiment_banner("coadaptive_encoder", config)
    pretrained_vqvae_path = config.training.pretrained_vqvae_path
    encoder_model_dir = config.training.encoder_model_dir
    os.makedirs(encoder_model_dir, exist_ok=True)
    encoder_save_path = os.path.join(encoder_model_dir, config.training.checkpoint_name)

    image_transform = config.data.build_transform()
    train_dataset = SpikeDataset1(
        img_path=config.data.dataset_root,
        transforms=image_transform,
        data_type=config.data.train_split,
        nuerons_nums=config.data.num_neurons,
        spike_times=config.data.spike_times,
    )
    val_dataset = SpikeDataset1(
        img_path=config.data.dataset_root,
        transforms=image_transform,
        data_type=config.data.val_split,
        nuerons_nums=config.data.num_neurons,
        spike_times=config.data.spike_times,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        pin_memory=config.data.pin_memory,
        num_workers=config.data.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        pin_memory=config.data.pin_memory,
        num_workers=config.data.num_workers,
    )

    virtual_brain = Model(
        config.model.num_hiddens,
        config.model.num_residual_layers,
        config.model.num_residual_hiddens,
        config.model.num_embeddings,
        config.model.embedding_dim,
        config.model.commitment_cost,
        config.model.decay,
        in_channels=config.model.in_channels,
        in_scale=config.model.in_scale,
    ).to(device)
    if not os.path.exists(pretrained_vqvae_path):
        raise FileNotFoundError(pretrained_vqvae_path)
    checkpoint_vqvae = torch.load(pretrained_vqvae_path, map_location=device)
    virtual_brain.load_state_dict(checkpoint_vqvae["model"])
    virtual_brain.eval()

    encoder = CoAdaptiveEncoder(out_shape=config.model.out_shape).to(device)
    framework = CoAdaptiveFramework(virtual_brain, encoder).to(device)

    l2_loss = nn.MSELoss().to(device)
    perceptual_loss = lpips.LPIPS(net="alex").to(device)
    ssim_loss = SSIM(data_range=1.0, size_average=True, channel=3).to(device)
    optimizer = optim.Adam(framework.encoder.parameters(), lr=config.training.learning_rate)
    warmup_scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: (epoch + 1) / config.schedule.warmup_epochs if epoch < config.schedule.warmup_epochs else 1,
    )
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.training.num_epochs - config.schedule.warmup_epochs,
        eta_min=config.schedule.eta_min,
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[config.schedule.warmup_epochs],
    )

    start_epoch = 0
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []
    learning_rates = []

    if os.path.exists(encoder_save_path):
        checkpoint = torch.load(encoder_save_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            encoder.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            best_val_loss = checkpoint["best_val_loss"]
            train_losses = checkpoint.get("train_losses", [])
            val_losses = checkpoint.get("val_losses", [])
            learning_rates = checkpoint.get("learning_rates", [])
        else:
            encoder.load_state_dict(checkpoint)
            start_epoch = 700

    try:
        for epoch in range(start_epoch, config.training.num_epochs):
            gumbel_tau = max(
                config.schedule.tau_end,
                config.schedule.tau_start
                - (config.schedule.tau_start - config.schedule.tau_end) * (epoch / config.schedule.tau_anneal_epochs),
            )

            framework.encoder.train()
            train_loss_meter = AverageMeter()
            for _, image in train_loader:
                original_images_gray = image.to(device)
                reconstructed_images, synthetic_spikes = framework(original_images_gray, gumbel_tau)
                original_images_rgb = original_images_gray.repeat(1, 3, 1, 1)
                loss_l2 = l2_loss(reconstructed_images, original_images_rgb)
                loss_ssim = 1 - ssim_loss(reconstructed_images, original_images_rgb)
                loss_perceptual = perceptual_loss(reconstructed_images, original_images_rgb).mean()
                loss_sparsity = torch.mean(synthetic_spikes)
                total_loss = (
                    config.loss.lambda_l2 * loss_l2
                    + config.loss.lambda_perceptual * loss_perceptual
                    + config.loss.lambda_ssim * loss_ssim
                    + config.loss.lambda_sparsity * loss_sparsity
                )
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                train_loss_meter.update(total_loss.item(), original_images_gray.size(0))

            framework.encoder.eval()
            val_loss_meter = AverageMeter()
            with torch.no_grad():
                for _, image in val_loader:
                    original_images_gray = image.to(device)
                    reconstructed_images, synthetic_spikes = framework(original_images_gray, gumbel_tau)
                    original_images_rgb = original_images_gray.repeat(1, 3, 1, 1)
                    loss_l2 = l2_loss(reconstructed_images, original_images_rgb)
                    loss_ssim = 1 - ssim_loss(reconstructed_images, original_images_rgb)
                    loss_perceptual = perceptual_loss(reconstructed_images, original_images_rgb).mean()
                    loss_sparsity = torch.mean(synthetic_spikes)
                    total_loss = (
                        config.loss.lambda_l2 * loss_l2
                        + config.loss.lambda_perceptual * loss_perceptual
                        + config.loss.lambda_ssim * loss_ssim
                        + config.loss.lambda_sparsity * loss_sparsity
                    )
                    val_loss_meter.update(total_loss.item(), original_images_gray.size(0))

            current_lr = optimizer.param_groups[0]["lr"]
            train_losses.append(train_loss_meter.avg)
            val_losses.append(val_loss_meter.avg)
            learning_rates.append(current_lr)
            scheduler.step()

            if val_loss_meter.avg < best_val_loss:
                best_val_loss = val_loss_meter.avg
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": framework.encoder.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "best_val_loss": best_val_loss,
                        "train_losses": train_losses,
                        "val_losses": val_losses,
                        "learning_rates": learning_rates,
                    },
                    encoder_save_path,
                )
    finally:
        if train_losses and val_losses and learning_rates:
            plt.figure(figsize=(18, 6))
            plt.subplot(1, 2, 1)
            plt.plot(train_losses, label="Train Loss", alpha=0.8)
            plt.plot(val_losses, label="Validation Loss", linewidth=2)
            best_epoch = val_losses.index(min(val_losses))
            plt.scatter(best_epoch, min(val_losses), s=100, c="red", marker="*", label="Best Validation Loss")
            plt.title("Training and Validation Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.6)

            plt.subplot(1, 2, 2)
            plt.plot(learning_rates, label="Learning Rate", color="darkorange")
            plt.title("Learning Rate Schedule")
            plt.xlabel("Epoch")
            plt.ylabel("Learning Rate")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.6)

            plt.tight_layout()
            plt.savefig(os.path.join(encoder_model_dir, config.training.training_curve_name), dpi=300)


def main(config: CoAdaptiveConfig | None = None, device: str | None = None) -> None:
    train(config or get_coadaptive_config(), resolve_device(device))


if __name__ == "__main__":
    main()
