import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from coadaptive_vqvae.config.defaults import VQVAEConfig, get_vqvae_config
from coadaptive_vqvae.data.datasets import SpikeDataset1
from coadaptive_vqvae.training.common import print_component_mapping, print_experiment_banner, resolve_device
from coadaptive_vqvae.models.vqvae import ModelNoVQ
from coadaptive_vqvae.utils.metrics import AverageMeter


def train(config: VQVAEConfig, device: torch.device | None = None) -> None:
    os.environ["CUDA_LAUNCH_BLOCKING"] = config.training.cuda_launch_blocking
    device = device or resolve_device()
    print_experiment_banner("vqvae", config)
    print_component_mapping("vqvae", ["Spike Pattern Attention Module (SPAM)", "Latent Space Projection", "Vector Quantizer"])
    writer = SummaryWriter(config.training.log_dir)
    model_dir = config.training.model_dir
    os.makedirs(model_dir, exist_ok=True)
    best_checkpoint_name = config.training.best_checkpoint_name
    latest_checkpoint_name = config.training.latest_checkpoint_name

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
        dataset=train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )

    model = ModelNoVQ(
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
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate, amsgrad=False)

    start_epoch = 0
    best_val_loss = float("inf")
    best_epoch = 0
    best_checkpoint_path = os.path.join(model_dir, best_checkpoint_name)
    latest_checkpoint_path = os.path.join(model_dir, latest_checkpoint_name)

    if os.path.exists(best_checkpoint_path):
        checkpoint = torch.load(best_checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["best_val_loss"]
        best_epoch = checkpoint["best_epoch"]

    for epoch in range(start_epoch, config.training.num_epochs + start_epoch):
        model.train()
        loss_meter = AverageMeter()
        recon_meter = AverageMeter()
        perplexity_meter = AverageMeter()

        for spike, image in train_loader:
            spike = spike.to(device)
            image = image.to(device)
            optimizer.zero_grad()
            vq_loss, reconstruction, perplexity = model(spike)
            recon_loss = F.mse_loss(reconstruction, image) / config.training.data_variance
            total_loss = recon_loss + vq_loss
            total_loss.backward()
            optimizer.step()
            loss_meter.update(total_loss.item(), image.size(0))
            recon_meter.update(recon_loss.item(), image.size(0))
            perplexity_meter.update(perplexity.item(), image.size(0))

        writer.add_scalar("loss/train", loss_meter.avg, epoch)
        writer.add_scalar("loss/train_reconstruction", recon_meter.avg, epoch)
        writer.add_scalar("metrics/train_perplexity", perplexity_meter.avg, epoch)

        model.eval()
        val_loss_meter = AverageMeter()
        with torch.no_grad():
            for spike, image in val_loader:
                spike = spike.to(device)
                image = image.to(device)
                vq_loss, reconstruction, _ = model(spike)
                recon_loss = F.mse_loss(reconstruction, image) / config.training.data_variance
                total_loss = recon_loss + vq_loss
                val_loss_meter.update(total_loss.item(), image.size(0))

        writer.add_scalar("loss/val", val_loss_meter.avg, epoch)

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
        }
        torch.save(checkpoint, latest_checkpoint_path)

        if val_loss_meter.avg < best_val_loss:
            best_val_loss = val_loss_meter.avg
            best_epoch = epoch
            checkpoint["best_epoch"] = best_epoch
            checkpoint["best_val_loss"] = best_val_loss
            torch.save(checkpoint, best_checkpoint_path)

        if (epoch - best_epoch) > config.training.early_stop_patience:
            break


def main(config: VQVAEConfig | None = None, device: str | None = None) -> None:
    train(config or get_vqvae_config(), resolve_device(device))


if __name__ == "__main__":
    main()
