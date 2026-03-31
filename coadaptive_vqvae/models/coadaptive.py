import torch
import torch.nn as nn
import torch.nn.functional as F

from .paper_modules import SpectralFeatureExtractor
from .paper_modules import SpikeFusionUnit
from .paper_modules import TopologicalFeatureExtractor

FrequencyFeatureExtractor = SpectralFeatureExtractor


class CoAdaptiveEncoder(nn.Module):
    def __init__(self, out_shape=(1, 150, 60)) -> None:
        super().__init__()
        self.topological_feature_extractor = TopologicalFeatureExtractor(out_channels=1024)
        self.spectral_feature_extractor = SpectralFeatureExtractor(out_channels=256, out_size=(2, 2))
        self.spike_fusion_unit = SpikeFusionUnit(topological_channels=1024, spectral_channels=256, fused_channels=1024)
        self.decoder_head = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(32, 512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(32, 256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(32, 128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(32, 64),
            nn.ReLU(True),
            nn.Upsample(size=out_shape[1:], mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(16, 32),
            nn.ReLU(True),
            nn.Conv2d(32, 2, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, image: torch.Tensor, gumbel_tau: float = 1.0) -> torch.Tensor:
        topological_features = self.topological_feature_extractor(image)
        spectral_features = self.spectral_feature_extractor(image)
        fused_features = self.spike_fusion_unit(topological_features, spectral_features)
        spike_logits = self.decoder_head(fused_features)
        spike_distribution = F.gumbel_softmax(spike_logits, tau=gumbel_tau, hard=True, dim=1)
        return spike_distribution[:, 1:2, :, :]


class CoAdaptiveFramework(nn.Module):
    def __init__(self, virtual_brain_model: nn.Module, encoder_model: nn.Module) -> None:
        super().__init__()
        self.virtual_brain = virtual_brain_model
        self.encoder = encoder_model
        for parameter in self.virtual_brain.parameters():
            parameter.requires_grad = False

    def forward(self, original_image: torch.Tensor, gumbel_tau: float = 1.0):
        synthetic_spikes = self.encoder(original_image, gumbel_tau)
        _, reconstructed_image, _ = self.virtual_brain(synthetic_spikes)
        return reconstructed_image, synthetic_spikes
