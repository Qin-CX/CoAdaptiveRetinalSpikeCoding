import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnext50_32x4d


class FrequencyFeatureExtractor(nn.Module):
    def __init__(self, out_channels: int = 256, out_size=(2, 2)) -> None:
        super().__init__()
        self.feature_net = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(16, 64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32, 128),
            nn.ReLU(True),
            nn.Conv2d(128, out_channels, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(out_size),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        fft_result = torch.fft.fft2(inputs, norm="ortho")
        shifted_result = torch.fft.fftshift(fft_result, dim=(-2, -1))
        frequency_tensor = torch.cat([torch.real(shifted_result), torch.imag(shifted_result)], dim=1)
        return self.feature_net(frequency_tensor)


class CoAdaptiveEncoder(nn.Module):
    def __init__(self, out_shape=(1, 150, 60)) -> None:
        super().__init__()
        spatial_backbone = resnext50_32x4d(weights=None)
        original_conv = spatial_backbone.conv1
        spatial_backbone.conv1 = nn.Conv2d(
            1,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False,
        )
        self.spatial_branch = nn.Sequential(*list(spatial_backbone.children())[:-3])
        self.frequency_branch = FrequencyFeatureExtractor(out_channels=256, out_size=(2, 2))
        self.fusion_layer = nn.Sequential(
            nn.Conv2d(1280, 1024, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(32, 1024),
            nn.ReLU(True),
        )
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
        spatial_features = self.spatial_branch(image)
        frequency_features = self.frequency_branch(image)
        fused_features = self.fusion_layer(torch.cat([spatial_features, frequency_features], dim=1))
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
