import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnext50_32x4d


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction_ratio: int = 16) -> None:
        super().__init__()
        reduced_channels = max(1, channels // reduction_ratio)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(),
            nn.Linear(reduced_channels, channels, bias=False),
        )
        self.activation = nn.Sigmoid()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, channels, _, _ = inputs.shape
        max_features = self.max_pool(inputs).view(batch_size, channels)
        avg_features = self.avg_pool(inputs).view(batch_size, channels)
        weights = self.activation(self.mlp(max_features) + self.mlp(avg_features)).view(batch_size, channels, 1, 1)
        return inputs * weights


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.activation = nn.Sigmoid()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        avg_features = torch.mean(inputs, dim=1, keepdim=True)
        max_features, _ = torch.max(inputs, dim=1, keepdim=True)
        fused_features = torch.cat([avg_features, max_features], dim=1)
        return self.activation(self.conv(fused_features))


class SpikePatternAttentionModule(nn.Module):
    def __init__(self, channels: int, reduction_ratio: int = 16, kernel_size: int = 7) -> None:
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        channel_refined = self.channel_attention(inputs)
        return channel_refined * self.spatial_attention(channel_refined)


class LatentSpaceProjection(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=1),
            nn.GroupNorm(32, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, output_dim, kernel_size=1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class TopologicalFeatureExtractor(nn.Module):
    def __init__(self, out_channels: int = 1024) -> None:
        super().__init__()
        backbone = resnext50_32x4d(weights=None)
        original_conv = backbone.conv1
        backbone.conv1 = nn.Conv2d(
            1,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False,
        )
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-3])
        self.out_channels = out_channels

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(inputs)


class SpectralFeatureExtractor(nn.Module):
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
        self.out_channels = out_channels

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        fft_result = torch.fft.fft2(inputs, norm="ortho")
        shifted_result = torch.fft.fftshift(fft_result, dim=(-2, -1))
        spectral_representation = torch.cat([torch.real(shifted_result), torch.imag(shifted_result)], dim=1)
        return self.feature_net(spectral_representation)


class SpikeFusionUnit(nn.Module):
    def __init__(self, topological_channels: int, spectral_channels: int, fused_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(topological_channels + spectral_channels, fused_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(32, fused_channels),
            nn.ReLU(True),
        )

    def forward(self, topological_features: torch.Tensor, spectral_features: torch.Tensor) -> torch.Tensor:
        return self.layers(torch.cat([topological_features, spectral_features], dim=1))


SPAM = SpikePatternAttentionModule
