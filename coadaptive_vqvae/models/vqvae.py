import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
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
        return self.activation(self.conv(torch.cat([avg_features, max_features], dim=1)))


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction_ratio: int = 16, kernel_size: int = 7) -> None:
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        channel_refined = self.channel_attention(inputs)
        return channel_refined * self.spatial_attention(channel_refined)


class FeedForwardSpikeEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        input_shape=(150, 60),
        out_channels: int = 64,
        output_shape=(8, 8),
    ) -> None:
        super().__init__()
        input_height, input_width = input_shape
        output_height, output_width = output_shape
        flattened_input = in_channels * input_height * input_width
        flattened_output = out_channels * output_height * output_width
        half_output = flattened_output // 2
        quarter_output = out_channels * max(1, output_height // 2) * max(1, output_width // 2)
        self.out_channels = out_channels
        self.output_shape = output_shape
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_input, half_output),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.15),
            nn.Linear(half_output, quarter_output),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.15),
            nn.Linear(quarter_output, half_output),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.15),
            nn.Linear(half_output, flattened_output),
            nn.Dropout(0.15),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.layers(inputs)
        output_height, output_width = self.output_shape
        return outputs.view(-1, self.out_channels, output_height, output_width)


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)
        self.commitment_cost = commitment_cost

    def forward(self, inputs: torch.Tensor):
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.embedding_dim)
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_input, self.embedding.weight.t())
        )
        encoding_indices = torch.argmin(distances, dim=1, keepdim=True)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape)
        encoder_loss = F.mse_loss(quantized.detach(), inputs)
        codebook_loss = F.mse_loss(quantized, inputs.detach())
        loss = codebook_loss + self.commitment_cost * encoder_loss
        quantized = inputs + (quantized - inputs).detach()
        average_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(average_probs * torch.log(average_probs + 1e-10)))
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings


class VectorQuantizerEMA(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float,
        decay: float,
        epsilon: float = 1e-5,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.normal_()
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
        self.ema_weight = nn.Parameter(torch.Tensor(num_embeddings, embedding_dim))
        self.ema_weight.data.normal_()

    def forward(self, inputs: torch.Tensor):
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.embedding_dim)
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_input, self.embedding.weight.t())
        )
        encoding_indices = torch.argmin(distances, dim=1, keepdim=True)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape)

        if self.training:
            self.ema_cluster_size = self.ema_cluster_size * self.decay + (1 - self.decay) * torch.sum(encodings, 0)
            total_count = torch.sum(self.ema_cluster_size.data)
            self.ema_cluster_size = (
                (self.ema_cluster_size + self.epsilon)
                / (total_count + self.num_embeddings * self.epsilon)
                * total_count
            )
            delta_weight = torch.matmul(encodings.t(), flat_input)
            self.ema_weight = nn.Parameter(self.ema_weight * self.decay + (1 - self.decay) * delta_weight)
            self.embedding.weight = nn.Parameter(self.ema_weight / self.ema_cluster_size.unsqueeze(1))

        encoder_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self.commitment_cost * encoder_loss
        quantized = inputs + (quantized - inputs).detach()
        average_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(average_probs * torch.log(average_probs + 1e-10)))
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, residual_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels, residual_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(32, residual_channels),
            nn.ReLU(True),
            nn.Conv2d(residual_channels, hidden_channels, kernel_size=1, stride=1, bias=False),
            nn.GroupNorm(32, hidden_channels),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.layers(inputs)


class ResidualStack(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int, residual_channels: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [ResidualBlock(in_channels, hidden_channels, residual_channels) for _ in range(num_layers)]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = inputs
        for layer in self.layers:
            outputs = layer(outputs)
        return F.relu(outputs)


class ConvEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int, residual_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels // 2, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels // 2, hidden_channels, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1)
        self.residual_stack = ResidualStack(hidden_channels, hidden_channels, num_layers, residual_channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = F.relu(self.conv1(inputs))
        outputs = F.relu(self.conv2(outputs))
        outputs = self.conv3(outputs)
        return self.residual_stack(outputs)


class ConvDecoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int, residual_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1)
        self.residual_stack = ResidualStack(hidden_channels, hidden_channels, num_layers, residual_channels)
        self.deconv1 = nn.ConvTranspose2d(hidden_channels, hidden_channels // 2, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(hidden_channels // 2, 3, kernel_size=4, stride=2, padding=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.conv1(inputs)
        outputs = self.residual_stack(outputs)
        outputs = F.relu(self.deconv1(outputs))
        return self.deconv2(outputs)


class VQVAEModel(nn.Module):
    def __init__(
        self,
        num_hiddens: int,
        num_residual_layers: int,
        num_residual_hiddens: int,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float,
        decay: float = 0.0,
        in_channels: int = 1,
        in_scale=(150, 60),
    ) -> None:
        super().__init__()
        self.encoder = FeedForwardSpikeEncoder(in_channels=in_channels, input_shape=in_scale, out_channels=num_hiddens)
        self.residual_stack = ResidualStack(num_hiddens, num_hiddens, num_residual_layers, num_residual_hiddens)
        self.attention = CBAM(num_hiddens)
        self.projection_head = ProjectionHead(num_hiddens, num_hiddens, embedding_dim)
        self.quantizer = (
            VectorQuantizerEMA(num_embeddings, embedding_dim, commitment_cost, decay)
            if decay > 0.0
            else VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        )
        self.decoder = ConvDecoder(embedding_dim, num_hiddens, num_residual_layers, num_residual_hiddens)

    def forward(self, inputs: torch.Tensor):
        latent = self.encoder(inputs)
        latent = self.residual_stack(latent)
        latent = self.attention(latent)
        latent = self.projection_head(latent)
        loss, quantized, perplexity, _ = self.quantizer(latent)
        reconstruction = self.decoder(quantized)
        return loss, reconstruction, perplexity


class ModelNoVQ(nn.Module):
    def __init__(
        self,
        num_hiddens: int,
        num_residual_layers: int,
        num_residual_hiddens: int,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float,
        decay: float = 0.0,
        in_channels: int = 1,
        in_scale=(150, 60),
    ) -> None:
        super().__init__()
        self.encoder = FeedForwardSpikeEncoder(in_channels=in_channels, input_shape=in_scale, out_channels=num_hiddens)
        self.residual_stack = ResidualStack(num_hiddens, num_hiddens, num_residual_layers, num_residual_hiddens)
        self.attention = CBAM(num_hiddens)
        self.projection_head = ProjectionHead(num_hiddens, num_hiddens, embedding_dim)
        self.decoder = ConvDecoder(embedding_dim, num_hiddens, num_residual_layers, num_residual_hiddens)

    def forward(self, inputs: torch.Tensor):
        latent = self.encoder(inputs)
        latent = self.residual_stack(latent)
        latent = self.attention(latent)
        latent = self.projection_head(latent)
        reconstruction = self.decoder(latent)
        zero_loss = torch.zeros((), device=inputs.device, dtype=latent.dtype)
        zero_perplexity = torch.zeros((), device=inputs.device, dtype=latent.dtype)
        return zero_loss, reconstruction, zero_perplexity


class VQConvVAE(nn.Module):
    def __init__(
        self,
        num_hiddens: int,
        num_residual_layers: int,
        num_residual_hiddens: int,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float,
        decay: float = 0.0,
        in_channels: int = 1,
        in_scale=(150, 60),
    ) -> None:
        super().__init__()
        input_height, input_width = in_scale
        self.pre_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels * input_height * input_width, 512),
            nn.Linear(512, 1024),
        )
        self.encoder = ConvEncoder(1, num_hiddens, num_residual_layers, num_residual_hiddens)
        self.pre_quantizer = nn.Conv2d(num_hiddens, embedding_dim, kernel_size=1, stride=1)
        self.quantizer = (
            VectorQuantizerEMA(num_embeddings, embedding_dim, commitment_cost, decay)
            if decay > 0.0
            else VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        )
        self.decoder = ConvDecoder(embedding_dim, num_hiddens, num_residual_layers, num_residual_hiddens)

    def forward(self, inputs: torch.Tensor):
        outputs = self.pre_encoder(inputs).view(-1, 1, 32, 32)
        outputs = self.encoder(outputs)
        outputs = self.pre_quantizer(outputs)
        loss, quantized, perplexity, _ = self.quantizer(outputs)
        reconstruction = self.decoder(quantized)
        return loss, reconstruction, perplexity


Model = VQVAEModel
