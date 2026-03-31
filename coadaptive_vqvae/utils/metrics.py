import cv2
import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.val = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count


def gaussian_kernel_1d(size: int, sigma: float) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    kernel = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel.unsqueeze(0).unsqueeze(0)


def gaussian_filter(inputs: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    outputs = F.conv2d(inputs, kernel, stride=1, padding=0, groups=inputs.shape[1])
    outputs = F.conv2d(outputs, kernel.transpose(2, 3), stride=1, padding=0, groups=inputs.shape[1])
    return outputs


def _ssim(
    inputs_a: torch.Tensor,
    inputs_b: torch.Tensor,
    kernel: torch.Tensor,
    data_range: float = 1023,
    size_average: bool = True,
    full: bool = False,
):
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    kernel = kernel.to(inputs_a.device, dtype=inputs_a.dtype)
    mu_a = gaussian_filter(inputs_a, kernel)
    mu_b = gaussian_filter(inputs_b, kernel)
    mu_a_sq = mu_a.pow(2)
    mu_b_sq = mu_b.pow(2)
    mu_ab = mu_a * mu_b
    sigma_a_sq = gaussian_filter(inputs_a * inputs_a, kernel) - mu_a_sq
    sigma_b_sq = gaussian_filter(inputs_b * inputs_b, kernel) - mu_b_sq
    sigma_ab = gaussian_filter(inputs_a * inputs_b, kernel) - mu_ab
    cs_map = (2 * sigma_ab + c2) / (sigma_a_sq + sigma_b_sq + c2)
    ssim_map = ((2 * mu_ab + c1) / (mu_a_sq + mu_b_sq + c1)) * cs_map
    if size_average:
        ssim_value = ssim_map.mean()
        cs_value = cs_map.mean()
    else:
        ssim_value = ssim_map.mean(-1).mean(-1).mean(-1)
        cs_value = cs_map.mean(-1).mean(-1).mean(-1)
    if full:
        return ssim_value, cs_value
    return ssim_value


def ssim(
    inputs_a: torch.Tensor,
    inputs_b: torch.Tensor,
    win_size: int = 11,
    win_sigma: float = 10,
    kernel: torch.Tensor = None,
    data_range: float = 1,
    size_average: bool = True,
    full: bool = False,
):
    if len(inputs_a.shape) != 4:
        raise ValueError("Input images must be 4D tensors.")
    if inputs_a.dtype != inputs_b.dtype:
        raise ValueError("Input images must use the same dtype.")
    if inputs_a.shape != inputs_b.shape:
        raise ValueError("Input images must use the same shape.")
    if win_size % 2 != 1:
        raise ValueError("Window size must be odd.")
    if kernel is None:
        kernel = gaussian_kernel_1d(win_size, win_sigma).repeat(inputs_a.shape[1], 1, 1, 1)
    ssim_value, cs_value = _ssim(
        inputs_a,
        inputs_b,
        kernel=kernel,
        data_range=data_range,
        size_average=False,
        full=True,
    )
    if size_average:
        ssim_value = ssim_value.mean()
        cs_value = cs_value.mean()
    if full:
        return ssim_value, cs_value
    return ssim_value


class SSIM(torch.nn.Module):
    def __init__(
        self,
        win_size: int = 11,
        win_sigma: float = 1.5,
        data_range: float = None,
        size_average: bool = True,
        channel: int = 3,
    ) -> None:
        super().__init__()
        self.kernel = gaussian_kernel_1d(win_size, win_sigma).repeat(channel, 1, 1, 1)
        self.data_range = data_range
        self.size_average = size_average

    def forward(self, inputs_a: torch.Tensor, inputs_b: torch.Tensor) -> torch.Tensor:
        return ssim(
            inputs_a,
            inputs_b,
            kernel=self.kernel,
            data_range=self.data_range,
            size_average=self.size_average,
        )


def in_ssim_grid(src_image: np.ndarray, div_num: int = 32, size=(256, 256)):
    resized_image = cv2.resize(src_image, size)
    image_height, image_width, _ = resized_image.shape
    height_step = image_height // div_num
    width_step = image_width // div_num
    image_grid = []
    for row in range(div_num):
        image_row = []
        for col in range(div_num):
            image_row.append(
                resized_image[(row * width_step) : ((row + 1) * width_step), (col * height_step) : ((col + 1) * height_step)]
            )
        image_grid.append(image_row)
    scores = np.zeros((div_num - 2, div_num - 2, 1), dtype=np.float32)
    block_average = 0.0
    for row in range(1, div_num - 1):
        for col in range(1, div_num - 1):
            neighbors = [
                structural_similarity(image_grid[row - 1][col - 1], image_grid[row][col], channel_axis=2),
                structural_similarity(image_grid[row - 1][col], image_grid[row][col], channel_axis=2),
                structural_similarity(image_grid[row - 1][col + 1], image_grid[row][col], channel_axis=2),
                structural_similarity(image_grid[row][col - 1], image_grid[row][col], channel_axis=2),
                structural_similarity(image_grid[row][col + 1], image_grid[row][col], channel_axis=2),
                structural_similarity(image_grid[row + 1][col - 1], image_grid[row][col], channel_axis=2),
                structural_similarity(image_grid[row + 1][col], image_grid[row][col], channel_axis=2),
                structural_similarity(image_grid[row + 1][col + 1], image_grid[row][col], channel_axis=2),
            ]
            neighbor_score = sum(neighbors) / 8.0
            scores[row - 1, col - 1] = neighbor_score
            block_average += neighbor_score
    average_score = block_average / ((div_num - 2) * (div_num - 2))
    return average_score, scores


def in_ssim_region(src_image: np.ndarray, div_num: int = 32, size=(256, 256)):
    resized_image = cv2.resize(src_image, size)
    image_height, image_width, _ = resized_image.shape
    height_step = image_height // div_num
    width_step = image_width // div_num
    scores = np.zeros((div_num - 2, div_num - 2, 1), dtype=np.float32)
    block_average = 0.0
    for row in range(1, div_num - 1):
        for col in range(1, div_num - 1):
            neighbors = [
                structural_similarity(
                    resized_image[((row - 1) * width_step) : (row * width_step), ((col - 1) * height_step) : (col * height_step)],
                    resized_image[(row * width_step) : ((row + 1) * width_step), (col * height_step) : ((col + 1) * height_step)],
                    channel_axis=2,
                ),
                structural_similarity(
                    resized_image[((row - 1) * width_step) : (row * width_step), (col * height_step) : ((col + 1) * height_step)],
                    resized_image[(row * width_step) : ((row + 1) * width_step), (col * height_step) : ((col + 1) * height_step)],
                    channel_axis=2,
                ),
                structural_similarity(
                    resized_image[((row - 1) * width_step) : (row * width_step), ((col + 1) * height_step) : ((col + 2) * height_step)],
                    resized_image[(row * width_step) : ((row + 1) * width_step), (col * height_step) : ((col + 1) * height_step)],
                    channel_axis=2,
                ),
                structural_similarity(
                    resized_image[(row * width_step) : ((row + 1) * width_step), ((col - 1) * height_step) : (col * height_step)],
                    resized_image[(row * width_step) : ((row + 1) * width_step), (col * height_step) : ((col + 1) * height_step)],
                    channel_axis=2,
                ),
                structural_similarity(
                    resized_image[(row * width_step) : ((row + 1) * width_step), ((col + 1) * height_step) : ((col + 2) * height_step)],
                    resized_image[(row * width_step) : ((row + 1) * width_step), (col * height_step) : ((col + 1) * height_step)],
                    channel_axis=2,
                ),
                structural_similarity(
                    resized_image[((row + 1) * width_step) : ((row + 2) * width_step), ((col - 1) * height_step) : (col * height_step)],
                    resized_image[(row * width_step) : ((row + 1) * width_step), (col * height_step) : ((col + 1) * height_step)],
                    channel_axis=2,
                ),
                structural_similarity(
                    resized_image[((row + 1) * width_step) : ((row + 2) * width_step), (col * height_step) : ((col + 1) * height_step)],
                    resized_image[(row * width_step) : ((row + 1) * width_step), (col * height_step) : ((col + 1) * height_step)],
                    channel_axis=2,
                ),
                structural_similarity(
                    resized_image[((row + 1) * width_step) : ((row + 2) * width_step), ((col + 1) * height_step) : ((col + 2) * height_step)],
                    resized_image[(row * width_step) : ((row + 1) * width_step), (col * height_step) : ((col + 1) * height_step)],
                    channel_axis=2,
                ),
            ]
            neighbor_score = sum(neighbors) / 8.0
            scores[row - 1, col - 1] = neighbor_score
            block_average += neighbor_score
    average_score = block_average / ((div_num - 2) * (div_num - 2))
    return average_score, scores
