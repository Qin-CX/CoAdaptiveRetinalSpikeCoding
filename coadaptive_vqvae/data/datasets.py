from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif"}
DEFAULT_DATA_ROOT = Path(r"E:\0619vqvae_sustech\Retinal-spike-train-decoder\dataset\cifar10_2_0718\gray_32_10000")
DEFAULT_SPIKE_FILE = DEFAULT_DATA_ROOT / "spike.npz"
SPLIT_RATIOS = {
    "train": (0.0, 0.7),
    "val": (0.7, 0.9),
    "valid": (0.7, 0.9),
    "test": (0.9, 1.0),
    None: (0.0, 1.0),
}


class SpikeImageDataset(Dataset):
    def __init__(
        self,
        image_root: Optional[str] = None,
        spike_file: Optional[str] = None,
        transform: Optional[Callable] = None,
        split: Optional[str] = "train",
        num_neurons: int = 150,
        spike_times: int = 60,
    ) -> None:
        self.image_root = Path(image_root) if image_root else DEFAULT_DATA_ROOT
        self.spike_file = Path(spike_file) if spike_file else DEFAULT_SPIKE_FILE
        self.transform = transform
        self.num_neurons = num_neurons
        self.spike_times = spike_times
        self.spike_length = num_neurons * spike_times

        if split not in SPLIT_RATIOS:
            raise ValueError(f"Unsupported split: {split}")

        self.image_ids = sorted(
            file.name for file in self.image_root.iterdir() if file.suffix.lower() in IMAGE_SUFFIXES
        )
        spike_data = np.load(self.spike_file)["arr_0"].reshape(len(self.image_ids), self.spike_length)
        start_ratio, end_ratio = SPLIT_RATIOS[split]
        start_index = int(len(self.image_ids) * start_ratio)
        end_index = int(len(self.image_ids) * end_ratio)
        self.image_ids = self.image_ids[start_index:end_index]
        self.spike_data = spike_data[start_index:end_index]

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int):
        spike = torch.tensor(self.spike_data[index], dtype=torch.float32).view(1, self.num_neurons, self.spike_times)
        image = Image.open(self.image_root / self.image_ids[index]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return spike, image


class SpikeDataset1(SpikeImageDataset):
    def __init__(
        self,
        img_path: Optional[str] = None,
        transforms: Optional[Callable] = None,
        data_type: Optional[str] = "train",
        nuerons_nums: int = 150,
        spike_times: int = 60,
    ) -> None:
        spike_file = Path(img_path) / "spike.npz" if img_path else None
        super().__init__(
            image_root=img_path,
            spike_file=str(spike_file) if spike_file else None,
            transform=transforms,
            split=data_type,
            num_neurons=nuerons_nums,
            spike_times=spike_times,
        )
