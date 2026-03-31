import json
from dataclasses import asdict

import torch


def resolve_device(explicit_device: str | None = None) -> torch.device:
    if explicit_device:
        return torch.device(explicit_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_experiment_banner(name: str, config) -> None:
    print(f"Experiment: {name}")
    print(json.dumps(asdict(config), indent=2, ensure_ascii=False))


def print_component_mapping(name: str, components: list[str]) -> None:
    print(f"Components for {name}:")
    for component in components:
        print(f"- {component}")
