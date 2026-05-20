from __future__ import annotations


from typing import Callable, Optional

import torch
from torch.utils.data import DataLoader

from .data import get_batch, get_dataloader


def train_model(
    model: torch.nn.Module,
    data: Optional[torch.Tensor] = None,
    dataloader: Optional[DataLoader] = None,
    *,
    steps: Optional[int] = None,
    epochs: Optional[int] = None,
    batch_size: int = 64,
    block_size: int = 128,
    lr: float = 3e-4,
    device: torch.device = torch.device("cpu"),
    log_interval: int = 1000,
    dataloader_num_workers: int = 0,
) -> list[tuple[int, float]]:
    """Train `model`.

    - If `dataloader` is provided, run the simple epoch-loop style training:
        for epoch in range(epochs):
          for idx, targets in dataloader:
            idx, targets = idx.to(device), targets.to(device)
            ...

    - If `dataloader` is not provided, but `data` is, a dataloader will be
      created via `get_dataloader` and used the same way.

    - If neither `dataloader` nor `data` is provided, raises `ValueError`.

    - `steps` can be used to limit total training steps; training exits when
      that number of optimizer updates is reached.
    """

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_history: list[tuple[int, float]] = []

    if dataloader is None:
        if data is None:
            raise ValueError("either `data` or `dataloader` must be provided")
        dataloader = get_dataloader(
            data, batch_size=batch_size, block_size=block_size, device=device, num_workers=dataloader_num_workers
        )

    step = 0
    # Default epochs to 1 if not specified when using a dataloader
    if epochs is None:
        epochs = 1

    for epoch in range(epochs):
        for idx, targets in dataloader:
            idx, targets = idx.to(device), targets.to(device)

            _, loss = model(idx, targets)
            assert loss is not None

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_history.append((step, loss.item()))

            if log_interval and step % log_interval == 0:
                print(f"step={step} loss={loss.item():.4f}")

            step += 1
            if steps is not None and step >= steps:
                return loss_history

    return loss_history


def save_model(model: torch.nn.Module, path: str) -> None:
    torch.save(model.state_dict(), path)


def load_model(model: torch.nn.Module, path: str, device: torch.device) -> torch.nn.Module:
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    return model
