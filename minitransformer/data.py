from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader

@dataclass(frozen=True)
class Dataset:
    text: str
    chars: List[str]
    stoi: Dict[str, int]
    itos: Dict[int, str]
    data: torch.Tensor


def seed_everything(seed: int = 1337) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device(device: str | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_text(source: str | None = None, limit: int | None = None) -> str:
    if source == "tinystories" or source is None:
        from datasets import load_dataset

        dataset = load_dataset("roneneldan/TinyStories", split="train")
        count = len(dataset) if limit is None else min(limit, len(dataset))
        return "".join(dataset[i]["text"] for i in range(count))

    return Path(source).read_text(encoding="utf-8")


def build_dataset(text: str) -> Dataset:
    chars = sorted(set(text))
    stoi = {ch: idx for idx, ch in enumerate(chars)}
    itos = {idx: ch for ch, idx in stoi.items()}
    data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
    return Dataset(text=text, chars=chars, stoi=stoi, itos=itos, data=data)


class TorchTextDataset(TorchDataset):
    """A torch.utils.data.Dataset that yields (x, y) sequence pairs.

    Designed for use with `torch.utils.data.DataLoader` and multiple workers.
    Each item is a contiguous block of `block_size` tokens (x) and the
    next-token targets (y).
    """

    def __init__(self, data: torch.Tensor, block_size: int) -> None:
        if len(data) <= block_size:
            raise ValueError("dataset is too small for the selected block size")
        self.data = data
        self.block_size = block_size
        self.n_examples = len(data) - block_size

    def __len__(self) -> int:
        return self.n_examples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        i = int(idx)
        x = self.data[i : i + self.block_size].clone()
        y = self.data[i + 1 : i + self.block_size + 1].clone()
        return x, y


def get_dataloader(
    data: torch.Tensor,
    *,
    batch_size: int,
    block_size: int,
    device: Optional[torch.device] = None,
    num_workers: int = 4,
    pin_memory: bool = True,
    shuffle: bool = True,
    drop_last: bool = True,
    prefetch_factor: Optional[int] = None,
) -> DataLoader:
    """Create a DataLoader that yields batches already moved to `device`.

    - Use `num_workers>0` for parallel loading.
    - Set `pin_memory=True` for faster host->GPU transfers.
    - If `device` is provided and is CUDA, tensors are moved with
      `non_blocking=True` to enable asynchronous transfers.
    """

    if device is None:
        device = torch.device("cpu")

    dataset = TorchTextDataset(data, block_size=block_size)

    def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
        xs = torch.stack([b[0] for b in batch])
        ys = torch.stack([b[1] for b in batch])
        # Move to device if requested. Use non_blocking for CUDA with pinned memory.
        if device.type == "cuda":
            xs = xs.pin_memory().to(device, non_blocking=True)
            ys = ys.pin_memory().to(device, non_blocking=True)
        else:
            xs = xs.to(device)
            ys = ys.to(device)
        return xs, ys

    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=drop_last,
    )

    if prefetch_factor is not None:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    return DataLoader(dataset, **loader_kwargs)


def encode(text: str, stoi: Dict[str, int], fallback_char: str | None = None) -> torch.Tensor:
    if fallback_char is None:
        return torch.tensor([stoi[ch] for ch in text], dtype=torch.long)

    fallback_id = stoi[fallback_char]
    return torch.tensor([stoi.get(ch, fallback_id) for ch in text], dtype=torch.long)


def decode(tokens: torch.Tensor, itos: Dict[int, str]) -> str:
    return "".join(itos[int(token)] for token in tokens)


def get_batch(data: torch.Tensor, batch_size: int, block_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    max_start = len(data) - block_size
    if max_start <= 0:
        raise ValueError("dataset is too small for the selected block size")

    indices = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in indices])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in indices])
    return x, y
