from datetime import time
from typing import Optional

import torch
from torch.utils.data import DataLoader

from .data import get_batch, get_dataloader


def profile_model(
    model: torch.nn.Module,
    data: Optional[torch.Tensor] = None,
    batch_size: Optional[int] = 64,
    block_size: Optional[int] = 128,
    dataloader: Optional[DataLoader] = None,
    dataloader_num_workers: int = 0,
    steps: Optional[int] = None,
    device: torch.device = torch.device("cpu"),
):
    """Profile `model`.

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

    model.eval()
    loss_history: list[tuple[int, float]] = []

    if dataloader is None:
        if data is None:
            raise ValueError("either `data` or `dataloader` must be provided")
        dataloader = get_dataloader(
            data, batch_size=batch_size, block_size=block_size, device=device, num_workers=dataloader_num_workers
        )

    print("\nModel and input tensor created. Starting profiler...")
    print("-" * 50)

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA, # Only include if CUDA is available
        ],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("profile"),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        step = 0
        for idx, targets in dataloader:
            idx, targets = idx.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            
            model(idx, targets)

            prof.step() # Notify the profiler that a step is complete
            print("step: ", step)
            step += 1
            if step >= steps:
                break

        # step = 0
        # for idx, targets in dataloader:
        #     idx, targets = idx.to(device), targets.to(device)
        #     with torch.no_grad():
        #         model(idx, targets)
        #     prof.step() # Notify the profiler that a step is complete
        #     print("step: ", step)
        #     step += 1
        #     if step >= steps:
        #         break

    # --- Print Profiler Results ---
    print("Profiler run complete. Printing summary...")
    print("-" * 50)

    # Print a summary of the results to the console, grouped by our custom labels.
    # The `group_by_input_shape` is useful for seeing how different tensor sizes perform.
    # The `group_by_stack_n` helps attribute time to specific lines of code.
    print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=15))

    print("\n" + "-" * 50)
    print("To view the detailed trace, run the following command in your terminal:")
    print("tensorboard --logdir=profile")
    print("-" * 50)


