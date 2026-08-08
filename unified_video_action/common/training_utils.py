"""Small helpers for world-size-independent optimizer-step accounting."""

import math
from typing import Tuple


def local_batches_per_epoch(
    dataloader_batches: int,
    num_processes: int,
    split_batches: bool = False,
) -> int:
    """Return the number of batches consumed by one process after ``prepare``.

    Accelerate keeps the batch count unchanged when ``split_batches=True`` and
    shards whole batches otherwise.  The explicit formula is used before
    ``accelerator.prepare`` so the LR scheduler can be built with the same
    horizon it will see in the loop.
    """

    dataloader_batches = int(dataloader_batches)
    num_processes = int(num_processes)
    if dataloader_batches < 0:
        raise ValueError("dataloader_batches must be non-negative")
    if num_processes < 1:
        raise ValueError("num_processes must be >= 1")
    if split_batches:
        return dataloader_batches
    return math.ceil(dataloader_batches / num_processes)


def optimizer_steps_per_epoch(
    dataloader_batches: int,
    num_processes: int,
    accumulation_steps: int,
    split_batches: bool = False,
) -> int:
    """Return optimizer updates in one epoch.

    ``gradient_accumulate_every`` is intentionally an optimizer-update unit,
    not a raw micro-batch/global-step unit.
    """

    accumulation_steps = int(accumulation_steps)
    if accumulation_steps < 1:
        raise ValueError("accumulation_steps must be >= 1")
    local_batches = local_batches_per_epoch(
        dataloader_batches,
        num_processes,
        split_batches=split_batches,
    )
    return math.ceil(local_batches / accumulation_steps)


def accumulation_window(
    batch_idx: int,
    num_batches: int,
    accumulation_steps: int,
) -> Tuple[int, int, int, bool]:
    """Describe the accumulation window containing ``batch_idx``.

    Returns ``(start, end, size, is_update_step)`` where ``end`` is exclusive.
    The final short window is stepped and normalized by its actual number of
    micro-batches, so it cannot silently lose or gain a scale factor.
    """

    batch_idx = int(batch_idx)
    num_batches = int(num_batches)
    accumulation_steps = int(accumulation_steps)
    if num_batches < 1:
        raise ValueError("num_batches must be >= 1")
    if not 0 <= batch_idx < num_batches:
        raise IndexError(
            f"batch_idx={batch_idx} is outside [0, {num_batches})"
        )
    if accumulation_steps < 1:
        raise ValueError("accumulation_steps must be >= 1")

    start = (batch_idx // accumulation_steps) * accumulation_steps
    end = min(start + accumulation_steps, num_batches)
    size = end - start
    return start, end, size, batch_idx + 1 == end
