#!/usr/bin/env python3
"""Verify the 4-GPU and 8-GPU accumulation/LR semantics on a CPU oracle.

This intentionally simulates DDP's *average* reduction with the same global
batch in two layouts:

    4 ranks x local batch 8 x accumulation 4 = 128 samples/update
    8 ranks x local batch 16 x accumulation 1 = 128 samples/update

It does not need eight GPUs.  A double-precision, deterministic toy model
makes a scale error (such as summing four micro-batch means) immediately
visible while exercising the project's actual scheduler factory and step-count
helpers.
"""

from __future__ import annotations

import argparse
import copy
import math
import pathlib
import sys
from dataclasses import dataclass
from typing import Iterable

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
import torch.nn.functional as F
from torch import nn

from unified_video_action.common.training_utils import (
    accumulation_window,
    optimizer_steps_per_epoch,
)
from unified_video_action.model.common.lr_scheduler import get_scheduler


@dataclass(frozen=True)
class Layout:
    name: str
    world_size: int
    per_device_batch: int
    accumulation_steps: int

    @property
    def global_batch(self) -> int:
        return self.world_size * self.per_device_batch * self.accumulation_steps


LAYOUTS = (
    Layout("4 x 8 x accum-4", world_size=4, per_device_batch=8, accumulation_steps=4),
    Layout("8 x 16 x accum-1", world_size=8, per_device_batch=16, accumulation_steps=1),
)


def make_model() -> nn.Module:
    return nn.Sequential(
        nn.Linear(5, 9),
        nn.GELU(),
        nn.Linear(9, 3),
    ).double()


def loss_gradients(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    loss = F.mse_loss(model(x), y)
    return tuple(torch.autograd.grad(loss, tuple(model.parameters())))


def averaged_ddp_gradients(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    layout: Layout,
) -> tuple[torch.Tensor, ...]:
    """Mirror DDP averaging after per-window loss normalization."""

    if x.shape[0] != layout.global_batch or y.shape[0] != layout.global_batch:
        raise ValueError("Input must contain exactly one effective global batch")

    params = tuple(model.parameters())
    global_grads = [torch.zeros_like(param) for param in params]
    micro_batch = layout.per_device_batch

    # The order is irrelevant to the global mean, but this rank/micro-batch
    # partition matches the quantities used by the training loop.
    for rank in range(layout.world_size):
        rank_grads = [torch.zeros_like(param) for param in params]
        for micro_step in range(layout.accumulation_steps):
            offset = (rank * layout.accumulation_steps + micro_step) * micro_batch
            micro_x = x[offset : offset + micro_batch]
            micro_y = y[offset : offset + micro_batch]
            micro_grads = loss_gradients(model, micro_x, micro_y)
            for aggregate, grad in zip(rank_grads, micro_grads):
                aggregate.add_(grad / layout.accumulation_steps)
        for aggregate, rank_grad in zip(global_grads, rank_grads):
            aggregate.add_(rank_grad / layout.world_size)

    return tuple(global_grads)


def max_abs_difference(
    left: Iterable[torch.Tensor], right: Iterable[torch.Tensor]
) -> float:
    return max(
        (a.detach() - b.detach()).abs().max().item() for a, b in zip(left, right)
    )


def run_layout(
    initial_model: nn.Module,
    layout: Layout,
    updates: int,
    warmup_steps: int,
    total_updates: int,
) -> tuple[nn.Module, list[float], float]:
    model = copy.deepcopy(initial_model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        betas=(0.9, 0.95),
        weight_decay=0.02,
    )
    scheduler = get_scheduler(
        "cosine",
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_updates,
    )
    lr_history = [scheduler.get_last_lr()[0]]
    initial_grad_error = 0.0

    for update_idx in range(updates):
        # Every layout sees exactly the same effective global batch at each
        # update; only the rank/micro-batch partition differs.
        generator = torch.Generator().manual_seed(1000 + update_idx)
        x = torch.randn(layout.global_batch, 5, generator=generator, dtype=torch.float64)
        y = torch.randn(layout.global_batch, 3, generator=generator, dtype=torch.float64)

        ddp_grads = averaged_ddp_gradients(model, x, y, layout)
        if update_idx == 0:
            direct_grads = loss_gradients(model, x, y)
            initial_grad_error = max_abs_difference(ddp_grads, direct_grads)

        optimizer.zero_grad(set_to_none=True)
        for parameter, grad in zip(model.parameters(), ddp_grads):
            parameter.grad = grad
        optimizer.step()
        scheduler.step()
        lr_history.append(scheduler.get_last_lr()[0])

    return model, lr_history, initial_grad_error


def check_step_horizons(num_samples: int) -> tuple[int, int]:
    if num_samples < 1:
        raise ValueError("num_samples must be >= 1")

    updates = []
    for layout in LAYOUTS:
        # Before Accelerator.prepare(), DataLoader batch_size is per-device.
        dataloader_batches = math.ceil(num_samples / layout.per_device_batch)
        updates.append(
            optimizer_steps_per_epoch(
                dataloader_batches,
                layout.world_size,
                layout.accumulation_steps,
                split_batches=False,
            )
        )
    assert updates[0] == updates[1], updates
    return updates[0], math.ceil(num_samples / LAYOUTS[0].global_batch)


def check_short_final_window() -> None:
    windows = [accumulation_window(idx, 5, 4) for idx in range(5)]
    assert windows[:4] == [(0, 4, 4, False)] * 3 + [(0, 4, 4, True)]
    assert windows[4] == (4, 5, 1, True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=6)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--total-updates", type=int, default=20000)
    # 124165 is the current Libero10 train split length in this workspace.
    parser.add_argument("--num-samples", type=int, default=124165)
    args = parser.parse_args()

    if args.updates < 1:
        raise ValueError("--updates must be >= 1")
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be >= 0")
    if args.total_updates < args.updates:
        raise ValueError("--total-updates must be >= --updates")

    assert LAYOUTS[0].global_batch == LAYOUTS[1].global_batch == 128
    updates_per_epoch, expected_updates = check_step_horizons(args.num_samples)
    check_short_final_window()

    torch.manual_seed(77)
    initial_model = make_model()
    result_4 = run_layout(
        initial_model,
        LAYOUTS[0],
        args.updates,
        args.warmup_steps,
        args.total_updates,
    )
    result_8 = run_layout(
        initial_model,
        LAYOUTS[1],
        args.updates,
        args.warmup_steps,
        args.total_updates,
    )

    model_4, lr_4, direct_grad_error_4 = result_4
    model_8, lr_8, direct_grad_error_8 = result_8
    parameter_error = max_abs_difference(model_4.parameters(), model_8.parameters())
    lr_error = max(abs(a - b) for a, b in zip(lr_4, lr_8))
    tolerance = 1e-12

    assert direct_grad_error_4 < tolerance, direct_grad_error_4
    assert direct_grad_error_8 < tolerance, direct_grad_error_8
    assert parameter_error < tolerance, parameter_error
    assert lr_error < tolerance, lr_error

    print("Gradient/LR equivalence: PASS")
    print(
        f"  layouts: {LAYOUTS[0].name} == {LAYOUTS[1].name} "
        f"== global batch {LAYOUTS[0].global_batch}"
    )
    print(
        f"  scheduler horizon: {updates_per_epoch} updates/epoch "
        f"for {args.num_samples} samples (expected {expected_updates})"
    )
    print(
        f"  direct-global gradient max error: "
        f"4-GPU={direct_grad_error_4:.3e}, 8-GPU={direct_grad_error_8:.3e}"
    )
    print(f"  post-{args.updates}-update parameter max error: {parameter_error:.3e}")
    print(f"  LR trajectory max error: {lr_error:.3e}")
    print("  final short accumulation window: PASS")


if __name__ == "__main__":
    main()
