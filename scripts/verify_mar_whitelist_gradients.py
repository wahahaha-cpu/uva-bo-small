#!/usr/bin/env python3
"""CPU-only demonstration of MAR whitelisting and gradient flow.

This is an educational unit-style probe. It mirrors the freeze/filter rules used
by UnifiedVideoActionPolicy without loading the full MAR, dataset, or checkpoints.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn


EXACT_WHITELIST_NAMES = {
    "diffusion_temporal_embed",
    "diffusion_spatial_embed",
    "mask_token",
    "blank_token",
}


def is_mar_pos_or_fake_parameter(name: str) -> bool:
    """Mirror UnifiedVideoActionPolicy._is_mar_pos_or_fake_parameter()."""
    leaf_name = name.rsplit(".", 1)[-1]
    return (
        leaf_name.startswith("fake_")
        or leaf_name.endswith("_pos_embed")
        or leaf_name in EXACT_WHITELIST_NAMES
    )


class TinyStudent(nn.Module):
    """Stand-in for the trainable student tokenizer."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 12),
            nn.GELU(),
            nn.Linear(12, 4),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.net(observation)


class TinyMar(nn.Module):
    """MAR-shaped graph with frozen blocks and trainable interface parameters."""

    def __init__(self) -> None:
        super().__init__()
        self.z_proj_cond = nn.Linear(4, 8)
        self.encoder = nn.Sequential(nn.Linear(8, 8), nn.GELU())
        self.decoder = nn.Sequential(nn.Linear(8, 8), nn.GELU())
        self.action_head = nn.Linear(8, 3)

        self.fake_action_latent = nn.Parameter(torch.randn(1, 8) * 0.02)
        self.temporal_pos_embed = nn.Parameter(torch.randn(1, 8) * 0.02)

        # This parameter deliberately matches the whitelist but is not used in
        # forward(), so the DDP zero-term behavior can be demonstrated.
        self.unused_pos_embed = nn.Parameter(torch.randn(1, 8) * 0.02)

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        hidden = self.z_proj_cond(condition)
        hidden = hidden + self.fake_action_latent + self.temporal_pos_embed
        hidden = self.encoder(hidden)
        hidden = self.decoder(hidden)
        return self.action_head(hidden)


def configure_mar_trainability(mar: nn.Module) -> tuple[str, ...]:
    """Freeze all MAR parameters, then reopen the whitelist."""
    mar.requires_grad_(False)
    for name, param in mar.named_parameters():
        if is_mar_pos_or_fake_parameter(name):
            param.requires_grad = True

    trainable_names = tuple(
        name for name, param in mar.named_parameters() if param.requires_grad
    )
    if not trainable_names:
        raise RuntimeError("The MAR whitelist matched no parameters.")
    return trainable_names


def add_weight_decay(
    module: nn.Module,
    weight_decay: float,
) -> list[dict[str, object]]:
    """Mirror the policy's requires_grad filter and decay grouping."""
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []

    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]


def optimizer_parameter_ids(optimizer: torch.optim.Optimizer) -> set[int]:
    return {
        id(param)
        for group in optimizer.param_groups
        for param in group["params"]
    }


def module_grad_norm(module: nn.Module) -> float:
    total_squared = 0.0
    for param in module.parameters():
        if param.grad is not None:
            total_squared += param.grad.detach().float().pow(2).sum().item()
    return total_squared**0.5


def parameter_grad_norm(param: nn.Parameter) -> float | None:
    if param.grad is None:
        return None
    return param.grad.detach().float().norm().item()


def ddp_unused_term(parameters: Iterable[nn.Parameter]) -> torch.Tensor:
    """Attach parameters to a graph with an exactly zero derivative."""
    terms = [
        torch.nan_to_num(param, nan=0.0, posinf=0.0, neginf=0.0).sum() * 0.0
        for param in parameters
        if param.requires_grad
    ]
    if not terms:
        return torch.tensor(0.0)
    return torch.stack(terms).sum()


def assert_optimizer_membership(
    mar: TinyMar,
    student: TinyStudent,
    align_projector: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> None:
    optimizer_ids = optimizer_parameter_ids(optimizer)

    for name, param in mar.named_parameters():
        in_optimizer = id(param) in optimizer_ids
        assert in_optimizer == param.requires_grad, (
            f"MAR optimizer mismatch for {name}: "
            f"requires_grad={param.requires_grad}, in_optimizer={in_optimizer}"
        )

    for module_name, module in (
        ("student", student),
        ("align_projector", align_projector),
    ):
        for name, param in module.named_parameters():
            if param.requires_grad:
                assert id(param) in optimizer_ids, (
                    f"Trainable {module_name} parameter is missing: {name}"
                )


def clone_parameters(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: param.detach().clone()
        for name, param in module.named_parameters()
    }


def changed_parameter_names(
    module: nn.Module,
    before: dict[str, torch.Tensor],
) -> list[str]:
    return [
        name
        for name, param in module.named_parameters()
        if not torch.equal(before[name], param.detach())
    ]


def make_action_loss(
    student: TinyStudent,
    mar: TinyMar,
    observation: torch.Tensor,
    action_target: torch.Tensor,
) -> torch.Tensor:
    condition = student(observation)
    action_prediction = mar(condition)
    return F.mse_loss(action_prediction, action_target)


def main() -> None:
    torch.manual_seed(7)

    student = TinyStudent()
    mar = TinyMar()
    align_projector = nn.Linear(4, 4)
    trainable_mar_names = configure_mar_trainability(mar)

    optimizer_groups: list[dict[str, object]] = []
    for module in (mar, student, align_projector):
        optimizer_groups.extend(add_weight_decay(module, weight_decay=0.01))
    optimizer = torch.optim.AdamW(optimizer_groups, lr=1e-2)

    print("Trainable MAR whitelist:")
    for name in trainable_mar_names:
        print(f"  {name}")

    assert_optimizer_membership(mar, student, align_projector, optimizer)
    print("Optimizer membership: PASS")

    observation = torch.randn(5, 6)
    action_target = torch.randn(5, 3)

    mar_before = clone_parameters(mar)
    student_before = clone_parameters(student)

    optimizer.zero_grad(set_to_none=True)
    action_loss = make_action_loss(student, mar, observation, action_target)
    action_loss.backward()

    student_grad = module_grad_norm(student)
    frozen_with_grad = [
        name
        for name, param in mar.named_parameters()
        if not param.requires_grad and param.grad is not None
    ]
    active_whitelist_grads = {
        name: parameter_grad_norm(param)
        for name, param in mar.named_parameters()
        if param.requires_grad and (parameter_grad_norm(param) or 0.0) > 0.0
    }

    assert student_grad > 0.0, "Action loss did not reach the student."
    assert not frozen_with_grad, frozen_with_grad
    assert active_whitelist_grads, "No active whitelist parameter received a gradient."
    assert mar.unused_pos_embed.grad is None
    assert module_grad_norm(align_projector) == 0.0

    print(f"Action-only student grad norm: {student_grad:.6e}")
    print("Action-active MAR whitelist gradients:")
    for name, grad_norm in active_whitelist_grads.items():
        print(f"  {name}: {grad_norm:.6e}")
    print("Frozen MAR gradients: PASS (all None)")
    print("Unused align projector gradient: PASS (None)")

    optimizer.step()
    changed_mar_names = changed_parameter_names(mar, mar_before)
    changed_student_names = changed_parameter_names(student, student_before)
    changed_frozen_names = [
        name
        for name in changed_mar_names
        if not dict(mar.named_parameters())[name].requires_grad
    ]

    assert changed_student_names, "The student did not update."
    assert not changed_frozen_names, changed_frozen_names
    assert "fake_action_latent" in changed_mar_names
    assert "temporal_pos_embed" in changed_mar_names
    assert "unused_pos_embed" not in changed_mar_names
    print("Optimizer step: PASS (student/active whitelist changed, frozen MAR unchanged)")

    # Repeat backward with the same zero-valued graph attachments used for DDP
    # safety. This deliberately turns an otherwise unused parameter's grad from
    # None into an all-zero tensor.
    optimizer.zero_grad(set_to_none=True)
    action_loss = make_action_loss(student, mar, observation, action_target)
    loss_with_ddp_safety = action_loss
    loss_with_ddp_safety = loss_with_ddp_safety + ddp_unused_term(mar.parameters())
    loss_with_ddp_safety = loss_with_ddp_safety + ddp_unused_term(student.parameters())
    loss_with_ddp_safety = loss_with_ddp_safety + ddp_unused_term(
        align_projector.parameters()
    )
    loss_with_ddp_safety.backward()

    unused_grad = parameter_grad_norm(mar.unused_pos_embed)
    projector_grad = module_grad_norm(align_projector)
    assert unused_grad == 0.0
    assert all(param.grad is not None for param in align_projector.parameters())
    assert projector_grad == 0.0

    print(
        "DDP zero-term demonstration: PASS "
        "(grad exists for unused parameters, but norm is zero)"
    )
    print("PASS: whitelist optimizer and frozen-MAR gradient flow are correct.")


if __name__ == "__main__":
    main()
