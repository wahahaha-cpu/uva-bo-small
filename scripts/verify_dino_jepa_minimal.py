#!/usr/bin/env python3
"""One-real-batch verification for complementary DINO and V-JEPA losses."""

import argparse
import copy
import pathlib
import random
import sys
from typing import Dict, Iterable

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import hydra
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate

from unified_video_action.common.pytorch_util import dict_apply
from unified_video_action.utils.data_utils import process_data, resize_image


def _clone_batch(batch):
    return dict_apply(batch, lambda value: value.clone())


def _gradient_norm(
    loss: torch.Tensor, parameters: Iterable[torch.nn.Parameter]
) -> float:
    parameters = tuple(parameter for parameter in parameters if parameter.requires_grad)
    if not loss.requires_grad or not parameters:
        return 0.0
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    squared_norm = loss.new_zeros((), dtype=torch.float32)
    for gradient in gradients:
        if gradient is not None:
            squared_norm = squared_norm + gradient.detach().float().square().sum()
    return squared_norm.sqrt().item()


def _module_gradient_summary(module: torch.nn.Module) -> Dict[str, float]:
    parameters = tuple(parameter for parameter in module.parameters() if parameter.requires_grad)
    gradients = tuple(
        parameter.grad for parameter in parameters if parameter.grad is not None
    )
    finite = all(torch.isfinite(gradient).all().item() for gradient in gradients)
    squared_norm = sum(
        gradient.detach().float().square().sum().item()
        for gradient in gradients
    )
    return {
        "trainable_tensors": len(parameters),
        "gradient_tensors": len(gradients),
        "gradient_norm": float(squared_norm**0.5),
        "finite": bool(finite),
    }


def _assert_teacher_frozen(teacher: torch.nn.Module, label: str) -> None:
    assert not teacher.training, f"{label} wrapper left training mode"
    assert all(not parameter.requires_grad for parameter in teacher.parameters())
    assert all(parameter.grad is None for parameter in teacher.parameters())


def _check_optimizer_membership(policy, optimizer) -> None:
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    for teacher in (policy.dinov2_teacher, policy.jepa_teacher):
        if teacher is not None:
            assert not any(id(parameter) in optimizer_ids for parameter in teacher.parameters())
    for module in (
        policy.student_tokenizer,
        policy.student_to_dino_projector,
        policy.temporal_fusion,
        policy.student_to_jepa_projector,
    ):
        if module is not None:
            assert all(
                id(parameter) in optimizer_ids
                for parameter in module.parameters()
                if parameter.requires_grad
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-name", default="uva_libero10_dino_jepa_minimal"
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--check-ema", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    device = torch.device(args.device)

    config_dir = str((ROOT_DIR / "unified_video_action" / "config").resolve())
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name=args.config_name)
    OmegaConf.resolve(cfg)

    dataset = hydra.utils.instantiate(cfg.task.dataset)
    normalizer = dataset.get_normalizer()
    batch = default_collate([dataset[args.sample_index]])
    batch = resize_image(cfg, batch)
    batch = dict_apply(batch, lambda value: value.to(device))

    policy = hydra.utils.instantiate(
        cfg.model.policy,
        task_name=cfg.task.name,
        task_modes=cfg.task.task_modes,
        normalizer_type=cfg.task.dataset.normalizer_type,
        language_emb_model=cfg.task.dataset.language_emb_model,
    )
    policy.set_normalizer(normalizer)
    policy.to(device)
    policy.train()

    if args.check_ema:
        ema_policy = copy.deepcopy(policy)
        ema_policy.eval()
        policy_state = policy.state_dict()
        ema_policy.load_state_dict(policy_state, strict=True)
        for module_name in (
            "student_to_dino_projector",
            "temporal_fusion",
            "student_to_jepa_projector",
        ):
            if getattr(policy, module_name) is not None:
                assert any(
                    key.startswith(f"{module_name}.") for key in policy_state
                ), f"{module_name} is missing from the checkpoint state"
        del ema_policy
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    optimizer = policy.get_optimizer(**cfg.model.policy.optimizer)
    _check_optimizer_membership(policy, optimizer)
    del optimizer

    processed_video, _, selected_indices = process_data(
        _clone_batch(batch),
        task_name=policy.task_name,
        **policy.kwargs,
    )
    history_clip, future_clip = torch.chunk(processed_video, 2, dim=2)
    assert history_clip.shape[2] == future_clip.shape[2] == 4
    assert torch.all(selected_indices[1:] > selected_indices[:-1])

    result = policy.compute_loss(
        _clone_batch(batch), return_debug_components=True
    )
    total_loss, (video_loss, action_loss), components = result
    for label, value in components.items():
        assert torch.isfinite(value).all(), f"Non-finite {label}: {value}"
    assert torch.isfinite(video_loss).all()
    assert torch.isfinite(action_loss).all()
    alignment_metrics = {}
    for key in (
        "dino_cos",
        "dino_mse",
        "dino_stats",
        "jepa_dynamics_cos",
        "jepa_dynamics_mse",
        "jepa_dynamics_stats",
    ):
        value = policy._last_align_metrics.get(key)
        if value is not None:
            assert torch.isfinite(value).all(), f"Non-finite {key}: {value}"
            alignment_metrics[key] = value.detach().float().item()

    student_parameters = tuple(policy.student_tokenizer.parameters())
    component_gradient_norms = {
        "grad_norm_base": _gradient_norm(
            components["base_loss"], student_parameters
        ),
        "grad_norm_dino": _gradient_norm(
            components["weighted_dino_loss"], student_parameters
        ),
        "grad_norm_jepa": _gradient_norm(
            components["weighted_jepa_loss"], student_parameters
        ),
    }

    policy.zero_grad(set_to_none=True)
    total_loss.backward()
    module_summaries = {
        "student_tokenizer": _module_gradient_summary(policy.student_tokenizer)
    }
    if policy.student_to_dino_projector is not None:
        module_summaries["student_to_dino_projector"] = _module_gradient_summary(
            policy.student_to_dino_projector
        )
    if policy.temporal_fusion is not None:
        module_summaries["temporal_fusion"] = _module_gradient_summary(
            policy.temporal_fusion
        )
    if policy.student_to_jepa_projector is not None:
        module_summaries["student_to_jepa_projector"] = _module_gradient_summary(
            policy.student_to_jepa_projector
        )

    has_auxiliary_teacher = any(
        module is not None
        for module in (
            policy.student_to_dino_projector,
            policy.temporal_fusion,
            policy.student_to_jepa_projector,
        )
    )
    if has_auxiliary_teacher:
        assert module_summaries["student_tokenizer"]["gradient_norm"] > 0.0
    for label, summary in module_summaries.items():
        assert summary["gradient_tensors"] > 0, f"No gradient for {label}"
        if has_auxiliary_teacher or label != "student_tokenizer":
            assert summary["gradient_norm"] > 0.0, f"Zero gradient for {label}"
        assert summary["finite"], f"Non-finite gradient for {label}"

    if policy.dinov2_teacher is not None:
        _assert_teacher_frozen(policy.dinov2_teacher, "DINOv2")
    if policy.jepa_teacher is not None:
        _assert_teacher_frozen(policy.jepa_teacher, "V-JEPA")
    teacher_status = {}
    for label, teacher in (
        ("dinov2", policy.dinov2_teacher),
        ("vjepa", policy.jepa_teacher),
    ):
        if teacher is not None:
            teacher_status[label] = {
                "training": teacher.training,
                "trainable_tensors": sum(
                    parameter.requires_grad for parameter in teacher.parameters()
                ),
                "gradient_tensors": sum(
                    parameter.grad is not None for parameter in teacher.parameters()
                ),
            }

    temporal_metrics = {}
    if policy.temporal_fusion is not None:
        with torch.no_grad():
            _, history_student_tokens = policy.student_tokenizer(history_clip)
            s0, s1 = history_student_tokens[:, :2].unbind(dim=1)
            fused_forward = policy.temporal_fusion(s0, s1)
            fused_swapped = policy.temporal_fusion(s1, s0)
            fusion_order_difference = (
                fused_forward - fused_swapped
            ).abs().mean()
        assert fusion_order_difference.item() > 1e-7
        temporal_metrics["fusion_swap_mean_abs_diff"] = (
            fusion_order_difference.item()
        )

        with torch.no_grad():
            delta_forward, _, _ = policy._extract_jepa_dynamics_target(
                history_clip, return_metadata=True
            )
            delta_reverse, _, _ = policy._extract_jepa_dynamics_target(
                history_clip.flip(2), return_metadata=True
            )
            jepa_reverse_difference = (delta_forward - delta_reverse).abs().mean()
            jepa_reverse_cosine = torch.nn.functional.cosine_similarity(
                delta_forward.flatten(1), delta_reverse.flatten(1), dim=-1
            ).mean()
        assert jepa_reverse_difference.item() > 1e-7
        temporal_metrics["jepa_reverse_mean_abs_diff"] = (
            jepa_reverse_difference.item()
        )
        temporal_metrics["jepa_reverse_cosine"] = jepa_reverse_cosine.item()

    print("--- one-real-batch verification ---")
    print(f"config: {args.config_name}")
    print(f"raw_batch_image: {tuple(batch['obs']['image'].shape)}")
    print(f"selected_frame_indices: {selected_indices.tolist()}")
    print(f"history_clip: {tuple(history_clip.shape)}")
    print(f"future_clip: {tuple(future_clip.shape)}")
    print(
        "losses: "
        f"base={components['base_loss'].detach().float().item():.8f}, "
        f"dino={components['dino_loss'].detach().float().item():.8f}, "
        f"jepa={components['jepa_loss'].detach().float().item():.8f}, "
        f"weighted_dino={components['weighted_dino_loss'].detach().float().item():.8f}, "
        f"weighted_jepa={components['weighted_jepa_loss'].detach().float().item():.8f}, "
        f"total={components['total_loss'].detach().float().item():.8f}"
    )
    print(f"component_gradient_norms: {component_gradient_norms}")
    print(f"alignment_metrics: {alignment_metrics}")
    print(f"module_gradients: {module_summaries}")
    print(f"teacher_status: {teacher_status}")
    print(f"temporal_sanity: {temporal_metrics}")
    print(f"{args.config_name} verification: PASS")


if __name__ == "__main__":
    main()
