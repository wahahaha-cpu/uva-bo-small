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
        _clone_batch(batch),
        return_debug_components=True,
        return_debug_outputs=True,
    )
    total_loss, (video_loss, action_loss), components, debug_outputs = result
    video_prediction = debug_outputs["video_prediction"]
    for label, value in components.items():
        assert torch.isfinite(value).all(), f"Non-finite {label}: {value}"
    assert torch.isfinite(video_loss).all()
    assert torch.isfinite(action_loss).all()
    alignment_metrics = {}
    for key in (
        "dino_cos",
        "dino_mse",
        "dino_stats",
        "jepa_kl",
        "jepa_teacher_row_sum_mean",
        "jepa_teacher_row_sum_max_error",
        "jepa_student_row_sum_mean",
        "jepa_student_row_sum_max_error",
        "jepa_teacher_probability_min",
        "jepa_teacher_probability_max",
        "jepa_student_probability_min",
        "jepa_student_probability_max",
        "jepa_teacher_entropy",
        "jepa_student_entropy",
        "jepa_diffusion_timestep_mean",
        "jepa_diffusion_timestep_min",
        "jepa_diffusion_timestep_max",
        "jepa_future_mask_fraction",
    ):
        value = policy._last_align_metrics.get(key)
        if value is not None:
            assert torch.isfinite(value).all(), f"Non-finite {key}: {value}"
            alignment_metrics[key] = value.detach().float().item()
    if policy.jepa_teacher is not None:
        assert video_prediction is not None
        predicted_future_latents = video_prediction["predicted_future_latents"]
        diffusion_timesteps = video_prediction["diffusion_timesteps"]
        future_mask = video_prediction["future_mask"]
        assert predicted_future_latents.shape == (1, 4, 256, 16)
        assert predicted_future_latents.requires_grad
        assert diffusion_timesteps.shape == (1,)
        assert torch.all(
            diffusion_timesteps
            == policy.model.diffloss.train_diffusion.num_timesteps - 1
        )
        assert future_mask.shape == (1, 4, 256)
        assert torch.all(future_mask == 1)
        assert alignment_metrics["jepa_future_mask_fraction"] == 1.0
        assert abs(alignment_metrics["jepa_teacher_row_sum_mean"] - 1.0) < 1e-5
        assert alignment_metrics["jepa_teacher_row_sum_max_error"] < 1e-5
        assert abs(alignment_metrics["jepa_student_row_sum_mean"] - 1.0) < 1e-5
        assert alignment_metrics["jepa_student_row_sum_max_error"] < 1e-5

        decoder_probe = video_prediction["decoder_condition"].float().square().mean()
        future_gradient, history_gradient = torch.autograd.grad(
            decoder_probe,
            (
                video_prediction["future_input_latents"],
                video_prediction["conditioning_input_latents"],
            ),
            retain_graph=True,
            allow_unused=True,
        )
        future_gradient_norm = (
            0.0
            if future_gradient is None
            else future_gradient.detach().float().norm().item()
        )
        history_gradient_norm = (
            0.0
            if history_gradient is None
            else history_gradient.detach().float().norm().item()
        )
        assert future_gradient_norm < 1e-10
        assert history_gradient_norm > 0.0

        prediction_scale = (
            predicted_future_latents.detach()
            .float()
            .square()
            .mean()
            .sqrt()
            .clamp_min(1e-6)
        )
        prediction_probe = (
            predicted_future_latents.float() / prediction_scale
        ).square().mean()
        prediction_future_gradient, prediction_history_gradient = torch.autograd.grad(
            prediction_probe,
            (
                video_prediction["future_input_latents"],
                video_prediction["conditioning_input_latents"],
            ),
            retain_graph=True,
            allow_unused=True,
        )
        prediction_future_gradient_norm = (
            0.0
            if prediction_future_gradient is None
            else prediction_future_gradient.detach().float().norm().item()
        )
        prediction_history_gradient_norm = (
            0.0
            if prediction_history_gradient is None
            else prediction_history_gradient.detach().float().norm().item()
        )
        assert prediction_future_gradient_norm < 1e-10
        assert prediction_history_gradient_norm > 0.0

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
    if policy.jepa_teacher is not None:
        component_gradient_norms["grad_norm_base_to_video_diffusion"] = (
            _gradient_norm(
                components["base_loss"],
                policy.model.diffloss.parameters(),
            )
        )
        component_gradient_norms["grad_norm_jepa_to_video_diffusion"] = (
            _gradient_norm(
                components["weighted_jepa_loss"],
                policy.model.diffloss.parameters(),
            )
        )
        assert component_gradient_norms[
            "grad_norm_jepa_to_video_diffusion"
        ] > 0.0

    policy.zero_grad(set_to_none=True)
    total_loss.backward()
    module_summaries = {
        "student_tokenizer": _module_gradient_summary(policy.student_tokenizer),
        "video_diffusion_head": _module_gradient_summary(policy.model.diffloss),
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
    assert module_summaries["video_diffusion_head"]["gradient_norm"] > 0.0
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
            predicted_future_latents = video_prediction[
                "predicted_future_latents"
            ]
            f0, f1 = predicted_future_latents[:, :2].unbind(dim=1)
            fused_forward = policy.temporal_fusion(f0, f1)
            fused_swapped = policy.temporal_fusion(f1, f0)
            fusion_order_difference = (
                fused_forward - fused_swapped
            ).abs().mean()
        assert fusion_order_difference.item() > 1e-7
        temporal_metrics["fusion_swap_mean_abs_diff"] = (
            fusion_order_difference.item()
        )

        with torch.no_grad():
            (
                probability_forward,
                relation_forward,
                teacher_tokens,
                metadata,
            ) = policy._extract_jepa_relational_target(
                future_clip, return_metadata=True
            )
            probability_reverse, relation_reverse, _, _ = (
                policy._extract_jepa_relational_target(
                    future_clip.flip(2), return_metadata=True
                )
            )
            assert teacher_tokens.shape == (1, 2, 576, 768)
            assert relation_forward.shape == probability_forward.shape == (1, 576, 576)
            assert torch.allclose(
                probability_forward.sum(dim=-1),
                torch.ones_like(probability_forward[..., 0]),
                atol=1e-5,
                rtol=1e-5,
            )
            jepa_reverse_difference = (
                probability_forward - probability_reverse
            ).abs().mean()
            jepa_reverse_cosine = torch.nn.functional.cosine_similarity(
                relation_forward.flatten(1), relation_reverse.flatten(1), dim=-1
            ).mean()
        assert jepa_reverse_difference.item() > 1e-7
        temporal_metrics["jepa_relation_reverse_mean_abs_diff"] = (
            jepa_reverse_difference.item()
        )
        temporal_metrics["jepa_relation_reverse_cosine"] = jepa_reverse_cosine.item()
        temporal_metrics["jepa_patch_embed_shape"] = metadata["patch_embed_shape"]
        temporal_metrics["video_diffusion_pred_x0_shape"] = tuple(
            predicted_future_latents.shape
        )
        temporal_metrics["future_mask_fraction"] = (
            video_prediction["future_mask"].float().mean().item()
        )
        temporal_metrics["diffusion_timesteps"] = (
            video_prediction["diffusion_timesteps"].tolist()
        )
        temporal_metrics["decoder_grad_wrt_future_input"] = future_gradient_norm
        temporal_metrics["decoder_grad_wrt_history_input"] = history_gradient_norm
        temporal_metrics["prediction_grad_wrt_future_input"] = (
            prediction_future_gradient_norm
        )
        temporal_metrics["prediction_grad_wrt_history_input"] = (
            prediction_history_gradient_norm
        )

    print("--- one-real-batch verification ---")
    print(f"config: {args.config_name}")
    print(f"raw_batch_image: {tuple(batch['obs']['image'].shape)}")
    print(f"selected_frame_indices: {selected_indices.tolist()}")
    print(f"history_clip: {tuple(history_clip.shape)}")
    print(f"future_clip: {tuple(future_clip.shape)}")
    if video_prediction is not None:
        print(
            "video_prediction: "
            f"pred_x0={tuple(video_prediction['predicted_future_latents'].shape)}, "
            f"decoder_condition={tuple(video_prediction['decoder_condition'].shape)}, "
            f"mask={tuple(video_prediction['future_mask'].shape)}, "
            f"timesteps={video_prediction['diffusion_timesteps'].tolist()}"
        )
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
