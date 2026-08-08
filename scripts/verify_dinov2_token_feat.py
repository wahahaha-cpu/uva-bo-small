#!/usr/bin/env python3
"""Offline DINOv2/student token-feature alignment smoke test."""

import argparse
import os
import pathlib
import sys

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch

from unified_video_action.model.common.dinov2_teacher import DINOv2Teacher
from unified_video_action.model.common.student_tokenizer import StudentLatentTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="pretrained_models/dinov2/dinov2_vits14_pretrain.pth",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    assert os.path.isfile(args.checkpoint), args.checkpoint

    device = torch.device(args.device)
    teacher = DINOv2Teacher(
        model_name="dinov2_vits14",
        img_size=224,
        model_img_size=518,
        checkpoint_path=args.checkpoint,
        loader="timm",
    ).to(device)
    student = StudentLatentTokenizer(
        img_size=256,
        patch_size=16,
        in_channels=3,
        latent_channels=16,
        hidden_dim=304,
        depth=5,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.0,
        use_temporal_mixer=True,
        temporal_kernel_size=3,
    ).to(device)
    projector = torch.nn.Sequential(
        torch.nn.Linear(304, 512),
        torch.nn.SiLU(),
        torch.nn.Linear(512, 512),
        torch.nn.SiLU(),
        torch.nn.Linear(512, 384),
    ).to(device)

    x = torch.randn(1, 3, 2, 256, 256, device=device)
    with torch.no_grad():
        teacher_tokens = teacher.extract_tokens(x)
    _, student_tokens = student(x)
    projected = projector(student_tokens)
    assert teacher_tokens.shape == (1, 2, 256, 384), teacher_tokens.shape
    assert student_tokens.shape == (1, 2, 256, 304), student_tokens.shape
    assert projected.shape == teacher_tokens.shape, projected.shape

    loss = 1.0 - torch.nn.functional.cosine_similarity(
        projected.float(), teacher_tokens.float(), dim=-1
    ).mean()
    loss.backward()
    assert torch.isfinite(loss), loss
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in student.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in projector.parameters())
    assert all(p.grad is None for p in teacher.parameters())
    print(
        "DINOv2 token-feature smoke test: PASS "
        f"teacher={tuple(teacher_tokens.shape)} "
        f"student={tuple(student_tokens.shape)} "
        f"projected={tuple(projected.shape)} loss={loss.item():.6f}"
    )


if __name__ == "__main__":
    main()
