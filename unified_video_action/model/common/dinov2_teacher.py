"""Frozen DINOv2 teacher used for patch-token feature alignment.

The Libero pipeline supplies images in ``[-1, 1]``.  DINOv2 expects RGB values
in ``[0, 1]`` followed by ImageNet normalization, so the conversion is kept in
this module instead of changing the policy/MAR image convention.
"""

import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DINOV2_HUB_TO_TIMM = {
    "dinov2_vits14": "vit_small_patch14_dinov2.lvd142m",
    "dinov2_vitb14": "vit_base_patch14_dinov2.lvd142m",
    "dinov2_vitl14": "vit_large_patch14_dinov2.lvd142m",
    "dinov2_vitg14": "vit_giant_patch14_dinov2.lvd142m",
}


def _resolve_timm_model_name(model_name: str) -> str:
    if model_name in DINOV2_HUB_TO_TIMM:
        return DINOV2_HUB_TO_TIMM[model_name]
    if model_name in DINOV2_HUB_TO_TIMM.values():
        return model_name
    raise ValueError(
        f"Unsupported DINOv2 model_name={model_name!r}. Expected one of "
        f"{list(DINOV2_HUB_TO_TIMM)} or the corresponding timm names."
    )


def _load_timm_model(
    model_name: str,
    img_size: int,
    model_img_size: int,
    checkpoint_path: Optional[str],
):
    """Build a timm DINOv2 model and load a local checkpoint without network IO.

    The bundled DINOv2-S/14 checkpoint was trained at 518px and contains a
    37x37 positional grid.  Constructing the model at that native size avoids
    a state-dict shape mismatch; ``dynamic_img_size`` then interpolates the
    grid for the 224px inputs used by this experiment.
    """
    import timm

    timm_name = _resolve_timm_model_name(model_name)
    if checkpoint_path is not None and str(checkpoint_path).strip() != "":
        checkpoint_path = os.path.abspath(str(checkpoint_path))
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"DINOv2 checkpoint not found: {checkpoint_path}")

    # All official timm DINOv2 LVD-142M weights use a 518px native grid.  Keep
    # this independent from the runtime inference size (usually 224px).
    create_kwargs = {
        "pretrained": False,
        "num_classes": 0,
        "img_size": int(model_img_size),
        "dynamic_img_size": True,
    }
    try:
        model = timm.create_model(timm_name, **create_kwargs)
    except TypeError:  # older timm versions may not expose dynamic_img_size
        create_kwargs.pop("dynamic_img_size")
        model = timm.create_model(timm_name, **create_kwargs)

    if checkpoint_path is None or str(checkpoint_path).strip() == "":
        # A network download in every distributed worker is both surprising and
        # fragile.  Require an explicit local file for reproducible experiments.
        raise FileNotFoundError(
            "DINOv2 requires dinov2_teacher_params.checkpoint_path."
        )

    payload = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    if not isinstance(payload, dict):
        raise TypeError(
            f"Expected a DINOv2 state dict at {checkpoint_path}, got {type(payload)}"
        )

    # Loading at native resolution makes the bundled 1370-token pos_embed fit
    # exactly.  Do not silently accept a partially loaded teacher.
    msg = model.load_state_dict(payload, strict=True)
    if msg.missing_keys or msg.unexpected_keys:
        raise RuntimeError(
            "DINOv2 checkpoint did not load exactly: "
            f"missing={msg.missing_keys}, unexpected={msg.unexpected_keys}"
        )
    print(
        f"Loaded DINOv2 teacher via timm: {timm_name} "
        f"(checkpoint={checkpoint_path}, native_img_size={model_img_size}, "
        f"inference_img_size={img_size})",
        flush=True,
    )
    return model


def _extract_patch_tokens(model: nn.Module, features: torch.Tensor) -> torch.Tensor:
    if isinstance(features, dict):
        tokens = features.get("x_norm_patchtokens")
        if tokens is None:
            raise RuntimeError("DINOv2 forward_features output has no patch tokens")
        return tokens

    prefix_tokens = int(getattr(model, "num_prefix_tokens", 1))
    if features.ndim != 3 or features.shape[1] <= prefix_tokens:
        raise RuntimeError(f"Unexpected DINOv2 feature shape: {tuple(features.shape)}")
    return features[:, prefix_tokens:, :]


class DINOv2Teacher(nn.Module):
    """Frozen DINOv2 patch-token extractor.

    Input is ``[B, C, T, H, W]`` in ``[-1, 1]``. Output is
    ``[B, T, S, D]`` where ``S`` is the square patch grid and ``D`` is the
    backbone embedding width (384 for ViT-S/14).
    """

    def __init__(
        self,
        model_name: str = "dinov2_vits14",
        img_size: int = 224,
        model_img_size: int = 518,
        checkpoint_path: Optional[str] = None,
        loader: str = "timm",
    ):
        super().__init__()
        if str(loader).lower() != "timm":
            raise ValueError("DINOv2 teacher currently supports loader='timm' only.")
        self.model_name = model_name
        self.img_size = int(img_size)
        self.model_img_size = int(model_img_size)
        if self.img_size <= 0 or self.model_img_size <= 0:
            raise ValueError("DINOv2 image sizes must be positive")

        self.model = _load_timm_model(
            model_name,
            self.img_size,
            self.model_img_size,
            checkpoint_path,
        )
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.feat_dim = int(
            getattr(self.model, "embed_dim", getattr(self.model, "num_features", 0))
        )
        if self.feat_dim <= 0:
            raise RuntimeError("Could not infer DINOv2 feature dimension")
        patch_size = getattr(getattr(self.model, "patch_embed", None), "patch_size", 14)
        if isinstance(patch_size, tuple):
            patch_size = patch_size[0]
        self.patch_size = int(patch_size)
        if self.img_size % self.patch_size != 0:
            raise ValueError(
                f"DINOv2 img_size={self.img_size} must be divisible by patch_size={self.patch_size}"
            )
        self.num_spatial_tokens = (self.img_size // self.patch_size) ** 2

        self.register_buffer(
            "mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False
        )

    def train(self, mode: bool = True):
        # ``policy.train()`` recursively reaches frozen teachers. Keep DINO in
        # eval mode so its stochastic/dropout behavior cannot change by epoch.
        super().train(mode)
        self.model.eval()
        return self

    @torch.no_grad()
    def extract_tokens(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected [B,C,T,H,W], got {tuple(x.shape)}")
        x = x.float()
        batch_size, channels, timesteps, height, width = x.shape
        if channels != 3:
            raise ValueError(f"DINOv2 expects 3 channels, got {channels}")

        frames = x.permute(0, 2, 1, 3, 4).reshape(
            batch_size * timesteps, channels, height, width
        )
        frames = F.interpolate(
            frames,
            size=(self.img_size, self.img_size),
            mode="bilinear",
            align_corners=False,
        )
        frames = (frames + 1.0) * 0.5
        frames = (frames - self.mean) / self.std

        features = self.model.forward_features(frames)
        tokens = _extract_patch_tokens(self.model, features)
        if tokens.shape[1] != self.num_spatial_tokens:
            raise RuntimeError(
                "DINOv2 patch grid mismatch: "
                f"expected {self.num_spatial_tokens}, got {tokens.shape[1]}"
            )
        if tokens.shape[2] != self.feat_dim:
            raise RuntimeError(
                f"DINOv2 feature width mismatch: expected {self.feat_dim}, got {tokens.shape[2]}"
            )
        return tokens.reshape(batch_size, timesteps, tokens.shape[1], tokens.shape[2])
