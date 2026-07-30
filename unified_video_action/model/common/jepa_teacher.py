import os
import time
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
VJEPA2_HUB_REPO = "facebookresearch/vjepa2"

# V-JEPA 2.1 Base distilled from ViT-G, at its native resolution.
VJEPA2_HUB_MODELS = {
    "vjepa2_1_vit_base_384": (768, 384),
}


def _resolve_checkpoint_path(checkpoint_path: Optional[str]) -> str:
    if checkpoint_path is None or str(checkpoint_path).strip() == "":
        raise FileNotFoundError(
            "V-JEPA 2.1 alignment requires jepa_teacher_params.checkpoint_path."
        )
    resolved = os.path.abspath(str(checkpoint_path))
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"JEPA checkpoint not found: {resolved}")
    return resolved


def _torch_hub_load_locked(
    repo: str, model: str, pretrained: bool = False, timeout_s: int = 600
):
    """Serialize torch.hub initialization across distributed workers."""
    hub_dir = torch.hub.get_dir()
    lock_dir = os.path.join(hub_dir, "locks")
    os.makedirs(lock_dir, exist_ok=True)
    lock_path = os.path.join(lock_dir, repo.replace("/", "_") + ".lock")

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            try:
                return torch.hub.load(repo, model, pretrained=pretrained)
            finally:
                try:
                    os.remove(lock_path)
                except FileNotFoundError:
                    pass
        except FileExistsError:
            time.sleep(1.0)

    raise TimeoutError(
        f"Timed out waiting for torch.hub lock: {lock_path}. "
        "Another process may still be loading V-JEPA 2.1."
    )


def _load_vjepa2_hub_encoder(model_name: str, pretrained: bool = False):
    try:
        import einops  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "V-JEPA 2.1 requires `einops`. Install it with: pip install einops"
        ) from exc

    hub_out = _torch_hub_load_locked(VJEPA2_HUB_REPO, model_name, pretrained=pretrained)
    return hub_out[0] if isinstance(hub_out, tuple) else hub_out


def _clean_vjepa_encoder_state_dict(state_dict):
    cleaned = {}
    for key, value in state_dict.items():
        key = key.replace("module.", "").replace("backbone.", "")
        cleaned[key] = value
    return cleaned


def _load_vjepa2_encoder_from_checkpoint(
    encoder: nn.Module, checkpoint_path: str, checkpoint_key: str = "ema_encoder"
):
    payload = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(payload, dict) and checkpoint_key in payload:
        encoder_state = payload[checkpoint_key]
    elif isinstance(payload, dict) and "encoder" in payload:
        encoder_state = payload["encoder"]
    else:
        encoder_state = payload

    encoder_state = _clean_vjepa_encoder_state_dict(encoder_state)
    msg = encoder.load_state_dict(encoder_state, strict=False)
    print(f"Loaded V-JEPA 2.1 encoder weights from {checkpoint_path}")
    if msg.missing_keys:
        print("V-JEPA 2.1 encoder missing keys:", msg.missing_keys)
    if msg.unexpected_keys:
        print("V-JEPA 2.1 encoder unexpected keys:", msg.unexpected_keys)


class JEPATeacher(nn.Module):
    """Frozen V-JEPA 2.1 encoder used only for latent alignment."""

    def __init__(
        self,
        model_name: str = "vjepa2_1_vit_base_384",
        img_size: int = 384,
        tubelet_size: int = 2,
        checkpoint_path: Optional[str] = None,
        checkpoint_key: str = "ema_encoder",
        loader: str = "checkpoint",
    ):
        super().__init__()
        self.model_name = model_name
        self.img_size = int(img_size)
        self.tubelet_size = int(tubelet_size)

        if model_name not in VJEPA2_HUB_MODELS:
            raise ValueError(
                f"Unsupported JEPA model_name={model_name!r}. "
                f"Expected one of {list(VJEPA2_HUB_MODELS.keys())}."
            )
        if str(loader).lower() != "checkpoint":
            raise ValueError("V-JEPA 2.1 alignment only supports loader='checkpoint'.")
        if self.tubelet_size != 2:
            raise ValueError("V-JEPA 2.1 Base requires tubelet_size=2.")

        default_feat_dim, default_img_size = VJEPA2_HUB_MODELS[model_name]
        if self.img_size != default_img_size:
            raise ValueError(
                f"JEPA model {model_name} requires img_size={default_img_size}, "
                f"got {self.img_size}."
            )

        checkpoint_path = _resolve_checkpoint_path(checkpoint_path)
        self.encoder = _load_vjepa2_hub_encoder(model_name, pretrained=False)
        _load_vjepa2_encoder_from_checkpoint(
            self.encoder, checkpoint_path, checkpoint_key=checkpoint_key
        )

        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.feat_dim = int(getattr(self.encoder, "embed_dim", default_feat_dim))
        self.patch_size = int(getattr(self.encoder, "patch_size", 16))
        self.grid_size = self.img_size // self.patch_size
        self.num_spatial_tokens = self.grid_size * self.grid_size

        self.register_buffer(
            "mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1, 1), persistent=False
        )
        self.register_buffer(
            "std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1, 1), persistent=False
        )

    def _resize_video(self, x: torch.Tensor) -> torch.Tensor:
        bsz, channels, timesteps, height, width = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(bsz * timesteps, channels, height, width)
        x = F.interpolate(
            x,
            size=(self.img_size, self.img_size),
            mode="bilinear",
            align_corners=False,
        )
        return x.view(bsz, timesteps, channels, self.img_size, self.img_size).permute(
            0, 2, 1, 3, 4
        )

    @staticmethod
    def _resample_tokens(tokens: torch.Tensor, target_tokens: int) -> torch.Tensor:
        batch_size, num_tokens, feat_dim = tokens.shape
        side = int(num_tokens**0.5)
        target_side = int(target_tokens**0.5)
        if side * side != num_tokens or target_side * target_side != target_tokens:
            raise ValueError(
                f"Token grids must be square, got {num_tokens} and {target_tokens}."
            )

        token_map = tokens.transpose(1, 2).reshape(batch_size, feat_dim, side, side)
        token_map = F.interpolate(
            token_map,
            size=(target_side, target_side),
            mode="bilinear",
            align_corners=False,
        )
        return token_map.flatten(2).transpose(1, 2)

    @torch.no_grad()
    def extract_tokens(self, x: torch.Tensor) -> torch.Tensor:
        x = self._resize_video(x.float())
        # process_data supplies policy images in [-1, 1], while V-JEPA 2.1 was
        # trained with ImageNet normalization applied to RGB values in [0, 1].
        x = (x + 1.0) * 0.5
        x = (x - self.mean) / self.std
        bsz, channels, timesteps, _, _ = x.shape

        # Encode every frame as a two-frame clip, matching the model tubelet size.
        frames = x.permute(0, 2, 1, 3, 4).reshape(
            bsz * timesteps, channels, self.img_size, self.img_size
        )
        clips = torch.stack([frames, frames], dim=2)
        tokens = self.encoder(clips)
        if tokens.shape[1] != self.num_spatial_tokens:
            tokens = self._resample_tokens(tokens, self.num_spatial_tokens)

        return tokens.reshape(bsz, timesteps, tokens.shape[1], tokens.shape[2])
