import os
import socket
import time
from typing import Dict, Optional, Tuple, Union

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


def _find_cached_hub_repo(hub_dir: str, repo: str) -> Optional[str]:
    owner, name = repo.split("/", 1)
    prefix = f"{owner}_{name}_"
    candidates = [
        os.path.join(hub_dir, prefix + ref)
        for ref in ("main", "master")
    ]
    candidates.extend(
        os.path.join(hub_dir, entry)
        for entry in sorted(os.listdir(hub_dir))
        if entry.startswith(prefix)
    )
    for candidate in candidates:
        if os.path.isfile(os.path.join(candidate, "hubconf.py")):
            return candidate
    return None


def _read_lock_metadata(lock_path: str) -> Dict[str, str]:
    try:
        with open(lock_path, "r", encoding="utf-8") as lock_file:
            lines = lock_file.read().splitlines()
    except (FileNotFoundError, OSError):
        return {}

    metadata = {}
    for line in lines:
        key, separator, value = line.partition("=")
        if separator:
            metadata[key.strip()] = value.strip()
    return metadata


def _lock_is_stale(lock_path: str, stale_after_s: int) -> bool:
    try:
        age_s = time.time() - os.path.getmtime(lock_path)
    except FileNotFoundError:
        return False

    metadata = _read_lock_metadata(lock_path)
    owner_host = metadata.get("host")
    owner_pid = metadata.get("pid")
    if owner_pid and (owner_host is None or owner_host == socket.gethostname()):
        try:
            os.kill(int(owner_pid), 0)
        except (ProcessLookupError, ValueError):
            return True
        except PermissionError:
            return False
        return False
    return age_s > stale_after_s


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
            with os.fdopen(fd, "w", encoding="utf-8") as lock_file:
                lock_file.write(
                    f"pid={os.getpid()}\nhost={socket.gethostname()}\n"
                )
            try:
                cached_repo = _find_cached_hub_repo(hub_dir, repo)
                if cached_repo is not None:
                    return torch.hub.load(
                        cached_repo,
                        model,
                        source="local",
                        pretrained=pretrained,
                    )
                return torch.hub.load(repo, model, pretrained=pretrained)
            finally:
                try:
                    os.remove(lock_path)
                except FileNotFoundError:
                    pass
        except FileExistsError:
            if _lock_is_stale(lock_path, stale_after_s=timeout_s):
                try:
                    os.remove(lock_path)
                except FileNotFoundError:
                    pass
                continue
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
    """Frozen V-JEPA 2.1 encoder with explicit temporal token structure."""

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
        encoder_tubelet_size = int(
            getattr(self.encoder, "tubelet_size", self.tubelet_size)
        )
        if encoder_tubelet_size != self.tubelet_size:
            raise RuntimeError(
                "Configured/loaded V-JEPA tubelet sizes disagree: "
                f"configured={self.tubelet_size}, encoder={encoder_tubelet_size}"
            )
        self.grid_size = self.img_size // self.patch_size
        self.num_spatial_tokens = self.grid_size * self.grid_size

        self.register_buffer(
            "mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1, 1), persistent=False
        )
        self.register_buffer(
            "std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1, 1), persistent=False
        )

    def train(self, mode: bool = True):
        super().train(False)
        self.encoder.eval()
        return self

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

    @torch.no_grad()
    def extract_temporal_tokens(
        self, x: torch.Tensor, return_metadata: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, object]]]:
        if x.ndim != 5:
            raise ValueError(f"Expected [B,C,T,H,W], got {tuple(x.shape)}")
        if x.shape[1] != 3:
            raise ValueError(f"V-JEPA expects 3 channels, got {x.shape[1]}")
        if x.shape[2] % self.tubelet_size != 0:
            raise ValueError(
                f"Frame count {x.shape[2]} must be divisible by tubelet_size="
                f"{self.tubelet_size}."
            )

        x = self._resize_video(x.float())
        x = (x + 1.0) * 0.5
        x = (x - self.mean) / self.std
        preprocessed_shape = tuple(x.shape)

        patch_projection = getattr(
            getattr(self.encoder, "patch_embed", None), "proj", None
        )
        if patch_projection is None:
            raise RuntimeError("V-JEPA encoder does not expose patch_embed.proj")

        patch_embed_shape = {}

        def _capture_patch_shape(_module, _inputs, output):
            patch_embed_shape["shape"] = tuple(output.shape)

        hook = patch_projection.register_forward_hook(_capture_patch_shape)
        try:
            raw_tokens = self.encoder(x)
        finally:
            hook.remove()

        if not torch.is_tensor(raw_tokens) or raw_tokens.ndim != 3:
            raise RuntimeError(
                "Unexpected V-JEPA encoder output: "
                f"{type(raw_tokens)} "
                f"{getattr(raw_tokens, 'shape', None)}"
            )
        if "shape" not in patch_embed_shape:
            raise RuntimeError("V-JEPA patch projection hook did not run")

        patch_shape = patch_embed_shape["shape"]
        if len(patch_shape) != 5:
            raise RuntimeError(
                f"Expected Conv3d patch output [B,D,T,H,W], got {patch_shape}"
            )
        batch_size, feature_dim, temporal_tokens, grid_h, grid_w = patch_shape
        expected_token_count = temporal_tokens * grid_h * grid_w
        if raw_tokens.shape != (
            batch_size,
            expected_token_count,
            feature_dim,
        ):
            raise RuntimeError(
                "V-JEPA token/patch shapes disagree: "
                f"patch_embed={patch_shape}, encoder={tuple(raw_tokens.shape)}"
            )

        # Conv3d returns [B,D,T,H,W], and PatchEmbed3D uses flatten(2), so the
        # encoder sequence is explicitly ordered as temporal, height, width.
        tokens = raw_tokens.reshape(
            batch_size,
            temporal_tokens,
            grid_h * grid_w,
            feature_dim,
        )
        metadata = {
            "input_shape": preprocessed_shape,
            "patch_embed_shape": patch_shape,
            "raw_output_shape": tuple(raw_tokens.shape),
            "temporal_tokens": temporal_tokens,
            "spatial_grid": (grid_h, grid_w),
            "reshaped_output_shape": tuple(tokens.shape),
            "flatten_order": "temporal,height,width",
        }
        if return_metadata:
            return tokens, metadata
        return tokens

    @torch.no_grad()
    def extract_tokens(
        self, x: torch.Tensor, return_metadata: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, object]]]:
        return self.extract_temporal_tokens(x, return_metadata=return_metadata)
