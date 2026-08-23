import torch
import torch.nn as nn
import torch.nn.functional as F


class StudentLatentTokenizer(nn.Module):
    """
    Lightweight student encoder that maps RGB frames to MAR-compatible latents.

    Input:
        x: [B, C, T, H, W]
    Output:
        latent: [B, T, C_latent, H_patch, W_patch]
        token_feat: [B, T, S, D]
    """

    def __init__(
        self,
        img_size=256,
        patch_size=16,
        in_channels=3,
        latent_channels=16,
        hidden_dim=384,
        depth=6,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.0,
        use_temporal_mixer=True,
        temporal_kernel_size=3,
        early_patchify=False,
    ):
        super().__init__()
        assert img_size % patch_size == 0, "image size must be divisible by patch size"
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.hidden_dim = hidden_dim
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.use_temporal_mixer = use_temporal_mixer
        self.early_patchify = bool(early_patchify)

        if self.early_patchify:
            # Avoid retaining full-resolution feature maps during backward.
            self.stem = nn.Sequential()
            patch_in_channels = in_channels
        else:
            self.stem = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    hidden_dim // 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.GELU(),
                nn.Conv2d(
                    hidden_dim // 2,
                    hidden_dim // 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.GELU(),
            )
            patch_in_channels = hidden_dim // 2
        self.patch_embed = nn.Conv2d(
            patch_in_channels,
            hidden_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
            bias=True,
        )

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_dim))

        blocks = []
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        for _ in range(depth):
            blocks.append(
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=mlp_hidden_dim,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
            )
        self.blocks = nn.ModuleList(blocks)

        if self.use_temporal_mixer:
            padding = temporal_kernel_size // 2
            # Token-wise temporal depthwise convolution.
            self.temporal_mixer = nn.Conv1d(
                hidden_dim,
                hidden_dim,
                kernel_size=temporal_kernel_size,
                padding=padding,
                groups=hidden_dim,
                bias=True,
            )

        self.norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, latent_channels)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.pos_embed, std=0.02)
        for module in self.stem:
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

        nn.init.xavier_uniform_(self.patch_embed.weight)
        if self.patch_embed.bias is not None:
            nn.init.constant_(self.patch_embed.bias, 0.0)

        if self.use_temporal_mixer:
            nn.init.dirac_(self.temporal_mixer.weight)
            if self.temporal_mixer.bias is not None:
                nn.init.constant_(self.temporal_mixer.bias, 0.0)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

    def _mix_temporal(self, token_feat: torch.Tensor) -> torch.Tensor:
        # token_feat: [B, T, S, D]
        bsz, timesteps, num_tokens, channels = token_feat.shape
        x = token_feat.permute(0, 2, 3, 1).reshape(bsz * num_tokens, channels, timesteps)
        x = self.temporal_mixer(x)
        x = x.reshape(bsz, num_tokens, channels, timesteps).permute(0, 3, 1, 2)
        return x

    def forward(self, x):
        x = x.float()
        bsz, channels, timesteps, height, width = x.shape
        assert channels == self.in_channels, (
            f"Expected {self.in_channels} channels, but got {channels}"
        )
        assert height == self.img_size and width == self.img_size, (
            f"Expected image size {self.img_size}, but got {height}x{width}"
        )

        x = x.permute(0, 2, 1, 3, 4).reshape(bsz * timesteps, channels, height, width)
        x = self.stem(x)
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed

        for block in self.blocks:
            x = block(x)
        token_feat = self.norm(x)
        token_feat = token_feat.reshape(bsz, timesteps, self.num_patches, self.hidden_dim)

        if self.use_temporal_mixer and timesteps > 1:
            token_feat = token_feat + self._mix_temporal(token_feat)

        token_latent = self.out_proj(token_feat)
        latent = token_latent.permute(0, 1, 3, 2).reshape(
            bsz, timesteps, self.latent_channels, self.grid_size, self.grid_size
        )

        return latent, token_feat
