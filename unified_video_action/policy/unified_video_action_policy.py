import torch
import os
from typing import Dict, Tuple
import torch.nn.functional as F
import random
import numpy as np

from unified_video_action.model.common.normalizer import LinearNormalizer
from unified_video_action.policy.base_image_policy import BaseImagePolicy
from unified_video_action.common.pytorch_util import dict_apply

from unified_video_action.utils.data_utils import (
    process_data,
    extract_latent_autoregressive,
    get_trajectory,
    get_vae_latent,
    resize_image_eval,
)
from unified_video_action.utils.data_utils import (
    normalize_action,
    normalize_obs,
    normalize_past_action,
    unnormalize_future_action,
)
from unified_video_action.model.autoregressive import mar_con_unified as mar
from unified_video_action.vae.vaekl import AutoencoderKL
from unified_video_action.utils.language_model import (
    get_text_model,
    extract_text_features,
)
from unified_video_action.model.common.dinov2_teacher import DINOv2Teacher
from unified_video_action.model.common.jepa_teacher import JEPATeacher
from unified_video_action.model.common.student_tokenizer import StudentLatentTokenizer


class UnifiedVideoActionPolicy(BaseImagePolicy):
    def __init__(
        self,
        vae_model_params,
        autoregressive_model_params,
        action_model_params,
        shape_meta: dict,
        n_action_steps,
        shift_action=True,
        language_emb_model=None,
        task_name=None,
        task_modes=[],
        **kwargs
    ):
        super().__init__()

        self.task_name = task_name
        self.task_modes = task_modes
        self.autoregressive_model_params = autoregressive_model_params
        self.n_action_steps = n_action_steps
        self.shift_action = shift_action
        self.language_emb_model = language_emb_model
        self.action_dim = shape_meta.action.shape[0]

        self.kwargs = kwargs
        self.normalizer_type = kwargs["normalizer_type"]
        self.selected_training_mode = kwargs["selected_training_mode"]

        self.use_history_action = kwargs["use_history_action"]
        self.use_proprioception = kwargs["use_proprioception"]
        self.use_student_tokenizer = bool(kwargs.get("use_student_tokenizer", False))
        self.student_tokenizer_params = kwargs.get("student_tokenizer_params", None)
        self.align_params = kwargs.get("align_params", {})
        self.teacher_type = str(kwargs.get("teacher_type", "vae")).lower()
        self.dinov2_teacher_params = kwargs.get("dinov2_teacher_params", {})
        self.jepa_teacher_params = kwargs.get("jepa_teacher_params", {})
        self.freeze_mar = bool(kwargs.get("freeze_mar", False))
        self.keep_mar_pos_and_fake_trainable = bool(
            kwargs.get("keep_mar_pos_and_fake_trainable", False)
        )
        self.keep_mar_action_head_trainable = bool(
            kwargs.get("keep_mar_action_head_trainable", False)
        )
        self.mar_trainable_parameter_names = ()

        if (
            self.keep_mar_pos_and_fake_trainable
            or self.keep_mar_action_head_trainable
        ) and not self.freeze_mar:
            raise ValueError(
                "MAR trainable whitelists require freeze_mar=True."
            )
        if self.keep_mar_action_head_trainable and not bool(
            action_model_params.get("predict_action", False)
        ):
            raise ValueError(
                "keep_mar_action_head_trainable=True requires predict_action=True."
            )

        # Alignment defaults are intentionally conservative for stable joint training.
        self.use_alignment = bool(self.align_params.get("enable", False))
        self.align_coeff = float(self.align_params.get("coeff", 0.0))
        self.align_loss_type = str(self.align_params.get("loss_type", "cosine")).lower()
        self.align_teacher_mode = str(
            self.align_params.get("teacher_mode", "sample")
        ).lower()
        self.align_use_projector = bool(self.align_params.get("use_projector", True))
        self.align_projector_dim = int(self.align_params.get("projector_dim", 512))
        self.align_mse_coeff = float(self.align_params.get("mse_coeff", 0.25))
        self.align_stats_coeff = float(self.align_params.get("stats_coeff", 0.1))
        default_align_on = "latent" if self.teacher_type == "jepa" else "token_feat"
        self.align_on = str(
            self.align_params.get("align_on", default_align_on)
        ).lower()
        self._last_align_metrics = {}

        if self.teacher_type not in ("vae", "jepa", "dinov2"):
            raise ValueError(
                f"Unsupported teacher_type={self.teacher_type!r}. "
                "Expected 'vae', 'jepa', or 'dinov2'."
            )
        if self.teacher_type in ("jepa", "dinov2"):
            if not self.use_student_tokenizer:
                raise ValueError(
                    f"teacher_type={self.teacher_type!r} requires "
                    "use_student_tokenizer=True."
                )
            if self.align_on not in ("latent", "token_feat"):
                raise ValueError(
                    f"{self.teacher_type} alignment requires "
                    "align_on='latent' or 'token_feat'."
                )
        if self.teacher_type == "dinov2" and self.align_on != "token_feat":
            raise ValueError(
                "This DINOv2 experiment requires align_on='token_feat'."
            )

        # Student alignment consumes RGB directly, but video generation and FVD
        # still encode/decode through the VAE even for JEPA/DINO teachers.
        self.vae_model = None
        if self._requires_vae_model(
            use_student_tokenizer=self.use_student_tokenizer,
            teacher_type=self.teacher_type,
            predict_video=bool(autoregressive_model_params.predict_video),
        ):
            with torch.no_grad():
                self.vae_model = AutoencoderKL(**vae_model_params)
            self.vae_model.eval()
            for param in self.vae_model.parameters():
                param.requires_grad = False

        # =========================== frozen alignment teacher ===========================
        self.dinov2_teacher = None
        self.jepa_teacher = None
        self.teacher_feat_dim = None
        self.teacher_latent_projector = None
        if self.teacher_type == "jepa":
            self.jepa_teacher = JEPATeacher(**self.jepa_teacher_params)
            self.teacher_feat_dim = self.jepa_teacher.feat_dim
        elif self.teacher_type == "dinov2":
            self.dinov2_teacher = DINOv2Teacher(**self.dinov2_teacher_params)
            self.teacher_feat_dim = self.dinov2_teacher.feat_dim

        # =========================== student tokenizer ===========================
        self.student_tokenizer = None
        self.align_projector = None
        if self.use_student_tokenizer:
            if self.student_tokenizer_params is None:
                raise ValueError(
                    "use_student_tokenizer=True but student_tokenizer_params is not provided."
                )
            self.student_tokenizer = StudentLatentTokenizer(**self.student_tokenizer_params)
            if self.use_alignment and self.teacher_type in ("jepa", "dinov2"):
                self.align_use_projector = self.align_on == "token_feat"
                if self.align_on == "latent":
                    latent_dim = int(
                        self.student_tokenizer_params.get(
                            "latent_channels", autoregressive_model_params.vae_embed_dim
                        )
                    )
                    self.teacher_latent_projector = torch.nn.Sequential(
                        torch.nn.Linear(
                            self.teacher_feat_dim, self.align_projector_dim
                        ),
                        torch.nn.SiLU(),
                        torch.nn.Linear(
                            self.align_projector_dim, self.align_projector_dim
                        ),
                        torch.nn.SiLU(),
                        torch.nn.Linear(self.align_projector_dim, latent_dim),
                    )
            if self.use_alignment and self.align_use_projector:
                hidden_dim = int(self.student_tokenizer_params.get("hidden_dim", 384))
                latent_dim = int(
                    self.student_tokenizer_params.get(
                        "latent_channels", autoregressive_model_params.vae_embed_dim
                    )
                )
                align_target_dim = latent_dim
                if self.teacher_type in ("jepa", "dinov2"):
                    align_target_dim = self.teacher_feat_dim
                self.align_projector = torch.nn.Sequential(
                    torch.nn.Linear(hidden_dim, self.align_projector_dim),
                    torch.nn.SiLU(),
                    torch.nn.Linear(self.align_projector_dim, self.align_projector_dim),
                    torch.nn.SiLU(),
                    torch.nn.Linear(self.align_projector_dim, align_target_dim),
                )

        ## =========================== load language model ===========================
        self.text_model, self.tokenizer, self.max_length = get_text_model(
            task_name, language_emb_model
        )
        if self.text_model is not None:
            self.text_model.eval()
            for param in self.text_model.parameters():
                param.requires_grad = False

        ## =========================== main model ===========================
        self.model = mar.__dict__[autoregressive_model_params.model_size](
            img_size=autoregressive_model_params.img_size,
            vae_stride=autoregressive_model_params.vae_stride,
            patch_size=autoregressive_model_params.patch_size,
            vae_embed_dim=autoregressive_model_params.vae_embed_dim,
            mask_ratio_min=autoregressive_model_params.mask_ratio_min,
            label_drop_prob=autoregressive_model_params.label_drop_prob,
            attn_dropout=autoregressive_model_params.attn_dropout,
            proj_dropout=autoregressive_model_params.proj_dropout,
            diffloss_d=autoregressive_model_params.diffloss_d,
            diffloss_w=autoregressive_model_params.diffloss_w,
            diffloss_act_d=autoregressive_model_params.diffloss_act_d,
            diffloss_act_w=autoregressive_model_params.diffloss_act_w,
            num_sampling_steps=autoregressive_model_params.num_sampling_steps,
            diffusion_batch_mul=autoregressive_model_params.diffusion_batch_mul,
            grad_checkpointing=autoregressive_model_params.grad_checkpointing,
            predict_video=autoregressive_model_params.predict_video,
            act_diff_training_steps=self.autoregressive_model_params.act_diff_training_steps,
            act_diff_testing_steps=self.autoregressive_model_params.act_diff_testing_steps,
            action_model_params=action_model_params,
            use_history_action=kwargs["use_history_action"],
            action_mask_ratio=kwargs["action_mask_ratio"],
            use_proprioception=kwargs["use_proprioception"],
            predict_wrist_img=kwargs["predict_wrist_img"],
            different_history_freq=kwargs["different_history_freq"],
            predict_proprioception=kwargs["predict_proprioception"],
            task_name=self.task_name,
            language_emb_model=language_emb_model,
            shape_meta=shape_meta,
        )

        ## =========================== load pretrained model ===========================
        self.pretrained_model_path = autoregressive_model_params.pretrained_model_path
        if self.pretrained_model_path is not None:
            if os.path.exists(self.pretrained_model_path):
                self.load_pretrained_model()
            else:
                print('pretrained model not found: ', self.pretrained_model_path)

        self._configure_mar_trainability()
        
        self.normalizer = LinearNormalizer()

        if self.selected_training_mode is None:
            if len(self.task_modes) == 0:
                self.task_modes = [
                    "video_model",
                    "dynamic_model",
                    "policy_model",
                    "inverse_model",
                    "full_dynamic_model",
                ]
        else:
            if self.selected_training_mode == "policy_model_full_dynamics_model":
                self.task_modes = ["policy_model", "full_dynamic_model"]
            else:
                self.task_modes = [self.selected_training_mode]
        print("----------------------------------------------------------------------")
        print("task_modes", self.task_modes)
        print("----------------------------------------------------------------------")

    @staticmethod
    def _is_mar_pos_or_fake_parameter(name: str) -> bool:
        leaf_name = name.rsplit(".", 1)[-1]
        return (
            leaf_name.startswith("fake_")
            or leaf_name.endswith("_pos_embed")
            or leaf_name in {
                "diffusion_temporal_embed",
                "diffusion_spatial_embed",
                "mask_token",
                "blank_token",
            }
        )

    @staticmethod
    def _requires_vae_model(
        *, use_student_tokenizer: bool, teacher_type: str, predict_video: bool
    ) -> bool:
        return (
            not use_student_tokenizer
            or teacher_type == "vae"
            or predict_video
        )

    @staticmethod
    def _is_mar_action_head_parameter(name: str) -> bool:
        return name == "diffactloss" or name.startswith("diffactloss.")

    def _configure_mar_trainability(self) -> None:
        if not self.freeze_mar:
            return

        # Keep the MAR graph differentiable with respect to student latents; only
        # parameter gradients are disabled for the frozen modules.
        self.model.requires_grad_(False)
        pos_and_fake_parameter_names = []
        action_head_parameter_names = []
        for name, param in self.model.named_parameters():
            if (
                self.keep_mar_pos_and_fake_trainable
                and self._is_mar_pos_or_fake_parameter(name)
            ):
                param.requires_grad = True
                pos_and_fake_parameter_names.append(name)
            if (
                self.keep_mar_action_head_trainable
                and self._is_mar_action_head_parameter(name)
            ):
                param.requires_grad = True
                action_head_parameter_names.append(name)

        self.mar_trainable_parameter_names = tuple(
            name for name, param in self.model.named_parameters() if param.requires_grad
        )
        if self.keep_mar_pos_and_fake_trainable and not pos_and_fake_parameter_names:
            raise RuntimeError("No trainable MAR position or fake parameters were found.")
        if self.keep_mar_action_head_trainable and not action_head_parameter_names:
            raise RuntimeError("No trainable MAR action-head parameters were found.")

    def load_pretrained_model(self):
        print("----------------------------------------------------------------------")
        print("Loading pretrained model: ", self.pretrained_model_path)
        print("----------------------------------------------------------------------")

        pretrained_diffusion_model_ckpt = torch.load(
            self.pretrained_model_path, map_location="cpu", weights_only=False
        )

        if "state_dicts" in pretrained_diffusion_model_ckpt:
            if "ema_model" in pretrained_diffusion_model_ckpt["state_dicts"]:
                print("load from previous ema model")
                ## load from previous checkpoint
                pretrained_diffusion_model_ckpt_ = {
                    k[6:]: v
                    for k, v in pretrained_diffusion_model_ckpt["state_dicts"][
                        "ema_model"
                    ].items()
                    if k.startswith("model.")
                }  # remove 'model.'

                model_state_dict = self.model.state_dict()
                pretrained_state_dict = {
                    k: v
                    for k, v in pretrained_diffusion_model_ckpt_.items()
                    if k in model_state_dict and model_state_dict[k].size() == v.size()
                }
                
                pretrained_state_dict_mismatch = {
                    k: v
                    for k, v in model_state_dict.items()
                    if k not in pretrained_diffusion_model_ckpt_
                    or pretrained_diffusion_model_ckpt_[k].size() != v.size()
                }
                
                print("----------------------------------------------------------------------")
                print(
                    "pretrained_state_dict_mismatch: ",
                    pretrained_state_dict_mismatch.keys(),
                )
                print("----------------------------------------------------------------------")
                
                assert len(model_state_dict) > 0
                assert len(pretrained_state_dict) > 0
                model_state_dict.update(pretrained_state_dict)

                missing_keys, unexpected_keys = self.model.load_state_dict(
                    model_state_dict, strict=False
                )
            else:
                raise NotImplementedError

        elif "model_ema" in pretrained_diffusion_model_ckpt:
            ## load from MAR pretrained mdoel
            pretrained_diffusion_model_ckpt_ = pretrained_diffusion_model_ckpt[
                "model_ema"
            ]

            model_state_dict = self.model.state_dict()
            pretrained_state_dict = {
                k: v
                for k, v in pretrained_diffusion_model_ckpt_.items()
                if k in model_state_dict and model_state_dict[k].size() == v.size()
            }
            assert len(model_state_dict) > 0
            assert len(pretrained_state_dict) > 0
            model_state_dict.update(pretrained_state_dict)

            missing_keys, unexpected_keys = self.model.load_state_dict(
                model_state_dict, strict=False
            )

        else:
            raise NotImplementedError

        print("---------------------------------------------------------------")
        print("Model Missing keys:", missing_keys)
        print("Model Unexpected keys:", unexpected_keys)
        print("---------------------------------------------------------------")


    def predict_action(
        self, obs_dict: Dict[str, torch.Tensor], language_goal=None
    ) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        
        obs_dict = resize_image_eval(self.task_name, obs_dict)
        B, T, C, H, W = obs_dict["image"].shape

        ## language goal
        text_latents = None
        if self.language_emb_model is not None:
            if "umi" in self.task_name:
                text_latents = language_goal
            else:
                print("predict_action language_goal: ", language_goal)
                print(self.task_name, "max_length", self.max_length)

                if self.language_emb_model == "clip":
                    text_tokens = self.tokenizer(
                        language_goal,
                        padding="max_length",
                        max_length=self.max_length,
                        return_tensors="pt",
                    ).to(self.device)
                    text_latents = extract_text_features(
                        self.text_model,
                        text_tokens,
                        language_emb_model=self.language_emb_model,
                    )
                else:
                    text_latents = None

        ## history action
        history_nactions = None
        if self.use_history_action:
            if "past_action" in obs_dict:
                history_nactions = normalize_past_action(
                    normalizer=self.normalizer,
                    normalizer_type=self.normalizer_type,
                    actions=obs_dict["past_action"],
                )
                del obs_dict["past_action"]

        ## normalize observations
        batch = normalize_obs(
            normalizer=self.normalizer,
            normalizer_type=self.normalizer_type,
            batch={"obs": obs_dict},
        )
        obs_dict = batch["obs"]

        c, proprioception_input, _ = process_data(
            {"obs": obs_dict}, task_name=self.task_name, eval=True, **self.kwargs
        )

        if self.use_student_tokenizer and self.student_tokenizer is not None:
            if self.use_proprioception and proprioception_input is not None:
                if "second_image" in proprioception_input:
                    second_image_z, _ = self._encode_student_latent(
                        proprioception_input["second_image"]
                    )
                    proprioception_input["second_image_z"] = second_image_z
            c, _ = self._encode_student_latent(c.detach())
        else:
            if self.use_proprioception and proprioception_input is not None:
                if "second_image" in proprioception_input:
                    second_image_z, _ = extract_latent_autoregressive(
                        self.vae_model, proprioception_input["second_image"]
                    )
                    proprioception_input["second_image_z"] = second_image_z
            c, _ = extract_latent_autoregressive(self.vae_model, c.detach())

        z, act_out = self.model.sample_tokens(
            bsz=B,
            cond=c,
            text_latents=text_latents,
            num_iter=self.autoregressive_model_params.num_iter,
            cfg=self.autoregressive_model_params.cfg,
            cfg_schedule=self.autoregressive_model_params.cfg_schedule,
            temperature=self.autoregressive_model_params.temperature,
            history_nactions=history_nactions,
            proprioception_input=proprioception_input,
            task_mode="policy_model",
            vae_model=self.vae_model,
        )

        # unnormalize prediction
        Da = self.action_dim

        naction_pred = act_out[..., :Da]

        ## unnormalize action
        action_pred = unnormalize_future_action(
            normalizer=self.normalizer,
            normalizer_type=self.normalizer_type,
            actions=naction_pred,
        )

        action = action_pred[:, : self.n_action_steps]

        result = {
            "action": action,
            "action_pred": action_pred,
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def add_weight_decay(self, model, weight_decay=1e-5, skip_list=()):
        decay = []
        no_decay = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
                no_decay.append(param)  # no weight decay on bias, norm and diffloss
            else:
                decay.append(param)

        return [
            {"params": no_decay, "weight_decay": 0.0},
            {"params": decay, "weight_decay": weight_decay},
        ]

    def get_optimizer(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: Tuple[float, float],
    ) -> torch.optim.Optimizer:

        optim_groups = self.add_weight_decay(self.model, weight_decay=weight_decay)
        if self.use_student_tokenizer and self.student_tokenizer is not None:
            optim_groups.extend(
                self.add_weight_decay(self.student_tokenizer, weight_decay=weight_decay)
            )
        if self.align_projector is not None:
            optim_groups.extend(
                self.add_weight_decay(self.align_projector, weight_decay=weight_decay)
            )
        if self.teacher_latent_projector is not None:
            optim_groups.extend(
                self.add_weight_decay(
                    self.teacher_latent_projector, weight_decay=weight_decay
                )
            )
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)

        # Manually set 'initial_lr' for each parameter group (assuming a base learning rate)
        for param_group in optimizer.param_groups:
            if "initial_lr" not in param_group:
                param_group["initial_lr"] = param_group[
                    "lr"
                ]  # or set a specific initial learning rate

        return optimizer

    def _extract_teacher_latent(self, x: torch.Tensor) -> torch.Tensor:
        """
        Teacher path (frozen VAE): x [B, C, T, H, W] -> z [B, T, C_lat, H_lat, W_lat]
        """
        x = x.float()
        bsz, channels, timesteps, height, width = x.size()
        with torch.no_grad():
            x_flat = x.permute(0, 2, 1, 3, 4).reshape(
                bsz * timesteps, channels, height, width
            )
            posterior = self.vae_model.encode(x_flat)
            if self.align_teacher_mode == "mode" and hasattr(posterior, "mode"):
                z = posterior.mode()
            elif self.align_teacher_mode == "mix" and hasattr(posterior, "mode"):
                z = 0.5 * (posterior.mode() + posterior.sample())
            else:
                z = posterior.sample()
            z = z.mul_(0.2325)
            z = z.reshape(bsz, timesteps, z.shape[1], z.shape[2], z.shape[3])
        return z

    def _latent_to_tokens(self, z: torch.Tensor) -> torch.Tensor:
        # z [B, T, C, H, W] -> [B, T, S, C]
        bsz, timesteps, channels, height, width = z.shape
        return z.permute(0, 1, 3, 4, 2).reshape(bsz, timesteps, height * width, channels)

    @staticmethod
    def _resample_teacher_tokens(
        student_tokens: torch.Tensor, teacher_tokens: torch.Tensor
    ) -> torch.Tensor:
        if student_tokens.shape[1:3] == teacher_tokens.shape[1:3]:
            return teacher_tokens

        bsz, timesteps, student_token_count, _ = student_tokens.shape
        if bsz != teacher_tokens.shape[0] or timesteps != teacher_tokens.shape[1]:
            raise ValueError(
                "Teacher/student batch or time mismatch: "
                f"student={student_tokens.shape}, teacher={teacher_tokens.shape}"
            )

        teacher_token_count = teacher_tokens.shape[2]
        feat_dim = teacher_tokens.shape[3]
        student_side = int(student_token_count**0.5)
        teacher_side = int(teacher_token_count**0.5)
        if student_side * student_side != student_token_count:
            raise ValueError(
                f"Student token count {student_token_count} is not a square grid."
            )
        if teacher_side * teacher_side != teacher_token_count:
            raise ValueError(
                f"Teacher token count {teacher_token_count} is not a square grid."
            )

        teacher_map = teacher_tokens.reshape(
            bsz, timesteps, teacher_side, teacher_side, feat_dim
        )
        teacher_map = teacher_map.permute(0, 1, 4, 2, 3).reshape(
            bsz * timesteps, feat_dim, teacher_side, teacher_side
        )
        teacher_map = F.interpolate(
            teacher_map,
            size=(student_side, student_side),
            mode="bilinear",
            align_corners=False,
        )
        return teacher_map.reshape(
            bsz, timesteps, feat_dim, student_side, student_side
        ).permute(0, 1, 3, 4, 2).reshape(
            bsz, timesteps, student_token_count, feat_dim
        )

    def _encode_student_latent(self, x: torch.Tensor):
        latent, token_feat = self.student_tokenizer(x)
        return latent, token_feat

    def _compute_alignment_loss(
        self, student_tokens: torch.Tensor, teacher_tokens: torch.Tensor
    ):
        student_tokens_raw = student_tokens
        if self.align_projector is not None:
            student_tokens = self.align_projector(student_tokens)

        assert student_tokens.shape == teacher_tokens.shape, (
            student_tokens.shape,
            teacher_tokens.shape,
        )

        student_tokens = student_tokens.float()
        teacher_tokens = teacher_tokens.float()

        eps = 1e-6
        student_norm = F.normalize(student_tokens, dim=-1, eps=eps)
        teacher_norm = F.normalize(teacher_tokens, dim=-1, eps=eps)
        cosine = (student_norm * teacher_norm).sum(dim=-1).mean()
        cosine_loss = 1.0 - cosine
        mse_loss = F.mse_loss(student_tokens, teacher_tokens)

        student_mu = student_tokens.mean(dim=(0, 1, 2))
        teacher_mu = teacher_tokens.mean(dim=(0, 1, 2))
        student_std = student_tokens.std(dim=(0, 1, 2)).clamp_min(eps)
        teacher_std = teacher_tokens.std(dim=(0, 1, 2)).clamp_min(eps)
        mean_loss = F.mse_loss(student_mu, teacher_mu)
        std_loss = F.mse_loss(student_std, teacher_std)
        stats_loss = mean_loss + std_loss

        if self.align_loss_type == "mse":
            base_loss = mse_loss
        elif self.align_loss_type == "hybrid":
            base_loss = 0.5 * (cosine_loss + mse_loss)
        else:
            base_loss = cosine_loss

        total_align_loss = (
            base_loss + self.align_mse_coeff * mse_loss + self.align_stats_coeff * stats_loss
        )
        metrics = {
            "align_cos": cosine.detach(),
            "align_mse": mse_loss.detach(),
            "align_stats": stats_loss.detach(),
            "student_norm": student_tokens_raw.detach().norm(dim=-1).mean(),
            "teacher_norm": teacher_tokens.detach().norm(dim=-1).mean(),
        }
        return total_align_loss, metrics

    def compute_loss(self, batch, **kwargs):
        B, T, C, H, W = batch["obs"]["image"].size()

        text_latents = None
        if self.language_emb_model == "clip":
            if "language" in batch["obs"]:
                language_goal = batch["obs"]["language"]
                del batch["obs"]["language"]
                text_tokens = {
                    "input_ids": language_goal[:, 0].long()[:, 0],
                    "attention_mask": language_goal[:, 0].long()[:, 1],
                }
                text_latents = extract_text_features(
                    self.text_model,
                    text_tokens,
                    language_emb_model=self.language_emb_model,
                )
            elif "language_latents" in batch:
                text_latents = batch["language_latents"]
            else:
                raise NotImplementedError

        nactions = normalize_action(
            normalizer=self.normalizer,
            normalizer_type=self.normalizer_type,
            actions=batch["action"],
        )
        batch = normalize_obs(
            normalizer=self.normalizer,
            normalizer_type=self.normalizer_type,
            batch=batch,
        )

        if self.use_history_action:
            batch = dict_apply(batch, lambda x: x[:, 1:])

        x, proprioception_input, _ = process_data(
            batch, task_name=self.task_name, **self.kwargs
        )
        align_loss = torch.tensor(0.0, device=x.device)
        if self.use_student_tokenizer and self.student_tokenizer is not None:
            c_img, x_img = torch.chunk(x, 2, dim=2)
            z, z_token_feat = self._encode_student_latent(x_img)
            c, c_token_feat = self._encode_student_latent(c_img)

            if proprioception_input is not None:
                if "second_image" in proprioception_input:
                    second_image_z, _ = self._encode_student_latent(
                        proprioception_input["second_image"]
                    )
                    proprioception_input["second_image_z"] = second_image_z
                if "pred_second_image" in proprioception_input:
                    pred_second_image_z, _ = self._encode_student_latent(
                        proprioception_input["pred_second_image"]
                    )
                    proprioception_input["pred_second_image_z"] = pred_second_image_z

            if self.use_alignment:
                if self.teacher_type in ("jepa", "dinov2"):
                    teacher = (
                        self.jepa_teacher
                        if self.teacher_type == "jepa"
                        else self.dinov2_teacher
                    )
                    teacher_z_tokens = teacher.extract_tokens(x_img)
                    teacher_c_tokens = teacher.extract_tokens(c_img)
                    if self.align_on == "latent":
                        student_z_tokens = self._latent_to_tokens(z)
                        student_c_tokens = self._latent_to_tokens(c)
                        teacher_z_tokens = self.teacher_latent_projector(
                            teacher_z_tokens
                        )
                        teacher_c_tokens = self.teacher_latent_projector(
                            teacher_c_tokens
                        )
                    else:
                        student_z_tokens = z_token_feat
                        student_c_tokens = c_token_feat
                    teacher_z_tokens = self._resample_teacher_tokens(
                        student_z_tokens, teacher_z_tokens
                    )
                    teacher_c_tokens = self._resample_teacher_tokens(
                        student_c_tokens, teacher_c_tokens
                    )
                else:
                    student_z_tokens = z_token_feat
                    student_c_tokens = c_token_feat
                    teacher_z = self._extract_teacher_latent(x_img)
                    teacher_c = self._extract_teacher_latent(c_img)
                    teacher_z_tokens = self._latent_to_tokens(teacher_z)
                    teacher_c_tokens = self._latent_to_tokens(teacher_c)
                align_z, metrics_z = self._compute_alignment_loss(
                    student_z_tokens, teacher_z_tokens
                )
                align_c, metrics_c = self._compute_alignment_loss(
                    student_c_tokens, teacher_c_tokens
                )
                align_loss = 0.5 * (align_z + align_c)
                self._last_align_metrics = {
                    "align_loss": align_loss.detach(),
                    "align_cos": 0.5 * (metrics_z["align_cos"] + metrics_c["align_cos"]),
                    "align_mse": 0.5 * (metrics_z["align_mse"] + metrics_c["align_mse"]),
                    "align_stats": 0.5
                    * (metrics_z["align_stats"] + metrics_c["align_stats"]),
                    "student_norm": 0.5
                    * (metrics_z["student_norm"] + metrics_c["student_norm"]),
                    "teacher_norm": 0.5
                    * (metrics_z["teacher_norm"] + metrics_c["teacher_norm"]),
                }
        else:
            x, z, c, _, proprioception_input = get_vae_latent(
                x, self.vae_model, eval=False, proprioception_input=proprioception_input
            )
        history_trajectory, trajectory = get_trajectory(
            nactions, T, self.shift_action, use_history_action=self.use_history_action
        )

        selected_mode = random.choice(self.task_modes)

        loss, video_loss, act_loss = self.model(
            z,
            c,
            history_trajectory,
            trajectory,
            text_latents,
            task_mode=selected_mode,
            proprioception_input=proprioception_input,
        )
        if self.use_student_tokenizer and self.use_alignment:
            loss = loss + self.align_coeff * align_loss

        # DDP safety: always attach every trainable parameter to graph.
        # Checking `param.grad is None` inside forward is unstable across iterations.
        def _ddp_unused_term(param: torch.Tensor) -> torch.Tensor:
            return torch.nan_to_num(param, nan=0.0, posinf=0.0, neginf=0.0).sum().mul(0.0)

        for param in self.model.parameters():
            if param.requires_grad:
                loss = loss + _ddp_unused_term(param)

        if self.student_tokenizer is not None:
            for param in self.student_tokenizer.parameters():
                if param.requires_grad:
                    loss = loss + _ddp_unused_term(param)

        if self.align_projector is not None:
            for param in self.align_projector.parameters():
                if param.requires_grad:
                    loss = loss + _ddp_unused_term(param)

        if self.teacher_latent_projector is not None:
            for param in self.teacher_latent_projector.parameters():
                if param.requires_grad:
                    loss = loss + _ddp_unused_term(param)

        return loss, (video_loss, act_loss)

    def forward(self, batch, **kwargs):
        return self.compute_loss(batch, **kwargs)
