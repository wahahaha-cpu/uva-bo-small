import torch
import os
from contextlib import contextmanager
from typing import Dict, Optional, Tuple
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
from unified_video_action.model.common.temporal_fusion import TemporalFusionMLP


@contextmanager
def _isolated_rng(seed=None, preserve_cuda: bool = False):
    """Run initialization/teacher work without perturbing shared RNG state."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = None
    # torch.manual_seed also seeds every CUDA device.  Save those states for
    # seeded module initialization as well as for teacher/auxiliary forward.
    if torch.cuda.is_available() and (seed is not None or preserve_cuda):
        cuda_states = torch.cuda.get_rng_state_all()
    try:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


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

        self.dual_teacher_params = kwargs.get("dual_teacher_params", {})
        self.use_dual_teacher_alignment = bool(
            self.dual_teacher_params.get("enable", False)
        )
        self.enable_dino_spatial = self.use_dual_teacher_alignment and bool(
            self.dual_teacher_params.get("enable_dino", True)
        )
        self.enable_jepa_dynamics = self.use_dual_teacher_alignment and bool(
            self.dual_teacher_params.get("enable_jepa", True)
        )
        self.lambda_dino = float(
            self.dual_teacher_params.get("lambda_dino", 0.02)
        )
        self.lambda_jepa = float(
            self.dual_teacher_params.get("lambda_jepa", 0.05)
        )
        self.dual_projector_dim = int(
            self.dual_teacher_params.get("projector_dim", 512)
        )
        self.dual_init_seed = int(
            self.dual_teacher_params.get("init_seed", 4200)
        )
        self.dual_required_frames = int(
            self.dual_teacher_params.get("required_frames", 4)
        )
        self.dino_loss_type = str(
            self.dual_teacher_params.get("dino_loss_type", "hybrid")
        ).lower()
        self.dino_mse_coeff = float(
            self.dual_teacher_params.get("dino_mse_coeff", 0.25)
        )
        self.dino_stats_coeff = float(
            self.dual_teacher_params.get("dino_stats_coeff", 0.1)
        )
        self.jepa_relation_temperature = float(
            self.dual_teacher_params.get("jepa_relation_temperature", 0.1)
        )
        self.log_dual_teacher_shapes = bool(
            self.dual_teacher_params.get("log_shapes_once", True)
        )
        self._dual_teacher_shapes_logged = False

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
        if self.use_dual_teacher_alignment:
            if not self.use_student_tokenizer:
                raise ValueError(
                    "dual_teacher_params.enable=True requires "
                    "use_student_tokenizer=True."
                )
            if not (self.enable_dino_spatial or self.enable_jepa_dynamics):
                raise ValueError(
                    "Dual-teacher alignment must enable DINO, JEPA, or both."
                )
            if self.dual_required_frames != 4:
                raise ValueError(
                    "The minimal DINO+JEPA experiment requires exactly 4 frames."
                )
            if self.lambda_dino < 0.0 or self.lambda_jepa < 0.0:
                raise ValueError("Teacher loss coefficients must be non-negative.")
            if self.enable_jepa_dynamics and self.jepa_relation_temperature <= 0.0:
                raise ValueError("jepa_relation_temperature must be positive.")
            if self.enable_jepa_dynamics and not bool(
                autoregressive_model_params.predict_video
            ):
                raise ValueError(
                    "JEPA future-latent supervision requires predict_video=True."
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
        if self.use_dual_teacher_alignment:
            with _isolated_rng():
                if self.enable_jepa_dynamics:
                    self.jepa_teacher = JEPATeacher(**self.jepa_teacher_params)
                if self.enable_dino_spatial:
                    self.dinov2_teacher = DINOv2Teacher(
                        **self.dinov2_teacher_params
                    )
        else:
            if self.teacher_type == "jepa":
                self.jepa_teacher = JEPATeacher(**self.jepa_teacher_params)
            if self.teacher_type == "dinov2":
                self.dinov2_teacher = DINOv2Teacher(**self.dinov2_teacher_params)
        if self.teacher_type == "jepa":
            self.teacher_feat_dim = self.jepa_teacher.feat_dim
        elif self.teacher_type == "dinov2":
            self.teacher_feat_dim = self.dinov2_teacher.feat_dim
        for teacher in (self.dinov2_teacher, self.jepa_teacher):
            if teacher is not None:
                teacher.eval()
                teacher.requires_grad_(False)

        # =========================== student tokenizer ===========================
        self.student_tokenizer = None
        self.align_projector = None
        self.student_to_dino_projector = None
        self.temporal_fusion = None
        self.student_to_jepa_projector = None
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

            hidden_dim = int(self.student_tokenizer_params.get("hidden_dim", 384))
            future_latent_channels = int(
                self.student_tokenizer_params.get(
                    "latent_channels", autoregressive_model_params.vae_embed_dim
                )
            )
            future_latent_dim = future_latent_channels * int(
                autoregressive_model_params.patch_size
            ) ** 2
            if self.enable_dino_spatial:
                with _isolated_rng(self.dual_init_seed + 1):
                    self.student_to_dino_projector = torch.nn.Sequential(
                        torch.nn.Linear(hidden_dim, self.dual_projector_dim),
                        torch.nn.SiLU(),
                        torch.nn.Linear(
                            self.dual_projector_dim, self.dual_projector_dim
                        ),
                        torch.nn.SiLU(),
                        torch.nn.Linear(
                            self.dual_projector_dim, self.dinov2_teacher.feat_dim
                        ),
                    )
            if self.enable_jepa_dynamics:
                with _isolated_rng(self.dual_init_seed + 2):
                    self.temporal_fusion = TemporalFusionMLP(future_latent_dim)
                with _isolated_rng(self.dual_init_seed + 3):
                    self.student_to_jepa_projector = torch.nn.Sequential(
                        torch.nn.LayerNorm(future_latent_dim),
                        torch.nn.Linear(
                            future_latent_dim, self.dual_projector_dim
                        ),
                        torch.nn.GELU(),
                        torch.nn.Linear(
                            self.dual_projector_dim, self.jepa_teacher.feat_dim
                        ),
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
            causal_future_mask=bool(
                getattr(autoregressive_model_params, "causal_future_mask", False)
            ),
            shared_video_diffusion_timestep=bool(
                getattr(
                    autoregressive_model_params,
                    "shared_video_diffusion_timestep",
                    False,
                )
            ),
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
        if self.enable_jepa_dynamics:
            video_training_modes = {
                "video_model",
                "dynamic_model",
                "full_dynamic_model",
            }
            invalid_modes = [
                mode for mode in self.task_modes if mode not in video_training_modes
            ]
            if invalid_modes:
                raise ValueError(
                    "JEPA future-latent supervision requires a video training "
                    f"mode, got {invalid_modes}."
                )
            if not self.model.causal_future_mask:
                raise ValueError(
                    "JEPA future-latent supervision requires "
                    "causal_future_mask=True."
                )
            if not self.model.shared_video_diffusion_timestep:
                raise ValueError(
                    "JEPA future-latent supervision requires one shared "
                    "diffusion timestep per video."
                )
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
        for module in (
            self.student_to_dino_projector,
            self.temporal_fusion,
            self.student_to_jepa_projector,
        ):
            if module is not None:
                optim_groups.extend(
                    self.add_weight_decay(module, weight_decay=weight_decay)
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
    def _resample_spatial_tokens(
        tokens: torch.Tensor, target_token_count: int
    ) -> torch.Tensor:
        """Interpolate only a token tensor's explicit H-W grid."""
        if tokens.ndim not in (3, 4):
            raise ValueError(
                "Spatial token resampling expects [B,N,D] or [B,T,N,D], got "
                f"{tuple(tokens.shape)}"
            )

        source_token_count = tokens.shape[-2]
        if source_token_count == target_token_count:
            return tokens

        source_side = int(source_token_count**0.5)
        target_side = int(target_token_count**0.5)
        if source_side * source_side != source_token_count:
            raise ValueError(
                f"Source token count {source_token_count} is not a square grid."
            )
        if target_side * target_side != target_token_count:
            raise ValueError(
                f"Target token count {target_token_count} is not a square grid."
            )

        feature_dim = tokens.shape[-1]
        if tokens.ndim == 4:
            batch_size, timesteps = tokens.shape[:2]
            token_map = tokens.reshape(
                batch_size * timesteps,
                source_side,
                source_side,
                feature_dim,
            ).permute(0, 3, 1, 2)
            token_map = F.interpolate(
                token_map,
                size=(target_side, target_side),
                mode="bilinear",
                align_corners=False,
            )
            return token_map.permute(0, 2, 3, 1).reshape(
                batch_size, timesteps, target_token_count, feature_dim
            )

        batch_size = tokens.shape[0]
        token_map = tokens.reshape(
            batch_size, source_side, source_side, feature_dim
        ).permute(0, 3, 1, 2)
        token_map = F.interpolate(
            token_map,
            size=(target_side, target_side),
            mode="bilinear",
            align_corners=False,
        )
        return token_map.permute(0, 2, 3, 1).reshape(
            batch_size, target_token_count, feature_dim
        )

    @classmethod
    def _resample_teacher_tokens(
        cls, student_tokens: torch.Tensor, teacher_tokens: torch.Tensor
    ) -> torch.Tensor:
        if student_tokens.ndim != 4 or teacher_tokens.ndim != 4:
            raise ValueError("Legacy teacher alignment expects [B,T,N,D] tokens.")
        if student_tokens.shape[:2] != teacher_tokens.shape[:2]:
            raise ValueError(
                "Teacher/student batch or time mismatch: "
                f"student={student_tokens.shape}, teacher={teacher_tokens.shape}"
            )
        return cls._resample_spatial_tokens(
            teacher_tokens, target_token_count=student_tokens.shape[2]
        )

    @staticmethod
    def _expand_jepa_regions_for_legacy_alignment(
        teacher_tokens: torch.Tensor,
        target_timesteps: int,
        tubelet_size: int,
    ) -> torch.Tensor:
        if teacher_tokens.shape[1] == target_timesteps:
            return teacher_tokens
        if teacher_tokens.shape[1] * tubelet_size != target_timesteps:
            raise ValueError(
                "Cannot map V-JEPA temporal regions to legacy frame alignment: "
                f"teacher={tuple(teacher_tokens.shape)}, "
                f"target_timesteps={target_timesteps}, tubelet_size={tubelet_size}"
            )
        return teacher_tokens.repeat_interleave(tubelet_size, dim=1)

    def _encode_student_latent(self, x: torch.Tensor):
        latent, token_feat = self.student_tokenizer(x)
        return latent, token_feat

    @staticmethod
    def _compute_feature_alignment_loss(
        student_tokens: torch.Tensor,
        teacher_tokens: torch.Tensor,
        loss_type: str,
        mse_coeff: float,
        stats_coeff: float,
    ):
        if student_tokens.shape != teacher_tokens.shape:
            raise ValueError(
                "Student/teacher token shapes must match, got "
                f"{tuple(student_tokens.shape)} and {tuple(teacher_tokens.shape)}"
            )
        student_tokens = student_tokens.float()
        teacher_tokens = teacher_tokens.float()

        eps = 1e-6
        student_norm = F.normalize(student_tokens, dim=-1, eps=eps)
        teacher_norm = F.normalize(teacher_tokens, dim=-1, eps=eps)
        cosine = (student_norm * teacher_norm).sum(dim=-1).mean()
        cosine_loss = 1.0 - cosine
        mse_loss = F.mse_loss(student_tokens, teacher_tokens)

        reduction_dims = tuple(range(student_tokens.ndim - 1))
        student_mu = student_tokens.mean(dim=reduction_dims)
        teacher_mu = teacher_tokens.mean(dim=reduction_dims)
        # Keep the legacy alignment statistic (torch.std's unbiased estimate)
        # unchanged while allowing the helper to serve both teacher branches.
        student_std = student_tokens.std(dim=reduction_dims).clamp_min(eps)
        teacher_std = teacher_tokens.std(dim=reduction_dims).clamp_min(eps)
        mean_loss = F.mse_loss(student_mu, teacher_mu)
        std_loss = F.mse_loss(student_std, teacher_std)
        stats_loss = mean_loss + std_loss

        if loss_type == "mse":
            base_loss = mse_loss
        elif loss_type == "hybrid":
            base_loss = 0.5 * (cosine_loss + mse_loss)
        else:
            base_loss = cosine_loss

        total_align_loss = base_loss + mse_coeff * mse_loss + stats_coeff * stats_loss
        metrics = {
            "cosine": cosine.detach(),
            "mse": mse_loss.detach(),
            "stats": stats_loss.detach(),
            "student_norm": student_tokens.detach().norm(dim=-1).mean(),
            "teacher_norm": teacher_tokens.detach().norm(dim=-1).mean(),
        }
        return total_align_loss, metrics

    def _compute_alignment_loss(
        self, student_tokens: torch.Tensor, teacher_tokens: torch.Tensor
    ):
        student_tokens_raw = student_tokens
        if self.align_projector is not None:
            student_tokens = self.align_projector(student_tokens)
        align_loss, metrics = self._compute_feature_alignment_loss(
            student_tokens,
            teacher_tokens,
            loss_type=self.align_loss_type,
            mse_coeff=self.align_mse_coeff,
            stats_coeff=self.align_stats_coeff,
        )
        metrics = {
            "align_cos": metrics["cosine"],
            "align_mse": metrics["mse"],
            "align_stats": metrics["stats"],
            "student_norm": student_tokens_raw.detach().norm(dim=-1).mean(),
            "teacher_norm": metrics["teacher_norm"],
        }
        return align_loss, metrics

    def _compute_dino_spatial_alignment(
        self, video: torch.Tensor, student_tokens: torch.Tensor
    ):
        with _isolated_rng(preserve_cuda=True):
            teacher_tokens = self.dinov2_teacher.extract_tokens(video)
        projected_student = self.student_to_dino_projector(student_tokens)
        projected_student = self._resample_spatial_tokens(
            projected_student, target_token_count=teacher_tokens.shape[2]
        )
        dino_loss, metrics = self._compute_feature_alignment_loss(
            projected_student,
            teacher_tokens,
            loss_type=self.dino_loss_type,
            mse_coeff=self.dino_mse_coeff,
            stats_coeff=self.dino_stats_coeff,
        )
        return dino_loss, metrics, teacher_tokens, projected_student

    def _compute_future_latent_temporal_regions(
        self, predicted_future_latents: torch.Tensor
    ):
        if predicted_future_latents.ndim != 4:
            raise ValueError(
                "Expected video-diffusion future latents [B,T,N,D], got "
                f"{tuple(predicted_future_latents.shape)}"
            )
        if predicted_future_latents.shape[1] != self.dual_required_frames:
            raise ValueError(
                f"Expected {self.dual_required_frames} predicted frames, got "
                f"{predicted_future_latents.shape[1]}"
            )
        f0, f1, f2, f3 = predicted_future_latents.unbind(dim=1)
        h01 = self.temporal_fusion(f0, f1)
        h23 = self.temporal_fusion(f2, f3)
        q01 = self.student_to_jepa_projector(h01)
        q23 = self.student_to_jepa_projector(h23)
        return {
            "future0": f0,
            "future1": f1,
            "future2": f2,
            "future3": f3,
            "h01": h01,
            "h23": h23,
            "q01": q01,
            "q23": q23,
        }

    @staticmethod
    def _cross_temporal_relation_logits(
        earlier_tokens: torch.Tensor, later_tokens: torch.Tensor
    ) -> torch.Tensor:
        # Match cross-temporal patch relations instead of absolute teacher features.
        if earlier_tokens.ndim != 3 or later_tokens.ndim != 3:
            raise ValueError(
                "Cross-temporal relations expect [B,N,D] tokens, got "
                f"{tuple(earlier_tokens.shape)} and {tuple(later_tokens.shape)}"
            )
        if earlier_tokens.shape != later_tokens.shape:
            raise ValueError(
                "Earlier/later token shapes must match, got "
                f"{tuple(earlier_tokens.shape)} and {tuple(later_tokens.shape)}"
            )
        earlier_tokens = F.normalize(earlier_tokens.float(), p=2, dim=-1)
        later_tokens = F.normalize(later_tokens.float(), p=2, dim=-1)
        return torch.matmul(later_tokens, earlier_tokens.transpose(-1, -2))

    @torch.no_grad()
    def _extract_jepa_relational_target(
        self, video: torch.Tensor, return_metadata: bool = False
    ):
        with _isolated_rng(preserve_cuda=True):
            teacher_result = self.jepa_teacher.extract_temporal_tokens(
                video, return_metadata=return_metadata
            )
        if return_metadata:
            teacher_tokens, metadata = teacher_result
        else:
            teacher_tokens = teacher_result
            metadata = None
        if teacher_tokens.shape[1] != 2:
            raise RuntimeError(
                "Four frames with tubelet_size=2 must produce two temporal "
                f"regions, got {tuple(teacher_tokens.shape)}"
            )
        j01 = teacher_tokens[:, 0]
        j23 = teacher_tokens[:, 1]
        relation_logits = self._cross_temporal_relation_logits(j01, j23)
        probabilities = F.softmax(
            relation_logits / self.jepa_relation_temperature, dim=-1
        )
        return probabilities, relation_logits, teacher_tokens, metadata

    def _compute_jepa_dynamics_alignment(
        self,
        video: torch.Tensor,
        predicted_future_latents: torch.Tensor,
        return_metadata: bool = False,
    ):
        (
            teacher_probabilities,
            teacher_relations,
            teacher_tokens,
            metadata,
        ) = self._extract_jepa_relational_target(video, return_metadata=return_metadata)
        temporal_regions = self._compute_future_latent_temporal_regions(
            predicted_future_latents
        )
        q01 = self._resample_spatial_tokens(
            temporal_regions["q01"],
            target_token_count=teacher_tokens.shape[2],
        )
        q23 = self._resample_spatial_tokens(
            temporal_regions["q23"],
            target_token_count=teacher_tokens.shape[2],
        )
        student_relations = self._cross_temporal_relation_logits(q01, q23)
        student_log_probabilities = F.log_softmax(
            student_relations / self.jepa_relation_temperature, dim=-1
        )
        jepa_loss = F.kl_div(
            student_log_probabilities,
            teacher_probabilities.detach(),
            reduction="batchmean",
        )
        # batchmean sums over query patches; average those rows so the
        # auxiliary scale does not grow with the JEPA spatial resolution.
        jepa_loss = jepa_loss / teacher_probabilities.shape[-2]

        student_probabilities = student_log_probabilities.exp()
        teacher_row_sums = teacher_probabilities.sum(dim=-1)
        student_row_sums = student_probabilities.sum(dim=-1)
        teacher_entropy = -(
            teacher_probabilities
            * teacher_probabilities.clamp_min(torch.finfo(torch.float32).tiny).log()
        ).sum(dim=-1).mean()
        student_entropy = -(
            student_probabilities * student_log_probabilities
        ).sum(dim=-1).mean()
        metrics = {
            "kl": jepa_loss.detach(),
            "teacher_row_sum_mean": teacher_row_sums.detach().mean(),
            "teacher_row_sum_max_error": teacher_row_sums.detach()
            .sub(1.0)
            .abs()
            .max(),
            "student_row_sum_mean": student_row_sums.detach().mean(),
            "student_row_sum_max_error": student_row_sums.detach()
            .sub(1.0)
            .abs()
            .max(),
            "teacher_probability_min": teacher_probabilities.detach().min(),
            "teacher_probability_max": teacher_probabilities.detach().max(),
            "student_probability_min": student_probabilities.detach().min(),
            "student_probability_max": student_probabilities.detach().max(),
            "teacher_entropy": teacher_entropy.detach(),
            "student_entropy": student_entropy.detach(),
        }
        temporal_regions.update(
            {
                "q01": q01,
                "q23": q23,
                "student_relations": student_relations,
                "student_log_probabilities": student_log_probabilities,
                "student_probabilities": student_probabilities,
            }
        )
        return (
            jepa_loss,
            metrics,
            teacher_tokens,
            teacher_relations,
            teacher_probabilities,
            temporal_regions,
            metadata,
        )

    def _compute_dual_teacher_pair_losses(
        self,
        conditioning_video: torch.Tensor,
        conditioning_student: torch.Tensor,
        target_video: torch.Tensor,
        target_student: torch.Tensor,
        video_prediction: Optional[Dict[str, torch.Tensor]] = None,
        capture_shapes: bool = False,
    ):
        """Keep DINO on tokenizer features and JEPA on predicted future latents."""
        for video in (conditioning_video, target_video):
            if video.shape[2] != self.dual_required_frames:
                raise ValueError(
                    "Foundation-teacher clips must contain "
                    f"{self.dual_required_frames} frames, got {tuple(video.shape)}"
                )

        zero = target_student.new_zeros(())
        dino_loss = zero
        jepa_loss = zero
        metrics = {}
        shapes = {
            "conditioning_rgb": tuple(conditioning_video.shape),
            "future_rgb": tuple(target_video.shape),
            "student_token_feat": tuple(target_student.shape),
        }

        with _isolated_rng(preserve_cuda=True):
            if self.enable_dino_spatial:
                dino_c, metrics_c, _, _ = self._compute_dino_spatial_alignment(
                    conditioning_video, conditioning_student
                )
                (
                    dino_z,
                    metrics_z,
                    dino_tokens,
                    projected_dino,
                ) = self._compute_dino_spatial_alignment(
                    target_video, target_student
                )
                dino_loss = 0.5 * (dino_c + dino_z)
                metrics.update(
                    {
                        "dino_cos": 0.5
                        * (metrics_c["cosine"] + metrics_z["cosine"]),
                        "dino_mse": 0.5
                        * (metrics_c["mse"] + metrics_z["mse"]),
                        "dino_stats": 0.5
                        * (metrics_c["stats"] + metrics_z["stats"]),
                        "dino_student_norm": 0.5
                        * (
                            metrics_c["student_norm"]
                            + metrics_z["student_norm"]
                        ),
                        "dino_teacher_norm": 0.5
                        * (
                            metrics_c["teacher_norm"]
                            + metrics_z["teacher_norm"]
                        ),
                    }
                )
                if capture_shapes:
                    shapes["dino"] = tuple(dino_tokens.shape)
                    shapes["student_to_dino"] = tuple(projected_dino.shape)

            if self.enable_jepa_dynamics:
                if video_prediction is None:
                    raise RuntimeError(
                        "JEPA supervision requires video-diffusion future latents."
                    )
                predicted_future_latents = video_prediction[
                    "predicted_future_latents"
                ]
                (
                    jepa_loss,
                    jepa_metrics,
                    jepa_tokens,
                    teacher_relations,
                    teacher_probabilities,
                    temporal_regions,
                    jepa_metadata,
                ) = self._compute_jepa_dynamics_alignment(
                    target_video,
                    predicted_future_latents,
                    return_metadata=capture_shapes,
                )
                metrics.update(
                    {
                        f"jepa_{key}": value
                        for key, value in jepa_metrics.items()
                    }
                )
                diffusion_timesteps = video_prediction["diffusion_timesteps"]
                metrics.update(
                    {
                        "jepa_diffusion_timestep_mean": diffusion_timesteps.float()
                        .mean()
                        .detach(),
                        "jepa_diffusion_timestep_min": diffusion_timesteps.min()
                        .detach(),
                        "jepa_diffusion_timestep_max": diffusion_timesteps.max()
                        .detach(),
                        "jepa_future_mask_fraction": video_prediction["future_mask"]
                        .float()
                        .mean()
                        .detach(),
                    }
                )
                if capture_shapes:
                    shapes.update(
                        {
                            "video_decoder_condition": tuple(
                                video_prediction["decoder_condition"].shape
                            ),
                            "video_diffusion_pred_x0": tuple(
                                predicted_future_latents.shape
                            ),
                            "future_mask": tuple(
                                video_prediction["future_mask"].shape
                            ),
                            "diffusion_timesteps": tuple(
                                diffusion_timesteps.shape
                            ),
                            "jepa_input": jepa_metadata["input_shape"],
                            "jepa_patch_embed": jepa_metadata[
                                "patch_embed_shape"
                            ],
                            "raw_jepa": jepa_metadata["raw_output_shape"],
                            "reshaped_jepa": tuple(jepa_tokens.shape),
                            "j01": tuple(jepa_tokens[:, 0].shape),
                            "j23": tuple(jepa_tokens[:, 1].shape),
                            "q01": tuple(temporal_regions["q01"].shape),
                            "q23": tuple(temporal_regions["q23"].shape),
                            "r_jepa": tuple(teacher_relations.shape),
                            "p_jepa": tuple(teacher_probabilities.shape),
                            "r_student": tuple(
                                temporal_regions["student_relations"].shape
                            ),
                            "p_student": tuple(
                                temporal_regions["student_probabilities"].shape
                            ),
                            "future0": tuple(temporal_regions["future0"].shape),
                            "future1": tuple(temporal_regions["future1"].shape),
                            "future2": tuple(temporal_regions["future2"].shape),
                            "future3": tuple(temporal_regions["future3"].shape),
                            "h01": tuple(temporal_regions["h01"].shape),
                            "h23": tuple(temporal_regions["h23"].shape),
                        }
                    )
        return dino_loss, jepa_loss, metrics, shapes

    def _log_dual_teacher_first_forward(
        self,
        shapes: Dict[str, tuple],
        base_loss: torch.Tensor,
        dino_loss: torch.Tensor,
        jepa_loss: torch.Tensor,
        total_loss: torch.Tensor,
        alignment_metrics: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        if self._dual_teacher_shapes_logged:
            return
        self._dual_teacher_shapes_logged = True
        if not self.log_dual_teacher_shapes or os.environ.get("RANK", "0") != "0":
            return

        print("--- foundation-teacher first-forward shapes ---", flush=True)
        for label in (
            "conditioning_rgb",
            "future_rgb",
            "student_token_feat",
            "dino",
            "student_to_dino",
            "video_decoder_condition",
            "video_diffusion_pred_x0",
            "future_mask",
            "diffusion_timesteps",
            "jepa_input",
            "jepa_patch_embed",
            "raw_jepa",
            "reshaped_jepa",
            "j01",
            "j23",
            "future0",
            "future1",
            "future2",
            "future3",
            "h01",
            "h23",
            "q01",
            "q23",
            "r_jepa",
            "r_student",
            "p_jepa",
            "p_student",
        ):
            if label in shapes:
                print(f"{label}: {shapes[label]}", flush=True)
        print(
            "losses: "
            f"base={base_loss.detach().float().item():.6f}, "
            f"dino={dino_loss.detach().float().item():.6f}, "
            f"jepa={jepa_loss.detach().float().item():.6f}, "
            f"total={total_loss.detach().float().item():.6f}",
            flush=True,
        )
        if alignment_metrics:
            metric_names = (
                "video_loss",
                "action_loss",
                "dino_cos",
                "dino_mse",
                "dino_stats",
                "jepa_kl",
                "jepa_teacher_row_sum_mean",
                "jepa_teacher_row_sum_max_error",
                "jepa_teacher_probability_min",
                "jepa_teacher_probability_max",
                "jepa_student_row_sum_mean",
                "jepa_student_row_sum_max_error",
                "jepa_student_probability_min",
                "jepa_student_probability_max",
                "jepa_teacher_entropy",
                "jepa_student_entropy",
                "jepa_diffusion_timestep_mean",
                "jepa_diffusion_timestep_min",
                "jepa_diffusion_timestep_max",
                "jepa_future_mask_fraction",
            )
            logged_metrics = {
                name: alignment_metrics[name].detach().float().item()
                for name in metric_names
                if name in alignment_metrics
            }
            print(f"alignment_metrics: {logged_metrics}", flush=True)

    def compute_loss(self, batch, **kwargs):
        B, T, C, H, W = batch["obs"]["image"].size()
        self._last_align_metrics = {}

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
        align_loss = x.new_zeros(())
        dino_loss = x.new_zeros(())
        jepa_loss = x.new_zeros(())
        legacy_metrics = {}
        dual_metrics = {}
        dual_shapes = {}
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
                    if self.teacher_type == "jepa":
                        teacher_z_tokens = (
                            self._expand_jepa_regions_for_legacy_alignment(
                                teacher_z_tokens,
                                target_timesteps=x_img.shape[2],
                                tubelet_size=self.jepa_teacher.tubelet_size,
                            )
                        )
                        teacher_c_tokens = (
                            self._expand_jepa_regions_for_legacy_alignment(
                                teacher_c_tokens,
                                target_timesteps=c_img.shape[2],
                                tubelet_size=self.jepa_teacher.tubelet_size,
                            )
                        )
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
                legacy_metrics = {
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

        model_result = self.model(
            z,
            c,
            history_trajectory,
            trajectory,
            text_latents,
            task_mode=selected_mode,
            proprioception_input=proprioception_input,
            return_video_prediction=self.enable_jepa_dynamics,
        )
        if self.enable_jepa_dynamics:
            base_loss, video_loss, act_loss, video_prediction = model_result
        else:
            base_loss, video_loss, act_loss = model_result
            video_prediction = None

        if self.use_dual_teacher_alignment:
            capture_shapes = not self._dual_teacher_shapes_logged
            (
                dino_loss,
                jepa_loss,
                dual_metrics,
                dual_shapes,
            ) = self._compute_dual_teacher_pair_losses(
                c_img,
                c_token_feat,
                x_img,
                z_token_feat,
                video_prediction=video_prediction,
                capture_shapes=capture_shapes,
            )
            dual_metrics.update(
                {
                    "video_loss": video_loss.detach(),
                    "action_loss": act_loss.detach(),
                }
            )
        loss = base_loss
        if self.use_student_tokenizer and self.use_alignment:
            loss = loss + self.align_coeff * align_loss
        if self.use_dual_teacher_alignment:
            loss = (
                loss
                + self.lambda_dino * dino_loss
                + self.lambda_jepa * jepa_loss
            )

        total_loss_without_unused_terms = loss
        self._last_align_metrics = {
            **legacy_metrics,
            **dual_metrics,
            "base_loss": base_loss.detach(),
            "video_loss": video_loss.detach(),
            "action_loss": act_loss.detach(),
            "dino_loss": dino_loss.detach(),
            "jepa_loss": jepa_loss.detach(),
            "weighted_dino_loss": (self.lambda_dino * dino_loss).detach(),
            "weighted_jepa_loss": (self.lambda_jepa * jepa_loss).detach(),
            "total_loss": total_loss_without_unused_terms.detach(),
        }
        if self.use_dual_teacher_alignment:
            self._log_dual_teacher_first_forward(
                dual_shapes,
                base_loss,
                dino_loss,
                jepa_loss,
                total_loss_without_unused_terms,
                dual_metrics,
            )

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

        for module in (
            self.student_to_dino_projector,
            self.temporal_fusion,
            self.student_to_jepa_projector,
        ):
            if module is not None:
                for param in module.parameters():
                    if param.requires_grad:
                        loss = loss + _ddp_unused_term(param)

        if kwargs.get("return_debug_components", False):
            components = {
                "base_loss": base_loss,
                "video_loss": video_loss,
                "action_loss": act_loss,
                "legacy_align_loss": align_loss,
                "dino_loss": dino_loss,
                "jepa_loss": jepa_loss,
                "weighted_dino_loss": self.lambda_dino * dino_loss,
                "weighted_jepa_loss": self.lambda_jepa * jepa_loss,
                "total_loss": total_loss_without_unused_terms,
            }
            if kwargs.get("return_debug_outputs", False):
                return (
                    loss,
                    (video_loss, act_loss),
                    components,
                    {"video_prediction": video_prediction},
                )
            return loss, (video_loss, act_loss), components
        return loss, (video_loss, act_loss)

    def forward(self, batch, **kwargs):
        return self.compute_loss(batch, **kwargs)
