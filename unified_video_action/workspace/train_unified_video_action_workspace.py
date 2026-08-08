if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
import copy
import random
import tqdm
from torch.utils.data import DataLoader
import numpy as np
from accelerate import Accelerator
import pickle
from datetime import timedelta

from accelerate.utils import (
      DeepSpeedPlugin,
      DistributedDataParallelKwargs,
      InitProcessGroupKwargs,
)
from unified_video_action.workspace.base_workspace import BaseWorkspace
from unified_video_action.policy.unified_video_action_policy import (
    UnifiedVideoActionPolicy,
)
from unified_video_action.dataset.base_dataset import BaseImageDataset
from unified_video_action.dataset.umi_multi_dataset import UmiMultiDataset
from unified_video_action.common.checkpoint_util import TopKCheckpointManager
from unified_video_action.common.pytorch_util import dict_apply
from unified_video_action.common.training_utils import (
    accumulation_window,
    local_batches_per_epoch,
    optimizer_steps_per_epoch,
)
from unified_video_action.model.autoregressive.ema_model import EMAModel
from unified_video_action.model.common.lr_scheduler import get_scheduler
from unified_video_action.utils.load_env import load_env_runner, env_rollout
from unified_video_action.eval.eval import test_video_fvd, test_action_l2
from unified_video_action.utils.data_utils import resize_image

OmegaConf.register_new_resolver("eval", eval, replace=True)

timeout_kwargs = InitProcessGroupKwargs(
      timeout=timedelta(hours=4)
  )


class TrainUnifiedVideoActionWorkspace(BaseWorkspace):
    include_keys = ["global_step", "epoch"]

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure policy model
        language_emb_model = cfg.task.dataset.language_emb_model
        if (
            "deepspeed_config" in cfg.training
            and cfg.training.deepspeed_config is not None
        ):
            language_emb_model = (
                None  # HACK: When training umi dataset on multiple nodes
            )
        self.model: UnifiedVideoActionPolicy = hydra.utils.instantiate(
            cfg.model.policy,
            task_name=cfg.task.name,
            task_modes=cfg.task.task_modes,
            normalizer_type=cfg.task.dataset.normalizer_type,
            language_emb_model=language_emb_model,
        )

        self.ema_model: UnifiedVideoActionPolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        self.optimizer = self.model.get_optimizer(**cfg.model.policy.optimizer)

        # configure training state
        self.global_step = 0
        self.epoch = 0
        
        
    def run(self):
        cfg = copy.deepcopy(self.cfg)
        if ( "deepspeed_config" in cfg.training and cfg.training.deepspeed_config is not None):
            deepspeed_plugin = DeepSpeedPlugin(
                hf_ds_config=cfg.training.deepspeed_config
            )

            ddp_kwargs = DistributedDataParallelKwargs(
                find_unused_parameters=True
            )

            accelerator = Accelerator(
                log_with="wandb",
                mixed_precision=self.cfg.training.mixed_precision,
                deepspeed_plugin=deepspeed_plugin,
                kwargs_handlers=[ddp_kwargs, timeout_kwargs],
            )
        else:
            ddp_kwargs = DistributedDataParallelKwargs(
                find_unused_parameters=True
            )

            accelerator = Accelerator(
                log_with="wandb",
                mixed_precision=self.cfg.training.mixed_precision,
                kwargs_handlers=[ddp_kwargs, timeout_kwargs],
            )

        if accelerator.is_main_process:
            cfg.logging.name = self.output_dir.split("/")[-1]
            wandb_cfg = OmegaConf.to_container(cfg.logging, resolve=True)
            wandb_cfg.pop("project")
            wandb_cfg["resume"] = "allow"

            accelerator.init_trackers(
                project_name=cfg.logging.project,
                config=OmegaConf.to_container(cfg, resolve=True),
                init_kwargs={"wandb": wandb_cfg},
            )

        if cfg.task.task_type == "multiple_datasets":
            dataset: UmiMultiDataset
            dataset = hydra.utils.instantiate(cfg.task.dataset)
            train_dataloader = dataset.get_dataloader()
            val_dataset = dataset.split_unused_episodes()
            val_dataloader = val_dataset.get_dataloader()
            dataset.set_datasets_attribute("random_img_sampling", True)
            print(
                "train dataset:",
                len(dataset),
                "train dataloader:",
                len(train_dataloader),
            )
            print(
                "val dataset:", len(val_dataset), "val dataloader:", len(val_dataloader)
            )
        else:
            # configure dataset
            dataset: BaseImageDataset
            dataset = hydra.utils.instantiate(cfg.task.dataset)
            train_dataloader = DataLoader(dataset, **cfg.dataloader)

            # configure validation dataset
            val_dataset = dataset.get_validation_dataset()
            val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)
            print(
                "train dataset:",
                len(dataset),
                "train dataloader:",
                len(train_dataloader),
            )
            print(
                "val dataset:", len(val_dataset), "val dataloader:", len(val_dataloader)
            )

            # compute normalizer on the main process and save to disk
            normalizer_path = os.path.join(self.output_dir, "normalizer.pkl")
            if accelerator.is_main_process:
                normalizer = dataset.get_normalizer()
                pickle.dump(normalizer, open(normalizer_path, "wb"))

        if (
            "deepspeed_config" not in cfg.training
            or cfg.training.deepspeed_config is None
        ):
            accelerator.wait_for_everyone()

        # load normalizer on all processes
        if cfg.task.task_type == "single_dataset":
            normalizer = pickle.load(open(normalizer_path, "rb"))

            self.model.set_normalizer(normalizer)
            if cfg.training.use_ema:
                self.ema_model.set_normalizer(normalizer)
        

        # Debug overrides must be applied before deriving the scheduler horizon.
        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        accumulation_steps = int(cfg.training.gradient_accumulate_every)
        if accumulation_steps < 1:
            raise ValueError("training.gradient_accumulate_every must be >= 1")
        # ``len(train_dataloader)`` is global before prepare() and local after
        # prepare().  Derive the local count explicitly so scheduler horizon is
        # independent of world size while still matching Accelerate's sharding.
        batches_before_prepare = len(train_dataloader)
        local_batch_count = local_batches_per_epoch(
            batches_before_prepare,
            accelerator.num_processes,
            split_batches=accelerator.split_batches,
        )
        updates_per_epoch = optimizer_steps_per_epoch(
            batches_before_prepare,
            accelerator.num_processes,
            accumulation_steps,
            split_batches=accelerator.split_batches,
        )
        total_optimizer_steps = updates_per_epoch * int(cfg.training.num_epochs)
        if cfg.training.max_train_steps is not None:
            if int(cfg.training.max_train_steps) < 1:
                raise ValueError("training.max_train_steps must be >= 1 or null")
            total_optimizer_steps = min(
                total_optimizer_steps, int(cfg.training.max_train_steps)
            )

        warmup_steps = int(cfg.training.lr_warmup_steps)
        if warmup_steps < 0:
            raise ValueError("training.lr_warmup_steps must be >= 0")
        if total_optimizer_steps < 1:
            raise ValueError("Training must contain at least one optimizer update")
        accelerator.print(
            "Optimizer-step schedule: "
            f"batches_before_prepare={batches_before_prepare}, "
            f"local_batches_per_epoch={local_batch_count}, "
            f"accumulation_steps={accumulation_steps}, "
            f"updates_per_epoch={updates_per_epoch}, "
            f"warmup_updates={warmup_steps}, "
            f"total_updates={total_optimizer_steps}"
        )

        # Keep this scheduler unwrapped.  Accelerate's wrapped scheduler advances
        # once per process for a non-split dataloader, which makes a 4-GPU run and
        # an 8-GPU run follow different LR curves.  We call it exactly once per
        # optimizer update below, independent of world size.
        self.lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_optimizer_steps,
            last_epoch=self.global_step - 1,
        )

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                accelerator.print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)

        # configure env
        rollout_enabled = (
              cfg.model.policy.action_model_params.predict_action
              and "env_runner" in cfg.task
        )
        env_runners = None
        if rollout_enabled:
            if accelerator.is_main_process:
                env_runners = load_env_runner(cfg, self.output_dir)
            accelerator.wait_for_everyone()
        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, "checkpoints"), **cfg.checkpoint.topk
        )

        # accelerator (the scheduler intentionally stays a plain scheduler)
        (
            train_dataloader,
            val_dataloader,
            self.model,
            self.optimizer,
        ) = accelerator.prepare(
            train_dataloader,
            val_dataloader,
            self.model,
            self.optimizer,
        )
        actual_local_batches = len(train_dataloader)
        if actual_local_batches != local_batch_count:
            raise RuntimeError(
                "Accelerate dataloader sharding did not match the scheduler "
                f"horizon: predicted={local_batch_count}, "
                f"actual={actual_local_batches}."
            )

        device = self.model.device

        if self.ema_model is not None:
            self.ema_model.to(device)

        # training loop
        stop_training = (
            cfg.training.max_train_steps is not None
            and self.global_step >= int(cfg.training.max_train_steps)
        )
        for local_epoch_idx in range(cfg.training.num_epochs):
            if stop_training:
                break
            step_log = dict()
            print(self.output_dir)

            # ========= train for this epoch ==========
            train_losses = list()
            with tqdm.tqdm(
                train_dataloader,
                desc=f"Training epoch {self.epoch}",
                leave=False,
                mininterval=cfg.training.tqdm_interval_sec,
            ) as tepoch:
                num_batches = len(train_dataloader)
                self.optimizer.zero_grad(set_to_none=True)
                for batch_idx, batch in enumerate(tepoch):
                    _, _, window_size, is_update_step = accumulation_window(
                        batch_idx,
                        num_batches,
                        accumulation_steps,
                    )

                    # device transfer
                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                    # resize image
                    batch = resize_image(cfg, batch)
                    # compute loss
                    if (
                        "deepspeed_config" in cfg.training
                        and cfg.training.deepspeed_config is not None
                    ): 
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16): # You might need to change the device_type to str(device) for other versions of torch
                            raw_loss, (loss_diffusion, loss_action) = self.model(batch)
                    else:
                        raw_loss, (loss_diffusion, loss_action) = self.model(batch)

                    # Average gradients over the effective global batch.  The old
                    # code summed micro-batch means, multiplying the update by K
                    # for 4-GPU x accumulation-4 versus 8-GPU x accumulation-1.
                    scaled_loss = raw_loss / float(window_size)
                    if is_update_step:
                        accelerator.backward(scaled_loss)
                    else:
                        with accelerator.no_sync(self.model):
                            accelerator.backward(scaled_loss)

                    # One optimizer/LR/EMA update per completed accumulation window.
                    optimizer_step_succeeded = False
                    if is_update_step:
                        self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)
                        optimizer_step_succeeded = not bool(
                            getattr(self.optimizer, "step_was_skipped", False)
                        )
                        if optimizer_step_succeeded:
                            self.lr_scheduler.step()
                            if cfg.training.use_ema:
                                ema.step(accelerator.unwrap_model(self.model))

                    # logging
                    raw_loss_cpu = raw_loss.item()

                    tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                    train_losses.append(raw_loss_cpu)

                    if cfg.model.policy.autoregressive_model_params.predict_video:
                        loss_diffusion_cpu = loss_diffusion.item()
                    else:
                        loss_diffusion_cpu = 0.0

                    if cfg.model.policy.action_model_params.predict_action:
                        loss_action_cpu = loss_action.item()
                    else:
                        loss_action_cpu = 0.0

                    step_log = {
                        "train_loss": raw_loss_cpu,
                        "diffusion_loss": loss_diffusion_cpu,
                        "action_loss": loss_action_cpu,
                        "global_step": self.global_step,
                        "optimizer_step": self.global_step,
                        "micro_step": batch_idx,
                        "optimizer_step_skipped": float(
                            is_update_step and not optimizer_step_succeeded
                        ),
                        "epoch": self.epoch,
                        "lr": self.lr_scheduler.get_last_lr()[0],
                    }
                    policy_module = accelerator.unwrap_model(self.model)
                    if hasattr(policy_module, "_last_align_metrics"):
                        align_metrics = getattr(policy_module, "_last_align_metrics", {})
                        if isinstance(align_metrics, dict) and len(align_metrics) > 0:
                            for key, value in align_metrics.items():
                                if torch.is_tensor(value):
                                    step_log[key] = value.detach().float().item()
                                elif isinstance(value, (float, int)):
                                    step_log[key] = float(value)

                    if is_update_step:
                        accelerator.log(step_log, step=self.global_step)
                        if optimizer_step_succeeded:
                            self.global_step += 1

                    if (
                        cfg.training.max_train_steps is not None
                        and self.global_step >= int(cfg.training.max_train_steps)
                    ):
                        stop_training = True
                        break

            train_loss = np.mean(train_losses)
            step_log["train_loss"] = train_loss

            # ========= eval for this epoch ==========
            # policy = self.model
            policy = accelerator.unwrap_model(self.model)
            if cfg.training.use_ema:
                policy = self.ema_model
            policy.eval()

            # ========= evaluate val video generation =========
            if cfg.model.policy.autoregressive_model_params.predict_video:
                fvd_log = test_video_fvd(
                    cfg,
                    policy,
                    val_dataloader,
                    local_epoch_idx,
                    self.output_dir,
                    device,
                )
                step_log.update(fvd_log)

            # ========= evaluate val action error =========
            if (
                cfg.model.policy.action_model_params.predict_action
                and "env_runner" not in cfg.task
            ):
                ## if has similartor, skip this
                act_log = test_action_l2(
                    cfg,
                    policy,
                    val_dataloader,
                    local_epoch_idx,
                    self.output_dir,
                    device,
                )
                step_log.update(act_log)

            # ========= simulator: run rollout =========
            if rollout_enabled:
                if (self.epoch % cfg.training.rollout_every) == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        runner_log = env_rollout(cfg, env_runners, policy)
                        step_log.update(runner_log)
                    accelerator.wait_for_everyone()

            # ========= checkpoint =========
            if (
                self.epoch % cfg.training.checkpoint_every
            ) == 0 and accelerator.is_main_process:
                # unwrap the model to save ckpt
                model_ddp = self.model
                self.model = accelerator.unwrap_model(self.model)

                # checkpointing
                if cfg.checkpoint.save_last_ckpt:
                    self.save_checkpoint()

                if cfg.checkpoint.save_last_snapshot:
                    self.save_snapshot()

                # sanitize metric names
                metric_dict = dict()
                for key, value in step_log.items():
                    new_key = key.replace("/", "_")
                    metric_dict[new_key] = value

                # save topk checkpoints
                topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                if topk_ckpt_path is not None:
                    self.save_checkpoint(path=topk_ckpt_path)

                # recover the DDP model
                self.model = model_ddp

            # ========= eval end for this epoch ==========
            policy.train()
            accelerator.log(step_log, step=self.global_step)
            self.epoch += 1

        accelerator.end_training()
