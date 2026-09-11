# PushT-M VAE-unfrozen policy experiment

This experiment is the action-only UVA policy setting on the official PushT-M
multitask demonstrations. The data and all large runtime artifacts stay on the
`data1`-backed symlinks in this checkout.

The downloaded archive is:

```text
/data1/local_userdata/jinboning/uva-bo-small/data/pusht_multitask.zip
```

The extracted Zarr dataset is:

```text
/data1/local_userdata/jinboning/uva-bo-small/data/pusht_multitask
```

It contains 247 episodes and 35,927 transitions. The arrays are `img` with
shape `(35927, 96, 96, 3)`, `state` with `(35927, 5)`, and `action` with
`(35927, 2)`. The source is the PushT-M Google Drive archive linked by the
[official UVA repository](https://github.com/ShuangLI59/unified_video_action#dataset)
(Drive file id `14VqUC_LL411o9F_qdjVZlgiRBjZknw01`).

To download or verify the archive again, use:

```bash
bash scripts/prepare_pusht_m.sh
```

The script uses the existing `data` symlink, checks the ZIP before extraction,
and verifies the expected Zarr directories. Set `HTTPS_PROXY`, `HTTP_PROXY`,
and `ALL_PROXY` when the host requires an HTTP proxy for Google Drive.

The reproducible configuration is
[`uva_pusht_m_vae_policy.yaml`](../unified_video_action/config/uva_pusht_m_vae_policy.yaml)
and the launcher is
[`train_uva_pusht_m_vae_policy.sh`](../scripts/training/train_uva_pusht_m_vae_policy.sh).

Run it from the repository root:

```bash
cd /home/jinboning/project/uva-bo-small
bash scripts/training/train_uva_pusht_m_vae_policy.sh
```

The default layout is eight processes, per-device batch 8, and accumulation 4:

```text
8 GPUs x 8 samples x 4 accumulation = global batch 256
```

The launcher rejects any per-device batch other than 8 and rejects layouts that
do not produce global batch 256. For example, a four-GPU run must use
`GRAD_ACCUM_STEPS=8`. The official UVA PushT training command uses eight
processes with per-process batch 32 and accumulation 1, so its effective global
batch is 256; this experiment preserves that value while enforcing per-device
batch 8. The paper text does not spell out this number separately from the
official training command.

The alignment references are the [UVA paper](https://unified-video-action-model.github.io/static/UVA_paper.pdf)
and the [official PushT config](https://github.com/ShuangLI59/unified_video_action/blob/main/unified_video_action/config/uva_pusht.yaml).

The script sets `task.env_runner.fix_goal=False`, uses 50 test environments
plus one non-visual train environment (the existing rollout aggregator expects
one train score), and keeps `legacy_test=True`, matching the PushT-M evaluation
setup. Rollouts run every 10 epochs and the top checkpoint is selected by
`test_mean_score`.

The experiment loads the MAR-B checkpoint only as the MAR pretrained model:

```yaml
model:
  policy:
    autoregressive_model_params:
      pretrained_model_path: pretrained_models/mar/mar_base/checkpoint-last.pth
```

The VAE still needs its KL-f16 checkpoint as initialization. "Only MAR
pretrained parameters" means that no prior UVA/video/action workspace
checkpoint is resumed; `training.resume` is explicitly false.

The policy path is configured as follows:

```yaml
model:
  policy:
    selected_training_mode: policy_model
    use_student_tokenizer: true
    student_tokenizer_backend: vae
    vae_student_mode: sample
    vae_student_feature: latent
    freeze_mar: false
    action_model_params:
      predict_action: true
```

`policy_model` calls MAR's action loss and sets video loss to zero. The model
still keeps `predict_video: true` because that is part of the MAR architecture
and its pretrained checkpoint interface; it does not make the training a video
objective.

No production Python code was changed for this experiment. The current policy
already contains the VAE backend used by this configuration. The worktree has
other pre-existing Python edits; they are outside this experiment and are not
part of the configuration/script change below. The comparison is between the
official PushT-M video command (`uva_pusht.yaml` plus
`train_uva_pusht_multitask.sh`) and this action-only run:

```diff
--- official PushT-M video command
+++ uva_pusht_m_vae_policy.yaml
@@
-dataloader.batch_size: 32
-training.gradient_accumulate_every: 1
-training.resume: true
-task.dataset.dataset_path: data/pusht_multitask
-task.dataset.dataset_type: multitask
-task.env_runner.fix_goal: false
-task.env_runner.n_train: 6
-model.policy.selected_training_mode: video_model
-model.policy.action_model_params.predict_action: false
+dataloader.batch_size: 8
+training.gradient_accumulate_every: 4
+training.resume: false
+task.dataset.dataset_path: data/pusht_multitask
+task.dataset.dataset_type: multitask
+task.env_runner.fix_goal: false
+task.env_runner.n_train: 1
+task.env_runner.n_test: 50
+model.policy.selected_training_mode: policy_model
+model.policy.use_student_tokenizer: true
+model.policy.student_tokenizer_backend: vae
+model.policy.vae_student_feature: latent
+model.policy.action_model_params.predict_action: true
+model.policy.autoregressive_model_params.pretrained_model_path: pretrained_models/mar/mar_base/checkpoint-last.pth
+model.policy.optimizer.learning_rate: 1e-4
```

The dataset, `dataset_type`, and `fix_goal` lines on the old side are the
effective overrides supplied by the official multitask launcher. The new file
collects those values into one reproducible config and adds the policy/VAE
settings.

The production-code behavior being reused is equivalent to:

```python
# existing unified_video_action/policy/unified_video_action_policy.py
if self.student_tokenizer_backend == "vae":
    self.vae_model.encoder.requires_grad_(True)
    self.vae_model.quant_conv.requires_grad_(True)
    self.vae_model.encoder.train()
    self.vae_model.quant_conv.train()

# existing optimizer construction
vae_optim_groups = add_weight_decay(self.vae_model.encoder)
vae_optim_groups += add_weight_decay(self.vae_model.quant_conv)
optim_groups.extend(vae_optim_groups)
```

The VAE decoder and `post_quant_conv` remain frozen because only the encoding
path feeds the policy loss. MAR itself remains trainable (`freeze_mar: false`),
and the action diffusion head is enabled with `conv_fc`.

The first update after loading MAR can have zero VAE gradient because the
action diffusion head's final projection is initialized to zero. This is the
MAR initialization behavior; after the first action-head update, the policy
loss propagates into the VAE encoder. The policy construction and loss smoke
test verified this post-update gradient path.

Before starting a long run, print the exact launch command without allocating
GPUs:

```bash
DRY_RUN=1 bash scripts/training/train_uva_pusht_m_vae_policy.sh
```

After a checkpoint is written, evaluate it with the repository's PushT runner:

```bash
CUDA_VISIBLE_DEVICES=0 \
  /data1/local_userdata/jinboning/conda/envs/repa/bin/python eval_sim.py \
  --checkpoint checkpoints/<run-name>/checkpoints/latest.ckpt \
  --output_dir checkpoints/<run-name>/eval \
  --device cuda:0
```

The checkpoint config keeps `fix_goal=False` and `n_test=50`, so this evaluates
the same 50-test-environment protocol used during training. The output path is
also required to resolve under `data1`.

Outputs default under the `checkpoints` and `wandb` symlinks, both backed by
`/data1/local_userdata/jinboning/uva-bo/`. Override `RUN_DIR` only with another
absolute path on `data1` if a different storage location is needed.
