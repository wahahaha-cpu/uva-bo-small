# Minimal DINOv2 + V-JEPA 2.1 Student Experiment

This experiment keeps the existing UVA/MAR objective and adds two independent
auxiliary signals to the same Student visual tokenizer:

```text
L = L_base + lambda_dino * L_dino + lambda_jepa * L_jepa
```

The default coefficients are `lambda_dino=0.02` and `lambda_jepa=0.05`.
The DINO loss is the existing hybrid patch-feature loss. The JEPA loss is a
cosine loss on temporal changes only. No adaptive gate, VAE alignment, MAR
change, action-conditioned predictor, or temporal Transformer is used.

## Data And Forward Flow

Libero samples a 32-frame sequence. The existing `process_data` path selects
frames `[3, 7, 11, 15, 19, 23, 27, 31]`, keeps their chronological order, and
splits them into history/future clips of four frames each. No dataset code was
changed.

For either four-frame clip:

```text
RGB input                    [B, 3, 4, 256, 256]
Student token features       [B, 4, 256, 304]
```

DINOv2 receives the four frames as independent images after its normal resize
and ImageNet normalization:

```text
DINO tokens                  [B, 4, 256, 384]
student_to_dino_projector    [B, 4, 256, 384]
```

V-JEPA receives the complete clip exactly once:

```text
JEPA input                   [B, 3, 4, 384, 384]
tubelet_size                 2 (kept at the pretrained setting)
patch projection             [B, 768, 2, 24, 24]
raw encoder output           [B, 1152, 768]
explicit temporal tokens    [B, 2, 576, 768]
J01, J23                    [B, 576, 768]
delta_jepa = J23 - J01      [B, 576, 768]
```

The reshape is derived from an actual forward hook on the encoder's Conv3d
patch projection. `PatchEmbed3D` flattens the `[T, H, W]` output in temporal,
height, width order; the implementation validates that the flattened token
count agrees with the captured projection shape. It never treats `T*H*W` as a
single square image grid.

Student dynamics use one shared ordered-pair MLP:

```text
H01 = F([S0, S1, S1-S0])
H23 = F([S2, S3, S3-S2])
delta_student = H23 - H01
student_to_jepa_projector(delta_student)  [B, 256, 768]
spatial interpolation only                 [B, 576, 768]
```

The JEPA objective is:

```text
L_jepa = 1 - cosine(projected_delta_student, stopgrad(delta_jepa))
```

## Changed Files

| File | Change |
| --- | --- |
| `unified_video_action/model/common/jepa_teacher.py` | Removes duplicate-frame clips; performs one real video forward, captures patch-grid shape, restores `[B,T,N,D]`, keeps teacher frozen/eval, and recovers stale torch-hub locks. |
| `unified_video_action/model/common/temporal_fusion.py` | Adds the small shared ordered-pair `TemporalFusionMLP`. |
| `unified_video_action/policy/unified_video_action_policy.py` | Adds independent DINO spatial and JEPA dynamics branches, spatial-only resampling, auxiliary coefficients, optimizer/EMA/DDP registration, first-forward shape logging, and legacy JEPA compatibility mapping. |
| `unified_video_action/model/common/dinov2_teacher.py` | Ensures the frozen DINO wrapper remains eval when the parent policy enters train mode. |
| `scripts/verify_dino_jepa_minimal.py` | Real Libero one-batch forward/backward, component gradient norms, frozen-teacher checks, optimizer membership, EMA state-dict check, and temporal-order sanity checks. |
| `scripts/training/train_uva_libero10_dino_jepa_minimal_8gpu.sh` | Selectable baseline/DINO/JEPA/combined launcher with checkpoint preflight, global-batch checks, optional sanity check, and dry-run mode. |
| `unified_video_action/config/uva_libero10_dino_jepa_minimal*.yaml` | Baseline, DINO-only, JEPA-only, and DINO+JEPA ablations. Legacy alignment is disabled in all four. |

Teacher construction, auxiliary module construction, and teacher forward are
RNG-isolated. This keeps Student/MAR initialization and stochastic base-loss
sampling identical across ablations; corresponding projectors use fixed
sub-seeds.

## Ablations

```bash
# Baseline
bash scripts/training/train_uva_libero10_dino_jepa_minimal_8gpu.sh baseline

# DINO spatial supervision only
bash scripts/training/train_uva_libero10_dino_jepa_minimal_8gpu.sh dino

# V-JEPA temporal-dynamics supervision only
bash scripts/training/train_uva_libero10_dino_jepa_minimal_8gpu.sh jepa

# Both complementary signals
bash scripts/training/train_uva_libero10_dino_jepa_minimal_8gpu.sh dino_jepa
```

The launcher defaults to `8 GPU x 16 samples x accumulation 1 = global batch
128`, matching the existing eight-GPU experiment. Set `RUN_SANITY_CHECK=0` to
skip the one-real-batch preflight, or `DRY_RUN=1` to print the exact command.

## Real One-Batch Results

Command:

```bash
CUDA_VISIBLE_DEVICES=3 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /data1/local_userdata/jinboning/conda/envs/repa/bin/python3.9 \
  scripts/verify_dino_jepa_minimal.py \
  --config-name uva_libero10_dino_jepa_minimal --device cuda --check-ema
```

Observed combined loss values on the first real Libero sample:

```text
base_loss       = 0.94105399
dino_loss       = 5.16265297
jepa_loss       = 1.01007223
weighted_dino   = 0.10325306
weighted_jepa   = 0.05050361
total_loss      = 1.09481061
```

Component gradient norms on Student (before the total backward):

```text
grad_norm_base  = 0.00000000
grad_norm_dino  = 0.09952239
grad_norm_jepa  = 0.18807101
```

The zero first-batch base-to-Student norm is expected for the existing
video-only MAR checkpoint: its randomly initialized action diffusion head has
a zero-initialized final output layer. The base loss still trains the MAR/action
parameters; after that head updates, its conditioning path can contribute to
Student gradients. The auxiliary branches provide the initial Student signal.

After `total_loss.backward()`:

```text
Student tokenizer             gradient norm 0.20928845, finite
DINO projector                gradient norm 0.02491881, finite
TemporalFusionMLP             gradient norm 0.02793704, finite
JEPA projector                gradient norm 0.02022957, finite
DINO teacher                  0 trainable tensors, 0 gradient tensors, eval
V-JEPA teacher                0 trainable tensors, 0 gradient tensors, eval
```

Temporal checks:

```text
TemporalFusion(S0,S1) vs TemporalFusion(S1,S0): mean abs diff 0.00613579
JEPA delta forward vs reversed clip:               mean abs diff 0.34640241
JEPA delta forward/reversed cosine:                         -0.16564649
```

Baseline, DINO-only, JEPA-only, and combined configurations all pass the same
finite-loss/gradient contract. DINO-only and combined share the same first-batch
DINO loss (`5.16265297`) and base loss (`0.94105399`), confirming the ablation
RNG isolation.

## Verification Status

Passed:

1. Python 3.9 and 3.12 compilation for all changed Python files.
2. `bash -n` for the launcher and dry-run command generation for all four modes.
3. Real V-JEPA output/shape validation with `tubelet_size=2` and four frames.
4. Real Libero one-batch forward, finite losses, backward, module gradients, and frozen-teacher checks for all ablations.
5. EMA deep-copy and strict policy state-dict load check for the combined policy.
6. Existing `uva_libero10_jepa2_1_small_token_feat` one-batch compatibility smoke test after removing duplicate-frame JEPA input.
