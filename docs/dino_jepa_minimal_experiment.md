# DINO Token Features + JEPA Future-Latent Experiment

This experiment gives the two frozen teachers different jobs:

```text
DINOv2   -> Student tokenizer token_feat (per-frame spatial semantics)
V-JEPA2  -> video diffusion pred_x0 (future temporal correspondence)
```

It uses the existing UVA/MAR action and video objectives:

```text
L_base = L_video_diffusion + L_action
L      = L_base + lambda_dino * L_dino + lambda_jepa * L_jepa
```

The default coefficients are `lambda_dino=0.02` and `lambda_jepa=0.02`.
There is no JEPA loss on tokenizer token features and no absolute JEPA feature
imitation.

## Forward Flow

The existing Libero data path selects frames
`[3, 7, 11, 15, 19, 23, 27, 31]` from each 32-frame sample, preserves their
order, and splits them into history and future clips:

```text
history RGB                         [B, 3, 4, 256, 256]
future RGB                          [B, 3, 4, 256, 256]
```

### DINO Spatial Branch

DINO behavior remains the same as the previous token-feature experiment. Both
four-frame clips are supervised independently and their losses are averaged:

```text
Student tokenizer token_feat       [B, 4, 256, 304]
DINO patch tokens                   [B, 4, 256, 384]
student_to_dino_projector           [B, 4, 256, 384]
L_dino                              hybrid(cosine, MSE, feature statistics)
```

DINO therefore continues to shape what each frame sees. It does not supervise
the video diffusion output.

### Causal Video-Diffusion Branch

The experiment uses `selected_training_mode=full_dynamic_model`, so the
existing video and action losses are both active. The four future MAR input
frames are fully masked before the encoder. The decoder condition therefore
depends on history but not on clean future tokens:

```text
history Student latent              [B, 4, 16, 16, 16]
future Student latent target        [B, 4, 16, 16, 16]
future MAR mask                     [B, 4, 256] = all ones
MAR decoder condition               [B, 1024, 768]
```

The base video diffusion loss remains standard: it noises the detached future
latent target to `x_t` and predicts epsilon. One shared training timestep is
used for every patch in a video.

The JEPA branch makes one additional differentiable diffusion-head call. It
starts from independent Gaussian noise at the highest training timestep
`t=999`, conditioned on the history-derived MAR decoder tokens, and converts
the predicted epsilon into `pred_x0`:

```text
video DiffLoss pred_x0              [B, 4, 256, 16]
```

This prediction does not consume the clean or noised future target. It is one
extra diffusion-head forward, but not the final result of the
non-differentiable 100-step sampling loop. The auxiliary noise generation is
RNG-isolated so action/video base losses remain identical across ablations.

### V-JEPA Future Target

Only the real future four-frame clip enters frozen V-JEPA 2.1:

```text
V-JEPA input                        [B, 3, 4, 384, 384]
tubelet_size                        2
patch projection                    [B, 768, 2, 24, 24]
raw encoder output                  [B, 1152, 768]
explicit temporal tokens            [B, 2, 576, 768]
J01, J23                            [B, 576, 768]
```

The temporal reshape comes from the real Conv3d patch projection. The encoder
sequence order is validated as temporal, height, width; `T*H*W` is never treated
as one square spatial grid.

### Relational KL

The diffusion future latents, rather than tokenizer features, enter the shared
ordered-pair fusion and JEPA projector:

```text
H01 = F([pred_x0_0, pred_x0_1, pred_x0_1 - pred_x0_0]) [B, 256, 16]
H23 = F([pred_x0_2, pred_x0_3, pred_x0_3 - pred_x0_2]) [B, 256, 16]
Q01, Q23 = P_J(H01), P_J(H23)                          [B, 256, 768]
spatial-only interpolation                            [B, 576, 768]
```

After L2 normalization:

```text
R_jepa    = J23 @ J01^T              [B, 576, 576]
R_student = Q23 @ Q01^T              [B, 576, 576]
P_jepa    = softmax(R_jepa / tau)
log_P_s   = log_softmax(R_student / tau)
```

The loss direction remains teacher to student:

```text
raw_kl = F.kl_div(log_P_s, P_jepa.detach(), reduction="batchmean")
L_jepa = raw_kl / number_of_query_patches
```

The row normalization is necessary because PyTorch `batchmean` sums all 576
query rows and divides only by `B`. Averaging the rows makes the auxiliary scale
independent of JEPA spatial resolution. `tau=0.1`.

## Causality Contract

The verifier checks the actual computation graph:

```text
||d decoder_condition / d clean_future_input|| = 0
||d decoder_condition / d history_input||       > 0
||d pure_noise_pred_x0 / d clean_future_input|| = 0
||d pure_noise_pred_x0 / d history_input||       > 0
```

This establishes history-to-future conditioning without clean future-token
leakage through the MAR encoder/decoder. It does not make the bidirectional MAR
decoder autoregressive within the four predicted future frames; those frames
are modeled jointly. It also does not add action-conditioned world modeling:
the existing `full_dynamic_model` path uses the MAR history condition for video
prediction and trains the existing action loss alongside it.

## Configuration And Ablations

All four configurations share the same full-dynamics base objective, causal
future mask, and one-timestep-per-video diffusion sampling:

```bash
# Base video + action training
bash scripts/training/train_uva_libero10_dino_jepa_minimal_8gpu.sh baseline

# Base + DINO token-feature hybrid alignment
bash scripts/training/train_uva_libero10_dino_jepa_minimal_8gpu.sh dino

# Base + JEPA future-latent relational KL
bash scripts/training/train_uva_libero10_dino_jepa_minimal_8gpu.sh jepa

# Base + both complementary teachers
bash scripts/training/train_uva_libero10_dino_jepa_minimal_8gpu.sh dino_jepa
```

The launcher defaults to `8 GPU x 16 samples x accumulation 1 = global batch
128`. It runs a real-batch sanity check unless `RUN_SANITY_CHECK=0` is set.

## Real One-Batch Result

Observed with the combined configuration:

```text
video_loss                         0.08798663
action_loss                        0.98115909
base_loss                          1.06914568
dino_loss                          5.16265297
jepa_loss (query-row mean)         0.66032153
weighted_dino                      0.10325306
weighted_jepa                      0.01320643
total_loss                         1.18560517
```

The same first-batch `base_loss=1.06914568` is observed for baseline,
DINO-only, JEPA-only, and combined runs.

Probability checks:

```text
teacher row-sum max error          2.38e-07
student row-sum max error          3.58e-07
teacher probability range          [3.28e-05, 7.997e-02]
student probability range          [5.47e-06, 4.011e-02]
teacher entropy                     5.88014221
student entropy                     6.09472132
```

Component gradients into the Student tokenizer:

```text
base                               5.46489286
weighted DINO                      0.09952239
weighted JEPA                      0.80731905
```

The base and JEPA gradient norms into the video diffusion head are
`0.75568020` and `1.48714137`, respectively. After the
combined backward, every trainable tensor in the Student tokenizer, video
diffusion head, DINO projector, TemporalFusionMLP, and JEPA projector has a
finite gradient. Both teachers remain in eval mode with zero trainable tensors
and zero gradients.

Causality and temporal checks:

```text
future mask fraction                         1.0
decoder gradient wrt clean future input      0.0
decoder gradient wrt history input           1.3041e-05
TemporalFusion forward/swap mean difference  0.16584623
JEPA forward/reverse probability difference  0.00055178
```

## Changed Files

| File | Role |
| --- | --- |
| `unified_video_action/model/autoregressive/diffusion_loss.py` | Adds the pure-noise differentiable `pred_x0` branch and one shared timestep per video. |
| `unified_video_action/model/autoregressive/mar_con_unified.py` | Applies the full future mask and returns video diffusion future-latent metadata. |
| `unified_video_action/policy/unified_video_action_policy.py` | Keeps DINO on token features and moves JEPA relational KL to diffusion `pred_x0`. |
| `unified_video_action/config/uva_libero10_dino_jepa_minimal*.yaml` | Defines common base behavior and four ablations. |
| `scripts/verify_dino_jepa_minimal.py` | Checks shapes, losses, optimizer/EMA state, gradients, frozen teachers, relations, and causal leakage. |
| `scripts/training/train_uva_libero10_dino_jepa_minimal_8gpu.sh` | Launches baseline, DINO-only, JEPA-only, or combined on eight GPUs. |

## Verification

Passed:

1. Python compilation and `git diff --check`.
2. Real V-JEPA four-frame shape recovery with `tubelet_size=2`.
3. Real Libero forward/backward for all four ablations.
4. Identical base loss across all four first-batch ablations.
5. Finite gradients into the intended Student, MAR, diffusion, and projector modules.
6. Frozen/eval teachers excluded from the optimizer.
7. Probability row sums, temporal-order sensitivity, full-mask, and causal-gradient checks.
