# DINOv2 token-feature 对齐与八卡训练修复

本文记录当前 `uva-bo-small` 分支从 JEPA token-feature 实验切换到 DINOv2 的实现、梯度/LR 修复、验证结果和启动方式。数据集、预训练权重和训练产物均不纳入 Git。

## 1. 实验定义

此前的 frozen-MAR action 配置（`uva_libero10_dinov2_small_token_feat_fully_frozen_action.yaml`）是：

```text
teacher: DINOv2 ViT-S/14
alignment: token_feat
student: small tokenizer, hidden_dim=304
projector: 304 -> 512 -> 512 -> 384
MAR: 从 `checkpoints/libero10.ckpt` 加载，全部冻结
action head: conv_ori，要求 checkpoint 中已有完整 action head
effective global batch: 128
```

当前实际训练改用文末的 video-pretrained/full-MAR 配置；DINOv2、student 和
token-feature 对齐实现不变。

DINOv2 的对齐对象不是 MAR latent，而是 student 最终 encoder 输出的 token feature：

```text
student token_feat       [B, T, 256, 304]
student align_projector  [B, T, 256, 304] -> [B, T, 256, 384]
DINOv2 patch tokens      [B, T, 256, 384]
```

`teacher_type=dinov2` 被强制要求 `align_on=token_feat`，误配成 latent 会在 policy 构造时直接报错。

## 2. DINOv2 加载和预处理

实现文件为 [`unified_video_action/model/common/dinov2_teacher.py`](../unified_video_action/model/common/dinov2_teacher.py)。加载方式遵循 `uva-bo` 的 timm 路径，但针对本工程的 Python 3.9 和本地 checkpoint 做了严格化处理：

1. `dinov2_vits14` 映射到 `vit_small_patch14_dinov2.lvd142m`。
2. 先以 checkpoint 的原生 `518 x 518` 位置网格构建模型，再用 `dynamic_img_size` 在推理时插值到 `224 x 224`。
3. 只接受显式本地文件 `pretrained_models/dinov2/dinov2_vits14_pretrain.pth`，严格加载 state dict；缺少或多出 key 会立即失败。
4. 从 `forward_features` 取 `x_norm_patchtokens`，去掉 CLS/prefix token。
5. Libero 的 `process_data()` 将图像变成 `[-1, 1]`。DINOv2 分支先恢复为 `[0, 1]`，再做 ImageNet mean/std normalize；没有改变 student/MAR 的图像约定。
6. teacher 的参数全部冻结并保持 eval mode；`policy.train()` 不会把 DINOv2 切回 train mode。

因此，224 输入得到 16×16=256 个 patch tokens，ViT-S 宽度为 384。

## 3. 梯度和 LR schedule 修复

旧 workspace 有四个会破坏四卡/八卡等价性的点：首个 micro-batch 提前 step、累积 loss 没有除以累积长度、EMA 每个 micro-batch 更新、scheduler 交给 Accelerate 后在非 split dataloader 下按进程数隐式多 step。

现在的规则是：

- 每个 accumulation window 的 loss 使用 `raw_loss / window_size`。
- 非最后一个 micro-batch 用 `accelerator.no_sync()`，最后一个才执行 DDP 梯度同步。
- 只有窗口完成时才执行 optimizer、scheduler 和 EMA；`global_step` 只表示成功的 optimizer update。
- scheduler 保持普通 diffusers scheduler，不交给 `accelerator.prepare()`；每个 optimizer update 显式调用一次。
- AMP 溢出导致 `step_was_skipped` 时，不推进 scheduler、EMA 或 `global_step`。
- 最后不足完整 accumulation 的窗口按实际 `window_size` 归一化，不丢弃也不放大梯度。

更新次数的计算使用 `unified_video_action/common/training_utils.py`：

```text
local_batches = ceil(batches_before_prepare / world_size)  # split_batches=False
updates_per_epoch = ceil(local_batches / gradient_accumulate_every)
```

当前 Libero10 训练集长度是 124165：

| 布局 | prepare 前 batch 数 | 每进程 batch 数 | accumulation | optimizer updates/epoch |
|---|---:|---:|---:|---:|
| 4 GPU × batch 8 | 15521 | 3881 | 4 | 971 |
| 8 GPU × batch 16 | 7761 | 971 | 1 | 971 |

因此两种布局都保持 global batch 128、每 epoch 971 次更新；3050 epoch 的 scheduler horizon 为 2,961,550 次 optimizer update。

本实验将 `lr_warmup_steps` 统一解释为 **optimizer update 次数**，DINO 配置和启动脚本都设置为 2000。它不再是 Accelerate 隐式放大的 per-process scheduler tick。旧 checkpoint 的 `global_step` 若来自修复前 workspace，不建议直接 resume；本 DINO action 实验明确使用 `training.resume=False`，从 action checkpoint 重新开始。

## 4. 修改文件

- `unified_video_action/model/common/dinov2_teacher.py`：本地 timm DINOv2-S/14 teacher。
- `unified_video_action/policy/unified_video_action_policy.py`：加入 `teacher_type=dinov2`，token-feature projector 和对齐分支。
- `unified_video_action/workspace/train_unified_video_action_workspace.py`：修复 accumulation、optimizer/scheduler/EMA/global_step 和 AMP skip 计数。
- `unified_video_action/common/training_utils.py`：统一更新次数和 accumulation window 公式。
- `unified_video_action/config/uva_libero10_dinov2_small_token_feat.yaml`：DINOv2 token-feature 配置，warmup=2000 optimizer updates。
- `unified_video_action/config/uva_libero10_dinov2_small_token_feat_fully_frozen_action.yaml`：冻结 MAR、加载 action head 的运行配置。
- `unified_video_action/config/uva_libero10_dinov2_small_token_feat_video_pretrained_action.yaml`：加载 `libero10_video.ckpt`、不冻结 MAR，并启用 `conv_ori` action head 的运行配置。
- `unified_video_action/config/uva_libero10_dinov2_small_token_feat_video_pretrained_conv_fc_action.yaml`：当前运行配置；继承前者，但改用 `conv_fc` action head。
- `scripts/training/train_uva_libero10_dinov2_small_token_feat_fully_frozen_action.sh`：检查权重/数据、自动选择空闲 GPU，并保持 global batch 128。
- `scripts/verify_dinov2_token_feat.py`：DINO/student/projector 离线 smoke test。
- `scripts/verify_gradient_lr_equivalence.py`：CPU 双布局梯度和 LR 等价性 oracle。

## 5. 已完成验证

### DINO token shape 和反向图

```bash
CUDA_VISIBLE_DEVICES=0 \
  /data1/local_userdata/jinboning/conda/envs/repa/bin/python3.9 \
  scripts/verify_dinov2_token_feat.py --device cuda
```

结果：

```text
DINOv2 token-feature smoke test: PASS
teacher=(1, 2, 256, 384)
student=(1, 2, 256, 304)
projected=(1, 2, 256, 384)
```

### 梯度和 LR 等价性

```bash
/data1/local_userdata/jinboning/conda/envs/repa/bin/python3.9 \
  scripts/verify_gradient_lr_equivalence.py
```

结果：

```text
Gradient/LR equivalence: PASS
scheduler horizon: 971 updates/epoch for 124165 samples
direct-global gradient max error: 2.776e-17
post-6-update parameter max error: 2.711e-20
LR trajectory max error: 0.000e+00
```

此外已在单卡上用真实 Libero sample 完成完整 DINOv2 + student + MAR + `libero10.ckpt` 前向/反向：`loss=0.380333`，student/projector 均有非零梯度，DINO 和冻结 MAR 参数无梯度，结果 PASS。

## 6. 启动方式

启动脚本默认扫描 `nvidia-smi`，选择显存不超过 1 GB 且 GPU 利用率不超过 10% 的卡。当前检查时 0–7 卡均空闲，因此八卡布局为：

```text
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
8 processes × batch 16 × accumulation 1 = global batch 128
```

直接运行：

```bash
cd /home/jinboning/project/uva-bo-small
bash scripts/training/train_uva_libero10_dinov2_small_token_feat_fully_frozen_action.sh
```

若只想指定卡或改变输出目录，可覆盖环境变量：

```bash
GPU_IDS=0,1,2,3,4,5,6,7 \
RUN_NAME=uva_libero10_dinov2_small_token_feat_fully_frozen_action_manual \
bash scripts/training/train_uva_libero10_dinov2_small_token_feat_fully_frozen_action.sh
```

训练输出目录由 `RUN_DIR` 控制，默认在 `checkpoints/`（该目录是本机外部存储的 symlink，不会提交到 Git）。

本次已实际启动：

```text
tmux session: uva_dinov2_8gpu
run directory: checkpoints/uva_libero10_dinov2_small_token_feat_fully_frozen_action_20260808_230315
layout: 8 GPU × batch 16 × accumulation 1
```

启动后已稳定通过初始化并在 epoch 0 跑过 130 个以上 optimizer update，八张卡均在计算。可用下面命令查看：

```bash
tmux attach -t uva_dinov2_8gpu
# 或不进入会话：
tmux capture-pane -pt uva_dinov2_8gpu:0 -S -80
```

## 7. 当前 video-pretrained、MAR 全量训练配置

由于 `checkpoints/libero10_video.ckpt` 是视频预训练权重（`predict_video=true`、
`predict_action=false`），本实验只需新增配置文件，不需要改训练脚本或模型代码：

```yaml
model:
  policy:
    freeze_mar: false
    keep_mar_pos_and_fake_trainable: false
    autoregressive_model_params:
      pretrained_model_path: checkpoints/libero10_video.ckpt
    action_model_params:
      predict_action: true
      act_model_type: conv_fc
    selected_training_mode: policy_model
training:
  resume: false
```

视频 checkpoint 不含 action head，因此 `conv_fc` action head 会随机初始化；MAR
主体从视频 checkpoint 加载后与 action head、student 和 alignment projector 一起训练。
`training.resume=false` 是必要的，避免误接上旧 frozen-MAR 或旧 scheduler 状态。

本次实际运行：

```text
config: unified_video_action/config/uva_libero10_dinov2_small_token_feat_video_pretrained_conv_fc_action.yaml
tmux session: uva_dinov2_4gpu_video_conv_fc
run directory: checkpoints/uva_libero10_dinov2_small_token_feat_video_pretrained_conv_fc_action_20260809_000346
layout: 4 GPU × batch 8 × accumulation 4 = global batch 128
```

查看训练：

```bash
tmux attach -t uva_dinov2_4gpu_video_conv_fc
# 或不进入会话：
tmux capture-pane -pt uva_dinov2_4gpu_video_conv_fc:0 -S -80
```

## 8. Git 同步状态

修改前基线已保存为 commit `1998d92`，并在最终修改前同步到 GitHub：

```bash
small/jepa2_1_token_feat_fully_frozen_mar -> 1998d92
```

当前 video-pretrained 配置和本文更新作为后续 commit 推送到同一分支，因此 GitHub
历史中仍能单独取出修改前基线与 frozen-MAR 版本。
