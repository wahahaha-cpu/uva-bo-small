# V-JEPA 2.1 对齐版本变更记录

> 本文档是 JEPA 对齐相关改动的统一记录。后续版本的设计、实现文件、验证结果和启动方式都追加在这里；不把训练产物、checkpoint 或无关脚本计入版本改动。

## 0. 版本总览

| 版本 | Git 分支 | 提交 | 对齐方式 | 状态 |
|---|---|---|---|---|
| V1 | `jepa2_1_latent_alignment` | `4ce6c11` | teacher JEPA token 投影到 16 维，与 student 最终 latent `z` 对齐 | 已完成 |
| V2 | `jepa2_1_token_feat_alignment` | `323ba5c` | student 最终 encoder `token_feat` 通过现有 `align_projector` 投影到 768 维，单方面对齐原始 JEPA token | 已完成 |
| V3 | `jepa2_1_token_feat_frozen_mar` | `2ef6f4f` | 保持 V2 token-feature 对齐，冻结 MAR blocks/heads，只训练 position embedding 与 fake/blank token 接口参数 | 已完成并保留 |
| V4 | `jepa2_1_token_feat_fully_frozen_mar` | 本分支 HEAD | 保持 V2 token-feature 对齐，严格冻结全部 MAR 参数，只训练 student 与 align projector | 已完成 |

### 0.1 版本保存约定

1. 每种对齐方案使用独立 Git 分支和独立 commit，保留上一版可复现状态。
2. 每次只提交代码、配置、启动脚本和本文档；不提交 `checkpoints`、数据集、缓存或无关脚本。
3. V2 是在 V1 基础上新增模式，不修改或删除 V1 的 `align_on: latent` 行为。
4. V3 在 V2 基础上只改变 MAR 参数的可训练范围，不改变 JEPA 对齐、loss、数据或推理结构。
5. V4 保留 V3 配置和脚本作为对照，新增严格全冻结配置；不覆盖任何旧实验入口。

## 1. V1 初始移植任务范围

- 参考仓库：`/home/jinboning/project/uva-bo`
- 修改目标：`/home/jinboning/project/uva-bo-small`
- 移植目标：把 `uva-bo` 中 V-JEPA 2.1 teacher 与 policy latent 对齐的必要逻辑移植到 `uva-bo-small`
- 核心约束：
  - 不修改现有动作生成结构
  - 不修改 MAR、action diffusion、轨迹切分和推理采样
  - 不引入 DINOv2、旧版 V-JEPA、token-feature JEPA 对齐或评估实验
  - 默认 VAE 对齐路径保持原行为
  - 只先支持 V-JEPA 2.1 ViT-Base/384 与本地 checkpoint

本次不是把 `uva-bo` 整个分支合并进来，而是从其实现中裁出 JEPA2.1 latent 对齐所需的最小闭环。

## 2. 文件映射

| 参考实现（`uva-bo`） | 目标实现（`uva-bo-small`） | 处理方式 |
|---|---|---|
| `unified_video_action/model/common/jepa_teacher.py` | `unified_video_action/model/common/jepa_teacher.py` | 只保留 JEPA2.1 Base、本地 checkpoint 与 token 提取 |
| `unified_video_action/policy/unified_video_action_policy.py` | `unified_video_action/policy/unified_video_action_policy.py` | 只加入 JEPA latent 对齐旁路 |
| `unified_video_action/config/uva_libero10_jepa_base_policy.yaml` | `unified_video_action/config/uva_libero10_jepa2_1_small.yaml` | 继承 small 现有配置，只覆盖 JEPA 必需项 |
| `scripts/training/train_uva_libero10_jepa_policy.sh` | `scripts/training/train_uva_libero10_jepa2_1_small.sh` | 结合 small 的运行环境，新增独立直接启动入口 |

## 3. 实现代码统计

以下统计不包含本文档：

| 文件 | 类型 | 新增 | 删除 |
|---|---|---:|---:|
| `unified_video_action/model/common/jepa_teacher.py` | 新文件 | 204 | 0 |
| `unified_video_action/policy/unified_video_action_policy.py` | 修改 | 116 | 4 |
| `unified_video_action/config/uva_libero10_jepa2_1_small.yaml` | 新文件 | 29 | 0 |
| `scripts/training/train_uva_libero10_jepa2_1_small.sh` | 新文件 | 76 | 0 |
| 合计 | 4 个文件 | 425 | 4 |

policy 中删除的 4 行没有删除原功能，只是把原 VAE teacher 的四行提取逻辑移入 `else` 分支，使 JEPA 与 VAE 根据 `teacher_type` 分流。

## 4. 新增 JEPA teacher

文件：`unified_video_action/model/common/jepa_teacher.py`

### 4.1 模型范围

只注册一个模型：

```python
"vjepa2_1_vit_base_384": (768, 384)
```

含义：

- JEPA token 维度为 768
- 原生输入分辨率为 384
- patch size 为 16 时，空间网格是 `24 × 24`
- 每帧得到 576 个空间 token

没有移植 `uva-bo` 中的旧 V-JEPA Large/Huge/Giant、JEPA2.1 Large/Giant/Gigantic，也没有移植自动下载权重的逻辑。

### 4.2 checkpoint 加载

配置使用：

```yaml
checkpoint_path: pretrained_models/jepa/vjepa2_1_vitb_dist_vitG_384.pt
checkpoint_key: ema_encoder
loader: checkpoint
```

加载逻辑：

1. 先检查 checkpoint 路径非空且文件存在。
2. 通过 V-JEPA2 torch hub 代码构建模型结构。
3. 从 checkpoint 的 `ema_encoder` 字段取权重。
4. 去掉权重名中的 `module.` 和 `backbone.` 前缀。
5. 以 `strict=False` 加载，并打印 missing/unexpected keys。

真实 checkpoint 已确认包含 `ema_encoder`，其中有 158 个参数项。

### 4.3 多进程初始化锁

四卡或两卡训练时，每个进程都会构建 policy。为避免多个进程同时初始化 torch hub 缓存，增加文件锁：

```text
~/.cache/torch/hub/locks/facebookresearch_vjepa2.lock
```

拿到锁的进程先完成模型结构加载，其他进程等待；锁最长等待 600 秒。

### 4.4 teacher 冻结

JEPA encoder 初始化后执行：

```python
self.encoder.eval()
for param in self.encoder.parameters():
    param.requires_grad = False
```

因此：

- JEPA 只产生监督目标
- JEPA 权重不进入优化器
- 对齐反向传播不会更新 JEPA
- 只有 student tokenizer、原 policy 和 teacher latent projector 会被更新

### 4.5 图像预处理与 token 提取

输入约定：

```text
x: [B, 3, T, 256, 256]
```

处理顺序：

1. policy 的 `process_data()` 已把原始 `[0, 1]` RGB 图像转换为 `[-1, 1]`，供 student/MAR 使用。
2. JEPA teacher 先将其恢复为 `(x + 1) * 0.5`，得到官方预处理要求的 `[0, 1]` RGB 值。
3. 每帧从 `256 × 256` 双线性缩放到 `384 × 384`。
4. 使用官方 V-JEPA 2.1 的 ImageNet mean/std：`(0.485, 0.456, 0.406)` / `(0.229, 0.224, 0.225)`。
5. 把每一帧复制成长度为 2 的短视频，匹配 JEPA 的 `tubelet_size=2`。
6. 将 `B × T` 个短视频一起送入 JEPA encoder。
7. 输出恢复为 `[B, T, 576, 768]`。

对应形状：

```text
[B, 3, T, 256, 256]
  -> [B*T, 3, 2, 384, 384]
  -> [B*T, 576, 768]
  -> [B, T, 576, 768]
```

`extract_tokens()` 使用 `torch.no_grad()`，不会为 teacher 建立反向图。

## 5. policy 中的修改

文件：`unified_video_action/policy/unified_video_action_policy.py`

### 5.1 teacher 配置入口

新增两个 policy 参数：

```python
self.teacher_type = str(kwargs.get("teacher_type", "vae")).lower()
self.jepa_teacher_params = kwargs.get("jepa_teacher_params", {})
```

`teacher_type` 默认仍为 `vae`，所以所有旧配置在不指定 JEPA 时继续走原逻辑。

V1 初始版本的约束：

- `teacher_type` 只允许 `vae` 或 `jepa`
- JEPA 必须启用 student tokenizer
- V1 只允许 JEPA 使用 `align_on=latent`

这些约束用于尽早发现配置错误，防止看似启动成功、实际却没有对齐最终 latent。V2 在保留该分支的同时新增 `align_on=token_feat`，见第 14 节。

### 5.2 对齐对象

原 VAE 对齐使用 student 最终 encoder 输出的 token feature：

```text
token_feat: [B, T, 256, 304]
```

JEPA 对齐改为使用 student 最终输出、也就是实际传给 MAR/action policy 的 latent：

```text
latent: [B, T, 16, 16, 16]
  -> flatten spatial
[B, T, 256, 16]
```

这里没有生成一套新的动作输入。用于对齐的 `z` 和 `c` 就是原本继续传入：

```python
self.model(z, c, history_trajectory, trajectory, ...)
```

的同一份 latent。

### 5.3 teacher latent projector

JEPA 输出是 768 维，而现有 student/action latent 是 16 维，因此新增 teacher 侧可训练 MLP：

```text
768
 -> Linear(768, 512)
 -> SiLU
 -> Linear(512, 512)
 -> SiLU
 -> Linear(512, 16)
```

输出：

```text
[B, T, 576, 768] -> [B, T, 576, 16]
```

配置中的：

```yaml
use_projector: false
```

表示关闭原来作用在 student token-feature 上的 projector；JEPA 的 teacher projector 仍然必须存在，因为 768 维和 16 维无法直接计算对齐损失。

### 5.4 空间 token 对齐

JEPA2.1 Base 的空间网格为：

```text
24 × 24 = 576 tokens
```

small student latent 的空间网格为：

```text
16 × 16 = 256 tokens
```

`_resample_teacher_tokens()` 将 teacher token 恢复成二维特征图：

```text
[B, T, 576, 16]
 -> [B*T, 16, 24, 24]
 -> bilinear interpolate
 -> [B*T, 16, 16, 16]
 -> [B, T, 256, 16]
```

函数同时检查：

- teacher/student 的 batch 是否一致
- 时间长度是否一致
- token 数是否能组成正方形网格

### 5.5 condition 与 target 同时对齐

训练输入在原 policy 中被分成：

- `c_img`：condition frames
- `x_img`：target frames

student 产生：

- `c`：condition latent
- `z`：target latent

JEPA 对两部分分别提取 teacher token：

```text
x_img -> JEPA -> teacher_z_tokens
c_img -> JEPA -> teacher_c_tokens
```

分别计算：

```text
align_z = alignment(z, teacher_z)
align_c = alignment(c, teacher_c)
```

最后取平均：

```text
align_loss = 0.5 * (align_z + align_c)
```

### 5.6 loss 逻辑

沿用仓库已有的 alignment loss，没有新造损失函数。

配置为：

```yaml
coeff: 0.05
loss_type: hybrid
mse_coeff: 0.25
stats_coeff: 0.1
```

每一侧的对齐损失包括：

```text
cosine_loss = 1 - mean(cosine(student, teacher))
mse_loss    = MSE(student, teacher)
stats_loss  = MSE(student_mean, teacher_mean)
            + MSE(student_std, teacher_std)

base_loss   = 0.5 * (cosine_loss + mse_loss)   # hybrid
side_loss   = base_loss
            + 0.25 * mse_loss
            + 0.10 * stats_loss
```

总训练损失：

```text
original_loss = 原视频/动作模型损失
align_loss    = 0.5 * (target_side_loss + condition_side_loss)

final_loss = original_loss + 0.05 * align_loss
```

原动作 loss 的计算方式、权重和结构没有改变，只是在最终 loss 上增加一个辅助项。

### 5.7 优化器与 DDP

新增的 `teacher_latent_projector` 被加入 AdamW 参数组，因此它能够学习把 JEPA 表征映射到当前 16 维 latent 空间。

冻结的 JEPA encoder 不会进入参数组。

同时沿用现有 DDP 安全模式，为 projector 参数附加数值为零的图连接，避免某些任务分支未使用参数时触发 DDP unused-parameter 问题。该项数值恒为零，不改变 loss。

## 6. 完整数据流

```text
                          ┌──────────────────────────────────────┐
RGB condition c_img ─────┤ small StudentLatentTokenizer          │
                          │ -> c [B,T,16,16,16]                  │
                          └──────────────────────────────────────┘
                                           │
                                           ├────> 原 MAR/action policy（保持不变）
                                           │
                                           └────> flatten -> [B,T,256,16]

                          ┌──────────────────────────────────────┐
RGB target x_img ─────────┤ small StudentLatentTokenizer          │
                          │ -> z [B,T,16,16,16]                  │
                          └──────────────────────────────────────┘
                                           │
                                           ├────> 原 MAR/action policy（保持不变）
                                           │
                                           └────> flatten -> [B,T,256,16]

c_img / x_img
  -> frozen V-JEPA2.1 Base
  -> [B,T,576,768]
  -> trainable teacher projector
  -> [B,T,576,16]
  -> spatial resample 24x24 -> 16x16
  -> [B,T,256,16]
  -> 与 student c/z latent 计算辅助对齐损失
```

关键点是：JEPA 分支只提供训练监督，不插入推理动作生成路径。

## 7. 新增实验配置

文件：`unified_video_action/config/uva_libero10_jepa2_1_small.yaml`

配置继承：

```yaml
defaults:
  - uva_libero10_student
  - override /model/student_tokenizer: small
```

因此复用所有现有 Libero10 policy/action 设置，只覆盖以下内容：

- student 使用 `small`：
  - `hidden_dim=304`
  - `depth=5`
  - `latent_channels=16`
- teacher 使用 `vjepa2_1_vit_base_384`
- checkpoint 使用 `ema_encoder`
- `align_on=latent`
- 日志项目使用 `uva-repa-jepa`

Hydra 展开后已经确认：

```text
predict_action: true
selected_training_mode: policy_model
teacher_type: jepa
model_name: vjepa2_1_vit_base_384
align_on: latent
student hidden_dim: 304
student latent_channels: 16
```

## 8. 明确未修改的内容

以下文件和逻辑没有修改：

- `unified_video_action/model/autoregressive/mar_con_unified.py`
- `unified_video_action/model/autoregressive/diffusion_action_loss.py`
- `unified_video_action/model/autoregressive/diffusion_loss.py`
- `unified_video_action/utils/data_utils.py`
- `predict_action()`
- `sample_tokens()`
- `get_trajectory()`
- action normalization/unnormalization
- MAR token 拼接顺序
- action diffusion 网络结构
- action diffusion 训练与采样步数
- student tokenizer 网络结构
- dataset、workspace、eval 与 env runner
- 所有原有默认 YAML 配置

对上述四个动作核心文件执行 `git diff --quiet` 的返回值为 0。

## 9. V1 没有移植的 uva-bo 内容

为满足最小改动要求，以下内容没有带入：

- DINOv2 teacher
- JEPA token-feature 对齐（V1 未包含；V2 后续作为独立本地增量实现）
- VAE latent-policy 实验配置
- 旧版 V-JEPA 模型
- JEPA2.1 Large/Giant/Gigantic
- 自动下载 checkpoint 的 setup 脚本
- shuffle alignment 评估脚本
- 可视化与实验结果
- workspace 中与 resume、normalizer、grad norm、rollout 有关的改动
- MAR/action diffusion 中的任何参考分支改动

## 10. 验证结果

已完成：

1. Python 语法检查通过。
2. `git diff --check` 通过。
3. Hydra 配置完整展开通过。
4. 本地 JEPA2.1 checkpoint 路径存在。
5. V-JEPA2 torch hub 代码缓存存在。
6. CLIP tokenizer/model 本地缓存存在。
7. Libero10 的 10 个 HDF5 数据文件与 zip cache 存在。
8. 真实 JEPA2.1 Base checkpoint 成功加载。
9. teacher 参数全部冻结。
10. 真实单帧前向输出为：

```text
[1, 1, 576, 768]
```

11. JEPA 投影与重采样后的对齐形状为：

```text
[2, 4, 256, 16]
```

12. 反向测试确认：
    - student 有梯度
    - teacher latent projector 有梯度
    - frozen JEPA teacher 无梯度
13. 原 VAE 对齐 loss 的回归梯度检查通过。
14. 动作核心文件无 diff。
15. 新启动脚本通过 `bash -n`。
16. 使用 `ACCELERATE_BIN=/bin/echo` 完成无训练 dry-run，启动参数展开正确。

尚未执行完整的多 GPU 长时间训练或 rollout。验证覆盖的是配置、真实权重加载、真实 JEPA 前向和对齐反传闭环。

## 11. 正确启动方式

### 11.1 直接启动脚本（默认 GPU 4、5、6、7）

新增的可执行脚本：

```text
scripts/training/train_uva_libero10_jepa2_1_small.sh
```

默认参数：

```text
GPU_IDS=4,5,6,7
NUM_PROCESSES=4
PER_DEVICE_BATCH=8
GRAD_ACCUM_STEPS=4
EXPECTED_GLOBAL_BATCH=128
LR_WARMUP_STEPS=2000
```

直接运行：

```bash
./scripts/training/train_uva_libero10_jepa2_1_small.sh
```

脚本固定选择：

```text
--config-name=uva_libero10_jepa2_1_small.yaml
```

并在启动前检查：

- GPU ID 数量是否等于进程数
- global batch 是否等于 128
- VAE checkpoint 是否存在
- MAR warm-start checkpoint 是否存在
- JEPA2.1 checkpoint 是否存在
- Libero10 数据目录是否存在
- Accelerate 可执行文件是否存在

每次启动默认生成带时间戳的新目录。也可以显式指定：

```bash
RUN_NAME=my_jepa2_1_run \
./scripts/training/train_uva_libero10_jepa2_1_small.sh
```

虽然默认卡号是 4、5、6、7，仍可以通过环境变量整体覆盖。例如：

```bash
GPU_IDS=0,1,2,3 \
./scripts/training/train_uva_libero10_jepa2_1_small.sh
```

修改 GPU 数量时，需要同时调整 `NUM_PROCESSES`、`PER_DEVICE_BATCH`，并保持预期 global batch 一致。

### 11.2 两卡手动启动

配置默认：

```text
2 GPUs × batch 16 × accumulation 4 = global batch 128
```

选择两张空闲 GPU 后运行：

```bash
CUDA_VISIBLE_DEVICES=<gpu_a>,<gpu_b> \
/data1/local_userdata/jinboning/conda/envs/repa/bin/accelerate launch \
  --num_processes=2 \
  train.py \
  --config-dir=unified_video_action/config \
  --config-name=uva_libero10_jepa2_1_small.yaml \
  hydra.run.dir=checkpoints/uva_libero10_jepa2_1_small_run1
```

每次实验应使用新的 `hydra.run.dir`，避免复用旧 checkpoint 或 normalizer。

### 11.3 原有四卡脚本

现有未跟踪脚本：

```text
scripts/training/train_uva_libero10_small_aligned.sh
```

不是本次创建或修改的文件。它目前仍指定：

```yaml
--config-name=uva_libero10.yaml
logging.project=uva-repa-new
```

因此不能原样用于 JEPA2.1 实验。若复用该脚本，至少要改成：

```diff
--- --config-name=uva_libero10.yaml
+++ --config-name=uva_libero10_jepa2_1_small.yaml

--- logging.project=uva-repa-new
+++ logging.project=uva-repa-jepa
```

该脚本已有：

```text
4 GPUs × batch 8 × accumulation 4 = global batch 128
```

所以其 batch 设置可继续使用。

## 12. 运行前依赖

- VAE：`pretrained_models/vae/kl16.ckpt`
- MAR warm start：`checkpoints/libero10_video.ckpt`
- JEPA2.1 Base：`pretrained_models/jepa/vjepa2_1_vitb_dist_vitG_384.pt`
- 数据：`data/libero_10`
- 数据 cache：`data/libero_10_clip.zarr.zip`
- Python/Accelerate：
  `/data1/local_userdata/jinboning/conda/envs/repa/bin/accelerate`
- `einops` 已在环境声明中
- torch hub 中需要已有 V-JEPA2 代码缓存，或机器能够首次获取该代码
- 必须选择显存足够且允许使用的空闲 GPU

## 13. 工作区中原有的未跟踪项

开始移植前，`uva-bo-small` 已存在：

```text
?? checkpoints
?? scripts/training/train_uva_libero10_small_aligned.sh
```

本次没有修改这两个项目；实现代码统计也没有把它们算入本次改动。

## 14. V2：仿照 VAE 的 JEPA token-feature 单侧对齐

### 14.1 最终需求与对齐职责

V2 的核心要求是仿照仓库原有 VAE 对齐方式：teacher 保持冻结且不经过可训练 projector，只让 student 通过现有 `align_projector` 学习贴近 teacher。

三条路径的区别如下：

| 路径 | student 对齐输入 | student projector | teacher 对齐目标 | teacher projector |
|---|---|---|---|---|
| 原 VAE | 最终 encoder `token_feat [B,T,256,304]` | `304 -> 512 -> 512 -> 16` | VAE latent `[B,T,256,16]` | 无 |
| V1 JEPA latent | latent `z/c [B,T,256,16]` | 无 | JEPA token 投影后的 `[B,T,256,16]` | `768 -> 512 -> 512 -> 16` |
| V2 JEPA token feature | 最终 encoder `token_feat [B,T,256,304]` | `304 -> 512 -> 512 -> 768` | 原始 JEPA token `[B,T,256,768]` | 无 |

这里的 `token_feat` 不是浅层或中间层特征。它在 tokenizer 中经过了全部编码层：

```text
stem -> patch_embed -> all Transformer blocks -> LayerNorm
     -> optional temporal mixer residual -> token_feat
     -> out_proj(304 -> 16) -> latent
```

`token_feat` 是最终 encoder 表征；称它位于 `out_proj` 前，只是在区分 304 维 encoder 输出和 16 维 latent head 输出。

因此 V2 明确满足：

- 只投影 student，不投影 JEPA teacher。
- 直接复用已有 `align_projector` 属性、优化器参数组、DDP 图连接和 `_compute_alignment_loss()`。
- 不新增 `teacher_token_projector`。
- V2 中 `teacher_latent_projector is None`。
- V1 中 `align_projector is None`，原 `teacher_latent_projector` 及其 16 维输出保持不变。

### 14.2 V2 完整数据流

condition 和 target 两侧仍分别对齐。以 target `x_img` 为例：

```text
x_img
  -> StudentLatentTokenizer
  -> z_token_feat [B,T,256,304]
  -> align_projector
       Linear(304,512) -> SiLU
       Linear(512,512) -> SiLU
       Linear(512,768)
  -> student_z_aligned [B,T,256,768]

x_img
  -> frozen V-JEPA2.1 Base
  -> raw teacher_z_tokens [B,T,576,768]
  -> reshape [B*T,768,24,24]
  -> bilinear interpolate to 16x16
  -> teacher_z_tokens [B,T,256,768]

student_z_aligned <-> teacher_z_tokens
  -> existing cosine/MSE/statistics alignment loss
```

condition `c_img` 同样产生 `c_token_feat` 和原始 `teacher_c_tokens`，最终：

```text
align_loss = 0.5 * (align_z + align_c)
final_loss = original_loss + align_coeff * align_loss
```

空间 `24x24 -> 16x16` 插值只匹配 token 网格，不改变特征维度，也不是可训练 projector。JEPA encoder 的 `extract_tokens()` 仍在 `torch.no_grad()` 下执行，teacher 参数保持冻结。

student tokenizer 产生的最终 `z` 和 `c` 仍原样传入 MAR/action policy；V2 使用最终 encoder `token_feat`（`out_proj` 前）计算辅助 loss，没有改变动作生成、rollout 或推理路径。

### 14.3 policy 实现

修改文件：`unified_video_action/policy/unified_video_action_policy.py`

具体修改：

1. JEPA 的 `align_on` 合法值从仅 `latent` 扩展为 `latent` 或 `token_feat`。
2. 当 `align_on=latent` 时，保持 V1：创建 `teacher_latent_projector(768 -> 16)`，关闭 student `align_projector`。
3. 当 `align_on=token_feat` 时，保持 teacher 原始 768 维 token，不创建 `teacher_latent_projector`，启用已有 student `align_projector`。
4. `align_projector` 对原 VAE teacher 仍输出 16 维；仅当 teacher 是 JEPA 且对齐 token feature 时输出 `self.jepa_teacher.feat_dim=768`。
5. compute loss 中把局部变量改为 `z_token_feat`、`c_token_feat`、`student_z_tokens`、`student_c_tokens`，避免再把最终 encoder token feature 和 latent head 输出混用。
6. teacher token 的空间重采样继续复用 `_resample_teacher_tokens()`。
7. loss、metrics、optimizer 和 DDP safety 继续复用现有代码，没有新增第二套逻辑。

本版 policy 相对 V1 为 51 行新增、29 行删除；删除主要来自把单一路径改成显式的 `latent/token_feat` 分支，并非删除 V1 功能。

### 14.4 V2 配置

新增文件：`unified_video_action/config/uva_libero10_jepa2_1_small_token_feat.yaml`（20 行）。

它继承 V1 配置：

```yaml
defaults:
  - uva_libero10_jepa2_1_small
  - _self_
```

只覆盖实验名和以下对齐开关：

```yaml
model:
  policy:
    align_params:
      align_on: token_feat
      use_projector: true
```

因此 JEPA checkpoint、ImageNet 预处理、small student、alignment loss 系数、动作模型和数据设置均继承 V1，不复制配置。

### 14.5 V2 训练脚本

新增可执行文件：`scripts/training/train_uva_libero10_jepa2_1_small_token_feat.sh`（76 行）。

默认参数：

```text
GPU_IDS=4,5,6,7
NUM_PROCESSES=4
PER_DEVICE_BATCH=8
GRAD_ACCUM_STEPS=4
EXPECTED_GLOBAL_BATCH=128
LR_WARMUP_STEPS=2000
training.resume=False
```

脚本固定使用：

```text
--config-name=uva_libero10_jepa2_1_small_token_feat.yaml
```

默认运行目录带 `uva_libero10_jepa2_1_small_token_feat_<timestamp>`，不会与 V1 或其他实验复用 checkpoint。脚本保留 V1 的 GPU 数量、global batch、依赖路径和 Accelerate 检查。

启动命令：

```bash
./scripts/training/train_uva_libero10_jepa2_1_small_token_feat.sh
```

### 14.6 V2 验证结果

已完成并通过：

1. policy 使用项目 Conda Python 执行 `py_compile`。
2. 新脚本执行 `bash -n`。
3. Hydra 完整合成 V2 配置，确认 `teacher_type=jepa`、`align_on=token_feat`、`use_projector=true`、`hidden_dim=304`、`latent_channels=16`。
4. 使用 `ACCELERATE_BIN=/bin/echo` 做无训练 dry-run，确认 4 进程、单卡 batch 8、累积 4、global batch 128、warmup 2000、resume false 和独立 run directory。
5. `git diff --check` 通过。
6. 轻量 policy 构造断言通过：
   - V2：`align_projector` 输入 304、输出 768，`teacher_latent_projector=None`。
   - V1：`align_projector=None`，`teacher_latent_projector` 输出 16。
   - 原 VAE：`align_projector` 输入 304、输出 16，teacher projector 不存在。
7. V2 形状检查通过：teacher `[1,2,576,768] -> [1,2,256,768]`，student `[1,2,256,304] -> [1,2,256,768]`。
8. V2、V1 和原 VAE 三条路径的 alignment loss 反向传播均通过；梯度落到各自应训练的 student/projector 参数。

尚未执行完整多 GPU 长时间训练或 rollout；当前验证覆盖语法、配置、启动展开、三种模块结构、关键维度和 loss 反向闭环。

### 14.7 V2 改动边界

V2 没有修改：

- JEPA teacher 实现与图像预处理。
- student tokenizer 网络结构。
- MAR、action diffusion、trajectory、rollout、eval 或 dataset。
- V1 配置和 V1 启动脚本。
- 用户原有未跟踪的 `checkpoints` 与 `scripts/training/train_uva_libero10_small_aligned.sh`。

V2 只在分支 `jepa2_1_token_feat_alignment` 本地保存；在用户明确要求前不推送远端。

## 15. V3：JEPA token-feature 对齐与选择性冻结 MAR

### 15.1 实验目的

V3 用于检验：当下游 MAR 主体和动作头不再更新时，action loss 与 JEPA token-feature alignment 是否能更集中地优化 student tokenizer。

V3 完整继承 V2：

- student 使用最终 encoder `token_feat [B,T,256,304]`。
- 现有 student `align_projector` 执行 `304 -> 512 -> 512 -> 768`。
- frozen JEPA teacher 保持原始 768 维 token，只做 `24x24 -> 16x16` 空间插值。
- condition 与 target 两侧的 alignment loss、系数和 metrics 不变。
- MAR/action 的前向结构、loss 计算、masking、dropout、EMA、rollout 和推理逻辑不变。

V3 唯一改变的是参数的 `requires_grad` 范围。

### 15.2 为什么不是整个 MAR 一刀切冻结

用户要求冻结 MAR encoder、decoder、动作头等重型结构，但保留 position embedding 和可学习的 fake/blank token。原因是这些参数承担输入位置、mask、空白 target/action 和 classifier-free guidance 的接口语义，可以少量适配新的 student 表征。

这意味着 V3 不是“只有 student 可学习”，而是：

```text
可学习：student tokenizer
       + student align_projector
       + MAR position embeddings
       + MAR fake/blank tokens

冻结：MAR 输入/条件线性投影
     + encoder blocks/norm
     + decoder embed/blocks/norm
     + video diffusion head
     + action diffusion/head
     + 其他 MAR 参数
```

位置向量和 fake token 不会清零或重新初始化。policy 先加载 `checkpoints/libero10_video.ckpt`，然后冻结参数并重新打开白名单，因此这些参数从 pretrained MAR 的值继续训练。

### 15.3 MAR 可学习参数白名单

新增判断 `_is_mar_pos_or_fake_parameter()`，当前规则允许：

- 参数叶子名以 `fake_` 开头。
- 参数叶子名以 `_pos_embed` 结尾。
- `diffusion_temporal_embed` 与 `diffusion_spatial_embed`。
- 为未来兼容预留的 `mask_token` 与 `blank_token`。

按当前 Libero10 配置（CLIP language、无 wrist/history/proprio 分支），真实 MAR 白名单是：

```text
fake_latent_x
fake_action_latent
fake_latent
temporal_pos_embed
spatial_pos_embed
text_pos_embed
decoder_temporal_pos_embed
decoder_spatial_pos_embed
decoder_text_pos_embed
diffusion_temporal_embed
diffusion_spatial_embed
```

以下容易混淆的模块仍然冻结：

```text
z_proj / z_proj_cond / action_proj_cond
text_proj_cond / proj_cond_x_layer / z_proj_ln
encoder_blocks / encoder_norm
decoder_embed / decoder_blocks / decoder_norm
diffloss / diffactloss
```

特意冻结这些线性适配层和 head，是为了避免 action loss 主要被下游大模块吸收，削弱对 student tokenizer 的训练压力。

### 15.4 action loss 的真实梯度路径

实现只修改 MAR 参数的 `requires_grad`，没有对 MAR 前向使用 `torch.no_grad()`，也没有 detach student 的 `z/c`。因此冻结的线性层、encoder、decoder 和 action head 仍对输入可微。

当前配置固定：

```text
selected_training_mode: policy_model
```

在 `policy_model` 分支中，MAR 的实际 action 梯度路径是：

```text
c_img
  -> student tokenizer
  -> condition latent c
  -> frozen z_proj_cond
  -> trainable position/fake interface params
  -> frozen encoder
  -> frozen decoder
  -> frozen action diffusion/head
  -> action loss
  -> gradient 返回 c 和共享的 student tokenizer 参数
```

需要特别注意：`policy_model` 中 target `x/z` 在 encoder 输入处由 `fake_latent_x` 替代，video loss 为 0，因此 action loss 不直接依赖 target latent `z`。target `z` 仍通过 JEPA alignment loss 获得监督；condition `c` 同时获得 action loss 和 JEPA alignment loss。

冻结 MAR 不会改变上述依赖关系，只会阻止 frozen MAR 参数更新。

### 15.5 policy 开关与兼容性

修改文件：`unified_video_action/policy/unified_video_action_policy.py`。

新增两个默认关闭的配置项：

```yaml
freeze_mar: false
keep_mar_pos_and_fake_trainable: false
```

V3 覆盖为：

```yaml
freeze_mar: true
keep_mar_pos_and_fake_trainable: true
```

执行顺序：

1. 构建 MAR。
2. 加载 pretrained MAR checkpoint。
3. `self.model.requires_grad_(False)` 冻结所有 MAR 参数。
4. 按白名单重新设置 position/fake 参数的 `requires_grad=True`。
5. 保存 `mar_trainable_parameter_names` 供检查。
6. 若要求保留白名单但没有匹配到任何参数，立即报错，防止未来参数重命名后静默全冻结。

默认值均为 false，所以 V1、V2、原 VAE 和所有旧配置的 MAR 可训练行为不变。

### 15.6 优化器、DDP 与 EMA

现有 `add_weight_decay()` 已跳过 `requires_grad=False` 参数，因此：

- frozen encoder/decoder/action head 不进入 AdamW。
- student tokenizer、`align_projector` 和 MAR 白名单参数进入 AdamW。
- 原 DDP safety term 只连接仍可训练的 MAR 白名单参数。
- frozen MAR 参数在 EMA step 中直接复制当前值；白名单、student 和 projector 继续做 EMA。

没有修改 workspace、optimizer 类、scheduler 或 EMA 实现。

### 15.7 V3 配置和脚本

新增配置：`unified_video_action/config/uva_libero10_jepa2_1_small_token_feat_frozen_mar.yaml`（23 行）。

它继承：

```yaml
defaults:
  - uva_libero10_jepa2_1_small_token_feat
  - _self_
```

新增脚本：`scripts/training/train_uva_libero10_jepa2_1_small_token_feat_frozen_mar.sh`（76 行，可执行）。

默认启动参数：

```text
GPU_IDS=0,1,2,3
NUM_PROCESSES=4
PER_DEVICE_BATCH=8
GRAD_ACCUM_STEPS=4
GLOBAL_BATCH=128
LR_WARMUP_STEPS=2000
training.resume=False
```

启动：

```bash
./scripts/training/train_uva_libero10_jepa2_1_small_token_feat_frozen_mar.sh
```

默认运行目录为 `checkpoints/uva_libero10_jepa2_1_small_token_feat_frozen_mar_<timestamp>`。

### 15.8 验证结果

已完成：

1. policy `py_compile` 通过。
2. V3 脚本 `bash -n` 通过。
3. Hydra 同时合成 V2/V3，确认 V2 默认 `freeze_mar=false`，V3 为 `freeze_mar=true`、`keep_mar_pos_and_fake_trainable=true`。
4. V3 继续是 `teacher_type=jepa`、`align_on=token_feat`、student hidden dim 304。
5. 脚本 dry-run 确认 GPU 0-3、4 进程、batch 8、累积 4、global batch 128、warmup 2000、resume false 和独立目录。
6. 轻量 MAR 白名单测试确认仅 position/fake/mask/blank 参数可训练。
7. 优化器测试确认 frozen encoder/decoder/action head 参数不在 AdamW 中，student、projector 和白名单参数在 AdamW 中。
8. 反向测试确认 action-like loss 能穿过 frozen MAR 到达 student；frozen 参数无梯度，白名单参数有梯度。
9. 旧 V2 mock policy 的全部 MAR 参数仍为可训练，兼容性检查通过。
10. `git diff --check` 通过。

尚未运行真实多 GPU 长时间训练或 rollout。

### 15.9 文件与改动边界

V3 源码改动（不含本文档）：

| 文件 | 类型 | 新增 | 删除 |
|---|---|---:|---:|
| `unified_video_action/policy/unified_video_action_policy.py` | 修改 | 47 | 0 |
| `unified_video_action/config/uva_libero10_jepa2_1_small_token_feat_frozen_mar.yaml` | 新增 | 23 | 0 |
| `scripts/training/train_uva_libero10_jepa2_1_small_token_feat_frozen_mar.sh` | 新增 | 76 | 0 |
| 合计 | 3 个文件 | 146 | 0 |

V3 没有修改 MAR 源文件、student tokenizer、JEPA teacher、dataset、workspace、action loss、rollout 或推理实现。

当前工作区中用户已有的 V2 配置 `logging.project` 和 V2 脚本 GPU 0-3 改动保持未暂存；`checkpoints` 与旧 aligned 脚本同样不纳入 V3 提交。

V3 位于本地分支 `jepa2_1_token_feat_frozen_mar`；用户明确要求前不推送远端。

### 15.10 逐处代码修改审核清单

本节以 V2 commit `323ba5c` 为基线，记录 V3 的全部代码修改。以下行号对应当前 V3 源文件；最终提交后以 `git diff 323ba5c..HEAD` 为权威差异。

#### 15.10.1 文件总览

| 文件 | 修改位置 | 内容 |
|---|---|---|
| `unified_video_action/policy/unified_video_action_policy.py` | 70-79 | 新增冻结开关、白名单名称缓存与非法组合检查 |
| 同上 | 221 | pretrained MAR 加载完成后调用冻结配置 |
| 同上 | 243-276 | 新增白名单判断与选择性冻结实现 |
| `unified_video_action/config/uva_libero10_jepa2_1_small_token_feat_frozen_mar.yaml` | 1-23 | 新增完整 V3 Hydra 配置 |
| `scripts/training/train_uva_libero10_jepa2_1_small_token_feat_frozen_mar.sh` | 1-76 | 新增完整 V3 启动脚本 |
| `docs/jepa2_1_latent_alignment_changes.md` | 版本表、第 15 节 | 新增 V3 设计、验证、运行与本审核清单 |

没有修改 `mar_con_unified.py`、workspace、optimizer、EMA、action loss、student tokenizer、JEPA teacher 或 dataset。

#### 15.10.2 policy：新增配置字段与校验

位置：`UnifiedVideoActionPolicy.__init__()` 第 70-79 行。

```python
self.freeze_mar = bool(kwargs.get("freeze_mar", False))
self.keep_mar_pos_and_fake_trainable = bool(
    kwargs.get("keep_mar_pos_and_fake_trainable", False)
)
self.mar_trainable_parameter_names = ()

if self.keep_mar_pos_and_fake_trainable and not self.freeze_mar:
    raise ValueError(
        "keep_mar_pos_and_fake_trainable=True requires freeze_mar=True."
    )
```

作用：两个开关默认都是 false，旧配置完全不受影响；禁止在不冻结 MAR 时单独开启白名单模式。

#### 15.10.3 policy：冻结调用时机

位置：`UnifiedVideoActionPolicy.__init__()` 第 213-221 行。

```python
self.pretrained_model_path = autoregressive_model_params.pretrained_model_path
if self.pretrained_model_path is not None:
    if os.path.exists(self.pretrained_model_path):
        self.load_pretrained_model()
    else:
        print("pretrained model not found: ", self.pretrained_model_path)

self._configure_mar_trainability()
```

关键点：冻结发生在 checkpoint 加载之后，position embedding 与 fake token 保留 pretrained 数值，不重新初始化。

#### 15.10.4 policy：完整白名单与冻结方法

位置：`UnifiedVideoActionPolicy` 第 243-276 行。

```python
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

def _configure_mar_trainability(self) -> None:
    if not self.freeze_mar:
        return

    # Keep the MAR graph differentiable with respect to student latents; only
    # parameter gradients are disabled for the frozen modules.
    self.model.requires_grad_(False)
    if self.keep_mar_pos_and_fake_trainable:
        for name, param in self.model.named_parameters():
            if self._is_mar_pos_or_fake_parameter(name):
                param.requires_grad = True

    self.mar_trainable_parameter_names = tuple(
        name for name, param in self.model.named_parameters() if param.requires_grad
    )
    if (
        self.keep_mar_pos_and_fake_trainable
        and not self.mar_trainable_parameter_names
    ):
        raise RuntimeError("No trainable MAR position or fake parameters were found.")
```

#### 15.10.5 未修改但直接决定行为的原代码

| 位置 | 原行为 | V3 中的结果 |
|---|---|---|
| policy 476-478 `add_weight_decay()` | 跳过 `requires_grad=False` 参数 | frozen MAR 自动不进入 AdamW |
| policy 496-504 `get_optimizer()` | 复用同一参数分组函数 | 白名单、student、align projector 进入 AdamW |
| policy 760-768 `self.model(...)` | 正常执行 MAR 前向 | 没有 `no_grad`，梯度仍返回 student latent |
| policy 777-779 DDP safety | 只连接 `requires_grad=True` 参数 | 只连接 MAR 白名单，不连接 frozen blocks/heads |
| workspace EMA step | frozen 参数直接复制、trainable 参数做 EMA | MAR 主体固定，白名单/student/projector 做 EMA |

#### 15.10.6 V3 配置全文

文件：`unified_video_action/config/uva_libero10_jepa2_1_small_token_feat_frozen_mar.yaml`。

```yaml
# V-JEPA 2.1 token-feature alignment with the MAR core and heads frozen.
name: uva_libero10_jepa2_1_small_token_feat_frozen_mar

defaults:
  - uva_libero10_jepa2_1_small_token_feat
  - _self_

model:
  policy:
    freeze_mar: true
    keep_mar_pos_and_fake_trainable: true

logging:
  project: uva-repa-jepa
  name: train_uva_libero10_jepa2_1_small_token_feat_frozen_mar
  tags:
    - train_uva_libero10_jepa2_1_small_token_feat_frozen_mar
    - libero10
    - jepa2_1_teacher
    - token_feat_alignment
    - frozen_mar
    - trainable_mar_pos_embed
    - trainable_mar_fake_tokens
```

#### 15.10.7 V3 脚本全部行段

文件：`scripts/training/train_uva_libero10_jepa2_1_small_token_feat_frozen_mar.sh`，整个 1-76 行均为新增。

| 行号 | 内容 |
|---|---|
| 1-2 | Bash 入口与 `set -euo pipefail` |
| 4-9 | CUDA、动态库与 `MUJOCO_EGL_DEVICE_ID=0` 默认环境 |
| 11-29 | GPU 0-3、4 进程、batch 8、累积 4、global batch 128 校验 |
| 31-42 | VAE、MAR checkpoint、JEPA checkpoint 与数据路径检查 |
| 44-54 | Accelerate 可执行文件选择和检查 |
| 56-62 | 独立的 frozen-MAR run name、run directory 与启动摘要 |
| 64-76 | Accelerate launch 参数和 V3 Hydra config 选择 |

最终实验参数与 launch 核心内容：

```bash
GPU_IDS="${GPU_IDS:-0,1,2,3}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-8}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-2000}"

RUN_NAME="${RUN_NAME:-uva_libero10_jepa2_1_small_token_feat_frozen_mar_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-checkpoints/${RUN_NAME}}"

--config-name=uva_libero10_jepa2_1_small_token_feat_frozen_mar.yaml
training.gradient_accumulate_every="${GRAD_ACCUM_STEPS}"
training.lr_warmup_steps="${LR_WARMUP_STEPS}"
training.resume=False
hydra.run.dir="${RUN_DIR}"
```

#### 15.10.8 推荐审核命令

V3 提交完成后，可用以下命令查看全部且仅包含 V3 的修改：

```bash
git diff 323ba5c..HEAD -- unified_video_action/policy/unified_video_action_policy.py
git diff 323ba5c..HEAD -- unified_video_action/config/uva_libero10_jepa2_1_small_token_feat_frozen_mar.yaml
git diff 323ba5c..HEAD -- scripts/training/train_uva_libero10_jepa2_1_small_token_feat_frozen_mar.sh
git diff 323ba5c..HEAD -- docs/jepa2_1_latent_alignment_changes.md
git show --stat --oneline HEAD
```

## 16. V4：JEPA token-feature 对齐与严格冻结全部 MAR 参数

### 16.1 决策与实验边界

V4 回答更严格的问题：当 pretrained MAR 中 **一个参数也不允许更新** 时，action loss 能否穿过 MAR 的普通可微前向，只训练 student tokenizer；JEPA token-feature loss 同时训练 student 与 `align_projector`。

V4 保留 V3，不覆盖原实验：

```text
V3:
    freeze_mar = true
    keep_mar_pos_and_fake_trainable = true
    MAR position/fake 参数可训练

V4:
    freeze_mar = true
    keep_mar_pos_and_fake_trainable = false
    MAR trainable parameter count = 0
```

V4 仍完整继承 V2 的对齐结构：

```text
student final token_feat [B,T,256,304]
  -> student align_projector 304 -> 512 -> 512 -> 768
  -> 对齐 frozen JEPA raw token [B,T,256,768]
```

V4 没有改变 dataset、图像归一化、student、JEPA teacher、alignment loss、MAR forward、action loss、optimizer 类、scheduler、EMA、rollout 或推理路径。

### 16.2 V4 全部文件改动总览

V4 基线是 commit `4c50e49`。本版全部文件改动如下：

| 文件 | 类型 | 精确位置 | 修改内容 |
|---|---|---:|---|
| `unified_video_action/config/uva_libero10_jepa2_1_small_token_feat_fully_frozen_mar.yaml` | 新增 | 1-23 | V4 Hydra 配置，严格冻结全部 MAR 参数 |
| `scripts/training/train_uva_libero10_jepa2_1_small_token_feat_fully_frozen_mar.sh` | 新增 | 1-76 | V4 独立启动脚本 |
| `scripts/verify_mar_whitelist_gradients.py` | 修改 | 75-91 | 测试 helper 增加“是否重开白名单”参数 |
| 同上 | 修改 | 211-214 | 原 V3 白名单测试显式传 `true` |
| 同上 | 修改 | 302-351 | 保留 V3 PASS，并新增 V4 零 MAR 参数梯度/optimizer/update 测试 |
| `docs/mar_whitelist_optimizer_and_gradient_testing_tutorial.md` | 修改 | 291-297 | 标记严格冻结方案为 V4 当前决策 |
| `docs/jepa2_1_latent_alignment_changes.md` | 修改 | 7-18、1175-1831 | 更新版本表并新增 V4 全部代码位置和验证记录 |

**V4 没有修改 `unified_video_action_policy.py`。** 严格冻结能力已经由 V3 的默认关闭白名单分支提供；V4 通过独立配置选中该分支。下面仍把所有决定冻结行为的 production 源码完整列出，便于逐行审核。

### 16.3 所有决定 MAR 冻结行为的 production 源码

#### 16.3.1 policy 配置字段

文件：`unified_video_action/policy/unified_video_action_policy.py` 第 70-79 行。

```python
self.freeze_mar = bool(kwargs.get("freeze_mar", False))
self.keep_mar_pos_and_fake_trainable = bool(
    kwargs.get("keep_mar_pos_and_fake_trainable", False)
)
self.mar_trainable_parameter_names = ()

if self.keep_mar_pos_and_fake_trainable and not self.freeze_mar:
    raise ValueError(
        "keep_mar_pos_and_fake_trainable=True requires freeze_mar=True."
    )
```

V4 解析结果：

```text
self.freeze_mar = True
self.keep_mar_pos_and_fake_trainable = False
self.mar_trainable_parameter_names 初始为 ()
```

两个字段默认都是 false，所以 V1、V2、原 VAE 配置不受 V4 影响。

#### 16.3.2 checkpoint 加载和冻结调用顺序

文件：同一 policy 第 213-221 行。

```python
self.pretrained_model_path = autoregressive_model_params.pretrained_model_path
if self.pretrained_model_path is not None:
    if os.path.exists(self.pretrained_model_path):
        self.load_pretrained_model()
    else:
        print('pretrained model not found: ', self.pretrained_model_path)

self._configure_mar_trainability()
```

顺序是：

```text
构建 MAR
  -> 加载 checkpoints/libero10_video.ckpt
  -> 冻结全部 MAR 参数
```

因此 fake token、position embedding 和所有 blocks/heads 都保留 checkpoint 数值，只是不再更新。没有清零或重新初始化。

#### 16.3.3 白名单判断函数

文件：同一 policy 第 243-255 行。

```python
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
```

该函数仍为 V3 保留，但 V4 的 `keep_mar_pos_and_fake_trainable=False`，所以 V4 不调用它重开任何参数。

#### 16.3.4 严格冻结执行方法

文件：同一 policy 第 257-276 行。

```python
def _configure_mar_trainability(self) -> None:
    if not self.freeze_mar:
        return

    # Keep the MAR graph differentiable with respect to student latents; only
    # parameter gradients are disabled for the frozen modules.
    self.model.requires_grad_(False)
    if self.keep_mar_pos_and_fake_trainable:
        for name, param in self.model.named_parameters():
            if self._is_mar_pos_or_fake_parameter(name):
                param.requires_grad = True

    self.mar_trainable_parameter_names = tuple(
        name for name, param in self.model.named_parameters() if param.requires_grad
    )
    if (
        self.keep_mar_pos_and_fake_trainable
        and not self.mar_trainable_parameter_names
    ):
        raise RuntimeError("No trainable MAR position or fake parameters were found.")
```

V4 逐行执行结果：

```text
第 258 行：freeze_mar=True，不 return
第 263 行：self.model.requires_grad_(False)，递归冻结全部 MAR Parameter
第 264 行：keep_mar_pos_and_fake_trainable=False，跳过整个白名单循环
第 269-271 行：MAR 可训练名称收集结果为 ()
第 272-276 行：keep=false，不触发“白名单没匹配”异常
```

关键保证：

```python
all(not param.requires_grad for param in self.model.parameters())
self.mar_trainable_parameter_names == ()
```

#### 16.3.5 optimizer 排除全部 MAR 参数

文件：同一 policy 第 472-511 行。

```python
def add_weight_decay(self, model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]
```

`get_optimizer()` 第 496-511 行先处理 MAR，再加入 student/projector：

```python
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
```

V4 中所有 MAR 参数都被 `continue` 跳过。AdamW 中仍包含：

```text
student tokenizer 参数
align_projector 参数
```

V4 的 token-feature 模式不创建 `teacher_latent_projector`。

#### 16.3.6 action loss 仍穿过 frozen MAR

文件：同一 policy 第 760-770 行。

```python
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
```

这里仍是普通 MAR forward。没有：

```python
torch.no_grad()
c.detach()
z.detach()
```

因此 V4 只是不给 MAR Parameter 累计梯度，autograd 仍使用 frozen MAR 权重计算 `d(action_loss)/d(c)`，并继续返回 student tokenizer。

当前固定 `selected_training_mode=policy_model`，实际路径是：

```text
c_img
  -> trainable student tokenizer
  -> c
  -> fully frozen MAR z_proj_cond
  -> fully frozen MAR encoder/decoder/action head
  -> action loss
  -> gradient 返回 c
  -> 更新 student tokenizer
```

target `z` 在 `policy_model` 下仍主要由 JEPA alignment loss 监督。

#### 16.3.7 DDP safety 不再连接任何 MAR 参数

文件：同一 policy 第 772-779 行。

```python
def _ddp_unused_term(param: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(
        param, nan=0.0, posinf=0.0, neginf=0.0
    ).sum().mul(0.0)

for param in self.model.parameters():
    if param.requires_grad:
        loss = loss + _ddp_unused_term(param)
```

V4 所有 MAR 参数 `requires_grad=False`，所以循环不会把任何 MAR 参数连接到 loss。student 和 align projector 的 DDP safety 循环仍正常执行。

#### 16.3.8 backward、optimizer step 和 EMA

文件：`unified_video_action/workspace/train_unified_video_action_workspace.py` 第 284-294 行。

```python
accelerator.backward(raw_loss)

if self.global_step % cfg.training.gradient_accumulate_every == 0:
    self.optimizer.step()
    self.optimizer.zero_grad()
    self.lr_scheduler.step()

if cfg.training.use_ema:
    ema.step(accelerator.unwrap_model(self.model))
```

V4 结果：

```text
backward:
    MAR Parameter.grad 全为 None
    student/projector 正常获得梯度

optimizer.step:
    MAR 不在 AdamW 中，逐元素不变
    student/projector 更新

EMA:
    frozen MAR 保持 source 数值
    student/projector 按原 EMA 逻辑更新
```

workspace 本身没有为 V4 修改。

### 16.4 严格冻结覆盖的 MAR 参数范围

V4 使用 `self.model.requires_grad_(False)`，不是名称黑名单，所以覆盖当前以及未来添加到 MAR 的全部 Parameter。当前范围包括：

```text
输入/条件投影:
    z_proj
    z_proj_cond
    action_proj_cond
    text_proj_cond
    proj_cond_x_layer
    z_proj_ln
    可选 wrist/history/proprioception 投影

所有 fake/null/mask 接口参数:
    fake_latent_x
    fake_action_latent
    fake_latent
    可选 fake_latent_wrist_x
    可选 fake_latent_history_action
    未来 mask_token / blank_token（如果加入 MAR）

所有 position embedding:
    temporal_pos_embed
    spatial_pos_embed
    text_pos_embed
    decoder_temporal_pos_embed
    decoder_spatial_pos_embed
    decoder_text_pos_embed
    diffusion_temporal_embed
    diffusion_spatial_embed

MAR 主体:
    encoder_blocks
    encoder_norm
    decoder_embed
    decoder_blocks
    decoder_norm

所有输出/loss head:
    diffloss
    diffactloss
    可选 diffloss_wrist
    可选 diffproploss
```

无论参数名称是否出现在上述说明中，只要属于 `self.model`，V4 都会冻结。

### 16.5 V4 配置完整内容

文件：`unified_video_action/config/uva_libero10_jepa2_1_small_token_feat_fully_frozen_mar.yaml` 第 1-23 行。

```yaml
# V-JEPA 2.1 token-feature alignment with every MAR parameter frozen.
name: uva_libero10_jepa2_1_small_token_feat_fully_frozen_mar

defaults:
  - uva_libero10_jepa2_1_small_token_feat
  - _self_

model:
  policy:
    freeze_mar: true
    keep_mar_pos_and_fake_trainable: false

logging:
  project: uva-repa-jepa
  name: train_uva_libero10_jepa2_1_small_token_feat_fully_frozen_mar
  tags:
    - train_uva_libero10_jepa2_1_small_token_feat_fully_frozen_mar
    - libero10
    - jepa2_1_teacher
    - token_feat_alignment
    - fully_frozen_mar
    - strict_mar_freeze
    - zero_trainable_mar_parameters
```

真正决定 V3/V4 差异的唯一配置行是第 11 行：

```yaml
keep_mar_pos_and_fake_trainable: false
```

### 16.6 V4 启动脚本完整内容

文件：`scripts/training/train_uva_libero10_jepa2_1_small_token_feat_fully_frozen_mar.sh` 第 1-76 行，全部为新增。

```bash
#!/usr/bin/env bash
set -euo pipefail

# Libero rollout needs MuJoCo/NVIDIA runtime libraries. Accelerate imports
# DeepSpeed while unwrapping the EMA model, which requires a CUDA toolkit root.
export CUDA_HOME="${CUDA_HOME:-/data1/local_userdata/jinboning/conda/envs/repa}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:/usr/lib/nvidia:/home/jinboning/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"

# Default experiment: GPUs 0,1,2,3 x batch 8 x accumulation 4 = global batch 128.
GPU_IDS="${GPU_IDS:-0,1,2,3}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-8}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
EXPECTED_GLOBAL_BATCH="${EXPECTED_GLOBAL_BATCH:-128}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-2000}"

IFS=',' read -r -a GPU_ID_ARRAY <<< "${GPU_IDS}"
if (( ${#GPU_ID_ARRAY[@]} != NUM_PROCESSES )); then
    echo "GPU_IDS count (${#GPU_ID_ARRAY[@]}) does not match NUM_PROCESSES (${NUM_PROCESSES})." >&2
    exit 2
fi

GLOBAL_BATCH=$((NUM_PROCESSES * PER_DEVICE_BATCH * GRAD_ACCUM_STEPS))
if (( GLOBAL_BATCH != EXPECTED_GLOBAL_BATCH )); then
    echo "Global batch mismatch: ${GLOBAL_BATCH}, expected ${EXPECTED_GLOBAL_BATCH}." >&2
    exit 2
fi

required_paths=(
    pretrained_models/vae/kl16.ckpt
    checkpoints/libero10_video.ckpt
    pretrained_models/jepa/vjepa2_1_vitb_dist_vitG_384.pt
    data/libero_10
)
for required_path in "${required_paths[@]}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "Required path not found: ${required_path}" >&2
        exit 1
    fi
done

if [[ -z "${ACCELERATE_BIN:-}" ]]; then
    if command -v accelerate >/dev/null 2>&1; then
        ACCELERATE_BIN="accelerate"
    else
        ACCELERATE_BIN="/data1/local_userdata/jinboning/conda/envs/repa/bin/accelerate"
    fi
fi
if [[ "${ACCELERATE_BIN}" != "accelerate" && ! -x "${ACCELERATE_BIN}" ]]; then
    echo "Accelerate executable not found: ${ACCELERATE_BIN}" >&2
    exit 1
fi

RUN_NAME="${RUN_NAME:-uva_libero10_jepa2_1_small_token_feat_fully_frozen_mar_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-checkpoints/${RUN_NAME}}"

echo "JEPA config: uva_libero10_jepa2_1_small_token_feat_fully_frozen_mar.yaml"
echo "GPUs: ${GPU_IDS}"
echo "Global batch: ${NUM_PROCESSES} x ${PER_DEVICE_BATCH} x ${GRAD_ACCUM_STEPS} = ${GLOBAL_BATCH}"
echo "Run directory: ${RUN_DIR}"

launch_args=(
    --num_processes="${NUM_PROCESSES}"
    train.py
    --config-dir=unified_video_action/config
    --config-name=uva_libero10_jepa2_1_small_token_feat_fully_frozen_mar.yaml
    dataloader.batch_size="${PER_DEVICE_BATCH}"
    val_dataloader.batch_size="${PER_DEVICE_BATCH}"
    training.gradient_accumulate_every="${GRAD_ACCUM_STEPS}"
    training.lr_warmup_steps="${LR_WARMUP_STEPS}"
    training.resume=False
    hydra.run.dir="${RUN_DIR}"
)
CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${ACCELERATE_BIN}" launch "${launch_args[@]}"
```

关键行：

```text
56: 独立 fully_frozen_mar run name
59: 输出 V4 配置名称
68: 选择 V4 Hydra 配置
73: resume=False，避免误接旧 optimizer/checkpoint 状态
```

默认启动命令：

```bash
./scripts/training/train_uva_libero10_jepa2_1_small_token_feat_fully_frozen_mar.sh
```

### 16.7 梯度验证脚本的全部 V4 修改

文件：`scripts/verify_mar_whitelist_gradients.py`。

#### 16.7.1 helper 支持严格模式，第 75-91 行

```python
def configure_mar_trainability(
    mar: nn.Module,
    keep_pos_and_fake_trainable: bool,
) -> tuple[str, ...]:
    """Freeze all MAR parameters and optionally reopen the interface whitelist."""
    mar.requires_grad_(False)
    if keep_pos_and_fake_trainable:
        for name, param in mar.named_parameters():
            if is_mar_pos_or_fake_parameter(name):
                param.requires_grad = True

    trainable_names = tuple(
        name for name, param in mar.named_parameters() if param.requires_grad
    )
    if keep_pos_and_fake_trainable and not trainable_names:
        raise RuntimeError("The MAR whitelist matched no parameters.")
    return trainable_names
```

#### 16.7.2 V3 测试显式保留白名单，第 211-214 行

```python
trainable_mar_names = configure_mar_trainability(
    mar,
    keep_pos_and_fake_trainable=True,
)
```

#### 16.7.3 V4 严格测试，第 304-351 行

```python
# Strict V4 mode: every MAR parameter, including fake/position parameters,
# is frozen and excluded from AdamW. The ordinary MAR forward must still
# propagate the action loss to the student condition representation.
strict_student = TinyStudent()
strict_mar = TinyMar()
strict_align_projector = nn.Linear(4, 4)
strict_trainable_mar_names = configure_mar_trainability(
    strict_mar,
    keep_pos_and_fake_trainable=False,
)
assert strict_trainable_mar_names == ()
assert not any(param.requires_grad for param in strict_mar.parameters())

strict_optimizer_groups: list[dict[str, object]] = []
for module in (strict_mar, strict_student, strict_align_projector):
    strict_optimizer_groups.extend(add_weight_decay(module, weight_decay=0.01))
strict_optimizer = torch.optim.AdamW(strict_optimizer_groups, lr=1e-2)
assert_optimizer_membership(
    strict_mar,
    strict_student,
    strict_align_projector,
    strict_optimizer,
)

strict_mar_before = clone_parameters(strict_mar)
strict_optimizer.zero_grad(set_to_none=True)
strict_action_loss = make_action_loss(
    strict_student,
    strict_mar,
    observation,
    action_target,
)
strict_action_loss.backward()

strict_student_grad = module_grad_norm(strict_student)
assert strict_student_grad > 0.0, (
    "Action loss did not cross the fully frozen MAR to reach the student."
)
assert all(param.grad is None for param in strict_mar.parameters())
assert module_grad_norm(strict_align_projector) == 0.0

strict_optimizer.step()
assert not changed_parameter_names(strict_mar, strict_mar_before)

print("Fully frozen MAR trainable parameter count: 0")
print(f"Fully frozen MAR student grad norm: {strict_student_grad:.6e}")
print("Fully frozen MAR optimizer/gradient/update checks: PASS")
print("PASS: selective and fully frozen MAR modes are both correct.")
```

该测试同时证明：

```text
MAR trainable count == 0
MAR 参数全部不在 AdamW
action-like loss 穿过 fully frozen MAR 到达 student
MAR Parameter.grad 全为 None
optimizer.step 后 MAR 参数逐元素完全不变
align projector 不接收 action-only loss
```

### 16.8 V4 预期参数集合

```text
MAR:
    trainable = 0
    optimizer membership = 0

student tokenizer:
    trainable > 0
    在 AdamW 中
    接收 action loss + JEPA alignment loss

align_projector:
    trainable > 0
    在 AdamW 中
    只接收 JEPA alignment loss

JEPA teacher:
    frozen
    不在 AdamW 中
```

### 16.9 验证结果

以下检查已实际执行并通过：

1. V4 Hydra 完整合成：`teacher_type=jepa`、`align_on=token_feat`、`freeze_mar=true`、`keep_mar_pos_and_fake_trainable=false`。
2. V3 Hydra 完整合成：`freeze_mar=true`、`keep_mar_pos_and_fake_trainable=true`，证明旧入口仍保留白名单行为。
3. `bash -n` 检查 V4 启动脚本：PASS。
4. V4 启动脚本以 `ACCELERATE_BIN=/bin/echo` dry-run：选择 V4 config，global batch 为 `4 x 8 x 4 = 128`，warmup 为 2000，`training.resume=False`。
5. Python 3.9 `py_compile` 检查梯度脚本：PASS。
6. CPU 反向测试：MAR 可训练参数数为 0，且全部 MAR `.grad is None`。
7. optimizer identity 测试：全部 MAR 参数均不在 AdamW，student 与 align projector 在 AdamW 中。
8. action-like loss 穿过 fully frozen MAR 后，student 梯度范数为 `1.354594e-02`，确认动作误差仍能回到 student。
9. action-only loss 下 align projector 梯度为 0；它只由 JEPA alignment loss 训练。
10. `optimizer.step()` 前后逐元素比较：全部 MAR 参数完全不变。

核心梯度测试输出：

```text
Fully frozen MAR trainable parameter count: 0
Fully frozen MAR student grad norm: 1.354594e-02
Fully frozen MAR optimizer/gradient/update checks: PASS
PASS: selective and fully frozen MAR modes are both correct.
```

提交前另执行 `git diff --check` 和 `git diff --cached --check`，结果记录在本版提交检查中。

### 16.10 审核命令

V4 提交后，以下命令会显示全部且仅包含 V4 的修改：

```bash
git diff 4c50e49..HEAD -- unified_video_action/config/uva_libero10_jepa2_1_small_token_feat_fully_frozen_mar.yaml
git diff 4c50e49..HEAD -- scripts/training/train_uva_libero10_jepa2_1_small_token_feat_fully_frozen_mar.sh
git diff 4c50e49..HEAD -- scripts/verify_mar_whitelist_gradients.py
git diff 4c50e49..HEAD -- docs/mar_whitelist_optimizer_and_gradient_testing_tutorial.md
git diff 4c50e49..HEAD -- docs/jepa2_1_latent_alignment_changes.md
git diff --name-status 4c50e49..HEAD
git show --stat --oneline HEAD
```

V4 分支：`jepa2_1_token_feat_fully_frozen_mar`。用户明确要求前不推送远端。
