# V-JEPA 2.1 对齐版本变更记录

> 本文档是 JEPA 对齐相关改动的统一记录。后续版本的设计、实现文件、验证结果和启动方式都追加在这里；不把训练产物、checkpoint 或无关脚本计入版本改动。

## 0. 版本总览

| 版本 | Git 分支 | 提交 | 对齐方式 | 状态 |
|---|---|---|---|---|
| V1 | `jepa2_1_latent_alignment` | `4ce6c11` | teacher JEPA token 投影到 16 维，与 student 最终 latent `z` 对齐 | 已完成 |
| V2 | `jepa2_1_token_feat_alignment` | 本分支 HEAD | student 最终 encoder `token_feat` 通过现有 `align_projector` 投影到 768 维，单方面对齐原始 JEPA token | 已完成 |

### 0.1 版本保存约定

1. 每种对齐方案使用独立 Git 分支和独立 commit，保留上一版可复现状态。
2. 每次只提交代码、配置、启动脚本和本文档；不提交 `checkpoints`、数据集、缓存或无关脚本。
3. V2 是在 V1 基础上新增模式，不修改或删除 V1 的 `align_on: latent` 行为。

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
