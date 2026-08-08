# `conv_ori` 与 `conv_fc` action conditioning 结构详解

## 1. 文档范围与结论

`conv_ori` 和 `conv_fc` 是 `DiffActLoss` 中的两种 **action conditioning adapter**。它们都接收 MAR decoder 输出的视觉 token，将其压缩/展开成 16 个 action condition token，再交给同一个 action diffusion MLP 预测动作。

它们不是两个完整、独立的 action diffusion head；二者只替换下图中间的 adapter：

```text
MAR decoder visual tokens
        [B, 4*256, C]
                 |
                 v
     conv_ori 或 conv_fc
                 |
                 v
action condition tokens
          [B, 16, C]
                 |
                 v
共享的 SimpleMLPAdaLN + action diffusion
                 |
                 v
        action [B, 16, A]
```

核心区别是：

- `conv_ori` 是较轻的线性适配器。它先用时域反卷积把每个输入帧展开为 4 个动作槽，再对空间做全局平均；不同输入帧产生的动作槽互不重叠。
- `conv_fc` 是更重、更非线性的适配器。它先提取每帧的粗粒度空间结构，再通过 `Linear(4 -> 16)` 让每一个动作槽显式融合全部 4 帧。
- 默认 `mar_base` 下 `C=768`，`conv_fc` adapter 约有 1652 万参数，`conv_ori` 约有 236 万参数，前者约为后者的 7 倍。这里不包含两者共享的 diffusion MLP。
- 两种结构输出形状相同，但参数名称和语义不同，checkpoint 不能直接互换。

主要实现位于：

- [`diffusion_action_loss.py`](../unified_video_action/model/autoregressive/diffusion_action_loss.py#L9)
- [`mar_con_unified.py`](../unified_video_action/model/autoregressive/mar_con_unified.py#L295)
- 默认配置 [`model/uva.yaml`](../unified_video_action/config/model/uva.yaml#L54)

## 2. 当前默认张量规格

以下形状来自当前默认 `mar_base` 和 Libero10 配置：

| 符号 | 当前值 | 来源 |
|---|---:|---|
| `B` | 每卡 batch size | DataLoader |
| `T` | 4 | `MAR.n_frames` 硬编码为 4 |
| `H, W` | 16, 16 | `img_size=256, vae_stride=16, patch_size=1` |
| `S` | 256 | `H * W` |
| `C` | 768 | `mar_base.decoder_embed_dim` |
| `L` | 16 | 训练动作序列长度 |
| `A` | Libero10 中为 10 | `task.shape_meta.action.shape` |

MAR 将 decoder 输出整理为：

```text
z: [B, T*S, C] = [B, 4*256, 768] = [B, 1024, 768]
```

Libero10 dataset 的 horizon 是 32；在默认 `shift_action=True`、不使用 history action 时，`get_trajectory()` 截取 16 个 future action，所以 action target 是：

```text
target: [B, L, A] = [B, 16, 10]
```

两种 adapter 都必须输出：

```text
condition: [B, L, C] = [B, 16, 768]
```

随后 condition 和 action target 都会展平到 batch/action-time 联合维度：

```text
condition -> [B*16, 768]
target    -> [B*16, 10]
```

共享的 `SimpleMLPAdaLN` 对这 `B*16` 行分别执行 diffusion denoising。它本身没有 action-time attention 或 action-time convolution；action 槽之间的时序信息是否提前融合，主要由 adapter 决定。

## 3. `conv_ori`：时域反卷积展开 + 全局空间平均

### 3.1 模块定义

```python
self.conv_transpose3d = nn.ConvTranspose3d(
    in_channels=C,
    out_channels=C,
    kernel_size=(4, 1, 1),
    stride=(4, 1, 1),
)
self.avg_pool = nn.AvgPool3d(kernel_size=(1, 16, 16))
```

### 3.2 逐层形状

```text
输入 z
[B, 4*256, C]

rearrange
[B, 4, 256, C]

恢复空间网格并把 channel 提到前面
[B, C, 4, 16, 16]

ConvTranspose3d(kernel=(4,1,1), stride=(4,1,1))
[B, C, 16, 16, 16]

AvgPool3d(kernel=(1,16,16))
[B, C, 16, 1, 1]

rearrange
[B, 16, C]
```

### 3.3 它实际学习了什么

时间维反卷积的 kernel 和 stride 都是 4，因此各输入帧在输出时间轴上没有重叠：

```text
输入 frame 0 -> action condition  0,  1,  2,  3
输入 frame 1 -> action condition  4,  5,  6,  7
输入 frame 2 -> action condition  8,  9, 10, 11
输入 frame 3 -> action condition 12, 13, 14, 15
```

每个 temporal kernel offset 都有独立的 `C -> C` channel mixing 矩阵，但：

- temporal kernel 不跨相邻输入帧重叠，所以 adapter 内没有显式跨帧融合；
- spatial kernel 是 `1x1`，不融合相邻空间位置；
- 后续 `16x16` 全局平均会丢掉显式空间布局；
- adapter 内没有 ReLU、SiLU 等非线性，因此整条 `conv_ori` adapter 是仿射映射。

更紧凑地写，对输入帧 `f` 和该帧对应的第 `r` 个动作槽，有近似形式：

```text
c[f*4+r] = W_r * mean_spatial(z[f]) + b_r
```

MAR decoder token 本身已经包含空间位置和上下文，因此全局平均后的表示并非完全没有空间信息；但 `conv_ori` adapter 不再显式保留“哪个特征来自哪个空间格点”。

### 3.4 参数量

只有 `ConvTranspose3d` 含参数：

```text
weight: C * C * 4 * 1 * 1
bias:   C

P_conv_ori = 4*C^2 + C
```

默认 `C=768`：

```text
P_conv_ori = 4*768^2 + 768
           = 2,360,064
```

## 4. `conv_fc`：逐帧空间编码 + 全帧时间展开

### 4.1 模块定义

```python
self.conv = nn.Sequential(
    nn.Conv2d(C, C, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d((4, 4)),
)

self.fc = nn.Sequential(
    nn.Linear(C * 4 * 4, C),
    nn.ReLU(),
    nn.Linear(C, C),
)

self.interpolate = nn.Linear(4, 16)

self.refine = nn.Sequential(
    nn.Linear(C, C),
    nn.ReLU(),
    nn.Linear(C, C),
)
```

### 4.2 逐层形状

```text
输入 z
[B, 4*256, C]

按帧拆开并恢复空间网格
[B*4, C, 16, 16]

3x3 Conv2d + ReLU + AdaptiveAvgPool(4,4)
[B*4, C, 4, 4]

flatten
[B*4, 16*C]

Linear(16*C -> C) + ReLU + Linear(C -> C)
[B*4, C]

恢复时间维
[B, 4, C]

转置后对每个 channel 做 Linear(4 -> 16)
[B, C, 4] -> [B, C, 16] -> [B, 16, C]

逐 action slot refine MLP
[B, 16, C]
```

### 4.3 它实际学习了什么

`conv_fc` 分成三个阶段：

1. **逐帧空间编码**：`3x3 Conv2d` 融合局部邻域；池化到 `4x4` 后仍保留粗空间布局。
2. **逐帧压缩**：将 `4x4` 网格展平，`Linear(16*C -> C)` 对不同粗空间位置使用不同权重，得到每帧一个 descriptor。
3. **跨帧展开到动作时间**：`Linear(4 -> 16)` 让每个输出动作槽读取全部 4 个输入帧。

其时间混合可写成：

```text
c_pre[j, channel] = sum_f W_time[j, f] * frame_feat[f, channel] + b[j]
```

这里同一个 `16x4` temporal weight matrix 在所有 channel 间共享；随后 `refine` MLP 再逐槽混合 channel。`refine` 对所有 16 个动作槽共享参数，不直接在动作槽之间通信。

相比 `conv_ori`，`conv_fc` 的每个动作 condition 都能显式依赖所有 4 帧，并保留更多粗空间结构。

### 4.4 参数量

各部分参数量为：

```text
Conv2d(C -> C, 3x3):       9*C^2 + C
Linear(16*C -> C):        16*C^2 + C
Linear(C -> C):            1*C^2 + C
Linear(4 -> 16):                   80
refine 两个 Linear(C -> C): 2*C^2 + 2*C
```

合计：

```text
P_conv_fc = 28*C^2 + 5*C + 80
```

默认 `C=768`：

```text
P_conv_fc = 28*768^2 + 5*768 + 80
          = 16,518,992
```

因此：

```text
P_conv_fc / P_conv_ori ~= 7.0
```

以上只统计 adapter。后面的 `SimpleMLPAdaLN`、diffusion schedule 和 action target projection对两者完全相同，未计入比较。

## 5. 结构对比

| 维度 | `conv_ori` | `conv_fc` |
|---|---|---|
| 输入 | `[B, 4*256, C]` | `[B, 4*256, C]` |
| 输出 | `[B, 16, C]` | `[B, 16, C]` |
| 空间处理 | `1x1` channel mixing 后全局平均 | `3x3` 卷积，保留到 `4x4` 粗网格后 FC |
| 显式空间邻域 | 无 | 有 |
| 输入帧间混合 | 无；每帧独立产生连续 4 个动作槽 | 有；每个动作槽混合全部 4 帧 |
| adapter 非线性 | 无 | 两处 ReLU |
| 动作槽间后续交互 | 无 | `Linear(4 -> 16)` 前融合输入帧，但输出槽之间无进一步交互 |
| 参数量，`C=768` | 2,360,064 | 16,518,992 |
| 表达能力 | 较低、归纳偏置强 | 较高、映射更灵活 |
| 计算/显存 | 通常更低 | 通常更高 |
| checkpoint 参数名 | `conv_transpose3d.*` | `conv.*`, `fc.*`, `interpolate.*`, `refine.*` |

## 6. 两者共享的 action diffusion 路径

adapter 输出 `[B,16,C]` 后，训练代码执行：

```python
target = target.reshape(B * 16, A)
condition = condition.reshape(B * 16, C)
t = randint(0, diffusion_steps, size=(B * 16,))
loss = diffusion.training_losses(net, target, t, {"c": condition})
loss = loss.mean()
```

共享的 `SimpleMLPAdaLN` 包含：

- action noisy input projection：`A -> diffloss_act_w`；
- diffusion timestep embedding；
- condition projection：`C -> diffloss_act_w`；
- 多个 AdaLN residual MLP block；
- 输出 action diffusion 的 epsilon/variance 参数。

因此，`conv_ori`/`conv_fc` 负责构造 condition，而不是直接回归最终 action。

训练和推理分别走 `DiffActLoss.forward()` 与 `DiffActLoss.sample()`；两处复制了相同的 adapter 变换。MAR 当前调用 action sampling 时把 `act_cfg` 固定为 `1.0`，所以实际 action 路径不使用 classifier-free guidance。

推理会生成完整的 16 步 `action_pred`，policy 再通过 `n_action_steps` 截取前若干步；Libero10 当前通常返回前 8 步执行。

## 7. 配置选择与当前实验

基础配置默认是：

```yaml
model:
  policy:
    action_model_params:
      predict_action: false
      act_model_type: conv_fc
```

启用 action 并选择结构可以使用 Hydra override：

```bash
model.policy.action_model_params.predict_action=True \
model.policy.action_model_params.act_model_type=conv_ori
```

或者：

```bash
model.policy.action_model_params.predict_action=True \
model.policy.action_model_params.act_model_type=conv_fc
```

当前工作区的 `train_uva_libero10_jepa2_1_small_token_feat_fully_frozen_action_8gpu.sh` 显式覆盖为 `conv_ori`，并在启动前检查 action checkpoint 中的 resolved config 也是 `conv_ori`。因此该脚本不受基础 YAML 中 `conv_fc` 默认值影响。

配置优先级可以理解为：

```text
model/uva.yaml 默认值
    < 继承配置中的覆盖
    < 启动命令行 override
```

## 8. Checkpoint 兼容性

两种 adapter 输出 shape 相同，但权重结构不同：

```text
conv_ori:
diffactloss.conv_transpose3d.weight
diffactloss.conv_transpose3d.bias

conv_fc:
diffactloss.conv.*
diffactloss.fc.*
diffactloss.interpolate.*
diffactloss.refine.*
```

共享的 `diffactloss.net.*` 权重形状相同，可以复用；adapter 权重不能互相转换。

项目的 `load_pretrained_model()` 只加载名称存在且 shape 相同的参数，然后用 `strict=False` 写入模型。这意味着，如果 checkpoint 是 `conv_ori` 而当前模型配置成 `conv_fc`：

- MAR 主干和共享 diffusion MLP 可能正常加载；
- `conv_ori` adapter 权重不会进入 `conv_fc` adapter；
- 新的 `conv_fc` adapter 保持随机初始化；
- 加载过程未必以 missing-key error 形式失败，因为代码先用当前模型 state dict 补齐了键。

这在 fully-frozen MAR 实验里尤其危险：如果随机初始化的 adapter 随后也被冻结，训练不会修复它。当前 8-GPU 脚本的 checkpoint preflight 正是在避免这一问题。

切换结构时应遵守：

1. 从头训练 action adapter；或明确只加载 MAR/shared diffusion 的兼容权重。
2. 不要把另一种结构的 action checkpoint 当成完整可复用 head。
3. fully frozen 前先检查 resolved `act_model_type` 和 checkpoint key。

## 9. 当前实现的固定假设与风险

### 9.1 固定 4 帧、16 动作槽

MAR 内部把 `n_frames` 硬编码为 4。

`conv_fc` 又单独硬编码：

```python
self.num_frames = 4
self.num_actions = 16
self.interpolate = nn.Linear(4, 16)
```

`conv_ori` 通过 `kernel_size=4, stride=4` 将 `T` 扩为 `4*T`；当前 `T=4` 时恰好是 16。

如果修改 observation frame 数或 action horizon，只改配置不足以保证这两个 adapter 正常工作。

### 9.2 固定 `16x16` visual grid

两者都写死 `w=16, h=16`：

- 输入 token 数默认必须是 `16*16=256`；
- `conv_ori` 的平均池化 kernel 也是 `16x16`；
- 改 `img_size`、VAE stride 或 patch size 时需要同步检查 adapter。

### 9.3 下游 diffusion 不做动作序列交互

condition 和 action target 被展平为 `[B*16, ...]`，`SimpleMLPAdaLN` 逐行处理。因此：

- `conv_ori` 的 action 0--3 主要依赖 frame 0，action 4--7 主要依赖 frame 1，依此类推；
- `conv_fc` 依靠 `Linear(4 -> 16)` 提前把四帧信息混到每个 action condition；
- 如果需要显式建模 action-to-action dependency，当前两种结构都没有 action-time Transformer/Conv1d。

### 9.4 其他共同实现细节

- `text_latents` 参数会传入 `DiffActLoss`，但当前 action diffusion 的 `model_kwargs` 只包含 `c=condition`，没有直接使用 text latent。
- `sample()` 用 `.cuda()` 创建 noise，而不是依据 `z.device`；非默认设备或非 CUDA 环境需要注意。
- adapter 前处理在 `forward()` 和 `sample()` 中重复实现，未来修改时必须保持两处一致。
- `conv_fc.self.h` 当前没有参与 `rearrange`；代码依靠 token 数与 `w=16` 推断另一空间维。

## 10. 如何选择

适合优先使用 `conv_ori` 的情况：

- 必须兼容已经训练好的 `conv_ori` action checkpoint；
- 希望 action adapter 参数更少、计算更轻；
- 任务中四帧与四段动作天然接近分段对应；
- MAR decoder 已经充分聚合空间和跨帧上下文。

适合评估 `conv_fc` 的情况：

- 希望每个动作条件显式访问全部四帧；
- 需要保留更多粗空间布局和局部空间关系；
- 可以接受约 7 倍 adapter 参数以及更高计算量；
- 有足够数据从头训练或微调新 adapter。

公平比较两种结构时，应保持以下条件一致：

- 相同 MAR/student/teacher 初始化；
- 相同有效 global batch、optimizer update 数和 LR schedule；
- 相同 action diffusion MLP 初始化；
- 两种 adapter 都允许训练，不能一边加载已训练 head、另一边随机初始化后冻结；
- 同时比较 action loss、rollout success、吞吐、显存和 adapter 梯度范数。

最终选择不应只看 action loss。`conv_fc` 表达力更强但可能更难训练或过拟合；`conv_ori` 更轻且约束更强，在已有 checkpoint 完整匹配时可能更稳。
