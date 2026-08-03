# MAR 白名单优化器与梯度回传测试教程

本文解释 V3 frozen-MAR 实验及当前 V2 token-feature run 中的四个核心问题：

1. 如何只训练 MAR 的 position embedding 和 fake/blank token，同时冻结 encoder、decoder 和 action head。
2. 如何证明 action loss 虽然不更新 frozen MAR 参数，但仍能穿过 MAR 返回 student tokenizer。
3. 三种 fake latent 为什么存在、能否删除，以及保持可训练意味着什么。
4. 当前 `student_norm` 为什么持续下降，以及如何区分尺度漂移和表征塌缩。

配套的 CPU 最小实验位于：

```bash
/data1/local_userdata/jinboning/conda/envs/repa/bin/python3.9 \
    scripts/verify_mar_whitelist_gradients.py
```

若已经激活 `repa` conda 环境，可简写为 `python scripts/verify_mar_whitelist_gradients.py`。系统 `/usr/bin/python3` 没有安装 PyTorch，不能直接用于该脚本。

该脚本用于验证 PyTorch 机制和本项目采用的判据，不加载 Libero 数据、MAR checkpoint 或 JEPA checkpoint。完整模型的真实 batch 检查方法见第 8 节。

## 1. 先区分四件不同的事

训练中经常把“冻结”“没有梯度”“不在优化器”“切断计算图”混为一谈。它们实际上是四个不同层次。

| 层次 | 控制方式 | 含义 |
|---|---|---|
| 参数是否需要梯度 | `param.requires_grad` | autograd 是否为该参数累计 `.grad` |
| 运算是否在计算图中 | 正常前向、`torch.no_grad()`、`detach()` | loss 是否还能对上游输入求导 |
| 参数是否交给优化器 | `optimizer.param_groups` | `optimizer.step()` 是否有机会更新该参数 |
| 参数最终是否发生变化 | 梯度、weight decay、学习率、数值精度 | 这是一次 step 后的结果，不能单独说明梯度来自哪里 |

V3 的目标组合是：

```text
student tokenizer:
    requires_grad=True
    在 optimizer 中
    接收 action loss + JEPA alignment loss

MAR 白名单参数:
    requires_grad=True
    在 optimizer 中
    可以更新

MAR encoder/decoder/action head:
    requires_grad=False
    不在 optimizer 中
    不更新自身参数
    但其普通前向仍保留在 autograd 图中
```

最后一行最重要：**冻结模块参数，不等于切断经过该模块的梯度。**

## 2. 为什么 frozen Linear 仍能把梯度传给输入

考虑一个最简单的冻结线性层：

```python
y = x @ W.T + b
loss = f(y)
```

即使 `W.requires_grad=False`、`b.requires_grad=False`，只要 `x` 来自可训练的 student，PyTorch 仍会计算：

```text
d loss / d x = d loss / d y @ W
```

PyTorch 只是不再为 `W` 和 `b` 累计：

```text
d loss / d W
d loss / d b
```

因此下面的路径是成立的：

```text
image
  -> student tokenizer (可训练)
  -> condition latent c
  -> frozen z_proj_cond
  -> frozen encoder/decoder/action head
  -> action loss
  -> 梯度穿过 frozen MAR
  -> 回到 c
  -> 回到 student tokenizer 参数
```

真正会切断这条路径的是：

```python
with torch.no_grad():
    mar_output = mar(c)
```

或者：

```python
mar_output = mar(c.detach())
```

V3 没有使用这两种写法。它只改变 MAR 参数的 `requires_grad`。

### 2.1 一个常见误解

`model.eval()` 也不会冻结参数或关闭梯度。它只改变 dropout、batch normalization 等模块的训练/推理行为。

```python
model.eval()                  # 不等于冻结
model.requires_grad_(False)  # 冻结参数梯度
torch.no_grad()              # 关闭上下文中的 autograd 记录
```

三者用途不同。

## 3. 当前项目如何建立 MAR 白名单

实现位于 `unified_video_action/policy/unified_video_action_policy.py`。

### 3.1 配置开关

policy 第 70-79 行读取两个开关：

```python
self.freeze_mar = bool(kwargs.get("freeze_mar", False))
self.keep_mar_pos_and_fake_trainable = bool(
    kwargs.get("keep_mar_pos_and_fake_trainable", False)
)
self.mar_trainable_parameter_names = ()
```

V3 配置为：

```yaml
freeze_mar: true
keep_mar_pos_and_fake_trainable: true
```

旧配置中二者默认都是 `false`，所以旧实验不会被自动冻结。

### 3.2 为什么先加载 checkpoint，再设置冻结状态

policy 第 213-221 行的顺序是：

```text
构建 MAR
  -> load_pretrained_model()
  -> _configure_mar_trainability()
```

因此白名单参数保留 pretrained checkpoint 中的数值。这里没有把位置向量或 fake token 清零，也没有重新初始化。

### 3.3 冻结全部，再重新打开白名单

policy 第 243-276 行的核心逻辑是：

```python
self.model.requires_grad_(False)

for name, param in self.model.named_parameters():
    if self._is_mar_pos_or_fake_parameter(name):
        param.requires_grad = True
```

采用“先全部关闭，再打开白名单”的优点是：

- 默认状态明确，不容易漏掉新增加的 encoder/head 参数。
- 审核时只需要检查允许列表。
- MAR 将来增加新模块时，新参数默认冻结，不会静默加入训练。

当前匹配规则检查参数的叶子名称：

```python
leaf_name = name.rsplit(".", 1)[-1]
```

允许：

```text
fake_*                        以 fake_ 开头
*_pos_embed                   以 _pos_embed 结尾
diffusion_temporal_embed      精确匹配
diffusion_spatial_embed       精确匹配
mask_token                    未来兼容
blank_token                   未来兼容
```

当前 Libero10 配置下，实际白名单为：

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

诸如 `z_proj_cond`、`encoder_blocks`、`decoder_blocks`、`diffloss` 和 `diffactloss` 都不匹配，因此保持冻结。

### 3.4 为什么需要 `fake_latent_x`、`fake_action_latent` 和 `fake_latent`

这些参数不是普通的补零，也不是为了让 tensor 形状勉强通过。它们是 **learned missing-modality token**：告诉同一个 MAR 编码器“这个输入槽位在当前 task mode 中不存在、被遮挡，或者被有意丢弃”。

| 参数 | 当前代码中的含义 | 使用位置 |
|---|---|---|
| `fake_latent_x` | 未知、被 mask 或当前任务不允许看到的视觉 target token | `policy_model` 的整段 `x`；`inverse_model` 的 `cond`；其他模式被 mask 的 `x` patch |
| `fake_action_latent` | 当前任务没有可作为输入的 action，尤其避免预测 action 时泄漏 ground-truth action | 除 `dynamic_model` 外的 action 输入槽位 |
| `fake_latent` | CLIP language 被 classifier-free guidance 随机丢弃时的 learned null-language token | training 下约 `label_drop_prob` 比例的 language 条件 |

#### `fake_latent_x`

在 `policy_model` 中，动作应由 condition frames `c` 和语言预测。target frames `x` 不能作为真实输入，否则任务定义会改变。因此代码使用：

```python
cond = self.z_proj_cond(cond)
x = self.fake_latent_x.unsqueeze(1).expand(B, cond.size(1), -1)
```

在其他 video/dynamics 模式中，它还承担 MAE `[MASK]` token 的作用：

```python
x[mask == 1] = fake_latent_expanded[mask == 1]
```

所以它的语义不是“随便填空”，而是统一表示 unavailable visual content。

#### `fake_action_latent`

只有 `dynamic_model` 会把真实 action 作为条件输入：

```python
if task_mode == "dynamic_model":
    action_latents = self.action_proj_cond(nactions)
else:
    action_latents = self.fake_action_latent.unsqueeze(0).repeat(B, 16, 1)
```

当前训练是 `policy_model`，目标就是预测 action。若把 ground-truth action 输入 encoder，会产生标签泄漏；因此 action 槽必须放 null-action token。

#### `fake_latent`

这个名字较模糊，但它只在 CLIP language 的 classifier-free guidance dropout 中使用：

```python
text_latents = (
    drop_latent_mask * self.fake_latent
    + (1 - drop_latent_mask) * text_latents
)
```

它表示“没有语言条件”，使模型在训练时同时学到 conditional 和 unconditional 分支。当前配置的 `label_drop_prob=0.1`。

#### 能不能直接不输入

对当前 pretrained MAR，不能只删除这些输入。当前无 wrist/history/proprio 的 Libero10 路径执行：

```python
parts = [x, cond, action_latents_expand]
x = torch.cat(parts, dim=-1)
x = self.proj_cond_x_layer(x)
```

`proj_cond_x_layer` 的输入宽度固定为三个 `encoder_embed_dim`。删除 `x` 或 action 槽后，最后一维会改变，立即与 pretrained linear weight 形状不匹配。

理论上有三种不同方案：

| 方案 | 能否运行 | 与 pretrained MAR 是否兼容 | 实验含义 |
|---|---|---|---|
| 保留 checkpoint 中的 learned fake token | 能 | 完全兼容 | 保留原 MAR 的 missing-input 语义 |
| 用同形状全零 tensor 替换 | 能 | 形状兼容，但数值分布改变 | null token 消融实验 |
| 从 concat 中彻底删除该槽 | 不能直接运行 | 不兼容 | 必须重建 `proj_cond_x_layer` 并重新训练 MAR |

“维度对上”只是必要条件，不是充分条件。learned fake token 经过 frozen `proj_cond_x_layer` 后会产生模型已经学会解释的偏置和接口语义；全零输入虽然形状相同，却会删掉这部分 pretrained 信号。

#### fake token 必须保持可训练吗

不必须。这取决于实验问题：

```text
实验 A：冻结 MAR blocks/heads，但允许接口适配
    student + align projector + position/fake token 可训练
    这就是当前 V3

实验 B：严格冻结整个 pretrained MAR
    只有 student + align projector 可训练
    position/fake token 保留 checkpoint 数值但 requires_grad=False
```

实验 A 更容易适应新 student latent 分布；实验 B 更严格地把 action loss 的调整压力放到 student。当前 fake/position 参数容量较小且对所有样本共享，不能单独记住每个样本的动作，但确实可以吸收一部分全局偏置。因此若研究问题是“动作误差是否只能推动 student”，实验 B 是更干净的对照组。

### 3.5 为什么还保存参数名称

代码把最终可训练的 MAR 参数保存为：

```python
self.mar_trainable_parameter_names = tuple(
    name for name, param in self.model.named_parameters() if param.requires_grad
)
```

这不是 optimizer 必需的数据，而是审核接口。可以直接打印：

```python
for name in policy.mar_trainable_parameter_names:
    print(name)
```

如果启用了白名单模式但一个参数也没匹配到，代码会立即报错。这样未来重命名 position embedding 时，不会在无人察觉的情况下变成“整个 MAR 全冻结”。

## 4. “白名单优化器”实际上如何工作

这里没有新写一种 optimizer。仍然使用 `torch.optim.AdamW`，白名单通过 `requires_grad` 和参数过滤共同实现。

### 4.1 第一道门：`requires_grad`

冻结配置决定参数自身是否接收梯度：

```python
param.requires_grad = False  # frozen MAR
param.requires_grad = True   # MAR 白名单
```

### 4.2 第二道门：构建 optimizer 参数组

policy 第 472-487 行的 `add_weight_decay()` 首先执行：

```python
for name, param in model.named_parameters():
    if not param.requires_grad:
        continue
```

所以 frozen MAR 参数根本不会进入 AdamW。

剩余可训练参数再分成两组：

```text
no_decay:
    一维参数、bias、skip_list 中的参数
    weight_decay = 0

decay:
    其他参数
    weight_decay = 配置值
```

随后 `get_optimizer()` 分别加入：

```text
MAR 中 requires_grad=True 的白名单参数
student tokenizer 的可训练参数
align_projector 的可训练参数
teacher_latent_projector（仅旧 latent 对齐模式可能存在）
```

最后仍然只有一个 AdamW：

```python
optimizer = torch.optim.AdamW(
    optim_groups,
    lr=learning_rate,
    betas=betas,
)
```

### 4.3 当前白名单没有独立学习率

当前实现只区分 weight decay，没有给 MAR 白名单、student 和 projector 设置不同学习率。它们共享 AdamW 的基础 `learning_rate`。

如果以后希望白名单使用更小学习率，需要显式增加带 `lr` 的参数组，例如：

```python
optimizer = torch.optim.AdamW(
    [
        {"params": student_params, "lr": base_lr},
        {"params": projector_params, "lr": base_lr},
        {"params": mar_whitelist_params, "lr": base_lr * 0.1},
    ],
    weight_decay=weight_decay,
)
```

这不是当前 V3 的行为，不能把它误写进实验说明。

## 5. optimizer 成员检查怎么做

不能只打印 `requires_grad`，还要验证 optimizer 里到底有哪些参数。最可靠的方法是比较 Python 对象 ID：

```python
optimizer_param_ids = {
    id(param)
    for group in optimizer.param_groups
    for param in group["params"]
}

for name, param in policy.model.named_parameters():
    in_optimizer = id(param) in optimizer_param_ids
    assert in_optimizer == param.requires_grad, (
        name,
        param.requires_grad,
        in_optimizer,
    )
```

为什么不用参数名检查？optimizer 保存的是 `Parameter` 对象，不保存稳定的模块参数名；对象 ID 可以直接回答“这个确切参数是否在参数组中”。

还应检查 student 和 projector 没被遗漏：

```python
for module in (policy.student_tokenizer, policy.align_projector):
    if module is None:
        continue
    for name, param in module.named_parameters():
        if param.requires_grad:
            assert id(param) in optimizer_param_ids, name
```

## 6. 此前的轻量梯度测试到底证明了什么

此前使用的是一个 **action-like 单元测试**，不是完整训练收敛实验。测试保留了真实问题的计算图结构：

```text
toy observation
  -> trainable student
  -> condition latent
  -> frozen MAR projection/encoder/decoder/action head
       + trainable position/fake parameters
  -> action-like MSE loss
  -> backward
```

它依次验证：

1. 冻结后只有白名单 MAR 参数为 `requires_grad=True`。
2. frozen MAR 参数不在 AdamW 中，白名单和 student 在 AdamW 中。
3. 只对 action-like loss 反向传播后，student 的梯度范数大于 0。
4. frozen MAR 参数的 `.grad` 全部为 `None`。
5. 实际参与该分支的白名单参数梯度范数大于 0。
6. `optimizer.step()` 后 frozen MAR 参数逐元素完全不变。

配套脚本就是这套测试的可运行、带输出版本：

```bash
/data1/local_userdata/jinboning/conda/envs/repa/bin/python3.9 \
    scripts/verify_mar_whitelist_gradients.py
```

预期最后看到：

```text
PASS: whitelist optimizer and frozen-MAR gradient flow are correct.
```

### 6.1 为什么用 action-like loss，而不是 raw loss

V3 的训练总损失是：

```text
raw_loss = action_loss + align_coeff * alignment_loss + DDP safety zero terms
```

如果只对 `raw_loss` backward，然后发现 student 有梯度，不能证明梯度来自 action loss，因为 JEPA alignment loss 本身也会训练 student。

因此验证“动作误差能否穿过 frozen MAR”时，必须隔离：

```python
action_loss.backward()
```

而不是只测试：

```python
raw_loss.backward()
```

这是整个梯度测试中最关键的实验设计。

## 7. 为什么 `grad is not None` 可能是假证据

policy 第 772-794 行有 DDP safety term：

```python
loss = loss + param.sum() * 0.0
```

其目的是让每个可训练参数都形式上连接到 DDP 图，即使当前随机 task/mask 分支没有真正使用该参数。

这会产生一个重要现象：

```text
param.grad is not None
param.grad.norm() == 0
```

所以：

```python
assert param.grad is not None
```

只能说明参数形式上连接了计算图，不能证明真实 action loss 给了它有效梯度。

更可靠的判据是：

```python
grad_norm = param.grad.detach().float().norm().item()
assert grad_norm > 0.0
```

配套脚本专门演示了这一点：一个未被 action 分支使用的白名单参数，在 action-only backward 后 `.grad is None`；加入 DDP safety zero term 后则变成“`.grad` 存在但范数为 0”。

### 7.1 `optimizer.step()` 后参数变化也不一定证明有真实梯度

AdamW 使用 decoupled weight decay。若 DDP safety 给一个参数制造了全零梯度 tensor，该参数仍可能因为 weight decay 发生变化。

因此下面的推断不严谨：

```text
参数变化了
=> action loss 一定给了它非零梯度
```

正确检查顺序是：

1. 在 `optimizer.step()` 之前查看 action-only 梯度范数。
2. 再执行 step，检查 frozen 参数确实保持不变。

## 8. 如何在完整 V3 模型上用真实 batch 检查

真实训练循环位于 `unified_video_action/workspace/train_unified_video_action_workspace.py` 第 268-290 行：

```python
raw_loss, (loss_diffusion, loss_action) = self.model(batch)
accelerator.backward(raw_loss)
self.optimizer.step()
```

为了隔离 action 梯度，建议在一个单独的 debug 进程中，取第一个真实 batch 后只做一次检查并退出，不要把诊断 backward 和正常训练 backward 混在一起。

核心检查代码如下：

```python
def module_grad_norm(module):
    total = 0.0
    for param in module.parameters():
        if param.grad is not None:
            total += param.grad.detach().float().pow(2).sum().item()
    return total ** 0.5


policy = accelerator.unwrap_model(self.model)
self.optimizer.zero_grad(set_to_none=True)

# 使用真实 batch 前向，但只反传 action loss。
raw_loss, (_, action_loss) = self.model(batch)
accelerator.backward(action_loss)

# 使用 fp16/GradScaler 时，先解除缩放再比较梯度数值。
accelerator.unscale_gradients(self.optimizer)

student_grad_norm = module_grad_norm(policy.student_tokenizer)
assert student_grad_norm > 0.0, student_grad_norm

frozen_with_grad = [
    name
    for name, param in policy.model.named_parameters()
    if not param.requires_grad and param.grad is not None
]
assert not frozen_with_grad, frozen_with_grad

active_whitelist_grads = {
    name: param.grad.detach().float().norm().item()
    for name, param in policy.model.named_parameters()
    if param.requires_grad
    and param.grad is not None
    and param.grad.detach().float().norm().item() > 0.0
}
assert active_whitelist_grads, "No MAR whitelist parameter received action gradient"

print("student action grad norm:", student_grad_norm)
print("active MAR whitelist grads:", active_whitelist_grads)

# 诊断结束，不执行 optimizer.step()，直接退出该 debug 进程。
raise SystemExit(0)
```

### 8.1 为什么这里检查 `loss_action`

`mar_con_unified.py` 第 756-761 行明确规定，在 `policy_model` 下：

```python
act_loss = self.diffactloss(...)
video_loss = 0
loss = act_loss
```

随后 policy 才把 JEPA alignment loss 加进 `raw_loss`。直接对返回的 `loss_action` backward，可以排除 alignment loss 的贡献。

### 8.2 完整模型测试的预期结果

在当前 `selected_training_mode: policy_model` 配置中，合理结果是：

| 对象 | action-only backward 预期 |
|---|---|
| student tokenizer | 总梯度范数 `> 0` |
| align projector | `.grad is None`，因为 action loss 不经过 JEPA projector |
| frozen MAR blocks/heads | `.grad is None` |
| 当前 action 分支使用的 MAR 白名单参数 | 至少一部分梯度范数 `> 0` |
| 当前分支未使用的白名单参数 | `.grad is None`，或在附加 DDP zero term 时为零 tensor |
| JEPA teacher | 无梯度，teacher 本来就是 frozen target |

不要强制断言“所有白名单参数在单个 batch 上都必须有非零梯度”。某些参数是否参与取决于 task mode、language、wrist、history action、mask 和 proprioception 分支。

## 9. 参数更新检查

若还要验证 step 行为，可以在一个独立测试中保存更新前参数：

```python
before = {
    name: param.detach().clone()
    for name, param in policy.model.named_parameters()
}

self.optimizer.step()

frozen_changed = []
for name, param in policy.model.named_parameters():
    if not param.requires_grad and not torch.equal(before[name], param.detach()):
        frozen_changed.append(name)

assert not frozen_changed, frozen_changed
```

对 frozen 参数可以使用严格的 `torch.equal`：它们既不在 optimizer 中，也不应被 step 改写。

对可训练参数不要要求每一个都变化。单个 batch 可能没有使用某些分支，学习率和 fp16 精度也可能使极小更新不可见。更合理的是检查 student 或至少一个活跃白名单参数发生变化。

## 10. 混合精度、梯度累积和 DDP 的注意事项

### 10.1 fp16 梯度缩放

`accelerator.backward()` 可能使用 GradScaler。若只检查是否非零，缩放通常不影响结论；若要比较梯度范数、检查 finite 或设置阈值，应先：

```python
accelerator.unscale_gradients(optimizer)
```

### 10.2 梯度累积

正式训练会累计多个 micro-batch。若检查前没有显式清空梯度，当前 `.grad` 可能包含之前 batch 的贡献。

诊断开始前必须：

```python
optimizer.zero_grad(set_to_none=True)
```

`set_to_none=True` 也能清楚地区分“本次 backward 没产生梯度”和“产生了全零梯度”。

### 10.3 DDP 包装

检查模块属性时应先取得原始 policy：

```python
policy = accelerator.unwrap_model(self.model)
```

但真实 batch 前向仍建议通过 `self.model(batch)`，保持 Accelerate/DDP 的正常调用方式。

### 10.4 不要在正式训练中重复 backward

如果先对 `loss_action` backward，又对同一图的 `raw_loss` backward，会遇到计算图已释放，或者在 `retain_graph=True` 时把梯度重复累计。

最稳妥的方法是：

- 单独启动一次 debug 任务。
- 取一个 batch。
- 只对 `loss_action` backward。
- 打印并断言。
- 不 step，直接退出。

## 11. 冻结 MAR 能省什么，不能省什么

冻结 MAR 参数通常可以减少：

- MAR 参数梯度存储。
- AdamW 对 frozen MAR 的一阶、二阶动量状态。
- frozen MAR 参数更新计算。

但为了让 action loss 返回 student，MAR 前向仍在计算图中，所以仍需保留反传到输入所需的部分激活。它不会像 `torch.no_grad()` 那样省掉整段 backward 图。

这是训练目标与显存之间的必要交换：若用 `no_grad()` 包住 MAR，显存会进一步下降，但 action loss 也无法训练 student，违背本实验目的。

## 12. 审核清单

每次修改白名单或 MAR 结构后，至少检查以下项目：

```text
[ ] checkpoint 在冻结配置之前加载
[ ] 真实 trainable MAR 名称与预期一致
[ ] 没有意外 trainable 的 encoder/decoder/head 参数
[ ] frozen MAR 参数不在 optimizer.param_groups
[ ] student 和 align_projector 没有从 optimizer 漏掉
[ ] action-only loss 能给 student 非零梯度
[ ] frozen MAR 参数 grad 全为 None
[ ] 至少一个实际参与 action 分支的白名单参数梯度非零
[ ] 不用 DDP zero term 的 grad 非空来冒充真实梯度
[ ] optimizer.step 后 frozen MAR 参数严格不变
[ ] 测试前清空累计梯度，并正确处理 AMP 缩放
```

## 13. 本次新增文件

本教程只新增以下文件，不修改训练行为：

| 文件 | 内容 |
|---|---|
| `docs/mar_whitelist_optimizer_and_gradient_testing_tutorial.md` | 当前教程、完整模型检查方法和审核清单 |
| `scripts/verify_mar_whitelist_gradients.py` | CPU 最小可运行实验，验证白名单、optimizer、action 梯度和 DDP zero-term 陷阱 |

生产实现仍以以下文件为准：

```text
unified_video_action/policy/unified_video_action_policy.py
unified_video_action/model/autoregressive/mar_con_unified.py
unified_video_action/workspace/train_unified_video_action_workspace.py
```

## 14. 当前 JEPA token-feature 实验的 `student_norm` 为什么持续下降

本节分析当前正在运行的 V2：

```text
config: uva_libero10_jepa2_1_small_token_feat.yaml
W&B run: run-20260731_153557-ounqx2dc
采样时间: 2026-08-03
最新纳入分析的 global_step: 534204
```

### 14.1 实际曲线

从本地 W&B history 提取到：

| 指标 | step 0 | step 534204 | 趋势 |
|---|---:|---:|---|
| `student_norm` | 17.4643 | 15.5390 | 下降约 11.0% |
| `teacher_norm` | 28.6584 | 28.7743 | 基本不变 |
| `align_cos` | 0.0331 | 0.9848 | 显著改善 |
| `align_mse` | 1.0697 | 0.0317 | 显著改善 |
| `align_stats` | 1.0412 | 0.00134 | 显著改善 |
| `action_loss` | 0.9715 | 0.00741 | 显著改善 |

按每 20000 step 求均值后，`student_norm` 几乎单调下降；step 与 student norm 的 Pearson correlation 约为 `-0.994`。下降现象是真实的，不是单个 batch 噪声。

但这组数据 **不支持表征塌缩结论**。alignment 的方向、逐元素误差、通道统计和 action loss 都在同步改善。

### 14.2 当前 `student_norm` 记录的不是对齐后的向量

`_compute_alignment_loss()` 的执行顺序是：

```python
student_tokens_raw = student_tokens        # 304 维 token_feat
student_tokens = align_projector(
    student_tokens_raw
)                                           # 768 维 projected token

# loss 全部使用 projected student
cosine_loss = cosine(student_tokens, teacher_tokens)
mse_loss = mse(student_tokens, teacher_tokens)
stats_loss = stats(student_tokens, teacher_tokens)

# 但 metric 使用 projector 前的 student
student_norm = norm(student_tokens_raw)
teacher_norm = norm(teacher_tokens)
```

所以 W&B 当前并列展示的是：

```text
student_norm: 304 维、projector 前的最终 student encoder token_feat
teacher_norm: 768 维、JEPA 原始 token
```

它们不在同一维度，也不在同一特征空间，不能要求数值相等。

### 14.3 初始 norm 恰好解释了维度差异

student 的 `token_feat` 先经过 `LayerNorm(hidden_dim=304)`。若每个通道方差约为 1，则单 token 的典型 L2 norm 是：

```text
sqrt(304) = 17.4356
```

当前第一个 batch 的 `student_norm=17.4643`，几乎正好是这个数。

同理，768 维、每通道方差约 1 的 token 典型 norm 是：

```text
sqrt(768) = 27.7128
```

当前 teacher norm 约 28.75，也符合这个量级。因此初始的 17 对 28 主要来自 304 和 768 的维度差异，不表示 student 比 teacher“弱”。

### 14.4 loss 真正约束的是 projected student

当前配置为：

```yaml
loss_type: hybrid
mse_coeff: 0.25
stats_coeff: 0.1
coeff: 0.05
```

单次 alignment loss 实际为：

```text
0.5 * (1 - cosine)
+ 0.75 * MSE
+ 0.1 * stats_loss
```

这些项全部作用于 `304 -> 512 -> 512 -> 768` projector 后的 student token。

因此 projector 前的 304 维 token scale 存在自由度：student raw feature 缩小一些，同时 projector 第一层和后续非线性重新调整，仍可产生正确尺度的 768 维输出。当前 `align_stats` 已降到约 0.0013，间接说明 projected student 的均值和标准差正在接近 teacher；遗憾的是代码没有直接记录 `projected_student_norm`。

### 14.5 raw norm 为什么会漂移

当前 raw token feature 的尺度受多个可训练模块共同影响：

```text
student final LayerNorm affine 参数
temporal mixer residual
student encoder
align projector
out_proj -> MAR/action 路径
```

V2 中 MAR 也全部可训练。因此 action 路径和 alignment 路径都可以通过上下游共同缩放来维持输出，loss 没有要求 projector 前的 raw norm 固定在 `sqrt(304)`。

从已有日志只能确认这种 scale drift 被目标函数允许，不能仅凭一条 norm 曲线断言是某一个权重造成的。要定位具体来源，还需要同时记录 final LayerNorm `weight` RMS、temporal mixer 权重范数、projector 第一层权重范数和 projected student norm。

### 14.6 为什么目前不像 collapse

如果所有 student token 都塌成同一个、接近零的向量，共享 projector 不可能为不同图像、时刻和 patch 恢复各自不同的 JEPA teacher token。通常会同时看到：

```text
align_cos 下降或停在较低水平
align_mse / align_stats 上升或无法继续下降
action loss 恶化
token 间方差或有效秩接近 0
```

当前观察恰好相反：`align_cos=0.985`，MSE、stats 和 action loss 都持续下降；raw norm 也仍有 15.54，而不是接近 0。因此更合理的结论是 **projector 前尺度缓慢收缩，而不是表征内容塌缩**。

### 14.7 下一版应该记录哪些指标

为了不再被这个名称误导，建议后续把 metric 拆成：

```text
student_preproj_norm       304 维 raw token_feat norm
student_projected_norm     768 维对齐输出 norm
teacher_norm               768 维 JEPA token norm
projected_teacher_ratio    projected_student_norm / teacher_norm
student_token_std          跨 batch/time/patch 的 token 方差
student_effective_rank     检查方向/维度塌缩，而不只检查长度
```

其中 `student_projected_norm` 才能和 `teacher_norm` 直接比较。当前 `student_norm` 最好重命名为 `student_preproj_norm`；这是监控语义问题，不是当前 alignment loss 的计算错误。

### 14.8 何时需要警惕

若后续出现以下组合，才应把 norm 下降视为严重问题：

```text
raw norm 快速逼近 0
+ projected norm 同时远离 teacher
+ align cosine 下降
+ MSE/stats 上升
+ action loss 或 rollout success 恶化
```

若只有 raw pre-projector norm 缓慢下降，而 projected alignment 和任务指标继续改善，则应先视为尺度重参数化，再通过 projected norm、token variance 和 effective rank 验证。
