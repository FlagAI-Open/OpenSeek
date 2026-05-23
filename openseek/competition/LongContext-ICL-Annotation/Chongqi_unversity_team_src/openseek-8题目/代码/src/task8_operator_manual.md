# Task8 答案中常用算子使用手册

本手册根据题目输入与答案代码中出现的 PyTorch/F API 总结，每个算子给出用途、场景和 demo。

## 高频算子统计

## `F.relu`

- **用途**：非线性激活，将负值截断为 0。
- **典型场景**：激活函数、ReLU+sqrt 等组合算子。
- **Demo**：

```python
import torch
import torch.nn.functional as F
y = F.relu(x, inplace=False)
```

## `torch.sqrt`

- **用途**：逐元素平方根。
- **典型场景**：sqrt、relu_sqrt、sqrt_tanh 等元素级任务。
- **Demo**：

```python
import torch
import torch.nn.functional as F
y = torch.sqrt(x)
```

## `torch.exp`

- **用途**：逐元素指数。
- **典型场景**：exp、softplus 手写、exp_mean 等。
- **Demo**：

```python
import torch
import torch.nn.functional as F
y = torch.exp(x)
```

## `torch.log`

- **用途**：逐元素自然对数。
- **典型场景**：log_tanh、softmax_log、数值变换。
- **Demo**：

```python
import torch
import torch.nn.functional as F
y = torch.log(x)
```

## `torch.sigmoid`

- **用途**：逐元素 Sigmoid。
- **典型场景**：sigmoid_argmax、mv_sigmoid_sub、门控激活。
- **Demo**：

```python
import torch
import torch.nn.functional as F
y = torch.sigmoid(x)
```

## `torch.tanh`

- **用途**：逐元素 tanh。
- **典型场景**：sqrt_tanh、tanh_linear、GELU 近似。
- **Demo**：

```python
import torch
import torch.nn.functional as F
y = torch.tanh(x)
```

## `F.gelu`

- **用途**：GELU 激活。
- **典型场景**：linear+gelu、bmm+rmsnorm+gelu、gelu_std。
- **Demo**：

```python
import torch
import torch.nn.functional as F
y = F.gelu(x, approximate="none")
```

## `F.silu`

- **用途**：SiLU/Swish 激活。
- **典型场景**：SwiGLU、silu_batch_norm、门控网络。
- **Demo**：

```python
import torch
import torch.nn.functional as F
y = F.silu(x)
```

## `F.elu`

- **用途**：ELU 激活。
- **典型场景**：elu_linear。
- **Demo**：

```python
import torch
import torch.nn.functional as F
y = F.elu(x, alpha=1.0, inplace=False)
```

## `F.softplus`

- **用途**：Softplus 平滑 ReLU。
- **典型场景**：softplus_linear。
- **Demo**：

```python
import torch
import torch.nn.functional as F
y = F.softplus(x, beta=1, threshold=20)
```

## `F.linear`

- **用途**：线性层 y=xA^T+b。
- **典型场景**：linear+activation 复合算子。
- **Demo**：

```python
import torch
import torch.nn.functional as F
y = F.linear(input, weight, bias)
```

## `torch.bmm`

- **用途**：批量矩阵乘法。
- **典型场景**：fused_bmm_rmsnorm_gelu_dropout_sub。
- **Demo**：

```python
import torch
import torch.nn.functional as F
y = torch.bmm(input1, input2)
```

## `torch.matmul`

- **用途**：通用矩阵/批量矩阵乘。
- **典型场景**：matmul、attention score、泛矩阵乘任务。
- **Demo**：

```python
import torch
import torch.nn.functional as F
y = torch.matmul(a, b)
```

## `torch.mm`

- **用途**：二维矩阵乘。
- **典型场景**：普通 2D matmul。
- **Demo**：

```python
import torch
import torch.nn.functional as F
y = torch.mm(a, b)
```

## `torch.mv`

- **用途**：矩阵向量乘。
- **典型场景**：fused_mv_sigmoid_sub。
- **Demo**：

```python
import torch
import torch.nn.functional as F
z = torch.mv(input, vec)
```

## `F.conv2d`

- **用途**：二维卷积。
- **典型场景**：conv2d_add、conv+bn+activation。
- **Demo**：

```python
import torch
import torch.nn.functional as F
y = F.conv2d(input, weight, bias, stride, padding, dilation, groups)
```

## `F.batch_norm`

- **用途**：BatchNorm。
- **典型场景**：conv/bn/activation 或 batch_norm+activation。
- **Demo**：

```python
import torch
import torch.nn.functional as F
y = F.batch_norm(x, running_mean, running_var, weight, bias, training, momentum, eps)
```

## `F.layer_norm`

- **用途**：LayerNorm。
- **典型场景**：Transformer、归一化复合算子。
- **Demo**：

```python
import torch
import torch.nn.functional as F
y = F.layer_norm(x, normalized_shape, weight, bias, eps)
```

## `F.instance_norm`

- **用途**：InstanceNorm。
- **典型场景**：图像/风格迁移相关归一化。
- **Demo**：

```python
import torch
import torch.nn.functional as F
y = F.instance_norm(x, running_mean, running_var, weight, bias, use_input_stats, momentum, eps)
```

## `F.group_norm`

- **用途**：GroupNorm。
- **典型场景**：小 batch 或分组通道归一化。
- **Demo**：

```python
import torch
import torch.nn.functional as F
y = F.group_norm(x, num_groups, weight, bias, eps)
```

## `F.max_pool2d`

- **用途**：二维最大池化。
- **典型场景**：conv 后降采样。
- **Demo**：

```python
import torch
import torch.nn.functional as F
y = F.max_pool2d(x, kernel_size, stride, padding)
```

## `F.avg_pool2d`

- **用途**：二维平均池化。
- **典型场景**：特征降采样。
- **Demo**：

```python
import torch
import torch.nn.functional as F
y = F.avg_pool2d(x, kernel_size, stride, padding)
```

## `F.adaptive_avg_pool2d`

- **用途**：自适应平均池化到指定输出尺寸。
- **典型场景**：输入尺寸不固定但输出尺寸固定。
- **Demo**：

```python
import torch
import torch.nn.functional as F
y = F.adaptive_avg_pool2d(x, output_size)
```

## `F.softmax`

- **用途**：softmax 概率归一化。
- **典型场景**：attention、softmax_mul。
- **Demo**：

```python
import torch
import torch.nn.functional as F
y = F.softmax(x, dim=dim, dtype=dtype)
```

## `F.log_softmax`

- **用途**：log softmax。
- **典型场景**：分类 log-prob 或 linear+log_softmax。
- **Demo**：

```python
import torch
import torch.nn.functional as F
y = F.log_softmax(x, dim=dim, dtype=dtype)
```

## `F.cross_entropy`

- **用途**：交叉熵损失。
- **典型场景**：分类损失融合任务。
- **Demo**：

```python
import torch
import torch.nn.functional as F
loss = F.cross_entropy(logits, target, reduction="mean")
```

## `F.dropout`

- **用途**：随机失活。
- **典型场景**：训练期正则；linear/gelu/dropout 复合。
- **Demo**：

```python
import torch
import torch.nn.functional as F
y = F.dropout(x, p=p, training=training)
```

## `torch.sum`

- **用途**：求和 reduction。
- **典型场景**：sum_std、统计类任务。
- **Demo**：

```python
import torch
import torch.nn.functional as F
y = torch.sum(x, dim=dim, keepdim=keepdim)
```

## `torch.mean`

- **用途**：均值 reduction。
- **典型场景**：add_mean、exp_mean、RMSNorm。
- **Demo**：

```python
import torch
import torch.nn.functional as F
y = torch.mean(x, dim=dim, keepdim=keepdim)
```

## `torch.std`

- **用途**：标准差。
- **典型场景**：sum_std、gelu_std。
- **Demo**：

```python
import torch
import torch.nn.functional as F
y = torch.std(x, dim=dim, keepdim=keepdim, correction=1)
```

## `torch.var`

- **用途**：方差。
- **典型场景**：norm/rms/统计任务。
- **Demo**：

```python
import torch
import torch.nn.functional as F
v = torch.var(x, dim=dim, keepdim=True)
```

## `torch.max`

- **用途**：最大值。
- **典型场景**：max reduction。
- **Demo**：

```python
import torch
import torch.nn.functional as F
vals = torch.max(x, dim=dim, keepdim=keepdim).values
```

## `torch.min`

- **用途**：最小值。
- **典型场景**：min_gelu、gelu_min。
- **Demo**：

```python
import torch
import torch.nn.functional as F
vals = torch.min(x, dim=dim, keepdim=keepdim).values
```

## `torch.argmax`

- **用途**：最大值索引。
- **典型场景**：sigmoid_argmax、分类索引。
- **Demo**：

```python
import torch
import torch.nn.functional as F
idx = torch.argmax(x, dim=dim, keepdim=keepdim)
```

## `torch.logsumexp`

- **用途**：稳定计算 log(sum(exp(x)))。
- **典型场景**：logsumexp reduction。
- **Demo**：

```python
import torch
import torch.nn.functional as F
y = torch.logsumexp(x, dim=dim, keepdim=keepdim)
```

## `torch.rsqrt`

- **用途**：平方根倒数。
- **典型场景**：RMSNorm、rsqrt 算子。
- **Demo**：

```python
import torch
import torch.nn.functional as F
y = torch.rsqrt(x)
```

## `torch.gather`

- **用途**：按 index gather。
- **典型场景**：gather_masked_fill。
- **Demo**：

```python
import torch
import torch.nn.functional as F
y = torch.gather(input, dim, index)
```

## `torch.index_select`

- **用途**：按 index 选择。
- **典型场景**：index_select_eq。
- **Demo**：

```python
import torch
import torch.nn.functional as F
y = torch.index_select(input, dim, index)
```

## `torch.masked_select`

- **用途**：根据 mask 拉平成选择元素。
- **典型场景**：masked_select_add_gelu。
- **Demo**：

```python
import torch
import torch.nn.functional as F
y = torch.masked_select(input, mask)
```

## `Tensor.masked_fill`

- **用途**：mask 位置填值。
- **典型场景**：gather 后填充或注意力 mask。
- **Demo**：

```python
import torch
import torch.nn.functional as F
y = x.masked_fill(mask, value)
```

## `torch.repeat_interleave`

- **用途**：重复元素。
- **典型场景**：repeat_interleave + log_softmax。
- **Demo**：

```python
import torch
import torch.nn.functional as F
y = torch.repeat_interleave(x, repeats, dim=dim)
```

## `F.embedding`

- **用途**：查表 embedding。
- **典型场景**：embedding_add_tanh。
- **Demo**：

```python
import torch
import torch.nn.functional as F
y = F.embedding(input, weight, padding_idx=padding_idx)
```

## `torch.linalg.solve`

- **用途**：解线性方程 AX=B。
- **典型场景**：solve、solve_and_add_scaled_vector。
- **Demo**：

```python
import torch
import torch.nn.functional as F
x = torch.linalg.solve(A, B)
```

## `torch.linalg.svd`

- **用途**：奇异值分解。
- **典型场景**：svd/reconstruct/low-rank。
- **Demo**：

```python
import torch
import torch.nn.functional as F
U, S, Vh = torch.linalg.svd(A, full_matrices=False)
```

## `torch.linalg.qr`

- **用途**：QR 分解。
- **典型场景**：least_squares_qr、det via qr。
- **Demo**：

```python
import torch
import torch.nn.functional as F
Q, R = torch.linalg.qr(A, mode="reduced")
```

## `torch.linalg.cholesky`

- **用途**：Cholesky 分解。
- **典型场景**：fused_cholesky_solve。
- **Demo**：

```python
import torch
import torch.nn.functional as F
L = torch.linalg.cholesky(A)
```

## `torch.linalg.inv`

- **用途**：矩阵逆。
- **典型场景**：invert_matrix_lu 等可用 fallback。
- **Demo**：

```python
import torch
import torch.nn.functional as F
A_inv = torch.linalg.inv(A)
```

## `torch.linalg.det`

- **用途**：行列式。
- **典型场景**：determinant_lu/via_qr。
- **Demo**：

```python
import torch
import torch.nn.functional as F
d = torch.linalg.det(A)
```

## `torch.linalg.pinv`

- **用途**：伪逆。
- **典型场景**：pseudoinverse_svd。
- **Demo**：

```python
import torch
import torch.nn.functional as F
P = torch.linalg.pinv(A, rcond=1e-15)
```

## `torch.linalg.lstsq`

- **用途**：最小二乘。
- **典型场景**：least_squares_qr。
- **Demo**：

```python
import torch
import torch.nn.functional as F
x = torch.linalg.lstsq(A, B).solution
```

## `F.normalize`

- **用途**：向量归一化。
- **典型场景**：normalize_pairwise_distance、cosine similarity。
- **Demo**：

```python
import torch
import torch.nn.functional as F
x_norm = F.normalize(x, p=2, dim=1, eps=1e-12)
```

## `F.pairwise_distance`

- **用途**：成对距离。
- **典型场景**：normalize_pairwise_distance。
- **Demo**：

```python
import torch
import torch.nn.functional as F
d = F.pairwise_distance(x1, x2, p=2, eps=1e-6)
```

## `F.cosine_similarity`

- **用途**：余弦相似度。
- **典型场景**：normalized_cosine_similarity。
- **Demo**：

```python
import torch
import torch.nn.functional as F
s = F.cosine_similarity(x1, x2, dim=1, eps=1e-8)
```

