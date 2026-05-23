# OpenSeek-8 每道题思路分析与拆解

说明：本文件将题目输入与答案代码对齐，逐题抽取 wrapper、任务类型、算子链、答案实现策略与解题步骤。

## 共性总结

- 总题数：166。
- 任务共同模式：自然语言功能描述 + Wrapper Entry Information + 参数/数学定义 → 生成同名 Python/Triton wrapper。
- 高稳策略：先保证函数名、import、参数兼容、out/inplace 支持，再用 PyTorch API 实现语义；复杂 Triton 仅在必要且简单时使用。
- 常见答案风格：`import torch`、`import torch.nn.functional as F`、`_write_out`、`def wrapper(*args, **kwargs)`、按算子链逐步组合。
- family 分布：linalg=58, matmul_linear=45, reduction=27, conv_norm_pool=24, attention_softmax_loss=6, indexing=4, activation=2

## 1. openseek-8-501f776ba20444458ac14dd7292cc913 — `fused_bmm_rmsnorm_gelu_dropout_sub`

- **任务类型**：matmul_linear
- **Wrapper**：`fused_bmm_rmsnorm_gelu_dropout_sub(input1, input2, other, normalized_shape, dropout_p=0.5, training=True, approximate='none', eps=1e-5, *, out=None) -> Tensor. Args: input1 (Tensor): First input tensor for batch matrix multiplication, of shape (B, N, M), where B is the batch size. input2 (Tensor): Second input tensor for batch matrix multiplication, of shape (B, M, P). other (Tensor): Tensor to subtract from the result after dropout, must be broadcastable to the shape of the output. normalized_s`
- **功能描述**：Performs a fused operation combining batch matrix multiplication, RMS normalization, GELU activation, dropout, and subtraction. The function takes three input tensors, performs batch matrix multiplication on the first two, applies RMS normalization, GELU activation, and dropout, and finally subtracts the third tensor from the result.
- **数学定义**：Given input tensors X, Y, and O, this function computes: \[ \begin{align*} Z &= \text{bmm}(X, Y) \\ Z_{\text{norm}} &= \text{RMSNorm}(Z, \epsilon) \\ G &= \text{GELU}(Z_{\text{norm}}) \\ D &= \text{Dropout}(G, p) \\ Y &= D - O \end{align*} \]
- **补充约束**：broadcastable to (B, N, P). Output: (B, N, P).
- **题目算子链**：F.linear, torch.bmm, torch.matmul, torch.mm, custom _rms_norm, F.dropout, F.gelu, torch.tanh, F.elu, torch.sqrt, torch.exp, torch.mean, torch.min, torch.linalg.vector_norm, torch.where, torch.linalg.qr
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `fused_bmm_rmsnorm_gelu_dropout_sub` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `matmul_linear` 类，优先把自然语言描述映射到 PyTorch 算子链：F.linear, torch.bmm, torch.matmul, torch.mm, custom _rms_norm, F.dropout, F.gelu, torch.tanh, F.elu, torch.sqrt, torch.exp, torch.mean, torch.min, torch.linalg.vector_norm, torch.where, torch.linalg.qr。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 2. openseek-8-82c29f05c3434437917c95f49fadff01 — `div`

- **任务类型**：reduction
- **Wrapper**：`div(input, other, *, rounding_mode=None, out=None) -> Tensor; input (Tensor): the dividend; other (Tensor or Number): the divisor; rounding_mode (str, optional): Type of rounding applied to the result; out (Tensor, optional): the output tensor`
- **功能描述**：Divides each element of the input tensor by the corresponding element of the other tensor, supporting broadcasting, type promotion, and handling integer, float, and complex inputs. Rounding behavior can be controlled with the rounding_mode parameter.
- **数学定义**：\text{out}_i = \frac{\text{input}_i}{\text{other}_i}
- **补充约束**：By default, performs a 'true' division like Python 3. Supports broadcasting to a common shape, type promotion, and integer, float, and complex inputs. Always promotes integer types to the default scalar type.
- **题目算子链**：torch.mm, custom _rms_norm, torch.exp, torch.min
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `div` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `reduction` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, custom _rms_norm, torch.exp, torch.min。
  1. 此类任务可短实现，重点处理 dim/keepdim/out/inplace/dtype 等参数。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 3. openseek-8-76a66f9a2bd5449fbb57b2b0a0bd7ec7 — `sigmoid_conv2d`

- **任务类型**：conv_norm_pool
- **Wrapper**：`sigmoid_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, out=None) -> Tensor`
- **功能描述**：Applies a 2D convolution over an input tensor with specified filters, followed by applying the sigmoid activation function element-wise to the result. This ensures that the convolutional output values are scaled between 0 and 1.
- **数学定义**：\text{out} = \sigma(\text{conv2d}(\text{input}, \text{weight})) where \sigma(x) = \frac{1}{1 + e^{-x}} is the sigmoid function.
- **补充约束**：The function combines 2D convolution and sigmoid activation, ensuring output values are between 0 and 1.
- **题目算子链**：F.conv2d, torch.mm, torch.sigmoid, torch.exp, torch.sin, torch.min, torch.where, torch.linalg.eig
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `sigmoid_conv2d` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `conv_norm_pool` 类，优先把自然语言描述映射到 PyTorch 算子链：F.conv2d, torch.mm, torch.sigmoid, torch.exp, torch.sin, torch.min, torch.where, torch.linalg.eig。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 4. openseek-8-616ca19cad034c8ba1763cbd4420b620 — `solve_multiple_lu`

- **任务类型**：matmul_linear
- **Wrapper**：`def solve_multiple_lu(A, Bs, *, pivot=True, out=None) -> Tensor - **A** (Tensor): Coefficient matrix of shape `(*, n, n)`, where `*` is zero or more batch dimensions. - **Bs** (Tensor): Right-hand side tensor of shape `(*, n, k)`, where `k` is the number of right-hand sides. - **pivot** (bool, optional): Controls whether to compute the LU decomposition with partial pivoting (`True`) or without pivoting (`False`). Default: `True`. - **out** (Tensor, optional): Output tensor. Ignored if `None`. De`
- **功能描述**：Solves multiple linear systems with the same coefficient matrix using LU decomposition. Given a square matrix A and multiple right-hand side vectors B, this function computes the solutions X to the linear systems A X = B by performing the LU decomposition of A and reusing it to solve for multiple right-hand sides efficiently. Supports batch dimensions.
- **数学定义**：LU Decomposition: A = P L U - P is a permutation matrix. - L is a lower triangular matrix with unit diagonal elements. - U is an upper triangular matrix.
- **补充约束**：This function efficiently reuses the LU decomposition of A to solve multiple linear systems with different right-hand sides. If `pivot=False`, no permutation is applied. Supports batch dimensions.
- **题目算子链**：F.linear, torch.mm, torch.exp, torch.sin, torch.min, torch.where, torch.linalg.solve
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `solve_multiple_lu` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `matmul_linear` 类，优先把自然语言描述映射到 PyTorch 算子链：F.linear, torch.mm, torch.exp, torch.sin, torch.min, torch.where, torch.linalg.solve。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 5. openseek-8-7cc09b51ea774c9d9e44cd680d32435c — `tanh`

- **任务类型**：activation
- **Wrapper**：`tanh(input, *, out=None) -> Tensor Args: input (Tensor): the input tensor. Keyword args: out (Tensor, optional): the output tensor.`
- **功能描述**：Returns a new tensor with the hyperbolic tangent of the elements of the input tensor.
- **数学定义**：\text{out}_{i} = \tanh(\text{input}_{i})
- **题目算子链**：torch.mm, torch.tanh, torch.exp, torch.min
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `tanh` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `activation` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, torch.tanh, torch.exp, torch.min。
  1. 此类任务可短实现，重点处理 dim/keepdim/out/inplace/dtype 等参数。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 6. openseek-8-a82af84ea5a14dc9b58d50f504ec8f5e — `relu_sqrt`

- **任务类型**：matmul_linear
- **Wrapper**：`def relu_sqrt(input, inplace=False, out=None) -> Tensor: input (Tensor): The input tensor. inplace (bool, optional): If True, modifies input in-place (if possible). Default is False. out (Tensor, optional): The output tensor.`
- **功能描述**：Applies the rectified linear unit (ReLU) function to each element in input, and then computes the square root of the result. This function ensures all negative values in input are set to zero before applying the square root.
- **数学定义**：\text{out}_i = \sqrt{\max(0, \text{input}_i)}
- **补充约束**：The function modifies input in-place if inplace is set to True.
- **题目算子链**：F.linear, torch.mm, F.relu, F.elu, torch.sqrt, torch.exp, torch.max, torch.min, torch.linalg.qr
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `relu_sqrt` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `matmul_linear` 类，优先把自然语言描述映射到 PyTorch 算子链：F.linear, torch.mm, F.relu, F.elu, torch.sqrt, torch.exp, torch.max, torch.min, torch.linalg.qr。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 7. openseek-8-215a58cbaf6d4e96a69284d61aeeaf3c — `sqrt`

- **任务类型**：linalg
- **Wrapper**：`sqrt(input, *, out=None) -> Tensor; Args: input (Tensor): the input tensor. Keyword args: out (Tensor, optional): the output tensor.`
- **功能描述**：Returns a new tensor with the square-root of the elements of the input tensor. It computes the square root element-wise.
- **数学定义**：\text{out}_{i} = \sqrt{\text{input}_{i}}
- **补充约束**：The function can handle negative inputs, resulting in NaN for those elements.
- **题目算子链**：torch.mm, torch.sqrt, torch.exp, torch.min, torch.linalg.qr
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `sqrt` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `linalg` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, torch.sqrt, torch.exp, torch.min, torch.linalg.qr。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 8. openseek-8-521e38ea57b1490a96e6dc76ff2f57b9 — `sigmoid_argmax`

- **任务类型**：linalg
- **Wrapper**：`sigmoid_argmax(input, dim=None, keepdim=False) -> LongTensor: input (Tensor): The input tensor. dim (int, optional): The dimension to reduce. Default is None, which computes the argmax over all elements. keepdim (bool, optional): Whether the output tensor has :attr:`dim` retained or not. Default is False.`
- **功能描述**：Applies the sigmoid (logistic) function to each element in the input and then computes the indices of the maximum values along the specified dimension or over all elements if no dimension is specified. If dim is not specified, it returns the index of the maximum value in the flattened tensor.
- **数学定义**：sigmoid(x) = 1 / (1 + e^{-x})
- **补充约束**：The function uses PyTorch tensor operations and returns a LongTensor containing indices.
- **题目算子链**：torch.mm, torch.sigmoid, torch.exp, torch.log, torch.argmax, torch.max, torch.min
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `sigmoid_argmax` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `linalg` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, torch.sigmoid, torch.exp, torch.log, torch.argmax, torch.max, torch.min。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 9. openseek-8-b41b0e3a84e4430887282bc3faed8b81 — `sub`

- **任务类型**：reduction
- **Wrapper**：`sub(input, other, *, alpha=1, out=None) -> Tensor; input (Tensor): the input tensor.; other (Tensor or Number): the tensor or number to subtract from input.; alpha (Number): the multiplier for other.; out (Tensor, optional): the output tensor.`
- **功能描述**：Subtracts :attr:`other`, scaled by :attr:`alpha`, from :attr:`input`. The operation is defined as: out_i = input_i - alpha * other_i. Supports broadcasting to a common shape, type promotion, and works with integer, float, and complex inputs.
- **数学定义**：out_i = input_i - alpha * other_i
- **补充约束**：Supports broadcasting, type promotion, and works with integer, float, and complex inputs.
- **题目算子链**：torch.mm, torch.exp, torch.min
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `sub` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `reduction` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, torch.exp, torch.min。
  1. 此类任务可短实现，重点处理 dim/keepdim/out/inplace/dtype 等参数。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 10. openseek-8-ca9d997e5aef49cf8d0bbf48b8a22fbd — `grid_sample`

- **任务类型**：matmul_linear
- **Wrapper**：`def grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=False) -> Tensor`
- **功能描述**：Computes output using input values and pixel locations from grid, supporting spatial (4-D) and volumetric (5-D) input. Interpolates output value at specified grid positions using nearest or bilinear interpolation. Grid values are normalized within [-1, 1] range, and values outside are handled by padding_mode. Often used with affine_grid to build Spatial Transformer Networks.
- **补充约束**：Note: NaN values in grid are interpreted as -1. align_corners=True changes sampled grid positions with image resolution. Default for align_corners changed to False since version 1.2.0. bicubic mode implemented using cubic convolution algorithm with alpha=-0.75; other packages might use different alpha values.
- **题目算子链**：F.linear, torch.mm, torch.exp, torch.sin, torch.min, torch.linalg.vector_norm
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `grid_sample` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `matmul_linear` 类，优先把自然语言描述映射到 PyTorch 算子链：F.linear, torch.mm, torch.exp, torch.sin, torch.min, torch.linalg.vector_norm。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 11. openseek-8-757113b30aed48eabfecadffd0aa1118 — `svd`

- **任务类型**：linalg
- **Wrapper**：`def linalg.svd(A, full_matrices=True, *, driver=None, out=None) -> (Tensor, Tensor, Tensor)`
- **功能描述**：Computes the singular value decomposition (SVD) of a matrix. Supports input of float, double, cfloat and cdouble dtypes. Also supports batches of matrices, and if A is a batch of matrices then the output has the same batch dimensions. The returned decomposition is a named tuple (U, S, Vh) which corresponds to U, S, V^{H} above. The singular values are returned in descending order. The parameter full_matrices chooses between the full (default) and reduced SVD. The driver kwarg may be used in CUDA with a cuSOLVER backend to choose the algorithm used to compute the SVD. The choice of a driver is a trade-off between accuracy and speed.
- **数学定义**：A = U \operatorname{diag}(S) V^{\text{H}} \mathrlap{\qquad U \in \mathbb{K}^{m \times m}, S \in \mathbb{R}^k, V \in \mathbb{K}^{n \times n}}
- **补充约束**：Differences with numpy.linalg.svd: Unlike numpy.linalg.svd, this function always returns a tuple of three tensors and it doesn't support compute_uv argument. Please use torch.linalg.svdvals, which computes only the singular values, instead of compute_uv=False. When full_matrices=True, the gradients with respect to U[..., :, min(m, n):] and Vh[..., min(m, n):, :] will be ignored, as those vectors can be arbitrary bases of the corresponding subspaces. The returned tensors U and V are not unique, n
- **题目算子链**：torch.mm, torch.exp, torch.sin, torch.min, torch.where, torch.linalg.svd, torch.linalg.solve
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `svd` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `linalg` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, torch.exp, torch.sin, torch.min, torch.where, torch.linalg.svd, torch.linalg.solve。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 12. openseek-8-188273dd82f7465dafa78d1430aeb9ee — `i0`

- **任务类型**：reduction
- **Wrapper**：`i0(input, *, out=None) -> Tensor; Args: input (Tensor): the input tensor; Keyword args: out (Tensor, optional): the output tensor.`
- **功能描述**：Computes the zeroth order modified Bessel function of the first kind for each element of the input tensor.
- **数学定义**：\text{out}_{i} = I_0(\text{input}_{i}) = \sum_{k=0}^{\infty} \frac{(\text{input}_{i}^2/4)^k}{(k!)^2}
- **补充约束**：The function calculates the zeroth order modified Bessel function of the first kind, which is a special mathematical function.
- **题目算子链**：torch.mm, torch.exp, torch.sum, torch.min
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `i0` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `reduction` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, torch.exp, torch.sum, torch.min。
  1. 此类任务可短实现，重点处理 dim/keepdim/out/inplace/dtype 等参数。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 13. openseek-8-e7f5dd02bec34352add8c7935b6d790f — `rsqrt`

- **任务类型**：linalg
- **Wrapper**：`rsqrt(input, *, out=None) -> Tensor; Args: input (Tensor): the input tensor.; Keyword args: out (Tensor, optional): the output tensor.`
- **功能描述**：Returns a new tensor with the reciprocal of the square-root of each of the elements of the input tensor.
- **数学定义**：\text{out}_{i} = \frac{1}{\sqrt{\text{input}_{i}}}
- **补充约束**：Note: The function will return 'nan' for negative input values.
- **题目算子链**：torch.mm, torch.sqrt, torch.exp, torch.rsqrt, torch.min, torch.linalg.qr
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `rsqrt` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `linalg` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, torch.sqrt, torch.exp, torch.rsqrt, torch.min, torch.linalg.qr。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 14. openseek-8-b36ca7e6da114f799ec8f9feaf26a769 — `dropout_relu_batch_norm_conv2d`

- **任务类型**：conv_norm_pool
- **Wrapper**：`dropout_relu_batch_norm_conv2d(input: torch.Tensor, weight: torch.Tensor, bias=None, stride=1, padding=0, dilation=1, groups=1, p=0.5, training=True, inplace=False) -> torch.Tensor; Args: input (Tensor): Input tensor of shape \(N, C_{in}, H, W\). weight (Tensor): Convolution filters of shape \(C_{out}, C_{in} / \text{groups}, kH, kW\). bias (Tensor, optional): Bias tensor of shape \(C_{out}\). Default is None. stride (int or tuple, optional): Stride of the convolution. Default: 1 padding (int, t`
- **功能描述**：Applies a 2D convolution followed by batch normalization, ReLU activation, and dropout. Sequentially applies conv2d, batch normalization for stabilizing training and reducing internal covariate shift, ReLU activation function, and dropout where some elements of the tensor are randomly zeroed with probability `p`.
- **补充约束**：Output tensor is returned after applying conv2d, batch normalization, ReLU, and dropout.
- **题目算子链**：F.conv2d, torch.mm, F.batch_norm, custom _rms_norm, F.dropout, F.relu, F.elu, torch.exp, torch.var, torch.min, torch.linalg.vector_norm, torch.where, torch.linalg.eig
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `dropout_relu_batch_norm_conv2d` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `conv_norm_pool` 类，优先把自然语言描述映射到 PyTorch 算子链：F.conv2d, torch.mm, F.batch_norm, custom _rms_norm, F.dropout, F.relu, F.elu, torch.exp, torch.var, torch.min, torch.linalg.vector_norm, torch.where, torch.linalg.eig。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 15. openseek-8-d4a55c1a2818498b9907bdaf461f3d0c — `fused_mv_logsoftmax_dropout`

- **任务类型**：matmul_linear
- **Wrapper**：`fused_mv_logsoftmax_dropout(input, vec, p=0.5, training=True, inplace=False, dim=0, *, out=None) -> Tensor`
- **功能描述**：Performs a fused operation combining matrix-vector multiplication, log-softmax activation, and dropout. The function first performs matrix-vector multiplication on the input matrix and vector. The result is then passed through a log-softmax activation function along the specified dimension. Finally, dropout is applied to the output of the log-softmax operation.
- **数学定义**：Given an input matrix A ∈ ℝ^(n × m) and a vector v ∈ ℝ^m, the function computes: z = A * v s = log(exp(z) / ∑_j exp(z_j)) y = Dropout(s, p) where log(exp(z) / ∑_j exp(z_j)) is the log-softmax function applied along dimension `dim`, and Dropout(s, p) randomly zeroes elements of s with probability p.
- **补充约束**：- The shapes of `input` and `vec` must be compatible for matrix-vector multiplication: the number of columns in `input` must match the size of `vec`. - The `dim` argument in `log_softmax` specifies the dimension along which the log-softmax is computed. Since `z` is a 1-D tensor of shape `(n,)`, `dim` should be `0` or `-1`. - The `dropout` is applied during training when `training=True`. Set `training=False` to disable dropout during evaluation. - This function supports autograd for gradient comp
- **题目算子链**：torch.mm, torch.mv, custom _rms_norm, F.log_softmax, F.softmax, F.dropout, torch.exp, torch.log, torch.sin, torch.max, torch.min, torch.where
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `fused_mv_logsoftmax_dropout` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `matmul_linear` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, torch.mv, custom _rms_norm, F.log_softmax, F.softmax, F.dropout, torch.exp, torch.log, torch.sin, torch.max, torch.min, torch.where。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 16. openseek-8-ddceed268b2546188db6e761c68b9522 — `add`

- **任务类型**：reduction
- **Wrapper**：`add(input, other, *, alpha=1, out=None) -> Tensor; input (Tensor): the input tensor.; other (Tensor or Number): the tensor or number to add to input.; alpha (Number): the multiplier for other.; out (Tensor, optional): the output tensor.`
- **功能描述**：Adds the tensor or number 'other', scaled by 'alpha', to the 'input' tensor. Supports broadcasting to a common shape, type promotion, and accepts integer, float, and complex inputs.
- **数学定义**：\text{{out}}_i = \text{{input}}_i + \text{{alpha}} \times \text{{other}}_i
- **补充约束**：Supports broadcasting and type promotion.
- **题目算子链**：torch.mm, torch.exp, torch.min
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `add` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `reduction` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, torch.exp, torch.min。
  1. 此类任务可短实现，重点处理 dim/keepdim/out/inplace/dtype 等参数。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 17. openseek-8-1a5b486bc85e4509a30bba465ae7a0f4 — `fused_silu_layer_norm_conv2d`

- **任务类型**：conv_norm_pool
- **Wrapper**：`fused_silu_layer_norm_conv2d(x: torch.Tensor, weight: torch.Tensor, conv_weight: torch.Tensor, conv_bias: torch.Tensor = None, conv_stride: int = 1, conv_padding: int = 0, conv_dilation: int = 1, conv_groups: int = 1, ln_eps: float = 1e-5) -> torch.Tensor`
- **功能描述**：Applies 2D Convolution, followed by Layer Normalization and SiLU activation to the input tensor `x`. Sequentially performs convolution on `x`, then applies layer normalization on the convolution output, followed by SiLU activation applied element-wise.
- **补充约束**：Convolution operation parameters include stride, padding, dilation, and groups. Layer Normalization uses an epsilon value. Default values are provided for optional parameters.
- **题目算子链**：F.conv2d, torch.mm, F.layer_norm, custom _rms_norm, F.silu, torch.exp, torch.min, torch.linalg.vector_norm, torch.linalg.eig
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `fused_silu_layer_norm_conv2d` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `conv_norm_pool` 类，优先把自然语言描述映射到 PyTorch 算子链：F.conv2d, torch.mm, F.layer_norm, custom _rms_norm, F.silu, torch.exp, torch.min, torch.linalg.vector_norm, torch.linalg.eig。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 18. openseek-8-1cc25388256c4207b53289a81921b12c — `fused_index_select_eq`

- **任务类型**：linalg
- **Wrapper**：`fused_index_select_eq(input, dim, index, other, *, out=None) -> Tensor. Args: input (Tensor): The input tensor X. dim (int): The dimension along which to index. index (IntTensor or LongTensor): The indices to select along dimension dim. other (Tensor or float): The tensor or value Y to compare with the selected tensor. out (Tensor, optional): Output tensor. Ignored if None. Default: None`
- **功能描述**：Performs a fused operation combining index selection and element-wise equality comparison. It selects elements from the input tensor along a specified dimension using provided indices and then performs an element-wise equality comparison between the selected elements and another tensor or scalar. The result is a boolean tensor of the same shape as the selected elements, indicating where the comparisons are true.
- **数学定义**：Given an input tensor X, dimension ext{dim}, index tensor I, and another tensor or scalar Y, the function computes: 1. **Index Selection:** Select elements from X along dimension ext{dim} using indices I: \[ S = \text{index\_select}(X, \text{dim}, I) \] 2. **Element-wise Equality Comparison:** Compare the selected tensor S with Y element-wise: \[ O = (S == Y) \] The output tensor O is a boolean tensor of the same shape as S.
- **补充约束**：- The shapes of the selected tensor S and other must be broadcastable for the element-wise comparison. - If other is a scalar, it is broadcasted to the shape of S. - The function supports autograd for gradient computation, although the output is a boolean tensor.
- **题目算子链**：torch.mm, custom _rms_norm, torch.exp, torch.sin, torch.min, torch.index_select, torch.where
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `fused_index_select_eq` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `linalg` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, custom _rms_norm, torch.exp, torch.sin, torch.min, torch.index_select, torch.where。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 19. openseek-8-bfb26a8289784475a2c5132dd723f6b3 — `argmax`

- **任务类型**：linalg
- **Wrapper**：`argmax(input, dim, keepdim=False) -> LongTensor`
- **功能描述**：Returns the indices of the maximum values of a tensor across a specified dimension. If the dimension is None, it returns the index of the maximum value in the flattened input tensor. The output tensor can retain the reduced dimension if keepdim is set to True.
- **补充约束**：This is the second value returned by torch.max. See its documentation for the exact semantics of this method.
- **题目算子链**：torch.mm, torch.exp, torch.argmax, torch.max, torch.min
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `argmax` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `linalg` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, torch.exp, torch.argmax, torch.max, torch.min。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 20. openseek-8-d8c481c6232b4f68baba55a9f6fcfa8f — `fused_lu_solve`

- **任务类型**：matmul_linear
- **Wrapper**：`def fused_lu_solve(A: Tensor, b: Tensor) -> Tensor: A: The input matrix `A` of shape `(n, n)`. b: The right-hand side tensor `b` of shape `(n,)`.`
- **功能描述**：Computes the solution `x` to the equation `Ax = b` using LU decomposition. Given matrix `A`, this function performs LU decomposition and then solves for `x` in `L @ U @ x = b`, where `P`, `L`, and `U` are derived from the LU decomposition.
- **数学定义**：Solves `Ax = b` using LU decomposition, where `A = P @ L @ U` and `L @ U @ x = b`.
- **补充约束**：The function uses LU decomposition to solve linear equations.
- **题目算子链**：F.linear, torch.mm, custom _rms_norm, torch.exp, torch.sin, torch.min, torch.where, torch.linalg.solve
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `fused_lu_solve` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `matmul_linear` 类，优先把自然语言描述映射到 PyTorch 算子链：F.linear, torch.mm, custom _rms_norm, torch.exp, torch.sin, torch.min, torch.where, torch.linalg.solve。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 21. openseek-8-45c89b4e8cef4315bfaa885afd98c669 — `normalize_pairwise_distance`

- **任务类型**：linalg
- **Wrapper**：`normalize_pairwise_distance(x1, x2, p_distance=2.0, eps_distance=1e-6, keepdim=False, p_norm=2, dim_norm=1, eps_norm=1e-12) -> Tensor; x1 (Tensor): The first input tensor; x2 (Tensor): The second input tensor, must have the same shape as `x1`; p_distance (float): The norm degree for computing the pairwise distance. Default: 2.0; eps_distance (float): Small value to avoid division by zero in pairwise distance calculation. Default: 1e-6; keepdim (bool): Whether to keep the reduced dimensions in th`
- **功能描述**：Computes the pairwise distance between `x1` and `x2` using the specified norm, then normalizes the resulting distances along the specified dimension. This combined operation is useful for obtaining normalized distance values between two sets of vectors.
- **数学定义**：\text{distance} = \frac{\text{pairwise\_distance}(x1, x2)}{\max(\lVert \text{pairwise\_distance}(x1, x2) \rVert_p, \epsilon)}
- **补充约束**：The combined operation is useful for obtaining normalized distance values between two sets of vectors.
- **题目算子链**：torch.mm, torch.exp, torch.sin, torch.max, torch.min, torch.linalg.vector_norm
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `normalize_pairwise_distance` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `linalg` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, torch.exp, torch.sin, torch.max, torch.min, torch.linalg.vector_norm。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 22. openseek-8-eed93a70ff1546d8aa68ef247bc922bd — `max`

- **任务类型**：linalg
- **Wrapper**：`max(input, dim, keepdim=False, *, out=None) -> (Tensor, LongTensor) input (Tensor): the input tensor. dim (int): the dimension to reduce. keepdim (bool): whether the output tensor has :attr:`dim` retained or not. Default: ``False``. out (tuple, optional): the result tuple of two output tensors (max, max_indices).`
- **功能描述**：Returns a namedtuple (values, indices) where values is the maximum value of each row of the input tensor in the given dimension dim. Indices is the index location of each maximum value found (argmax). If keepdim is True, the output tensors are of the same size as input except in the dimension dim where they are of size 1. Otherwise, dim is squeezed, resulting in the output tensors having 1 fewer dimension than input. If there are multiple maximal values in a reduced row, the indices of the first maximal value are returned.
- **补充约束**：If there are multiple maximal values in a reduced row then the indices of the first maximal value are returned.
- **题目算子链**：torch.mm, torch.exp, torch.argmax, torch.max, torch.min, torch.where
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `max` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `linalg` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, torch.exp, torch.argmax, torch.max, torch.min, torch.where。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 23. openseek-8-f7521a91f83c43da8791d9f29ea31535 — `log_softmax_linear`

- **任务类型**：matmul_linear
- **Wrapper**：`log_softmax_linear(input, weight, bias=None, dim=-1, dtype=None) -> Tensor: input (Tensor): The input tensor of shape `(*, in_features)`, where `*` represents any number of additional dimensions. weight (Tensor): The weight matrix of shape `(out_features, in_features)`. bias (Tensor, optional): The optional bias tensor of shape `(out_features)`. Default: None. dim (int): The dimension along which log_softmax will be computed. Default: -1. dtype (:class:`torch.dtype`, optional): The desired data `
- **功能描述**：Applies a linear transformation to the input tensor followed by the log_softmax activation function. This combined operation is optimized to be numerically stable and efficient, applying both a linear transformation and log-softmax in one step.
- **数学定义**：\text{out} = \log\left(\frac{\exp(\text{linear}(\text{input}))}{\sum_j \exp(\text{linear}(\text{input})_j)}\right) y = xA^T + b
- **补充约束**：The values along the specified dimension represent log probabilities and sum to 1.
- **题目算子链**：F.linear, torch.mm, F.log_softmax, F.softmax, torch.exp, torch.log, torch.sum, torch.max, torch.min, torch.where, torch.linalg.eig
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `log_softmax_linear` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `matmul_linear` 类，优先把自然语言描述映射到 PyTorch 算子链：F.linear, torch.mm, F.log_softmax, F.softmax, torch.exp, torch.log, torch.sum, torch.max, torch.min, torch.where, torch.linalg.eig。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 24. openseek-8-3a629a95117a4ca18edda2c3bc560fc0 — `relu`

- **任务类型**：matmul_linear
- **Wrapper**：`relu(input, inplace=False) -> Tensor`
- **功能描述**：Applies the rectified linear unit function element-wise. This operation compares each element in the input tensor to zero and returns the element itself if it is greater than zero or zero otherwise. The operation can be performed in-place, modifying the input tensor directly if inplace=True.
- **数学定义**：ReLU(x) = (x)^+ = max(0, x)
- **补充约束**：See torch.nn.ReLU for more details.
- **题目算子链**：F.linear, torch.mm, F.relu, F.elu, torch.exp, torch.mean, torch.max, torch.min, torch.where
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `relu` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `matmul_linear` 类，优先把自然语言描述映射到 PyTorch 算子链：F.linear, torch.mm, F.relu, F.elu, torch.exp, torch.mean, torch.max, torch.min, torch.where。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 25. openseek-8-495afd5d58c84aff9fd26367f8c28f40 — `least_squares_qr`

- **任务类型**：matmul_linear
- **Wrapper**：`def least_squares_qr(A, b, *, mode='reduced', out=None) -> Tensor: A (Tensor): Coefficient matrix of shape (*, m, n), where * is zero or more batch dimensions. b (Tensor): Right-hand side vector or matrix of shape (*, m) or (*, m, k), where k is the number of right-hand sides. mode (str, optional): Determines the type of QR decomposition to use. One of 'reduced' (default) or 'complete'. See torch.linalg.qr for details. out (Tensor, optional): Output tensor. Ignored if None. Default: None.`
- **功能描述**：Solves the least squares problem for an overdetermined system of linear equations using QR decomposition. It computes the least squares solution x that minimizes the Euclidean 2-norm |Ax - b|_2, where A is the coefficient matrix and b is the right-hand side vector or matrix.
- **数学定义**：The QR decomposition of A is given by A = QR, where Q is a matrix with orthonormal columns and R is an upper triangular matrix. The least squares solution is x = R^{-1} Q^H b.
- **补充约束**：The function utilizes QR decomposition to efficiently solve overdetermined linear systems by finding the least squares solution.
- **题目算子链**：F.linear, torch.mm, torch.exp, torch.sin, torch.min, torch.linalg.vector_norm, torch.where, torch.linalg.qr, torch.linalg.solve
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `least_squares_qr` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `matmul_linear` 类，优先把自然语言描述映射到 PyTorch 算子链：F.linear, torch.mm, torch.exp, torch.sin, torch.min, torch.linalg.vector_norm, torch.where, torch.linalg.qr, torch.linalg.solve。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 26. openseek-8-9e761bfd59194360afd2b637f96c63a7 — `determinant_via_qr`

- **任务类型**：linalg
- **Wrapper**：`determinant_via_qr(A, *, mode='reduced', out=None) -> Tensor`
- **功能描述**：Computes the determinant of a square matrix using QR decomposition. It performs QR decomposition of a square matrix A in \mathbb{K}^{n imes n} (where \mathbb{K} is either \mathbb{R} or \mathbb{C}) and computes the determinant by taking the product of the diagonal elements of R.
- **数学定义**：The QR decomposition of A is: A = Q R, where Q is an orthogonal/unitary matrix, R is an upper triangular matrix. The determinant is given by: \det(A) = \det(Q)\cdot \prod_{i=1}^{n} R_{ii}. For real matrices, \det(Q) = \pm 1. For complex matrices, |\det(Q)| = 1.
- **补充约束**：Numerical stability considerations are important, especially for ill-conditioned matrices. The function explicitly computes \det(Q) to account for the sign. For complex matrices, the result may be complex.
- **题目算子链**：torch.mm, custom _rms_norm, torch.exp, torch.sin, torch.min, torch.where, torch.linalg.qr, torch.linalg.det
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `determinant_via_qr` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `linalg` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, custom _rms_norm, torch.exp, torch.sin, torch.min, torch.where, torch.linalg.qr, torch.linalg.det。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 27. openseek-8-cc45f93ffa22476c8063c1ef6a8207d6 — `fused_tile_exp`

- **任务类型**：reduction
- **Wrapper**：`fused_tile_exp(input, dims, *, out=None) -> Tensor; input (Tensor): The input tensor X whose elements are to be repeated and exponentiated.; dims (tuple of int): The number of repetitions for each dimension. If `dims` has fewer dimensions than `input`, ones are prepended to `dims` until all dimensions are specified.; out (Tensor, optional): Output tensor. Ignored if `None`. Default: `None`.`
- **功能描述**：Performs a fused operation combining tiling (repeating elements) and the exponential function. The input tensor is first repeated along each dimension according to the specified `dims` using the tiling operation, then the exponential function is applied element-wise to the resulting tensor.
- **数学定义**：Given an input tensor X and a tuple of dimensions ext{dims}, the function computes: 1. **Tiling:** The input tensor is repeated along each dimension according to the specified number of times in `dims`: Y = tile(X, dims) 2. **Exponential Function:** The exponential function is applied element-wise to the tiled tensor: Z = exp(Y)
- **补充约束**：The `dims` parameter controls how many times the input tensor is repeated along each dimension. If `dims` specifies fewer dimensions than `input`, ones are prepended to `dims` until all dimensions are specified. The function supports autograd for gradient computation. All operations are differentiable and support backpropagation.
- **题目算子链**：torch.mm, custom _rms_norm, torch.exp, torch.sin, torch.min
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `fused_tile_exp` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `reduction` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, custom _rms_norm, torch.exp, torch.sin, torch.min。
  1. 此类任务可短实现，重点处理 dim/keepdim/out/inplace/dtype 等参数。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 28. openseek-8-f08e0eca68224df896e459818a75b5e3 — `sqrt_tanh`

- **任务类型**：linalg
- **Wrapper**：`def sqrt_tanh(input, out=None) -> Tensor: input (Tensor): The input tensor. out (Tensor, optional): The output tensor.`
- **功能描述**：Computes the square root of each element in the input tensor, and then applies the hyperbolic tangent (tanh) function to the square-rooted values. The function returns a tensor where each element is the result of applying sqrt followed by tanh to each element of the input.
- **数学定义**：\text{out}_{i} = \tanh(\sqrt{\text{input}_{i}})
- **补充约束**：Using a tensor with some negative values results in NaN for those elements.
- **题目算子链**：torch.mm, torch.tanh, torch.sqrt, torch.exp, torch.sin, torch.min, torch.where, torch.linalg.qr
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `sqrt_tanh` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `linalg` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, torch.tanh, torch.sqrt, torch.exp, torch.sin, torch.min, torch.where, torch.linalg.qr。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 29. openseek-8-5b28a4d5afff45b1879d276494830c51 — `silu_batch_norm`

- **任务类型**：conv_norm_pool
- **Wrapper**：`silu_batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-5) -> Tensor; input (Tensor): The input tensor for Batch Normalization.; running_mean (Tensor): The running mean tensor (used during evaluation).; running_var (Tensor): The running variance tensor (used during evaluation).; weight (Tensor, optional): The weight tensor for Batch Normalization scaling. Default: None.; bias (Tensor, optional): The bias tensor for Batch Normalization. Defau`
- **功能描述**：Applies Batch Normalization over an input tensor across channels, followed by the Sigmoid Linear Unit (SiLU) activation function applied element-wise. This combined operation normalizes the input tensor and then applies a non-linear SiLU activation.
- **数学定义**：The combined operation is defined as: \text{out} = \text{silu}(\text{BatchNorm}(x)), where the SiLU function is defined as: \text{silu}(x) = x * \sigma(x), \text{where } \sigma(x) = \frac{1}{1 + \exp(-x)}
- **补充约束**：Returns: A tensor that has undergone batch normalization and SiLU activation.
- **题目算子链**：F.linear, torch.mm, F.batch_norm, F.silu, torch.sigmoid, torch.exp, torch.mean, torch.var, torch.min, torch.linalg.vector_norm, torch.where, torch.linalg.eig
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `silu_batch_norm` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `conv_norm_pool` 类，优先把自然语言描述映射到 PyTorch 算子链：F.linear, torch.mm, F.batch_norm, F.silu, torch.sigmoid, torch.exp, torch.mean, torch.var, torch.min, torch.linalg.vector_norm, torch.where, torch.linalg.eig。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 30. openseek-8-781a002b0f944a9989a836c8da9dec47 — `index_fill_`

- **任务类型**：linalg
- **Wrapper**：`index_fill_(dim, index, value) -> Tensor`
- **功能描述**：Fills the elements of the self tensor with a specified value by selecting the indices in the order given in the index tensor. The operation is performed along a specified dimension.
- **补充约束**：The function modifies the tensor in-place.
- **题目算子链**：torch.mm, torch.exp, torch.min, Tensor.index_fill_
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `index_fill_` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `linalg` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, torch.exp, torch.min, Tensor.index_fill_。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 31. openseek-8-6fc781d88f5743efaacedd57d654066f — `fused_cross_entropy_softmax_layernorm`

- **任务类型**：conv_norm_pool
- **Wrapper**：`fused_cross_entropy_softmax_layernorm(logits, targets, normalized_shape, weight=None, ignore_index=-100, reduction='mean', label_smoothing=0.0, eps=1e-5, *, out=None) -> Tuple[Tensor, Tensor] - logits (Tensor): Input logits of shape (N, C) or (N, C, *), where N is the batch size and C is the number of classes. - targets (Tensor): Ground truth class indices or class probabilities. If containing class indices: shape (N) or (N, *) with values 0 <= targets_i < C. If containing class probabilities: s`
- **功能描述**：Performs a fused operation combining cross-entropy loss computation, softmax activation, and layer normalization. It computes the cross-entropy loss for given logits and targets, applies softmax activation to the logits, and then applies layer normalization to the resulting probabilities.
- **数学定义**：Given input logits \mathbf{z} and target labels \mathbf{y}, the function computes: 1. **Cross-Entropy Loss:**
- **补充约束**：- The `logits` tensor should contain raw, unnormalized scores for each class. - The `targets` can be class indices or class probabilities matching the shape of `logits`. - The `normalized_shape` argument in `layer_norm` should correspond to the dimensions over which you want to apply normalization. - If `elementwise_affine` parameters (`weight` and `bias`) are needed in `layer_norm`, they can be defined and passed accordingly. - All operations support autograd for gradient computation.
- **题目算子链**：torch.mm, F.layer_norm, custom _rms_norm, F.softmax, F.cross_entropy, torch.sqrt, torch.exp, torch.log, torch.mean, torch.sum, torch.var, torch.max, torch.min, torch.linalg.vector_norm, torch.where, torch.linalg.qr, torch.linalg.eig
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `fused_cross_entropy_softmax_layernorm` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `conv_norm_pool` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, F.layer_norm, custom _rms_norm, F.softmax, F.cross_entropy, torch.sqrt, torch.exp, torch.log, torch.mean, torch.sum, torch.var, torch.max, torch.min, torch.linalg.vector_norm, torch.where, torch.linalg.qr, torch.linalg.eig。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 32. openseek-8-6895c51d60b848399f2d206e16b9c587 — `input`

- **任务类型**：linalg
- **Wrapper**：`input (Tensor): the input tensor. dim (int or tuple of ints): the dimension or dimensions to reduce. keepdim (bool): whether the output tensor has dim retained or not. dtype (torch.dtype, optional): the desired data type of returned tensor. If specified, the input tensor is casted to dtype before the operation is performed. This is useful for preventing data type overflows. Default: None. out (Tensor, optional): the output tensor.`
- **功能描述**：Returns the mean value of each row of the input tensor in the given dimension dim. If dim is a list of dimensions, reduce over all of them. If keepdim is True, the output tensor is of the same size as input except in the dimension(s) dim where it is of size 1. Otherwise, dim is squeezed, resulting in the output tensor having 1 (or len(dim)) fewer dimension(s).
- **补充约束**：See also torch.nanmean which computes the mean value of non-NaN elements.
- **题目算子链**：torch.mm, torch.exp, torch.mean, torch.min, torch.where
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `input` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `linalg` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, torch.exp, torch.mean, torch.min, torch.where。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 33. openseek-8-2a6dd4dd9e43480182989a6b5a5cac1d — `eig`

- **任务类型**：linalg
- **Wrapper**：`def linalg.eig(A, *, out=None) -> (Tensor, Tensor) Args: A (Tensor): tensor of shape `(*, n, n)` where `*` is zero or more batch dimensions consisting of diagonalizable matrices. Keyword args: out (tuple, optional): output tuple of two tensors. Ignored if `None`. Default: `None`.`
- **功能描述**：Computes the eigenvalue decomposition of a square matrix if it exists. Supports input of float, double, cfloat and cdouble dtypes. Also supports batches of matrices, and if A is a batch of matrices then the output has the same batch dimensions. The returned eigenvalues are not guaranteed to be in any specific order. The eigenvalues and eigenvectors of a real matrix may be complex. When inputs are on a CUDA device, this function synchronizes that device with the CPU. Assumes that A is diagonalizable. The returned eigenvectors are normalized to have norm 1. The eigenvectors of a matrix are not unique, nor are they continuous with respect to A. Gradients computed using the eigenvectors tensor will only be finite when A has distinct eigenvalues.
- **数学定义**：A = V \operatorname{diag}(\Lambda) V^{-1}\mathrlap{\qquad V \in \mathbb{C}^{n \times n}, \Lambda \in \mathbb{C}^n}
- **补充约束**：The eigenvalues and eigenvectors of a real matrix may be complex. When inputs are on a CUDA device, this function synchronizes that device with the CPU. Assumes that A is diagonalizable. The returned eigenvectors are normalized to have norm 1. The eigenvectors of a matrix are not unique, nor are they continuous with respect to A. Gradients computed using the eigenvectors tensor will only be finite when A has distinct eigenvalues.
- **题目算子链**：torch.mm, torch.exp, torch.sin, torch.sum, torch.min, torch.linalg.vector_norm, torch.where, torch.linalg.eig
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `eig` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `linalg` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, torch.exp, torch.sin, torch.sum, torch.min, torch.linalg.vector_norm, torch.where, torch.linalg.eig。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 34. openseek-8-71075dbf0e0a4c1592be4985ebd1ba1b — `logsumexp`

- **任务类型**：reduction
- **Wrapper**：`def logsumexp(input, dim, keepdim=False, *, out=None) -> Tensor`
- **功能描述**：This function computes the logarithm of the sum of exponentials of input elements along the specified dimension. It is useful for numerical stability when computing log probabilities.
- **数学定义**：logsumexp(x) = log(sum(exp(x)))
- **补充约束**：Alias for torch.logsumexp.
- **题目算子链**：torch.mm, torch.exp, torch.logsumexp, torch.log, torch.sum, torch.min
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `logsumexp` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `reduction` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, torch.exp, torch.logsumexp, torch.log, torch.sum, torch.min。
  1. 此类任务可短实现，重点处理 dim/keepdim/out/inplace/dtype 等参数。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 35. openseek-8-61ae6f57db984c43b3e82cbaa5f6753f — `fused_embedding_add_tanh`

- **任务类型**：linalg
- **Wrapper**：`fused_embedding_add_tanh(input_indices, weight, other, *, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False, out=None) -> Tensor; input_indices (LongTensor): Tensor containing indices into the embedding matrix, of arbitrary shape (*); weight (Tensor): The embedding matrix of shape (V, D), where V is the number of embeddings (vocabulary size), and D is the embedding dimension; other (Tensor): Tensor to be added to the embeddings, must be broadcastable to the s`
- **功能描述**：Performs a fused operation combining embedding lookup, element-wise addition, and tanh activation. The function retrieves embeddings from an embedding matrix using input indices, adds another tensor to these embeddings, and applies a tanh activation function to the result. It supports options for padding indices, max norm for embeddings, scaling gradients by frequency, and sparse gradients.
- **数学定义**：Given input indices \mathbf{i}, embedding weight matrix W, and tensor O, the function computes: \[ \begin{align*} E &= \text{Embedding}(\mathbf{i}, W) \\ S &= E + O \\ Y &= \tanh(S) \end{align*} \]
- **补充约束**：- The `other` tensor must be broadcastable to the shape of the embeddings retrieved by `torch.nn.functional.embedding`. - All parameters related to `torch.nn.functional.embedding` are passed through to allow for options like `padding_idx`, `max_norm`, etc. - This function supports autograd for gradient computation. - All operations are differentiable and support backpropagation.
- **题目算子链**：torch.mm, custom _rms_norm, torch.tanh, torch.exp, torch.sin, torch.max, torch.min, torch.linalg.vector_norm, F.embedding, torch.where, torch.linalg.inv, torch.linalg.eig
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `fused_embedding_add_tanh` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `linalg` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, custom _rms_norm, torch.tanh, torch.exp, torch.sin, torch.max, torch.min, torch.linalg.vector_norm, F.embedding, torch.where, torch.linalg.inv, torch.linalg.eig。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 36. openseek-8-b76244965a1e4ee78c7a2e17f1ab9fdb — `fused_mv_sigmoid_sub`

- **任务类型**：matmul_linear
- **Wrapper**：`fused_mv_sigmoid_sub(input, vec, other, alpha=1, *, out=None) -> Tensor; input (Tensor): Input matrix A of shape (n, m); vec (Tensor): Input vector \mathbf{v} of shape (m); other (Tensor or Number): Tensor or scalar b to subtract from the sigmoid output, scaled by \alpha; alpha (Number, optional): Scalar multiplier for other. Default: `1`; out (Tensor, optional): Output tensor. Ignored if `None`. Default: `None``
- **功能描述**：Performs a fused operation combining matrix-vector multiplication, sigmoid activation, and subtraction.
- **数学定义**：Given an input matrix A, a vector \mathbf{v}, and another tensor or scalar b, the function computes: \[ \begin{align*} \mathbf{z} &= A \mathbf{v} \\ \mathbf{s} &= \sigma(\mathbf{z}) = \frac{1}{1 + \exp(-\mathbf{z})} \\ \mathbf{y} &= \mathbf{s} - \alpha b \end{align*} \]
- **补充约束**：- The shapes of `input` and `vec` must be compatible for matrix-vector multiplication. - The `other` tensor must be broadcastable to the shape of the output from the sigmoid function. - The function supports autograd for gradient computation. - All operations are differentiable and support backpropagation.
- **题目算子链**：torch.mm, torch.mv, custom _rms_norm, torch.sigmoid, torch.exp, torch.min
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `fused_mv_sigmoid_sub` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `matmul_linear` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, torch.mv, custom _rms_norm, torch.sigmoid, torch.exp, torch.min。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 37. openseek-8-97a0096f362e4089a20674eaad0d6173 — `add_gelu`

- **任务类型**：matmul_linear
- **Wrapper**：`def add_gelu(input, other, alpha=1, approximate='none', out=None) -> Tensor: input (Tensor): The input tensor. other (Tensor or Number): The tensor or number to add to input. alpha (Number, optional): The multiplier for other. Default is 1. approximate (str, optional): The approximation method for GELU. Default is 'none'. out (Tensor, optional): The output tensor.`
- **功能描述**：Adds the tensor or number `other`, scaled by the multiplier `alpha`, to the input tensor `input`, and then applies the Gaussian Error Linear Units (GELU) activation function to the result.
- **数学定义**：\text{out}_i = \text{GELU}(\text{input}_i + \text{alpha} \times \text{other}_i) where GELU is defined as: - \text{GELU}(x) = x * \Phi(x) when approximate is 'none', - \text{GELU}(x) = 0.5 * x * (1 + \text{Tanh}(\sqrt{2 / \pi} * (x + 0.044715 * x^3))) when approximate is 'tanh'.
- **补充约束**：The GELU function is defined with two methods: an exact method using the Cumulative Distribution Function for Gaussian Distribution, and an approximate method using a tanh-based formula.
- **题目算子链**：F.linear, torch.mm, F.gelu, torch.tanh, F.elu, torch.sqrt, torch.exp, torch.sin, torch.min, torch.where, torch.linalg.qr
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `add_gelu` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `matmul_linear` 类，优先把自然语言描述映射到 PyTorch 算子链：F.linear, torch.mm, F.gelu, torch.tanh, F.elu, torch.sqrt, torch.exp, torch.sin, torch.min, torch.where, torch.linalg.qr。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 38. openseek-8-3eb056fc79754edcb5e161265c57732f — `fused_cosine_embedding_loss_with_normalization`

- **任务类型**：linalg
- **Wrapper**：`def fused_cosine_embedding_loss_with_normalization(input1: torch.Tensor, input2: torch.Tensor, target: torch.Tensor, margin: float = 0, reduction: str = 'mean') -> torch.Tensor: input1 (Tensor): First input tensor to be normalized and compared. input2 (Tensor): Second input tensor to be normalized and compared. target (Tensor): Tensor label with values 1 or -1, where 1 encourages similarity and -1 encourages dissimilarity. margin (float, optional): Margin for dissimilarity. Default: 0. reduction`
- **功能描述**：Computes cosine embedding loss between two normalized tensors. This function first normalizes the inputs along the specified dimension using L2 normalization and then calculates the cosine embedding loss. The loss encourages similarity when the target is 1 and dissimilarity when the target is -1. It accepts optional parameters margin for dissimilarity control and reduction method for output aggregation.
- **补充约束**：The inputs are first L2 normalized along dimension 1 before loss calculation. The reduction parameter can be 'none', 'mean', or 'sum', with default as 'mean'.
- **题目算子链**：torch.mm, torch.exp, torch.cos, torch.sin, torch.mean, torch.sum, torch.min, torch.linalg.vector_norm, F.embedding, torch.where
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `fused_cosine_embedding_loss_with_normalization` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `linalg` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, torch.exp, torch.cos, torch.sin, torch.mean, torch.sum, torch.min, torch.linalg.vector_norm, F.embedding, torch.where。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 39. openseek-8-a6024651bc2a4554b4bf6898bdd0a33e — `fused_transformer_block`

- **任务类型**：conv_norm_pool
- **Wrapper**：`fused_transformer_block(input, weight1, weight2, residual, dropout_p=0.1, eps=1e-5, *, out=None) -> Tensor; input (Tensor): Input tensor of shape (*, N, D_in), where * denotes any number of batch dimensions.; weight1 (Tensor): Weight matrix of shape (D_in, D_k).; weight2 (Tensor): Weight matrix of shape (D_k, D_out).; residual (Tensor): Residual tensor to be added before layer normalization, must be broadcastable to the shape of Z_4.; dropout_p (float, optional): Probability of an element to be `
- **功能描述**：Performs a sequence of operations commonly used in transformer models, combining matrix multiplication, softmax, dropout, another matrix multiplication, layer normalization, and addition (residual connection).
- **数学定义**：Given an input tensor X, weight matrices W_1 and W_2, and a residual tensor R, the function computes: \[ \begin{align*} Z_1 &= X W_1 \\ Z_2 &= \text{softmax}(Z_1) \\ Z_3 &= \text{dropout}(Z_2, p) \\ Z_4 &= Z_3 W_2 \\ Y &= \text{LayerNorm}(Z_4 + R, \gamma, \beta, \epsilon) \end{align*} \] where: - \text{softmax}(Z) is applied along the last dimension. - \text{dropout}(Z, p) randomly zeroes elements of Z with probability p. - \text{LayerNorm} applies layer normalization with learnable parameters \
- **补充约束**：- The dimensions of `input` and `weight1` must be compatible for matrix multiplication: the last dimension of `input` must match the first dimension of `weight1`. - The output of the first matrix multiplication has shape `(*, N, D_k)`. - The `softmax` is applied along the last dimension (`dim=-1`). - The `dropout` is applied during training. Set `training=False` to disable dropout during evaluation. - The `layer_norm` is applied over the last dimension of the input tensor. - The `residual` tenso
- **题目算子链**：torch.matmul, torch.mm, F.layer_norm, custom _rms_norm, F.softmax, F.dropout, torch.exp, torch.max, torch.min, torch.linalg.vector_norm, torch.where, torch.linalg.eig
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `fused_transformer_block` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `conv_norm_pool` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.matmul, torch.mm, F.layer_norm, custom _rms_norm, F.softmax, F.dropout, torch.exp, torch.max, torch.min, torch.linalg.vector_norm, torch.where, torch.linalg.eig。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 40. openseek-8-0b47029f5b3241c6ba2201b9babd9935 — `log1p`

- **任务类型**：linalg
- **Wrapper**：`log1p(input, *, out=None) -> Tensor; Args: input (Tensor): the input tensor. Keyword args: out (Tensor, optional): the output tensor.`
- **功能描述**：Returns a new tensor with the natural logarithm of (1 + input). This function is more accurate than torch.log for small values of input.
- **数学定义**：y_i = \log_{e} (x_i + 1)
- **补充约束**：This function is more accurate than torch.log for small values of input.
- **题目算子链**：torch.mm, torch.exp, torch.log, torch.min
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `log1p` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `linalg` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, torch.exp, torch.log, torch.min。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 41. openseek-8-f976b81fe08840119706d14983d18384 — `sigmoid_batch_norm`

- **任务类型**：conv_norm_pool
- **Wrapper**：`def sigmoid_batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-5) -> Tensor`
- **功能描述**：Applies Batch Normalization over the input tensor across each channel, followed by applying the sigmoid activation function element-wise to the normalized result. This is useful for scaling the output to a range between 0 and 1 after normalization.
- **数学定义**：\text{out} = \sigma\left(\frac{\text{input} - \text{mean}}{\sqrt{\text{var} + \epsilon}} * \gamma + \beta \right) where \sigma(x) = \frac{1}{1 + \exp(-x)} is the sigmoid function.
- **补充约束**：The function normalizes the input tensor using batch normalization and then applies the sigmoid activation function to scale the output between 0 and 1.
- **题目算子链**：torch.mm, F.batch_norm, torch.sigmoid, torch.sqrt, torch.exp, torch.sin, torch.mean, torch.var, torch.min, torch.linalg.vector_norm, torch.where, torch.linalg.qr, torch.linalg.eig
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `sigmoid_batch_norm` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `conv_norm_pool` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, F.batch_norm, torch.sigmoid, torch.sqrt, torch.exp, torch.sin, torch.mean, torch.var, torch.min, torch.linalg.vector_norm, torch.where, torch.linalg.qr, torch.linalg.eig。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 42. openseek-8-64b5d9d1883b4672a73b22b9040ef9e1 — `fused_hardsigmoid_batch_norm`

- **任务类型**：conv_norm_pool
- **Wrapper**：`fused_hardsigmoid_batch_norm(x: torch.Tensor, running_mean: torch.Tensor, running_var: torch.Tensor, weight: torch.Tensor = None, bias: torch.Tensor = None, training: bool = False, momentum: float = 0.1, eps: float = 1e-5, inplace: bool = False) -> torch.Tensor: Args: x (Tensor): Input tensor for batch normalization and activation. running_mean (Tensor): The running mean buffer (persistent). running_var (Tensor): The running variance buffer (persistent). weight (Tensor, optional): Learnable weig`
- **功能描述**：Applies Batch Normalization followed by the Hardsigmoid activation function on the input tensor `x`. This function performs batch normalization on `x` using the specified parameters and then applies Hardsigmoid activation element-wise on the normalized output.
- **补充约束**：The function includes optional parameters for learnable weight and bias, a training flag to update running estimates, momentum for running mean and variance, a small constant `eps` for numerical stability, and an `inplace` option for Hardsigmoid.
- **题目算子链**：torch.mm, F.batch_norm, custom _rms_norm, torch.sigmoid, F.hardsigmoid, torch.exp, torch.sin, torch.mean, torch.var, torch.min, torch.linalg.vector_norm, torch.linalg.eig
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `fused_hardsigmoid_batch_norm` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `conv_norm_pool` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, F.batch_norm, custom _rms_norm, torch.sigmoid, F.hardsigmoid, torch.exp, torch.sin, torch.mean, torch.var, torch.min, torch.linalg.vector_norm, torch.linalg.eig。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 43. openseek-8-289f73ed62e740c8b6cdf08f2ea929da — `zeta`

- **任务类型**：reduction
- **Wrapper**：`zeta(input, other, *, out=None) -> Tensor; Args: input (Tensor): the input tensor corresponding to `x`. other (Tensor): the input tensor corresponding to `q`. Keyword args: out (Tensor, optional): the output tensor.`
- **功能描述**：Computes the Hurwitz zeta function, elementwise. The function calculates the sum of the series for each element in the input tensors, which represent the parameters x and q of the Hurwitz zeta function. The Riemann zeta function is a special case when q equals 1.
- **数学定义**：\zeta(x, q) = \sum_{k=0}^{\infty} \frac{1}{(k + q)^x}
- **补充约束**：The Riemann zeta function corresponds to the case when `q = 1`
- **题目算子链**：torch.mm, torch.exp, torch.sum, torch.min, torch.special.zeta / finite sum
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `zeta` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `reduction` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, torch.exp, torch.sum, torch.min, torch.special.zeta / finite sum。
  1. 此类任务可短实现，重点处理 dim/keepdim/out/inplace/dtype 等参数。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 44. openseek-8-da7522a7c6924bf39ca8e4a82bb53c5a — `symmetric_matrix_vector_norm`

- **任务类型**：matmul_linear
- **Wrapper**：`def symmetric_matrix_vector_norm(A: torch.Tensor, x: torch.Tensor, alpha: float, beta: float, p: float = 2.0) -> torch.Tensor: A (Tensor): A symmetric matrix of shape `(n, n)`. x (Tensor): A vector of shape `(n,)`. alpha (float): Scalar multiplier for the matrix-vector product. beta (float): Scalar multiplier added to `y`. p (float, optional): Order of the norm. Default is 2.0 (Euclidean norm).`
- **功能描述**：Computes the matrix-vector product for a symmetric matrix `A` and a vector `x`, with scaling factors `alpha` and `beta`. Then calculates the norm of the resulting vector `y`. The operation performed is: 1. `y = alpha * torch.mv(A, x) + beta * y`, assuming `A` is symmetric. 2. `norm = torch.norm(y, p)`.
- **数学定义**：y = alpha * torch.mv(A, x) + beta * y norm = torch.norm(y, p)
- **补充约束**：Assumes `A` is symmetric.
- **题目算子链**：torch.mm, torch.mv, torch.exp, torch.sum, torch.min, torch.linalg.vector_norm
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `symmetric_matrix_vector_norm` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `matmul_linear` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, torch.mv, torch.exp, torch.sum, torch.min, torch.linalg.vector_norm。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 45. openseek-8-d4cabb13b39a499a9a6b23e853e99c8b — `softplus_linear`

- **任务类型**：matmul_linear
- **Wrapper**：`softplus_linear(input, weight, bias=None, beta=1, threshold=20) -> Tensor`
- **功能描述**：Applies a linear transformation to the input tensor, followed by the Softplus activation function applied element-wise. This combined operation first performs a linear transformation and then introduces non-linearity with Softplus, which is smoother than ReLU and approximates it for large values. The function is particularly designed to improve numerical stability by reverting to a linear function for values above a specified threshold.
- **数学定义**：The combined operation is defined as: out = Softplus(Linear(x)), where the Softplus function is defined as: Softplus(x) = (1/β) * log(1 + exp(β * x))
- **补充约束**：For values exceeding the threshold, the function helps maintain numerical stability by approximating a linear function, which enhances stability and prevents potential overflow.
- **题目算子链**：F.linear, torch.mm, custom _rms_norm, F.relu, F.elu, F.softplus, torch.exp, torch.log, torch.min, torch.where, torch.linalg.eig
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `softplus_linear` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `matmul_linear` 类，优先把自然语言描述映射到 PyTorch 算子链：F.linear, torch.mm, custom _rms_norm, F.relu, F.elu, F.softplus, torch.exp, torch.log, torch.min, torch.where, torch.linalg.eig。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 46. openseek-8-11320a88c62f40ce98b98080ba07ef61 — `fused_svd_reconstruct`

- **任务类型**：linalg
- **Wrapper**：`fused_svd_reconstruct(A: Tensor) -> Tensor: The input matrix `A` of shape `(m, n)`.`
- **功能描述**：Reconstructs the input matrix `A` using its Singular Value Decomposition (SVD). This function combines the Singular Value Decomposition (SVD) with matrix reconstruction. Given a matrix `A`, it performs the following operations: 1. Compute the SVD of `A`: A = U Σ V^H, where `U` and `Vh` are unitary matrices and `S` contains the singular values of `A`. 2. Reconstruct `A` as A_reconstructed = U Σ V^H.
- **数学定义**：A = U Σ V^H A_reconstructed = U diag(S) V^H
- **补充约束**：The function returns the reconstructed matrix `A` of shape `(m, n)`, approximating the original matrix.
- **题目算子链**：torch.mm, custom _rms_norm, torch.exp, torch.sin, torch.min, torch.where, torch.linalg.svd
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `fused_svd_reconstruct` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `linalg` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, custom _rms_norm, torch.exp, torch.sin, torch.min, torch.where, torch.linalg.svd。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 47. openseek-8-50d2585ad4334780a630fe1bf041fb18 — `fused_mul_add_logsoftmax_dropout_bmm`

- **任务类型**：matmul_linear
- **Wrapper**：`fused_mul_add_logsoftmax_dropout_bmm(input1, input2, other, mat2, p=0.5, training=True, inplace=False, dim=-1, *, out=None) -> Tensor`
- **功能描述**：Performs a fused operation combining element-wise multiplication, addition, log-softmax activation, dropout, and batch matrix multiplication.
- **数学定义**：Given input tensors X_1, X_2, O, and M, the function computes: \[ \begin{align*} Z &= X_1 \odot X_2 \\ S &= Z + O \\ L &= \log\left( \frac{\exp(S)}{\sum_j \exp(S_j)} \right) \\ D &= \text{Dropout}(L, p) \\ Y &= \text{bmm}(D, M) \end{align*} \]
- **补充约束**：- The shapes of `input1`, `input2`, and `other` must be broadcastable to each other. - The `mat2` tensor must have a shape compatible with the output of the dropout layer for batch matrix multiplication, i.e., `mat2` should have shape `(B, D_in, D_out)` if the dropout output has shape `(B, N, D_in)`. - The `log_softmax` function is applied along dimension `dim`, which should be the dimension of the features (typically `-1` for the last dimension). - The `dropout` is applied during training when 
- **题目算子链**：torch.bmm, torch.matmul, torch.mm, custom _rms_norm, F.log_softmax, F.softmax, F.dropout, torch.exp, torch.log, torch.sum, torch.max, torch.min, torch.where
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `fused_mul_add_logsoftmax_dropout_bmm` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `matmul_linear` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.bmm, torch.matmul, torch.mm, custom _rms_norm, F.log_softmax, F.softmax, F.dropout, torch.exp, torch.log, torch.sum, torch.max, torch.min, torch.where。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 48. openseek-8-6c82bb8c97874fcda4c650e922fb65a1 — `selu`

- **任务类型**：matmul_linear
- **Wrapper**：`selu(input, inplace=False) -> Tensor`
- **功能描述**：Applies the element-wise SELU (Scaled Exponential Linear Unit) function to the input tensor. The SELU function is defined as scale * (max(0, x) + min(0, alpha * (exp(x) - 1))), where the constants alpha and scale are fixed values with alpha approximately 1.673 and scale approximately 1.051.
- **数学定义**：SELU(x) = scale * (max(0,x) + min(0, alpha * (exp(x) - 1))), with alpha=1.6732632423543772848170429916717 and scale=1.0507009873554804934193349852946.
- **补充约束**：See torch.nn.SELU for more details.
- **题目算子链**：F.linear, torch.mm, F.elu, F.selu, torch.exp, torch.max, torch.min, torch.where
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `selu` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `matmul_linear` 类，优先把自然语言描述映射到 PyTorch 算子链：F.linear, torch.mm, F.elu, F.selu, torch.exp, torch.max, torch.min, torch.where。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 49. openseek-8-192450a134ba4b8db5cf15f591eb74b5 — `scaled_add_norm`

- **任务类型**：indexing
- **Wrapper**：`scaled_add_norm(y: Tensor, x: Tensor, alpha: float) -> Tensor: y (Tensor): The target tensor to be modified, of shape `(n,)`. x (Tensor): The tensor to be scaled and added to `y`, of shape `(n,)`. alpha (float): The scalar multiplier for `x`.`
- **功能描述**：Computes `y += alpha * x` and returns the 2-norm of the modified `y`. The function takes a target tensor `y`, a tensor `x` to be scaled by a scalar `alpha`, and adds the scaled `x` to `y`. It then calculates and returns the 2-norm of the updated `y`.
- **数学定义**：y += alpha * x norm = ||y||_2
- **补充约束**：The function modifies the input tensor `y` in place and calculates the 2-norm using `torch.norm`.
- **题目算子链**：torch.mm, torch.exp, torch.sin, torch.min, torch.linalg.vector_norm
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `scaled_add_norm` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `indexing` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, torch.exp, torch.sin, torch.min, torch.linalg.vector_norm。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 50. openseek-8-78523a7fa58b495091e13f92efb7b7eb — `leaky_relu_conv2d`

- **任务类型**：conv_norm_pool
- **Wrapper**：`def leaky_relu_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, negative_slope=0.01, inplace=False) -> Tensor`
- **功能描述**：Applies a 2D convolution over the input tensor, followed by applying the Leaky ReLU activation function element-wise to the result. This allows for both feature extraction and non-linear activation in one step.
- **数学定义**：The combined operation is defined as: .. math:: \text{out} = \text{LeakyReLU}(\text{conv2d}(\text{input})) where the Leaky ReLU function is applied element-wise as: .. math:: \text{LeakyReLU}(x) = \max(0, x) + \text{negative\_slope} \times \min(0, x)
- **补充约束**：The function combines 2D convolution and Leaky ReLU activation in one step, allowing for efficient computation.
- **题目算子链**：F.conv2d, F.linear, torch.mm, F.leaky_relu, F.relu, F.elu, torch.exp, torch.max, torch.min, torch.where, torch.linalg.eig
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `leaky_relu_conv2d` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `conv_norm_pool` 类，优先把自然语言描述映射到 PyTorch 算子链：F.conv2d, F.linear, torch.mm, F.leaky_relu, F.relu, F.elu, torch.exp, torch.max, torch.min, torch.where, torch.linalg.eig。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 51. openseek-8-dc20e66156654cfc8d1a558548ccb016 — `sqrt_exp`

- **任务类型**：linalg
- **Wrapper**：`def sqrt_exp(input, out=None) -> Tensor: input (Tensor): The input tensor. out (Tensor, optional): The output tensor.`
- **功能描述**：Computes the square root of each element in :attr:`input`, and then applies the exponential function to the square-rooted values. The combined operation is defined as: out_i = e^(sqrt(input_i))
- **数学定义**：out_i = e^(sqrt(input_i))
- **补充约束**：N/A
- **题目算子链**：torch.mm, torch.sqrt, torch.exp, torch.min, torch.linalg.qr
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `sqrt_exp` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `linalg` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, torch.sqrt, torch.exp, torch.min, torch.linalg.qr。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 52. openseek-8-125dc3fb47954939a752d1b89c38c022 — `cos_avg_pool1d`

- **任务类型**：linalg
- **Wrapper**：`def cos_avg_pool1d(input: torch.Tensor, kernel_size: int, stride: int = None, padding: int = 0, ceil_mode: bool = False, count_include_pad: bool = True) -> torch.Tensor input (Tensor): The input tensor of shape (minibatch, in_channels, iW). kernel_size (int): Size of the pooling window. stride (int, optional): Stride of the pooling window. Defaults to `kernel_size`. padding (int, optional): Zero-padding added to both sides of the input. Default is 0. ceil_mode (bool, optional): If True, uses cei`
- **功能描述**：Applies the cosine function element-wise to the input tensor, followed by a 1D average pooling. The function first computes the cosine of each element in the input tensor, then applies 1D average pooling over the resulting tensor with the specified kernel size, stride, padding, ceil mode, and padding inclusion.
- **数学定义**：\text{output} = \text{avg\_pool1d}(\cos(\text{input}))
- **补充约束**：The function involves computing the cosine transformation followed by pooling, and handles parameters like stride, padding, and ceil mode.
- **题目算子链**：torch.mm, torch.exp, torch.cos, torch.sin, torch.min
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `cos_avg_pool1d` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `linalg` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, torch.exp, torch.cos, torch.sin, torch.min。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 53. openseek-8-23e1c9547e2348be8fbf4e531d860e6e — `sum_std`

- **任务类型**：linalg
- **Wrapper**：`def sum_std(input, dim=None, keepdim=False, dtype=None, correction=1, out=None) -> Tensor: input (Tensor): The input tensor. dim (int or tuple of ints, optional): The dimension(s) to reduce. If None, all dimensions are reduced. keepdim (bool, optional): Whether the output tensor has dim retained or not. Default is False. dtype (torch.dtype, optional): The desired data type of the returned tensor. If specified, the input tensor is cast to dtype before the operation. Default: None. correction (int`
- **功能描述**：Computes the sum of elements in the input tensor along the specified dimension(s), followed by calculating the standard deviation of the summed values.
- **数学定义**：\text{sum} = \sum_{i=0}^{N-1} x_i \sigma = \sqrt{\frac{1}{\max(0,~N - \delta N)}\sum_{i=0}^{N-1}(x_i-\bar{x})^2}
- **补充约束**：The function uses Bessel's correction by default with a correction value of 1.
- **题目算子链**：torch.mm, torch.sqrt, torch.exp, torch.sum, torch.std, torch.max, torch.min, torch.linalg.qr
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `sum_std` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `linalg` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, torch.sqrt, torch.exp, torch.sum, torch.std, torch.max, torch.min, torch.linalg.qr。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 54. openseek-8-3c2783058b9a49deba902ae94007f399 — `mul_relu`

- **任务类型**：matmul_linear
- **Wrapper**：`def mul_relu(input, other, inplace=False, out=None) -> Tensor: input (Tensor): The input tensor to be multiplied. other (Tensor or Number): The tensor or number to multiply with `input`. inplace (bool, optional): If True, modifies `input` in-place, if possible. Default is False. out (Tensor, optional): The output tensor.`
- **功能描述**：This function performs element-wise multiplication of two inputs, input and other, and then applies the Rectified Linear Unit (ReLU) function to the result, which replaces all negative values with zero.
- **数学定义**：ReLU(x) = max(0, x); out_i = ReLU(input_i * other_i)
- **补充约束**：The function uses torch.mul for multiplication and F.relu for the ReLU operation.
- **题目算子链**：F.linear, torch.mm, custom _rms_norm, F.relu, F.elu, torch.exp, torch.max, torch.min
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `mul_relu` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `matmul_linear` 类，优先把自然语言描述映射到 PyTorch 算子链：F.linear, torch.mm, custom _rms_norm, F.relu, F.elu, torch.exp, torch.max, torch.min。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 55. openseek-8-27c7026034bc48de894b56c90b53633d — `gelu_conv2d`

- **任务类型**：conv_norm_pool
- **Wrapper**：`def gelu_conv2d(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None, stride: Union[int, Tuple[int, int]] = 1, padding: Union[int, Tuple[int, int], str] = 0, dilation: Union[int, Tuple[int, int]] = 1, groups: int = 1, approximate: str = 'none', out: Optional[Tensor] = None) -> Tensor`
- **功能描述**：Applies a 2D convolution over an input tensor with specified filters, followed by applying the Gaussian Error Linear Units (GELU) activation function element-wise to the result. This helps introduce non-linearity after the convolution operation.
- **数学定义**：The combined operation is defined as: .. math:: \text{out} = \text{GELU}(\text{conv2d}(\text{input}, \text{weight}))
- **补充约束**：The function combines 2D convolution and GELU activation, with options for approximation methods for GELU.
- **题目算子链**：F.conv2d, F.linear, torch.mm, F.gelu, torch.tanh, F.elu, torch.sqrt, torch.exp, torch.sin, torch.min, torch.where, torch.linalg.qr, torch.linalg.eig
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `gelu_conv2d` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `conv_norm_pool` 类，优先把自然语言描述映射到 PyTorch 算子链：F.conv2d, F.linear, torch.mm, F.gelu, torch.tanh, F.elu, torch.sqrt, torch.exp, torch.sin, torch.min, torch.where, torch.linalg.qr, torch.linalg.eig。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 56. openseek-8-3580ec85929944749e9fa41e2e02bc65 — `fused_instance_norm_selu_conv2d`

- **任务类型**：conv_norm_pool
- **Wrapper**：`fused_instance_norm_selu_conv2d(input: Tensor, weight: Tensor, bias=None, stride=1, padding=0, dilation=1, groups=1, num_features=None, eps=1e-5, momentum=0.1, affine=False, track_running_stats=False) -> Tensor: input (Tensor): Input tensor of shape (minibatch, in_channels, iH, iW). weight (Tensor): Weights for the convolution, shape (out_channels, in_channels / groups, kH, kW). bias (Tensor, optional): Bias for the convolution layer, shape (out_channels). stride (int or tuple, optional): Stride`
- **功能描述**：Applies a fused operation consisting of a 2D convolution followed by SELU activation and instance normalization on the input tensor.
- **补充约束**：The function combines convolution, SELU activation, and instance normalization in a single operation.
- **题目算子链**：F.conv2d, torch.mm, F.instance_norm, F.elu, F.selu, torch.exp, torch.sin, torch.mean, torch.var, torch.min, torch.linalg.vector_norm, torch.linalg.eig
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `fused_instance_norm_selu_conv2d` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `conv_norm_pool` 类，优先把自然语言描述映射到 PyTorch 算子链：F.conv2d, torch.mm, F.instance_norm, F.elu, F.selu, torch.exp, torch.sin, torch.mean, torch.var, torch.min, torch.linalg.vector_norm, torch.linalg.eig。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 57. openseek-8-988e490635214576b037e77930deb604 — `fused_fractional_max_pool2d_with_relu`

- **任务类型**：conv_norm_pool
- **Wrapper**：`def fused_fractional_max_pool2d_with_relu(input: torch.Tensor, kernel_size, output_size=None, output_ratio=None, return_indices=False) -> torch.Tensor: Input (Tensor): Input tensor. kernel_size (int or Tuple[int, int]): Size of the pooling window. output_size (Tuple[int, int], optional): Target output size (height, width). output_ratio (Tuple[float, float], optional): If set, output size is scaled as a ratio of the input size. return_indices (bool, optional): If `True`, return the max pooling in`
- **功能描述**：Applies a ReLU activation followed by 2D fractional max pooling over an input signal composed of multiple planes. The input is first rectified (non-negative) and then pooled using fractional max pooling.
- **补充约束**：The function combines ReLU activation with fractional max pooling, allowing for optional output size or ratio specification and the option to return pooling indices.
- **题目算子链**：torch.mm, F.max_pool2d, F.relu, F.elu, torch.exp, torch.sin, torch.max, torch.min, torch.linalg.eig
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `fused_fractional_max_pool2d_with_relu` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `conv_norm_pool` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, F.max_pool2d, F.relu, F.elu, torch.exp, torch.sin, torch.max, torch.min, torch.linalg.eig。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 58. openseek-8-c16a172f3dc240c88cbafacaedfe6221 — `chebyshev_polynomial_t`

- **任务类型**：reduction
- **Wrapper**：`chebyshev_polynomial_t(input, n, *, out=None) -> Tensor; Args: input (Tensor): the input tensor. n (Tensor): Degree of the polynomial. Keyword args: out (Tensor, optional): the output tensor.`
- **功能描述**：Computes the Chebyshev polynomial of the first kind T_n(input). If n = 0, returns 1. If n = 1, returns input. For n < 6 or |input| > 1, uses a recursive formula. Otherwise, uses an explicit trigonometric formula.
- **数学定义**：T_{n + 1}(input) = 2 \times input \times T_{n}(input) - T_{n - 1}(input) T_{n}(input) = \text{cos}(n \times \text{arccos}(x))
- **补充约束**：If n = 0, returns 1. If n = 1, returns input. Uses recursion for n < 6 or |input| > 1, otherwise uses trigonometric formula.
- **题目算子链**：torch.mm, torch.exp, torch.cos, torch.min, Chebyshev recurrence
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `chebyshev_polynomial_t` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `reduction` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, torch.exp, torch.cos, torch.min, Chebyshev recurrence。
  1. 此类任务可短实现，重点处理 dim/keepdim/out/inplace/dtype 等参数。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 59. openseek-8-497431032a4749f08d2c2b4eb92faed1 — `logit`

- **任务类型**：reduction
- **Wrapper**：`logit(input, eps=None, *, out=None) -> Tensor; input (Tensor): the input tensor.; eps (float, optional): the epsilon for input clamp bound. Default: None; out (Tensor, optional): the output tensor.`
- **功能描述**：Returns a new tensor with the logit of the elements of input. The input is clamped to [eps, 1 - eps] when eps is not None. When eps is None and input < 0 or input > 1, the function yields NaN.
- **数学定义**：y_{i} = \ln(\frac{z_{i}}{1 - z_{i}}); z_{i} = \begin{cases} x_{i} & \text{if eps is None} \\ \text{eps} & \text{if } x_{i} < \text{eps} \\ x_{i} & \text{if } \text{eps} \leq x_{i} \leq 1 - \text{eps} \\ 1 - \text{eps} & \text{if } x_{i} > 1 - \text{eps} \end{cases}
- **补充约束**：input is clamped to [eps, 1 - eps] when eps is not None. When eps is None and input < 0 or input > 1, the function yields NaN.
- **题目算子链**：torch.mm, torch.exp, torch.log, torch.min
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `logit` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `reduction` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, torch.exp, torch.log, torch.min。
  1. 此类任务可短实现，重点处理 dim/keepdim/out/inplace/dtype 等参数。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 60. openseek-8-d3eb7e27374b4ff881103691a389095d — `solve_symmetric_ldl`

- **任务类型**：matmul_linear
- **Wrapper**：`solve_symmetric_ldl(A, b, *, hermitian=False, out=None) -> Tensor A (Tensor): 形状为 (*, n, n) 的对称（或 Hermitian）矩阵，其中 * 是零个或多个批次维度。 b (Tensor): 形状为 (*, n) 或 (*, n, k) 的右端项张量。 hermitian (bool, 可选): 是否将 A 视为 Hermitian 矩阵。默认值：False。 out (Tensor, 可选): 输出张量。如果为 None，则忽略。默认值：None。`
- **功能描述**：Solves a symmetric (or Hermitian) linear system A x = b using LDL decomposition. The function first decomposes A into L and D through LDL decomposition, reconstructs matrix A, and then uses `torch.linalg.solve` to solve the linear system.
- **数学定义**：Given a symmetric (or Hermitian) matrix A in \mathbb{K}^{n \times n} (where \mathbb{K} is the real field \mathbb{R} or complex field \mathbb{C}), the LDL decomposition of A is represented as: A = L D L^{\mathrm{T}} or A = L D L^{\mathrm{H}}.
- **补充约束**：This function supports batch processing; all computations are performed across batch dimensions.
- **题目算子链**：F.linear, torch.mm, torch.exp, torch.sin, torch.min, torch.where, torch.linalg.solve
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `solve_symmetric_ldl` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `matmul_linear` 类，优先把自然语言描述映射到 PyTorch 算子链：F.linear, torch.mm, torch.exp, torch.sin, torch.min, torch.where, torch.linalg.solve。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 61. openseek-8-4f6f918252af440abb332a6351565838 — `exp_sqrt`

- **任务类型**：linalg
- **Wrapper**：`def exp_sqrt(input, out=None) -> Tensor; input (Tensor): The input tensor.; out (Tensor, optional): The output tensor.`
- **功能描述**：Computes the exponential of each element in the input tensor, followed by calculating the square root of the result. Returns a tensor where each element is the result of applying exponential followed by square root to each element of input.
- **数学定义**：\text{out}_i = \sqrt{e^{\text{input}_i}}
- **补充约束**：This function will return NaN for input elements that result in negative values after `exp` and `sqrt` due to overflow.
- **题目算子链**：torch.mm, torch.sqrt, torch.exp, torch.min, torch.where, torch.linalg.qr
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `exp_sqrt` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `linalg` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, torch.sqrt, torch.exp, torch.min, torch.where, torch.linalg.qr。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 62. openseek-8-171e992fdf344d2782f9673b7ed5a50d — `combined_activation`

- **任务类型**：matmul_linear
- **Wrapper**：`combined_activation(input, weight1, weight2, bias, *, out=None) -> Tensor; input (Tensor): Input tensor of shape (*, N, D_{in}), where * denotes any number of batch dimensions.; weight1 (Tensor): Weight matrix of shape (D_{in}, D_{out}).; weight2 (Tensor): Weight tensor for element-wise multiplication, must be broadcastable to the shape of the intermediate activation.; bias (Tensor): Bias tensor, must be broadcastable to the shape of the output.; out (Tensor, optional): Output tensor. Ignored if`
- **功能描述**：Performs a sequence of operations combining matrix multiplication, sigmoid, tanh, element-wise multiplication, and addition. It supports batches of inputs, where any leading batch dimensions in `input` will be preserved in the output. The function's operations are differentiable and support autograd. The function ensures the dimensions of `input` and `weight1` are compatible for matrix multiplication, and that `weight2` and `bias` are broadcastable to the shape of the output tensor.
- **数学定义**：Given an input tensor X, weight matrices W_1 and W_2, and a bias b, the function computes: Y = (tanh(sigmoid(X W_1)) ⊙ W_2) + b - σ(z) = 1 / (1 + exp(-z)) is the sigmoid function applied element-wise. - tanh(z) = (exp(z) - exp(-z)) / (exp(z) + exp(-z)) is the hyperbolic tangent function applied element-wise. - ⊙ denotes element-wise multiplication.
- **补充约束**：The function supports differentiable operations and autograd. It requires compatibility in dimensions for matrix multiplication and broadcasting for element-wise operations.
- **题目算子链**：torch.matmul, torch.mm, custom _rms_norm, torch.sigmoid, torch.tanh, torch.exp, torch.min, torch.where, torch.linalg.eig
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `combined_activation` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `matmul_linear` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.matmul, torch.mm, custom _rms_norm, torch.sigmoid, torch.tanh, torch.exp, torch.min, torch.where, torch.linalg.eig。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 63. openseek-8-6683d9566d4c4a498e388df775898fa2 — `scaled_add_dot`

- **任务类型**：reduction
- **Wrapper**：`def scaled_add_dot(y: Tensor, x: Tensor, alpha: float) -> Tensor: y (Tensor): The target tensor to be modified, of shape (n,). x (Tensor): The tensor to be scaled and added to y, of shape (n,). alpha (float): The scalar multiplier for x.`
- **功能描述**：Computes `y += alpha * x` and returns the dot product of the modified `y` with itself. This fused function performs two operations: 1. Scales `x` by a factor of `alpha` and adds the result to `y`. 2. Computes the dot product of the modified `y` with itself.
- **数学定义**：y += alpha * x dot_product = torch.dot(y, y)
- **补充约束**：The function modifies the input tensor `y` in place.
- **题目算子链**：torch.mm, custom _rms_norm, torch.exp, torch.min
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `scaled_add_dot` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `reduction` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, custom _rms_norm, torch.exp, torch.min。
  1. 此类任务可短实现，重点处理 dim/keepdim/out/inplace/dtype 等参数。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 64. openseek-8-f52b79791fe8432bbd6a43023a96c586 — `tensordot`

- **任务类型**：reduction
- **Wrapper**：`def tensordot(a: Tensor, b: Tensor, dims: Union[int, Tuple[List[int], List[int]], List[List[int]]]) -> Tensor:`
- **功能描述**：Returns a contraction of a and b over multiple dimensions. It implements a generalized matrix product.
- **数学定义**：r_{i_0,...,i_{m-d}, i_d,...,i_n} = \sum_{k_0,...,k_{d-1}} a_{i_0,...,i_{m-d},k_0,...,k_{d-1}} \times b_{k_0,...,k_{d-1}, i_d,...,i_n}.
- **补充约束**：The sizes in the contracted dimensions must match, but broadcasted dimensions are handled.
- **题目算子链**：torch.mm, torch.exp, torch.sum, torch.min
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `tensordot` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `reduction` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, torch.exp, torch.sum, torch.min。
  1. 此类任务可短实现，重点处理 dim/keepdim/out/inplace/dtype 等参数。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 65. openseek-8-270812dbf8a249af947034627990704d — `qr`

- **任务类型**：matmul_linear
- **Wrapper**：`qr(A, mode='reduced', *, out=None) -> (Tensor, Tensor) A (Tensor): tensor of shape `(*, m, n)` where `*` is zero or more batch dimensions. mode (str, optional): one of `'reduced'`, `'complete'`, `'r'`. Controls the shape of the returned tensors. Default: `'reduced'`. out (tuple, optional): output tuple of two tensors. Ignored if `None`. Default: `None`.`
- **功能描述**：Computes the QR decomposition of a matrix. Supports input of float, double, cfloat and cdouble dtypes. Also supports batches of matrices, and if A is a batch of matrices then the output has the same batch dimensions. The parameter mode chooses between the full and reduced QR decomposition. It is always differentiable for 'reduced' mode, differentiable for 'complete' mode when m <= n, and never differentiable for 'r' mode.
- **数学定义**：A = QR where Q is orthogonal in the real case and unitary in the complex case, and R is upper triangular with real diagonal. For tall matrices (m > n), the reduced QR decomposition is A = QR with Q in K^{m x n} and R in K^{n x n}.
- **补充约束**：Differences with numpy.linalg.qr: mode='raw' is not implemented. Unlike numpy.linalg.qr, this function always returns a tuple of two tensors. When mode='r', the Q tensor is an empty tensor. The elements in the diagonal of R are not necessarily positive, making the QR decomposition unique only up to the sign of the diagonal of R. The QR decomposition is only well-defined if the first k = min(m, n) columns of every matrix in A are linearly independent.
- **题目算子链**：F.linear, torch.mm, torch.exp, torch.min, torch.where, torch.linalg.qr
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `qr` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `matmul_linear` 类，优先把自然语言描述映射到 PyTorch 算子链：F.linear, torch.mm, torch.exp, torch.min, torch.where, torch.linalg.qr。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 66. openseek-8-ac530c5c1f78450ba681b4b4195f1d79 — `asin`

- **任务类型**：linalg
- **Wrapper**：`asin(input, *, out=None) -> Tensor: input (Tensor): the input tensor. out (Tensor, optional): the output tensor.`
- **功能描述**：Returns a new tensor with the arcsine of the elements of the input tensor. The function computes the inverse sine (arcsine) for each element in the input tensor.
- **数学定义**：\text{out}_{i} = \sin^{-1}(\text{input}_{i})
- **补充约束**：The function returns NaN for input values outside the range [-1, 1] as arcsine is not defined for those values.
- **题目算子链**：torch.mm, torch.exp, torch.sin, torch.min, torch.linalg.inv
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `asin` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `linalg` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, torch.exp, torch.sin, torch.min, torch.linalg.inv。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 67. openseek-8-0f3a3783fd594587aac20e01eff12abc — `fused_masked_select_add_gelu`

- **任务类型**：matmul_linear
- **Wrapper**：`fused_masked_select_add_gelu(input, mask, other, *, alpha=1, approximate='none', out=None) -> Tensor`
- **功能描述**：This function performs a fused operation combining masked selection, addition, and GELU activation. It first selects elements from the input tensor based on a boolean mask, then adds a scalar or tensor (scaled by alpha) to the selected values, and finally applies the GELU (Gaussian Error Linear Unit) activation function element-wise to the result.
- **数学定义**：Z = masked_select(X, M) S = Z + alpha * O Y = GELU(S)
- **补充约束**：The function is differentiable and supports autograd. The mask and other tensor must be broadcastable to the shape of the selected elements. The 'approximate' parameter can be set to 'tanh' for a faster, approximate GELU computation.
- **题目算子链**：F.linear, torch.mm, custom _rms_norm, F.gelu, torch.tanh, F.elu, torch.exp, torch.min, torch.masked_select
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `fused_masked_select_add_gelu` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `matmul_linear` 类，优先把自然语言描述映射到 PyTorch 算子链：F.linear, torch.mm, custom _rms_norm, F.gelu, torch.tanh, F.elu, torch.exp, torch.min, torch.masked_select。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 68. openseek-8-aa03d33964c946dc9c8062140df295d7 — `fused_pairwise_distance_adaptive_avg_pool2d`

- **任务类型**：conv_norm_pool
- **Wrapper**：`def fused_pairwise_distance_adaptive_avg_pool2d(x1: torch.Tensor, x2: torch.Tensor, output_size: int or tuple, p: float = 2.0, eps: float = 1e-6, keepdim: bool = False) -> torch.Tensor: x1 (Tensor): First input tensor for adaptive average pooling and distance calculation. x2 (Tensor): Second input tensor for adaptive average pooling and distance calculation. output_size (int or tuple): The target output size for the adaptive average pooling. p (float, optional): The norm degree for pairwise dist`
- **功能描述**：This function applies adaptive average pooling to the input tensors `x1` and `x2` to resize them to the specified `output_size`, and then computes the pairwise distance between the pooled outputs. The function first applies `adaptive_avg_pool2d` to each input tensor, and then calculates the pairwise distance using the specified norm `p`. A small value `eps` is added to avoid division by zero during distance calculation. The function can also retain the reduced dimension of the output via the `keepdim` parameter.
- **数学定义**：No explicit formula provided. The function applies adaptive average pooling followed by pairwise distance calculation with norm p and epsilon to avoid division by zero.
- **补充约束**：The function combines adaptive average pooling and pairwise distance calculation in a sequential manner.
- **题目算子链**：torch.mm, F.avg_pool2d, F.adaptive_avg_pool2d, torch.exp, torch.sin, torch.min, torch.linalg.vector_norm
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `fused_pairwise_distance_adaptive_avg_pool2d` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `conv_norm_pool` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, F.avg_pool2d, F.adaptive_avg_pool2d, torch.exp, torch.sin, torch.min, torch.linalg.vector_norm。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 69. openseek-8-7ece5967c99242b4b406be8a2dc82ef5 — `add_mean`

- **任务类型**：linalg
- **Wrapper**：`def add_mean(input, other, dim=None, alpha=1, keepdim=False, dtype=None, out=None) -> Tensor: input (Tensor): The input tensor. other (Tensor or Number): The tensor or number to add to input. dim (int or tuple of ints, optional): The dimension(s) to reduce. Default: None. alpha (Number, optional): The multiplier for other. Default: 1. keepdim (bool, optional): Whether the output tensor has dim retained or not. Default: False. dtype (torch.dtype, optional): The desired data type of returned tenso`
- **功能描述**：Adds the `other` tensor, scaled by `alpha`, to the `input` tensor and computes the mean value along the specified dimension. If no dimension is specified, it computes the mean over all elements. Supports broadcasting, type promotion, and works with integer, float, and complex inputs.
- **数学定义**：\text{out}_i = \text{mean}(\text{input}_i + \text{alpha} \times \text{other}_i)
- **补充约束**：Supports broadcasting to a common shape, type promotion, and integer, float, and complex inputs.
- **题目算子链**：torch.mm, torch.exp, torch.mean, torch.min
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `add_mean` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `linalg` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, torch.exp, torch.mean, torch.min。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 70. openseek-8-4f188313a3474339ace1e868ae3ec2d5 — `fused_layer_norm_relu_linear`

- **任务类型**：conv_norm_pool
- **Wrapper**：`fused_layer_norm_relu_linear(input: Tensor, weight: Tensor, bias=None, normalized_shape=None, eps=1e-5, elementwise_affine=True) -> Tensor: Input (Tensor): Input tensor with shape (*, in_features). Weight (Tensor): Weights for the linear transformation, shape (out_features, in_features). Bias (Tensor, optional): Bias for the linear transformation, shape (out_features). Normalized_shape (int or list or torch.Size, optional): Shape of the dimensions to normalize. Eps (float, optional): A value add`
- **功能描述**：Applies a fused operation consisting of a linear transformation followed by ReLU activation and layer normalization on the input tensor.
- **补充约束**：The function performs a sequence of operations: linear transformation, ReLU activation, and layer normalization. It supports optional bias and learnable parameters for layer normalization.
- **题目算子链**：F.linear, torch.mm, F.layer_norm, custom _rms_norm, F.relu, F.elu, torch.exp, torch.min, torch.linalg.vector_norm, torch.linalg.eig
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `fused_layer_norm_relu_linear` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `conv_norm_pool` 类，优先把自然语言描述映射到 PyTorch 算子链：F.linear, torch.mm, F.layer_norm, custom _rms_norm, F.relu, F.elu, torch.exp, torch.min, torch.linalg.vector_norm, torch.linalg.eig。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 71. openseek-8-dd5140f5d0104073bb68c57c7d334c88 — `fused_add_mul_groupnorm`

- **任务类型**：linalg
- **Wrapper**：`fused_add_mul_groupnorm(input1, input2, weight, bias, num_groups, eps=1e-5, *, out=None) -> Tensor; input1 (Tensor): The first input tensor X; input2 (Tensor): The second input tensor Y, must be broadcastable to the shape of X; weight (Tensor): Learnable weight parameter \gamma of shape (C,), where C is the number of channels; bias (Tensor): Learnable bias parameter \beta of shape (C,); num_groups (int): Number of groups to separate the channels into for group normalization; eps (float, optional`
- **功能描述**：Performs a fused operation combining element-wise addition, element-wise multiplication, and group normalization. It takes two input tensors, adds them element-wise, multiplies the result with the second tensor, and then applies group normalization using learnable parameters for scaling and shifting. The function supports autograd for gradient computation and all operations are differentiable.
- **数学定义**：Given two input tensors X and Y, and learnable parameters \gamma and \beta for group normalization, the function computes: \[ \begin{align*} Z &= X + Y \\ M &= Z \odot Y \\ O &= \text{GroupNorm}(M, \gamma, \beta, \text{num\_groups}, \epsilon) \end{align*} \]
- **补充约束**：- The shapes of `input1` and `input2` must be broadcastable to each other. - The `weight` and `bias` parameters must have shape `(C,)`, where `C` is the number of channels in the input tensors. - The `num_groups` parameter must divide the number of channels `C` evenly. - This function supports autograd for gradient computation. - All operations are differentiable and support backpropagation.
- **题目算子链**：torch.mm, custom _rms_norm, torch.exp, torch.sin, torch.min, torch.linalg.vector_norm, torch.where, torch.linalg.eig
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `fused_add_mul_groupnorm` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `linalg` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, custom _rms_norm, torch.exp, torch.sin, torch.min, torch.linalg.vector_norm, torch.where, torch.linalg.eig。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 72. openseek-8-837d48e849ab472cbd085950fb72c382 — `SGD`

- **任务类型**：linalg
- **Wrapper**：`def SGD(params, lr=1e-3, momentum=0, weight_decay=0, dampening=0, nesterov=False, maximize=False, foreach=None, differentiable=False, fused=None)`
- **功能描述**：Implements stochastic gradient descent, optionally with momentum, weight decay, dampening, and Nesterov momentum. It can maximize or minimize an objective function and supports different optimization algorithms for performance.
- **数学定义**：\begin{aligned} &g_t \leftarrow \nabla_{\theta} f_t (\theta_{t-1}) \\\ &\text{if} \: \lambda \neq 0 \\\ &g_t \leftarrow g_t + \lambda \theta_{t-1} \\\ &\text{if} \: \mu \neq 0 \\\ &\text{if} \: t > 1 \\\ &\textbf{b}_t \leftarrow \mu \textbf{b}_{t-1} + (1-\tau) g_t \\\ &\text{else} \\\ &\textbf{b}_t \leftarrow g_t \\\ &\text{if} \: \textit{nesterov} \\\ &g_t \leftarrow g_{t} + \mu \textbf{b}_t \\\ &\text{else} \\\ &g_t \leftarrow \textbf{b}_t \\\ &\text{if} \: \textit{maximize} \\\ &\theta_t \lef
- **补充约束**：Nesterov momentum is based on a research paper. The algorithm prioritizes different implementations based on performance. It differs from some traditional frameworks in its handling of momentum. The initial momentum buffer is set to the gradient value at the first step.
- **题目算子链**：torch.mm, torch.exp, torch.max, torch.min, torch.linalg.eig
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `SGD` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `linalg` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, torch.exp, torch.max, torch.min, torch.linalg.eig。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 73. openseek-8-83a297a664454392acae42b950384831 — `relu_batch_norm_conv2d`

- **任务类型**：conv_norm_pool
- **Wrapper**：`def relu_batch_norm_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, running_mean=None, running_var=None, bn_weight=None, bn_bias=None, training=False, momentum=0.1, eps=1e-5, inplace=False) -> Tensor`
- **功能描述**：Applies a 2D convolution over the input tensor, followed by batch normalization and then applies the ReLU activation function element-wise to the normalized result. This combined operation is useful for applying feature extraction, normalization, and non-linearity in one step, commonly used in convolutional neural networks (CNNs).
- **数学定义**：out = ReLU(BatchNorm(conv2d(input))) ReLU(x) = max(0, x) y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta
- **补充约束**：The function combines convolution, batch normalization, and ReLU activation in a single step, which is a common pattern in CNNs for efficient computation.
- **题目算子链**：F.conv2d, F.linear, torch.mm, F.batch_norm, custom _rms_norm, F.relu, F.elu, torch.sqrt, torch.exp, torch.sin, torch.mean, torch.var, torch.max, torch.min, torch.linalg.vector_norm, torch.linalg.qr, torch.linalg.eig
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `relu_batch_norm_conv2d` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `conv_norm_pool` 类，优先把自然语言描述映射到 PyTorch 算子链：F.conv2d, F.linear, torch.mm, F.batch_norm, custom _rms_norm, F.relu, F.elu, torch.sqrt, torch.exp, torch.sin, torch.mean, torch.var, torch.max, torch.min, torch.linalg.vector_norm, torch.linalg.qr, torch.linalg.eig。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 74. openseek-8-c4c5d71167f14729967dbe0df7067ee9 — `conv2d`

- **任务类型**：conv_norm_pool
- **Wrapper**：`conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1) -> Tensor Args: input: input tensor of shape (minibatch , in_channels , iH , iW) weight: filters of shape (out_channels , in_channels/groups , kH , kW) bias: optional bias tensor of shape (out_channels). Default: None stride: the stride of the convolving kernel. Can be a single number or a tuple (sH, sW). Default: 1 padding: implicit paddings on both sides of the input. Can be a string {'valid', 'same'}, single number or`
- **功能描述**：Applies a 2D convolution over an input image composed of several input planes. Supports TensorFloat32. May select a nondeterministic algorithm on CUDA with CuDNN for performance. Supports complex data types.
- **补充约束**：Supports TensorFloat32. May select a nondeterministic algorithm on CUDA with CuDNN. Supports complex data types.
- **题目算子链**：F.conv2d, torch.mm, torch.exp, torch.sin, torch.min, torch.linalg.eig
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `conv2d` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `conv_norm_pool` 类，优先把自然语言描述映射到 PyTorch 算子链：F.conv2d, torch.mm, torch.exp, torch.sin, torch.min, torch.linalg.eig。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 75. openseek-8-d11b2ee92ecd46eda022a00907af3242 — `normalized_cosine_similarity`

- **任务类型**：linalg
- **Wrapper**：`def normalized_cosine_similarity(x1: Tensor, x2: Tensor, dim: int = 1, eps_similarity: float = 1e-8, p_norm: float = 2, eps_norm: float = 1e-12) -> Tensor`
- **功能描述**：Computes the cosine similarity between two normalized input tensors `x1` and `x2`. This function normalizes `x1` and `x2` along a specified dimension using L_p normalization, and subsequently calculates the cosine similarity between these normalized tensors along the specified dimension. This involves ensuring vectors are scaled to avoid division by zero by introducing small epsilon values both during normalization and similarity computation.
- **数学定义**：The operation is defined as: similarity = \frac{\text{normalize}(x1) \cdot \text{normalize}(x2)}{\max(\lVert \text{normalize}(x1) \Vert _2, \epsilon) \cdot \max(\lVert \text{normalize}(x2) \Vert _2, \epsilon)} where the `normalize` function is defined as: v = \frac{v}{\max(\lVert v \rVert_p, \epsilon)}.
- **补充约束**：The function allows broadcasting x2 to match x1's shape. Default values are provided for dimension, normalization, and similarity thresholds to enhance robustness against division by zero.
- **题目算子链**：torch.mm, torch.exp, torch.cos, torch.sin, torch.max, torch.min, torch.linalg.vector_norm, torch.where
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `normalized_cosine_similarity` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `linalg` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, torch.exp, torch.cos, torch.sin, torch.max, torch.min, torch.linalg.vector_norm, torch.where。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 76. openseek-8-f3cef4ea85f0425a9bf4d76d144d22e5 — `fused_cholesky_solve`

- **任务类型**：linalg
- **Wrapper**：`def fused_cholesky_solve(A: Tensor, b: Tensor) -> Tensor: A: The symmetric positive-definite matrix `A` of shape `(n, n)`. b: The right-hand side tensor `b` of shape `(n, k)`.`
- **功能描述**：Computes the solution `x` to the equation `Ax = b` using the Cholesky decomposition. It first performs Cholesky decomposition on a symmetric positive-definite matrix `A` to obtain a lower triangular matrix `L` such that `A = L * L.T`, then solves for `x` in `Ax = b` using the Cholesky factorization.
- **数学定义**：Cholesky decomposition: A = L * L.T, Solve: Ax = b
- **补充约束**：The function assumes that the input matrix `A` is symmetric positive-definite.
- **题目算子链**：torch.mm, custom _rms_norm, torch.exp, torch.sin, torch.sum, torch.min, torch.linalg.cholesky, torch.linalg.solve
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `fused_cholesky_solve` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `linalg` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, custom _rms_norm, torch.exp, torch.sin, torch.sum, torch.min, torch.linalg.cholesky, torch.linalg.solve。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 77. openseek-8-bd29ed3d82ce4259a74b4daff2fad6e4 — `matmul`

- **任务类型**：matmul_linear
- **Wrapper**：`matmul(input, other, *, out=None) -> Tensor Arguments: input (Tensor): the first tensor to be multiplied other (Tensor): the second tensor to be multiplied`
- **功能描述**：Matrix product of two tensors. The behavior depends on the dimensionality of the tensors: 1D tensors return a dot product; 2D tensors return a matrix-matrix product; 1D and 2D tensors return a matrix-vector product; N-dimensional tensors (N > 2) return a batched matrix multiply with broadcasting support. Sparse layouts are supported for 2D matrix-matrix products. TensorFloat32 is supported. On certain ROCm devices, float16 inputs use different precision for backward. The 1D dot product version does not support an out parameter.
- **补充约束**：Sparse support is a beta feature and some layout(s)/dtype/device combinations may not be supported, or may not have autograd support. If you notice missing functionality please open a feature request.
- **题目算子链**：torch.matmul, torch.mm, torch.exp, torch.sin, torch.min
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `matmul` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `matmul_linear` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.matmul, torch.mm, torch.exp, torch.sin, torch.min。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 78. openseek-8-2de63b529046419bbdbe301183a782ca — `fused_gather_masked_fill`

- **任务类型**：linalg
- **Wrapper**：`fused_gather_masked_fill(input, dim, index, mask, value, *, sparse_grad=False, out=None) -> Tensor; input (Tensor): The input tensor X.; dim (int): The dimension along which to index.; index (LongTensor): The indices of elements to gather, of the same dimensionality as `input`.; mask (BoolTensor): A boolean mask tensor, broadcastable to the shape of the output tensor Y.; value (float): The value to fill in where `mask` is True.; sparse_grad (bool, optional): If True, gradient w.r.t. `input` will`
- **功能描述**：Performs a fused operation combining torch.gather and torch.Tensor.masked_fill. It first gathers values from the input tensor along a specified dimension using provided indices, and then replaces the gathered elements with a specified value where the mask is True.
- **数学定义**：Y = \text{gather}(X, \text{dim}, I) Y[M] = \text{value}
- **补充约束**：- The input and index tensors must have the same number of dimensions. - The size of index at each dimension d must not exceed the size of input at that dimension, except at dimension dim. - The mask tensor must be broadcastable to the shape of the gathered output. - The function supports autograd for gradient computation. - All operations are differentiable and support backpropagation.
- **题目算子链**：torch.mm, custom _rms_norm, torch.exp, torch.sin, torch.min, torch.gather, Tensor.masked_fill, torch.where
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `fused_gather_masked_fill` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `linalg` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, custom _rms_norm, torch.exp, torch.sin, torch.min, torch.gather, Tensor.masked_fill, torch.where。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 79. openseek-8-2a6303ed19dd441a8b9406303601283b — `fused_cross_entropy_log_softmax`

- **任务类型**：attention_softmax_loss
- **Wrapper**：`def fused_cross_entropy_log_softmax(input: torch.Tensor, target: torch.Tensor, dim: int = 1, weight: torch.Tensor = None, ignore_index: int = -100, reduction: str = 'mean', label_smoothing: float = 0.0) -> torch.Tensor`
- **功能描述**：This function computes the cross entropy loss with log softmax applied to the input logits. It combines log softmax activation and cross entropy loss calculation in a numerically stable way. The log softmax is applied to the input logits, and the cross entropy loss is computed between the normalized logits and the target. The function allows customization with options such as which dimension to apply the log softmax, manual rescaling weights for each class, handling of ignored targets, reduction method for loss aggregation, and label smoothing to modify the target distribution.
- **数学定义**：log_softmax(x_i) = log(exp(x_i) / sum(exp(x))) CE(y, p) = -sum(y * log(p))
- **补充约束**：The function integrates the log softmax and cross entropy loss computation into a single operation for numerical stability. The input and target tensors must be of compatible shapes, where the input is expected to have logits of size (N, C) and target should have size (N,) for class indices.
- **题目算子链**：torch.mm, F.log_softmax, F.softmax, F.cross_entropy, torch.exp, torch.log, torch.sin, torch.mean, torch.sum, torch.max, torch.min, torch.linalg.vector_norm, torch.where, torch.linalg.eig
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `fused_cross_entropy_log_softmax` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `attention_softmax_loss` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, F.log_softmax, F.softmax, F.cross_entropy, torch.exp, torch.log, torch.sin, torch.mean, torch.sum, torch.max, torch.min, torch.linalg.vector_norm, torch.where, torch.linalg.eig。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 80. openseek-8-b39378719da64c2d8300dc220b5a4232 — `addmm`

- **任务类型**：matmul_linear
- **Wrapper**：`addmm(input, mat1, mat2, *, beta=1, alpha=1, out=None) -> Tensor; input (Tensor): matrix to be added; mat1 (Tensor): the first matrix to be matrix multiplied; mat2 (Tensor): the second matrix to be matrix multiplied; beta (Number, optional): multiplier for input (β); alpha (Number, optional): multiplier for mat1 @ mat2 (α); out (Tensor, optional): the output tensor.`
- **功能描述**：Performs a matrix multiplication of the matrices mat1 and mat2. The matrix input is added to the final result. If mat1 is a (n x m) tensor, mat2 is a (m x p) tensor, then input must be broadcastable with a (n x p) tensor and out will be a (n x p) tensor. Alpha and beta are scaling factors on matrix-vector product between mat1 and mat2 and the added matrix input respectively. If beta is 0, then input will be ignored, and nan and inf in it will not be propagated. This operation supports sparse layouts. If input is sparse the result will have the same layout and if out is provided it must have the same layout as input. Sparse support is a beta feature and some layout(s)/dtype/device combinations may not be supported, or may not have autograd support. This operator supports TensorFloat32. On certain ROCm devices, when using float16 inputs this module will use different precision for backward.
- **数学定义**：out = β * input + α * (mat1 @ mat2)
- **补充约束**：Sparse support is a beta feature and some layout(s)/dtype/device combinations may not be supported, or may not have autograd support. This operator supports TensorFloat32. On certain ROCm devices, when using float16 inputs this module will use different precision for backward.
- **题目算子链**：torch.matmul, torch.mm, torch.addmm, custom _rms_norm, torch.exp, torch.sin, torch.min
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `addmm` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `matmul_linear` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.matmul, torch.mm, torch.addmm, custom _rms_norm, torch.exp, torch.sin, torch.min。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 81. openseek-8-37a45ee5d6dc4415b23c9dd30e61fe61 — `fused_qr_solve`

- **任务类型**：matmul_linear
- **Wrapper**：`def fused_qr_solve(A: Tensor, b: Tensor) -> Tensor: A: The matrix `A` of shape `(m, n)` where `m >= n`. b: The right-hand side tensor `b` of shape `(m, k)`.`
- **功能描述**：Solves the linear system `Ax = b` using QR decomposition. This function combines the QR decomposition with solving a linear system. Given a matrix `A` and a vector (or matrix) `b`, it performs the QR decomposition of `A` and computes the solution `x` using the formula `x = R^{-1} (Q^T b)`.
- **数学定义**：x = R^{-1} Q^T b
- **补充约束**：The function assumes `m >= n` for the matrix `A`.
- **题目算子链**：F.linear, torch.mm, custom _rms_norm, torch.exp, torch.sin, torch.sum, torch.min, torch.where, torch.linalg.qr, torch.linalg.solve
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `fused_qr_solve` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `matmul_linear` 类，优先把自然语言描述映射到 PyTorch 算子链：F.linear, torch.mm, custom _rms_norm, torch.exp, torch.sin, torch.sum, torch.min, torch.where, torch.linalg.qr, torch.linalg.solve。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 82. openseek-8-91b698430a3e4048ab3cb178db442b7b — `sigmoid_adaptive_avg_pool2d`

- **任务类型**：conv_norm_pool
- **Wrapper**：`def sigmoid_adaptive_avg_pool2d(input: Tensor, output_size: Union[int, Tuple[int, int]]) -> Tensor`
- **功能描述**：Applies a 2D adaptive average pooling over an input tensor, followed by the sigmoid activation function applied element-wise. This is used for downsampling a feature map to a specified output size and then normalizing the result with the sigmoid function.
- **数学定义**：out = σ(AdaptiveAvgPool2D(input)) Sigmoid(x) = 1 / (1 + exp(-x))
- **补充约束**：Each element in the resulting tensor is scaled to the range (0, 1) by the sigmoid activation.
- **题目算子链**：torch.mm, F.avg_pool2d, F.adaptive_avg_pool2d, torch.sigmoid, torch.exp, torch.min, torch.linalg.vector_norm
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `sigmoid_adaptive_avg_pool2d` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `conv_norm_pool` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, F.avg_pool2d, F.adaptive_avg_pool2d, torch.sigmoid, torch.exp, torch.min, torch.linalg.vector_norm。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 83. openseek-8-dac33da1e6774a2191262f47ed0e75af — `cos`

- **任务类型**：linalg
- **Wrapper**：`cos(input, *, out=None) -> Tensor; Args: input (Tensor): the input tensor.; Keyword args: out (Tensor, optional): the output tensor.`
- **功能描述**：Returns a new tensor with the cosine of the elements of the input tensor.
- **数学定义**：\text{out}_{i} = \cos(\text{input}_{i})
- **补充约束**：The function computes the cosine of each element in the input tensor and returns a new tensor with these values.
- **题目算子链**：torch.mm, torch.exp, torch.cos, torch.sin, torch.min
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `cos` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `linalg` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, torch.exp, torch.cos, torch.sin, torch.min。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 84. openseek-8-71b10a42ed2845888fc64dccdb2ee75c — `fused_bmm_dropout_gelu`

- **任务类型**：matmul_linear
- **Wrapper**：`fused_bmm_dropout_gelu(input1, input2, p=0.5, training=True, inplace=False, approximate='none', *, out=None) -> Tensor - **input1** (Tensor): First input tensor for batch matrix multiplication, of shape (B, N, M), where B is the batch size. - **input2** (Tensor): Second input tensor for batch matrix multiplication, of shape (B, M, P). - **p** (float, optional): Probability of an element to be zeroed in the dropout layer. Default: `0.5`. - **training** (bool, optional): Apply dropout if `True`. D`
- **功能描述**：Performs a fused operation combining batch matrix multiplication, dropout, and GELU activation. It computes the batch matrix multiplication of two input tensors, applies dropout to the result, and then applies the GELU activation function.
- **数学定义**：Given two input tensors X and Y, this function computes: \[ \begin{align*} Z &= \text{bmm}(X, Y) \\ D &= \text{Dropout}(Z, p) \\ O &= \text{GELU}(D) \end{align*} \]
- **补充约束**：- The shapes of `input1` and `input2` must be compatible for batch matrix multiplication: `input1` of shape `(B, N, M)` and `input2` of shape `(B, M, P)` result in an output of shape `(B, N, P)`. - The `dropout` is applied during training when `training=True`. Set `training=False` to disable dropout during evaluation. - The `GELU` activation is applied element-wise to the output of dropout. - All operations are differentiable and support autograd.
- **题目算子链**：F.linear, torch.bmm, torch.matmul, torch.mm, custom _rms_norm, F.dropout, F.gelu, torch.tanh, F.elu, torch.exp, torch.min, torch.where
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `fused_bmm_dropout_gelu` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `matmul_linear` 类，优先把自然语言描述映射到 PyTorch 算子链：F.linear, torch.bmm, torch.matmul, torch.mm, custom _rms_norm, F.dropout, F.gelu, torch.tanh, F.elu, torch.exp, torch.min, torch.where。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 85. openseek-8-91f086bdba4744609803d8cf9b2ab3f3 — `trunc`

- **任务类型**：linalg
- **Wrapper**：`trunc(input, *, out=None) -> Tensor`
- **功能描述**：Returns a new tensor with the truncated integer values of the elements of the input tensor. For integer inputs, it follows the array-api convention of returning a copy of the input tensor.
- **补充约束**：For integer inputs, follows the array-api convention of returning a copy of the input tensor.
- **题目算子链**：torch.mm, torch.exp, torch.min
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `trunc` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `linalg` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, torch.exp, torch.min。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 86. openseek-8-eda30a668f9b445f864fa81b512aa3e3 — `matrix_power_eig`

- **任务类型**：linalg
- **Wrapper**：`def matrix_power_eig(A, k, *, out=None) -> Tensor`
- **功能描述**：Computes the matrix power A^k of a square matrix A using eigendecomposition. It relies on A being diagonalizable and computes the power through the equation A^k = V diag(Λ^k) V^(-1), where Λ and V are the eigenvalues and eigenvectors of A. It allows for fractional powers of matrices and supports real or complex exponents. If A is not diagonalizable, the result may not be accurate.
- **数学定义**：A^k = V diag(Λ^k) V^{-1}, where A = V diag(Λ) V^{-1}, and Λ^k denotes the element-wise power of the eigenvalues.
- **补充约束**：Supports input of float, double, cfloat, and cdouble dtypes. Also supports batches of matrices, output has the same batch dimensions. Note that the computed A^k may be complex even if A is real, due to complex eigenvalues. Warning: If A is not diagonalizable, the result may not be accurate. Gradients might be numerically unstable if the distance between any two eigenvalues is close to zero.
- **题目算子链**：torch.mm, torch.exp, torch.sin, torch.min, torch.where, torch.linalg.eig, torch.linalg.matrix_power
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `matrix_power_eig` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `linalg` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, torch.exp, torch.sin, torch.min, torch.where, torch.linalg.eig, torch.linalg.matrix_power。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 87. openseek-8-7a073b2b19a54d098be7bbb0089c27cd — `log_tanh`

- **任务类型**：activation
- **Wrapper**：`def log_tanh(input, out=None) -> Tensor: input (Tensor): The input tensor. All elements must be positive for the log function. out (Tensor, optional): The output tensor.`
- **功能描述**：Computes the natural logarithm of each element in the input tensor, then applies the hyperbolic tangent (tanh) function to the result. This involves applying the logarithm first, which is only defined for positive numbers, and then applying tanh to transform the result between -1 and 1.
- **数学定义**：\text{out}_{i} = \tanh(\log(\text{input}_{i}))
- **补充约束**：All input elements must be positive for the logarithm function to be defined.
- **题目算子链**：torch.mm, torch.tanh, torch.exp, torch.log, torch.min
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `log_tanh` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `activation` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, torch.tanh, torch.exp, torch.log, torch.min。
  1. 此类任务可短实现，重点处理 dim/keepdim/out/inplace/dtype 等参数。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 88. openseek-8-88bb80e1e19f4e45974105bd5b4aa758 — `exp`

- **任务类型**：reduction
- **Wrapper**：`exp(input, *, out=None) -> Tensor input (Tensor): the input tensor. out (Tensor, optional): the output tensor.`
- **功能描述**：Returns a new tensor with the exponential of the elements of the input tensor.
- **数学定义**：y_{i} = e^{x_{i}}
- **题目算子链**：torch.mm, torch.exp, torch.min
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `exp` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `reduction` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, torch.exp, torch.min。
  1. 此类任务可短实现，重点处理 dim/keepdim/out/inplace/dtype 等参数。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 89. openseek-8-63b594e894014f1cb17357d2ca37b053 — `matrix_multiply_symmetric`

- **任务类型**：matmul_linear
- **Wrapper**：`matrix_multiply_symmetric(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, alpha: float, beta: float) -> torch.Tensor; Args: A (Tensor): The first input matrix of shape `(n, m)`. B (Tensor): The second input matrix of shape `(m, p)`. C (Tensor): The target matrix for the operations, shape `(n, p)`. alpha (float): Scalar multiplier for matrix products. beta (float): Scalar multiplier for adding to `C`. Example: A = torch.tensor([[1.0, 2.0], [3.0, 4.0]]), B = torch.tensor([[0.5, -1.0], [1.5, 2.0`
- **功能描述**：Computes two operations on matrix `C`: first, it performs the matrix-matrix product `C = alpha * torch.mm(A, B) + beta * C`, then updates `C` to be `C = alpha * torch.mm(C, C.T) + beta * C`. This function effectively performs two sequential matrix operations: a weighted sum of a matrix product and itself, followed by a weighted product of `C` and its transpose.
- **数学定义**：C = alpha * torch.mm(A, B) + beta * C C = alpha * torch.mm(C, C.T) + beta * C
- **补充约束**：This function performs a fused operation of matrix multiplication and symmetric update.
- **题目算子链**：torch.matmul, torch.mm, custom _rms_norm, torch.exp, torch.sum, torch.min, torch.linalg.eig
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `matrix_multiply_symmetric` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `matmul_linear` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.matmul, torch.mm, custom _rms_norm, torch.exp, torch.sum, torch.min, torch.linalg.eig。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 90. openseek-8-480648b79e3d4207ac10bf110b90f31f — `fused_avg_pool2d_cosine_similarity`

- **任务类型**：conv_norm_pool
- **Wrapper**：`fused_avg_pool2d_cosine_similarity(x1: torch.Tensor, x2: torch.Tensor, kernel_size: int, stride: int = None, padding: int = 0, eps: float = 1e-8) -> torch.Tensor`
- **功能描述**：Computes the cosine similarity between `x1` and `x2` along a specified dimension, adds a singleton dimension, and applies 2D average pooling. It first computes cosine similarity along dim=1 using `cosine_similarity`, then adds a singleton dimension using `unsqueeze`, and finally applies 2D average pooling using `avg_pool2d`.
- **补充约束**：The function provides an optional `stride` parameter which defaults to the value of `kernel_size` if not provided. The `eps` parameter is used to prevent division by zero in cosine similarity.
- **题目算子链**：torch.mm, F.avg_pool2d, torch.exp, torch.cos, torch.sin, torch.min
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `fused_avg_pool2d_cosine_similarity` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `conv_norm_pool` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, F.avg_pool2d, torch.exp, torch.cos, torch.sin, torch.min。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 91. openseek-8-67eda67e084f415db53beb4402320699 — `fused_hardshrink_dropout`

- **任务类型**：attention_softmax_loss
- **Wrapper**：`def fused_hardshrink_dropout(input: torch.Tensor, p: float = 0.5, training: bool = True, inplace: bool = False, lambd: float = 0.5) -> torch.Tensor`
- **功能描述**：Applies a fused operation consisting of dropout followed by hard shrinkage on the input tensor. The function first applies dropout to the input tensor, where each element is zeroed with a probability of p if training is True. The dropout can be applied in-place if specified. After dropout, a hard shrinkage operation is applied, which shrinks values towards zero based on the lambda parameter.
- **补充约束**：The function combines dropout and hard shrinkage operations, which are typically used in neural network training to prevent overfitting and to enforce sparsity, respectively.
- **题目算子链**：torch.mm, F.dropout, torch.exp, torch.min, torch.where
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `fused_hardshrink_dropout` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `attention_softmax_loss` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, F.dropout, torch.exp, torch.min, torch.where。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 92. openseek-8-6584c3ee8b14474983d820e65a4742a4 — `erfc_sqrt`

- **任务类型**：linalg
- **Wrapper**：`def erfc_sqrt(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]: The input tensor for which the erfc and square root are computed.`
- **功能描述**：Computes the complementary error function (erfc) and the square root of each element in the input tensor.
- **数学定义**：\text{erfc}(x) = 1 - \frac{2}{\sqrt{\pi}} \int_{0}^{x} e^{-t^2} dt \text{out}_{i} = \sqrt{\text{input}_{i}}
- **补充约束**：Returns a tuple containing the erfc result and the square root result for each element in the input tensor.
- **题目算子链**：torch.mm, torch.sqrt, torch.exp, torch.erfc, torch.min, torch.linalg.qr
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `erfc_sqrt` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `linalg` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, torch.sqrt, torch.exp, torch.erfc, torch.min, torch.linalg.qr。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 93. openseek-8-438651ab55e5428daa39a47005a42e63 — `tensordot_rsqrt`

- **任务类型**：linalg
- **Wrapper**：`def tensordot_rsqrt(a: torch.Tensor, b: torch.Tensor, dims) -> torch.Tensor: a (Tensor): Left tensor to contract. b (Tensor): Right tensor to contract. dims (int, Tuple[List[int], List[int]], or List[List[int]]): Dimensions for contraction, as per `torch.tensordot`.`
- **功能描述**：Returns the reciprocal of the square root of the tensordot product of two tensors `a` and `b`. This function performs a tensor contraction of `a` and `b` over the specified dimensions using `torch.tensordot`, and then applies the element-wise reciprocal square root to the resulting tensor. The operation involves computing the tensordot product first and then applying the reciprocal of the square root element-wise to the result.
- **数学定义**：\text{output} = \frac{1}{\sqrt{\sum_{k_0,...,k_{d-1}} a_{i_0,...,i_{m-d},k_0,...,k_{d-1}} \times b_{k_0,...,k_{d-1}, i_d,...,i_n}}}
- **补充约束**：The function applies the `torch.tensordot` and `torch.rsqrt` operations. The `dims` argument specifies the dimensions over which the contraction happens, similar to the `torch.tensordot` function.
- **题目算子链**：torch.mm, custom _rms_norm, torch.sqrt, torch.exp, torch.rsqrt, torch.sin, torch.sum, torch.min, torch.linalg.qr
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `tensordot_rsqrt` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `linalg` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, custom _rms_norm, torch.sqrt, torch.exp, torch.rsqrt, torch.sin, torch.sum, torch.min, torch.linalg.qr。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 94. openseek-8-829de8149cf149d782ba0cbad32c09b5 — `softmax_log`

- **任务类型**：attention_softmax_loss
- **Wrapper**：`def softmax_log(input, dim=-1, dtype=None) -> Tensor:`
- **功能描述**：Applies the natural logarithm element-wise on the input tensor, followed by applying the softmax function along the specified dimension. This combined operation scales input values to a range between 0 and 1, summing to 1 after the logarithmic transformation. It allows transformation of the input tensor into a probability distribution.
- **数学定义**：out = Softmax(log(input))
- **补充约束**：The function handles optional data type casting to prevent overflow and allows specifying the dimension for softmax application.
- **题目算子链**：torch.mm, F.softmax, torch.exp, torch.log, torch.sum, torch.max, torch.min, torch.where
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `softmax_log` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `attention_softmax_loss` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, F.softmax, torch.exp, torch.log, torch.sum, torch.max, torch.min, torch.where。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 95. openseek-8-a9aebe7cd5e741f9819610e210d594eb — `dropout_sigmoid_linear`

- **任务类型**：matmul_linear
- **Wrapper**：`def dropout_sigmoid_linear(input: torch.Tensor, weight: torch.Tensor, bias=None, p=0.5, training=True, inplace=False) -> torch.Tensor: Input tensor of shape :math:`(*, \text{in\_features})`. Weight tensor of shape :math:`(\text{out\_features}, \text{in\_features})`. Bias tensor of shape :math:`(\text{out\_features})`. Default is `None`. Probability of an element to be zeroed in dropout. Default: 0.5 If `True`, applies dropout during training. Default: `True` If `True`, performs the operation in-`
- **功能描述**：Applies a linear transformation followed by a sigmoid activation and dropout. This function sequentially applies a linear transformation to the input tensor, a sigmoid activation to scale the values between 0 and 1, and randomly zeroes some elements of the tensor with a specified probability during dropout.
- **数学定义**：`(*, \text{in\_features})`. Weight tensor of shape :math:`(\text{out\_features}, \text{in\_features})`. Bias tensor of shape :math:`(\text{out\_features})`. Default is `None`. Probability of an element to be zeroed in dropout. Default: 0.5 If `True`, applies dropout during training. Default: `True` If `True`, performs the operation in-place. Default: `False`
- **补充约束**：The function applies dropout only if the `training` parameter is set to `True`. The `inplace` parameter allows for in-place operations to save memory.
- **题目算子链**：F.linear, torch.mm, custom _rms_norm, F.dropout, torch.sigmoid, torch.exp, torch.min, torch.linalg.eig
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `dropout_sigmoid_linear` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `matmul_linear` 类，优先把自然语言描述映射到 PyTorch 算子链：F.linear, torch.mm, custom _rms_norm, F.dropout, torch.sigmoid, torch.exp, torch.min, torch.linalg.eig。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 96. openseek-8-6d7d7a1572de4ef19d1b20eeb4094268 — `batch_norm`

- **任务类型**：conv_norm_pool
- **Wrapper**：`def batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-05) -> Tensor`
- **功能描述**：Applies Batch Normalization for each channel across a batch of data. Batch Normalization is a technique to improve the training of deep neural networks by ensuring that each layer receives whitened input, which helps to stabilize the learning process and reduce the number of training epochs needed to converge.
- **补充约束**：This function is related to the BatchNorm classes like BatchNorm1d, BatchNorm2d, and BatchNorm3d, which are layers that handle this operation with additional features.
- **题目算子链**：torch.mm, F.batch_norm, torch.exp, torch.mean, torch.var, torch.min, torch.linalg.vector_norm, torch.linalg.eig
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `batch_norm` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `conv_norm_pool` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, F.batch_norm, torch.exp, torch.mean, torch.var, torch.min, torch.linalg.vector_norm, torch.linalg.eig。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 97. openseek-8-b211216562ce47218e4faefeb69a3284 — `gammaln`

- **任务类型**：linalg
- **Wrapper**：`gammaln(input, *, out=None) -> Tensor`
- **功能描述**：Computes the natural logarithm of the absolute value of the gamma function on the input tensor.
- **数学定义**：\text{out}_{i} = \ln \Gamma(|\text{input}_{i}|)
- **题目算子链**：torch.mm, torch.exp, torch.log, torch.min
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `gammaln` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `linalg` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, torch.exp, torch.log, torch.min。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 98. openseek-8-95acbc1a47824faaa34fb0d73a228b89 — `bitwise_and`

- **任务类型**：reduction
- **Wrapper**：`bitwise_and(input, other, *, out=None) -> Tensor; input: the first input tensor; other: the second input tensor; out (Tensor, optional): the output tensor.`
- **功能描述**：Computes the bitwise AND of input and other. The input tensor must be of integral or Boolean types. For bool tensors, it computes the logical AND.
- **补充约束**：the second input tensor; out (Tensor, optional): the output tensor.
- **题目算子链**：torch.mm, torch.exp, torch.log, torch.bitwise_and, torch.min
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `bitwise_and` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `reduction` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, torch.exp, torch.log, torch.bitwise_and, torch.min。
  1. 此类任务可短实现，重点处理 dim/keepdim/out/inplace/dtype 等参数。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 99. openseek-8-e4a7846ad75646708b931b6639175bfd — `sub_gelu`

- **任务类型**：matmul_linear
- **Wrapper**：`def sub_gelu(input, other, alpha=1, approximate='none', out=None) -> Tensor: input (Tensor): The input tensor. other (Tensor or Number): The tensor or number to subtract from input. alpha (Number, optional): The multiplier for other. Default is 1. approximate (str, optional): The approximation method for GELU. Default is 'none'. out (Tensor, optional): The output tensor.`
- **功能描述**：Subtracts 'other', scaled by 'alpha', from 'input', and then applies the Gaussian Error Linear Units (GELU) activation function to the result. The function supports two modes for GELU: exact and approximate using 'tanh'.
- **数学定义**：out_i = GELU(input_i - alpha * other_i) GELU(x) = x * Φ(x) when approximate is 'none' GELU(x) = 0.5 * x * (1 + Tanh(√(2/π) * (x + 0.044715 * x^3))) when approximate is 'tanh'
- **补充约束**：The function allows for an optional output tensor and supports both exact and approximate GELU calculations.
- **题目算子链**：F.linear, torch.mm, F.gelu, torch.tanh, F.elu, torch.exp, torch.sin, torch.min
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `sub_gelu` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `matmul_linear` 类，优先把自然语言描述映射到 PyTorch 算子链：F.linear, torch.mm, F.gelu, torch.tanh, F.elu, torch.exp, torch.sin, torch.min。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 100. openseek-8-02cc469192bb4412938dede63a8eedda — `gelu_std`

- **任务类型**：matmul_linear
- **Wrapper**：`def gelu_std(input, dim=None, keepdim=False, correction=1, approximate='none', out=None) -> Tensor: input (Tensor): The input tensor. dim (int or tuple of ints, optional): The dimension or dimensions to reduce. If None, computes over all dimensions. keepdim (bool, optional): Whether to retain the dimension(s) with size 1 after reduction. Default is False. correction (int, optional): The correction factor for standard deviation. Default is 1. approximate (str, optional): The approximation method `
- **功能描述**：Applies the Gaussian Error Linear Units (GELU) activation function to the elements of input, then computes the standard deviation along the specified dimension(s). The GELU function is applied element-wise to the input tensor, with an option to use an approximation method. After activation, the standard deviation of the result is calculated over specified dimensions, with options to keep reduced dimensions and apply a correction factor.
- **数学定义**：GELU(x) = x * Φ(x) (when approximate is 'none') GELU(x) = 0.5 * x * (1 + Tanh(√(2/π) * (x + 0.044715 * x^3))) (when approximate is 'tanh') σ = √(1/(max(0, N - δN)) * Σ(x_i - x̄)^2)
- **补充约束**：The function allows the use of a correction factor in the standard deviation calculation. It supports two methods for computing GELU: exact using CDF or approximate using a tanh-based formula.
- **题目算子链**：F.linear, torch.mm, F.gelu, torch.tanh, F.elu, torch.exp, torch.sin, torch.std, torch.max, torch.min
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `gelu_std` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `matmul_linear` 类，优先把自然语言描述映射到 PyTorch 算子链：F.linear, torch.mm, F.gelu, torch.tanh, F.elu, torch.exp, torch.sin, torch.std, torch.max, torch.min。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 101. openseek-8-0ed62fee44d9485ea80491be353d9dc6 — `permute_copy`

- **任务类型**：reduction
- **Wrapper**：`torch.permute_copy(input, dims) -> Tensor`
- **功能描述**：Performs the same operation as torch.permute, which rearranges the dimensions of the input tensor according to the specified dims, but all output tensors are freshly created instead of aliasing the input.
- **补充约束**：Freshly created output tensors mean that the function does not create views, so changes to the output will not affect the input.
- **题目算子链**：torch.mm, custom _rms_norm, torch.exp, torch.sin, torch.mean, torch.min
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `permute_copy` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `reduction` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, custom _rms_norm, torch.exp, torch.sin, torch.mean, torch.min。
  1. 此类任务可短实现，重点处理 dim/keepdim/out/inplace/dtype 等参数。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 102. openseek-8-1028736f1c1045d7ada072ce8e7b81a9 — `digamma`

- **任务类型**：reduction
- **Wrapper**：`digamma(input, *, out=None) -> Tensor; Args: input (Tensor): the tensor to compute the digamma function on; Keyword args: out (Tensor, optional): the output tensor.`
- **功能描述**：Computes the logarithmic derivative of the gamma function on input. This function is similar to SciPy's scipy.special.digamma. From PyTorch 1.8 onwards, the digamma function returns -Inf for 0, previously it returned NaN for 0.
- **数学定义**：\digamma(x) = \frac{d}{dx} \ln\left(\Gamma\left(x\right)\right) = \frac{\Gamma'(x)}{\Gamma(x)}
- **补充约束**：This function is similar to SciPy's scipy.special.digamma. From PyTorch 1.8 onwards, the digamma function returns -Inf for 0, previously it returned NaN for 0.
- **题目算子链**：torch.mm, torch.exp, torch.log, torch.min
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `digamma` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `reduction` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, torch.exp, torch.log, torch.min。
  1. 此类任务可短实现，重点处理 dim/keepdim/out/inplace/dtype 等参数。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 103. openseek-8-80ac379da8704c958ef03daed8d41b46 — `softmax_mul`

- **任务类型**：attention_softmax_loss
- **Wrapper**：`def softmax_mul(input, other, dim, dtype=None, out=None) -> Tensor: Applies the softmax function to the input tensor along the specified dimension, and then multiplies the softmaxed values by other. Args: input (Tensor): The input tensor to apply softmax on. other (Tensor or Number): The tensor or number to multiply with the softmaxed values. dim (int): The dimension along which softmax will be computed. dtype (torch.dtype, optional): The desired data type of returned tensor. If specified, the i`
- **功能描述**：Applies the softmax function to the input tensor along the specified dimension, and then multiplies the softmaxed values by another tensor or number. The softmax function re-scales the elements so that they lie in the range [0, 1] and sum to 1 along the specified dimension.
- **数学定义**：\text{out}_i = \text{Softmax}(\text{input}_i) \times \text{other}_i \text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}
- **补充约束**：Softmax re-scales the elements so that they lie in the range [0, 1] and sum to 1 along the specified dimension.
- **题目算子链**：torch.mm, F.softmax, torch.exp, torch.sum, torch.max, torch.min
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `softmax_mul` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `attention_softmax_loss` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, F.softmax, torch.exp, torch.sum, torch.max, torch.min。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 104. openseek-8-fb3ffb7be7524d2494d8cc837084eb6a — `bitwise_and_binomial`

- **任务类型**：linalg
- **Wrapper**：`def bitwise_and_binomial(input: torch.Tensor, other: torch.Tensor, total_count: torch.Tensor, probs: torch.Tensor = None, logits: torch.Tensor = None) -> torch.Tensor: input (Tensor): The first input tensor of integral or Boolean type. other (Tensor): The second input tensor of integral or Boolean type. total_count (Tensor): Number of Bernoulli trials, must be broadcastable with `probs` or `logits`. probs (Tensor, optional): Event probabilities. Only one of `probs` or `logits` should be provided`
- **功能描述**：Computes the bitwise AND operation between two tensors and then applies a Binomial distribution sampling based on the resulting tensor's values. First, it computes the bitwise AND of `input` and `other`. Then, the result is used as input for the Binomial distribution, with each element representing the number of trials with the probability specified in `probs` or `logits`.
- **数学定义**：\text{output} = \text{Binomial}( \text{bitwise\_and}(\text{input}, \text{other}))
- **补充约束**：torch.Tensor, total_count: torch.Tensor, probs: torch.Tensor = None, logits: torch.Tensor = None) -> torch.Tensor: input (Tensor): The first input tensor of integral or Boolean type. other (Tensor): The second input tensor of integral or Boolean type. total_count (Tensor): Number of Bernoulli trials, must be broadcastable with `probs` or `logits`. probs (Tensor, optional): Event probabilities. Only one of `probs` or `logits` should be provided. logits (Tensor, optional): Event log-odds.
- **题目算子链**：torch.mm, custom _rms_norm, torch.exp, torch.log, torch.bitwise_and, torch.min
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `bitwise_and_binomial` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `linalg` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, custom _rms_norm, torch.exp, torch.log, torch.bitwise_and, torch.min。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 105. openseek-8-352b77bb1ac149459fbcda6a1e61ec0c — `rad2deg_sqrt`

- **任务类型**：linalg
- **Wrapper**：`def rad2deg_sqrt(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]: Args: input (Tensor): The input tensor with angles in radians.`
- **功能描述**：This function computes the conversion of angles from radians to degrees and calculates the square root for each element in the input tensor. It returns a tuple where the first element is the converted degrees and the second is the square root of the input tensor elements.
- **数学定义**：\text{out}_{i} = \text{input}_{i} \times (180.0 / \pi) \text{out}_{i} = \sqrt{\text{input}_{i}}
- **补充约束**：The function uses torch's rad2deg and sqrt functions to perform the operations.
- **题目算子链**：torch.mm, torch.sqrt, torch.exp, torch.rad2deg, torch.min, torch.where, torch.linalg.qr
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `rad2deg_sqrt` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `linalg` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, torch.sqrt, torch.exp, torch.rad2deg, torch.min, torch.where, torch.linalg.qr。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 106. openseek-8-42e97cc21acd464bb7a9ec6323a4fe8c — `bessel_j1`

- **任务类型**：reduction
- **Wrapper**：`bessel_j1(input, *, out=None) -> Tensor Args: input (Tensor): the input tensor. Keyword args: out (Tensor, optional): the output tensor.`
- **功能描述**：Computes the Bessel function of the first kind of order 1 for each element of the input tensor.
- **数学定义**：Bessel function of the first kind of order :math:`1`.
- **补充约束**：The function supports an optional output tensor.
- **题目算子链**：torch.mm, torch.exp, torch.min
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `bessel_j1` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `reduction` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, torch.exp, torch.min。
  1. 此类任务可短实现，重点处理 dim/keepdim/out/inplace/dtype 等参数。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 107. openseek-8-42f721c18cde485fb32fbe1e29128328 — `lu`

- **任务类型**：linalg
- **Wrapper**：`lu(A, *, pivot=True, out=None) -> (Tensor, Tensor, Tensor) Args: A (Tensor): tensor of shape `(*, m, n)` where `*` is zero or more batch dimensions. pivot (bool, optional): Controls whether to compute the LU decomposition with partial pivoting or no pivoting. Default: `True`. Keyword args: out (tuple, optional): output tuple of three tensors. Ignored if `None`. Default: `None`.`
- **功能描述**：Computes the LU decomposition with partial pivoting of a matrix. If pivot=True, returns a permutation matrix P, a lower triangular matrix L, and an upper triangular matrix U such that A = PLU. If pivot=False and A is on GPU, computes the LU decomposition without pivoting, returning empty P, L and U such that A = LU. Supports float, double, cfloat, and cdouble dtypes, as well as batches of matrices. Outputs have the same batch dimensions as input.
- **数学定义**：A = PLU where P is a permutation matrix, L is lower triangular with ones on the diagonal, U is upper triangular. If pivot=False, A = LU.
- **补充约束**：LU decomposition is not unique; different platforms may yield different decompositions. Gradient computations are supported only if the matrix is full-rank.
- **题目算子链**：torch.mm, custom _rms_norm, torch.exp, torch.min, torch.where
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `lu` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `linalg` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, custom _rms_norm, torch.exp, torch.min, torch.where。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 108. openseek-8-da2be421bd4f4679a35aab46c5608101 — `gelu_min`

- **任务类型**：matmul_linear
- **Wrapper**：`gelu_min(input, approximate='none', dim=None, keepdim=False, out=None) -> Tensor or (Tensor, LongTensor)`
- **功能描述**：Applies the Gaussian Error Linear Units (GELU) activation function to each element in the input tensor, followed by computing the minimum value along the specified dimension. If no dimension is specified, it computes the minimum over all elements. The function supports two methods for computing GELU: exact ('none') and an approximation using 'tanh'.
- **数学定义**：When approximate is 'none': GELU(x) = x * Φ(x), where Φ(x) is the Cumulative Distribution Function for Gaussian Distribution. When approximate is 'tanh': GELU(x) = 0.5 * x * (1 + Tanh(√(2/π) * (x + 0.044715 * x^3)))
- **补充约束**：Returns a namedtuple (values, indices) if dim is specified, otherwise returns the minimum value tensor.
- **题目算子链**：F.linear, torch.mm, F.gelu, torch.tanh, F.elu, torch.exp, torch.sin, torch.min, torch.where
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `gelu_min` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `matmul_linear` 类，优先把自然语言描述映射到 PyTorch 算子链：F.linear, torch.mm, F.gelu, torch.tanh, F.elu, torch.exp, torch.sin, torch.min, torch.where。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 109. openseek-8-5f0ba656d54941d1b319a195df05031a — `grid_sample_with_affine`

- **任务类型**：matmul_linear
- **Wrapper**：`def grid_sample_with_affine(input: torch.Tensor, theta: torch.Tensor, size: torch.Size, mode: str = 'bilinear', padding_mode: str = 'zeros', align_corners: bool = False) -> torch.Tensor: Input tensor of shape (N, C, H_{in}, W_{in}) (4D). Affine transformation matrix of shape (N, 2, 3) for 2D transformations. Target output image size as a 4D size (N, C, H_{out}, W_{out}). Interpolation mode to calculate output values, 'bilinear', 'nearest', or 'bicubic'. Default is 'bilinear'. Defines how to hand`
- **功能描述**：This function applies an affine transformation to the input tensor followed by grid sampling. It first generates a 2D flow field (sampling grid) based on the input affine matrix `theta` using `affine_grid`. Then it uses the generated grid to sample from the input image using `grid_sample`. It supports multiple interpolation modes (such as 'bilinear', 'nearest', and 'bicubic'), different padding modes ('zeros', 'border', 'reflection'), and has an option to align corners for transformation consistency.
- **补充约束**：The function generates an affine transformation grid and applies grid sampling to the input tensor.
- **题目算子链**：F.linear, torch.mm, torch.exp, torch.sin, torch.min
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `grid_sample_with_affine` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `matmul_linear` 类，优先把自然语言描述映射到 PyTorch 算子链：F.linear, torch.mm, torch.exp, torch.sin, torch.min。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 110. openseek-8-177d413e25474275bfcd9471c75cb895 — `pseudoinverse_svd`

- **任务类型**：linalg
- **Wrapper**：`def pseudoinverse_svd(A, *, full_matrices=True, rcond=1e-15, out=None) -> Tensor`
- **功能描述**：Computes the Moore-Penrose pseudoinverse of a matrix using Singular Value Decomposition (SVD). It decomposes the input matrix A into its singular value components, inverts the non-zero singular values above a certain threshold to avoid numerical instability, and reconstructs the pseudoinverse using these components. Supports input of float, double, cfloat, and cdouble dtypes, and can handle batches of matrices.
- **数学定义**：A^{+} = V^{\mathrm{H}} \Sigma^{+} U^{\mathrm{H}}; \sigma_i^{+} = \begin{cases} \dfrac{1}{\sigma_i}, & \text{if } \sigma_i > \text{rcond} \times \sigma_{\max} \\ 0, & \text{otherwise} \end{cases}
- **补充约束**：Supports input of float, double, cfloat, and cdouble dtypes; Handles batches of matrices
- **题目算子链**：torch.mm, torch.exp, torch.sin, torch.max, torch.min, torch.where, torch.linalg.svd, torch.linalg.inv
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `pseudoinverse_svd` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `linalg` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, torch.exp, torch.sin, torch.max, torch.min, torch.where, torch.linalg.svd, torch.linalg.inv。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 111. openseek-8-685c416260624574b55e451f2644af7d — `exp_mean`

- **任务类型**：linalg
- **Wrapper**：`def exp_mean(input, dim=None, keepdim=False, dtype=None, out=None) -> Tensor`
- **功能描述**：Applies the exponential function to each element in the input tensor and then computes the mean value of the result along the specified dimension or over all elements if no dimension is specified.
- **数学定义**：The combined operation is defined as: out = mean(e^{input}) where the exponential function is defined as: y_{i} = e^{x_{i}}
- **补充约束**：The function first applies the exponential function to each element of the input tensor and then computes the mean of these exponential values. The function allows specifying dimensions to reduce, whether to keep dimensions, and the data type of the output.
- **题目算子链**：torch.mm, torch.exp, torch.mean, torch.min, torch.where
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `exp_mean` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `linalg` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, torch.exp, torch.mean, torch.min, torch.where。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 112. openseek-8-8a70c2f4fded4de79b5c5303cc5dc73c — `low_rank_svd_approximation`

- **任务类型**：linalg
- **Wrapper**：`def low_rank_svd_approximation(A, k, *, full_matrices=True, out=None) -> Tensor`
- **功能描述**：Computes a rank-k approximation of a matrix using its Singular Value Decomposition (SVD). The function retains the top-k singular values and corresponding singular vectors from the SVD of A to form the approximation Ak. This low-rank approximation minimizes the Frobenius norm of the difference between A and Ak among all rank-k matrices. Supports input of float, double, cfloat, and cdouble dtypes, and batches of matrices.
- **数学定义**：A \approx A_k = U_k \Sigma_k V_k^{\text{H}}; U_k \in \mathbb{K}^{m \times k}; \Sigma_k \in \mathbb{R}^{k \times k}; V_k^{\text{H}} \in \mathbb{K}^{k \times n}
- **补充约束**：Supports input of float, double, cfloat, and cdouble dtypes; Batches of matrices are supported.
- **题目算子链**：torch.mm, torch.exp, torch.sin, torch.min, torch.linalg.vector_norm, torch.where, torch.linalg.svd
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `low_rank_svd_approximation` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `linalg` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, torch.exp, torch.sin, torch.min, torch.linalg.vector_norm, torch.where, torch.linalg.svd。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 113. openseek-8-1bcc9abd9154461cb857951cc82f2789 — `min`

- **任务类型**：linalg
- **Wrapper**：`min(input, dim, keepdim=False, *, out=None) -> (Tensor, LongTensor) Args: input (Tensor): the input tensor. dim (int): the dimension to reduce. keepdim (bool): whether the output tensor has :attr:`dim` retained or not. Keyword args: out (tuple, optional): the tuple of two output tensors (min, min_indices)`
- **功能描述**：Returns the minimum value of each row of the input tensor in the given dimension dim, along with the index location of each minimum value found. If keepdim is True, the output tensors retain the same size as input except in the dimension dim where they are of size 1. Otherwise, dim is squeezed, resulting in the output tensors having 1 fewer dimension than input. If there are multiple minimal values in a reduced row, the indices of the first minimal value are returned. The function can also compare two tensors element-wise and return a tensor with the minimum values.
- **补充约束**：If there are multiple minimal values in a reduced row, the indices of the first minimal value are returned.
- **题目算子链**：torch.mm, torch.exp, torch.min, torch.where
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `min` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `linalg` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, torch.exp, torch.min, torch.where。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 114. openseek-8-1a96aa0c423349ba95e3564d6c9e8c3d — `symmetric_mm_and_abs_sum`

- **任务类型**：matmul_linear
- **Wrapper**：`symmetric_mm_and_abs_sum(A: torch.Tensor, C: torch.Tensor, alpha: float, beta: float) -> torch.Tensor`
- **功能描述**：Performs a symmetric matrix multiplication by multiplying matrix `A` with its transpose, scales the result by `alpha`, adds it to matrix `C` scaled by `beta`, and returns the sum of the absolute values of the resulting matrix.
- **数学定义**：1. `C = alpha * torch.mm(A, A.T) + beta * C`; 2. `asum = torch.sum(torch.abs(C))`
- **补充约束**：Returns a scalar tensor representing the sum of absolute values of the resulting matrix `C`.
- **题目算子链**：torch.matmul, torch.mm, custom _rms_norm, torch.exp, torch.sum, torch.min
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `symmetric_mm_and_abs_sum` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `matmul_linear` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.matmul, torch.mm, custom _rms_norm, torch.exp, torch.sum, torch.min。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 115. openseek-8-659c185115c548589643df14f1c77a25 — `determinant_lu`

- **任务类型**：linalg
- **Wrapper**：`determinant_lu(A, *, pivot=True, out=None) -> Tensor; A (Tensor): Tensor of shape `(*, n, n)` where `*` is zero or more batch dimensions consisting of square matrices. pivot (bool, optional): Controls whether to compute the LU decomposition with partial pivoting (`True`) or without pivoting (`False`). Default: `True`. out (Tensor, optional): Output tensor. Ignored if `None`. Default: `None`.`
- **功能描述**：Computes the determinant of a square matrix using LU decomposition. The function performs LU decomposition on a given square matrix A and calculates its determinant. It supports matrices over real or complex numbers and can handle batch dimensions. The determinant is computed as the product of the diagonal elements of the upper triangular matrix U from the LU decomposition, adjusted by the sign of the permutation matrix P if pivoting is used. The function assumes A is invertible and supports float, double, cfloat, and cdouble dtypes.
- **数学定义**：\det(A) = \det(P) \cdot \prod_{i=1}^{n} U_{ii}; When pivot=False: \det(A) = \prod_{i=1}^{n} U_{ii}
- **补充约束**：This method assumes that A is invertible. If A is singular, the determinant will be zero, and the function may return `inf` or `nan` due to division by zero or numerical instability.
- **题目算子链**：torch.mm, custom _rms_norm, torch.exp, torch.sin, torch.sum, torch.min, torch.where, torch.linalg.inv, torch.linalg.det
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `determinant_lu` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `linalg` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, custom _rms_norm, torch.exp, torch.sin, torch.sum, torch.min, torch.where, torch.linalg.inv, torch.linalg.det。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 116. openseek-8-f074ea9a5243428bac40a55e25ce18fa — `tanh_linear`

- **任务类型**：matmul_linear
- **Wrapper**：`def tanh_linear(input, weight, bias=None) -> Tensor: input (Tensor): The input tensor of shape `(*, in_features)`, where `*` represents any number of additional dimensions. weight (Tensor): The weight matrix of shape `(out_features, in_features)`. bias (Tensor, optional): The optional bias tensor of shape `(out_features)`. Default: None.`
- **功能描述**：Applies a linear transformation to the input tensor followed by a Tanh activation function. This combined operation is useful for introducing non-linearity after a linear transformation, helping to capture complex relationships in the data.
- **数学定义**：The combined operation is defined as: out = tanh(linear(input, weight, bias)) where the linear transformation is applied as y = xA^T + b and Tanh activation is applied element-wise as: Tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
- **补充约束**：A linear transformation followed by a Tanh activation helps capture complex relationships by introducing non-linearity.
- **题目算子链**：F.linear, torch.mm, torch.tanh, torch.exp, torch.min, torch.where, torch.linalg.eig
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `tanh_linear` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `matmul_linear` 类，优先把自然语言描述映射到 PyTorch 算子链：F.linear, torch.mm, torch.tanh, torch.exp, torch.min, torch.where, torch.linalg.eig。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 117. openseek-8-e5046812327840df84cf151a4a410978 — `sum`

- **任务类型**：indexing
- **Wrapper**：`def sum(input, dim, keepdim=False, *, dtype=None) -> Tensor; input (Tensor): the input tensor.; dim (int or tuple of ints, optional): the dimension or dimensions to reduce.; keepdim (bool): whether the output tensor has :attr:`dim` retained or not.; dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.`
- **功能描述**：Returns the sum of each row of the input tensor in the given dimension dim. If dim is a list of dimensions, reduce over all of them. If keepdim is True, the output tensor is of the same size as input except in the dimension(s) dim where it is of size 1. Otherwise, dim is squeezed, resulting in the output tensor having 1 (or len(dim)) fewer dimension(s).
- **补充约束**：If dim is a list of dimensions, reduce over all of them. If keepdim is True, the output tensor is of the same size as input except in the dimension(s) dim where it is of size 1. Otherwise, dim is squeezed.
- **题目算子链**：torch.mm, torch.exp, torch.sum, torch.min, torch.where
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `sum` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `indexing` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, torch.exp, torch.sum, torch.min, torch.where。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 118. openseek-8-ac47cee255454660b25d893807c4731d — `logspace`

- **任务类型**：linalg
- **Wrapper**：`logspace(start, end, steps, base=10.0, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor`
- **功能描述**：Creates a one-dimensional tensor of size 'steps' whose values are evenly spaced from base^start to base^end, inclusive, on a logarithmic scale with a specified base. The tensor values are generated in a logarithmic progression from base^start to base^end using the specified number of steps.
- **数学定义**：( ext{base}^{ ext{start}}, ext{base}^{( ext{start} + rac{ ext{end} - ext{start}}{ ext{steps} - 1})}, \ldots, ext{base}^{( ext{start} + ( ext{steps} - 2) * rac{ ext{end} - ext{start}}{ ext{steps} - 1})}, ext{base}^{ ext{end}})
- **补充约束**：From PyTorch 1.11, the 'steps' argument is required. Use steps=100 to restore the previous behavior. The function allows specifying various properties of the output tensor such as dtype, layout, and device.
- **题目算子链**：torch.mm, torch.exp, torch.log, torch.sin, torch.var, torch.min
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `logspace` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `linalg` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, torch.exp, torch.log, torch.sin, torch.var, torch.min。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 119. openseek-8-31089932de764b6a93545a1ca1f976e5 — `solve_and_add_scaled_vector`

- **任务类型**：matmul_linear
- **Wrapper**：`def solve_and_add_scaled_vector(A: torch.Tensor, b: torch.Tensor, y: torch.Tensor, alpha: float) -> torch.Tensor: A (Tensor): A triangular matrix of shape `(n, n)`. b (Tensor): Right-hand side vector or matrix of shape `(n,)` or `(n, k)`. y (Tensor): Vector to be scaled and added, must have shape `(n,)` or broadcastable to `(n,)`. alpha (float): Scaling factor for the vector y.`
- **功能描述**：Solves the triangular system of linear equations Ax = b, where A is a triangular matrix. Then, adds a scaled version of the vector y to the solution x. The operations performed are: 1. Solve the triangular system Ax = b using torch.linalg.solve_triangular with A as an upper triangular matrix. 2. Add the scaled vector alpha * y to the solution x.
- **数学定义**：x = torch.linalg.solve_triangular(A, b, upper=True) x += alpha * y
- **补充约束**：The function assumes A is an upper triangular matrix.
- **题目算子链**：F.linear, torch.mm, torch.exp, torch.sin, torch.sum, torch.min, torch.where, torch.linalg.solve
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `solve_and_add_scaled_vector` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `matmul_linear` 类，优先把自然语言描述映射到 PyTorch 算子链：F.linear, torch.mm, torch.exp, torch.sin, torch.sum, torch.min, torch.where, torch.linalg.solve。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 120. openseek-8-fabf0f38be3c48e385547bb1eb32ae71 — `pixel_shuffle_conv2d`

- **任务类型**：conv_norm_pool
- **Wrapper**：`def pixel_shuffle_conv2d(input: torch.Tensor, weight: torch.Tensor, bias=None, stride=1, padding=0, dilation=1, groups=1, upscale_factor=2) -> torch.Tensor: Input tensor of shape (minibatch, in_channels, iH, iW). Convolution filter tensor of shape (out_channels, in_channels/groups, kH, kW). Optional bias tensor of shape (out_channels). Stride of the convolving kernel. Padding added to all four sides of the input. Spacing between kernel elements. Number of blocked connections from input channels `
- **功能描述**：Applies a 2D convolution followed by pixel shuffle upscaling to rearrange the spatial dimensions. This function sequentially applies a 2D convolution operation and then rearranges the elements of the convolution output to increase the spatial resolution by the upscale_factor.
- **补充约束**：The function first applies a 2D convolution and then uses pixel shuffle to upscale the spatial dimensions by the given upscale_factor.
- **题目算子链**：F.conv2d, torch.mm, F.pixel_shuffle, torch.exp, torch.min, torch.linalg.eig
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `pixel_shuffle_conv2d` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `conv_norm_pool` 类，优先把自然语言描述映射到 PyTorch 算子链：F.conv2d, torch.mm, F.pixel_shuffle, torch.exp, torch.min, torch.linalg.eig。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 121. openseek-8-3b11d4629e6e4254acc225208c9959bb — `matrix_vector_dot`

- **任务类型**：matmul_linear
- **Wrapper**：`def matrix_vector_dot(A: Tensor, x: Tensor, y: Tensor, alpha: float, beta: float) -> Tensor:`
- **功能描述**：Computes the matrix-vector product `y = alpha * torch.mv(A, x) + beta * y` and then returns the dot product `torch.dot(y, x)`. The function first computes a scaled matrix-vector product and updates `y`, then calculates the dot product of the updated `y` with `x`. It requires an input matrix `A` of shape `(n, m)`, an input vector `x` of shape `(m,)`, and a target vector `y` of shape `(n,)` that is modified in-place. The scalar `alpha` is a multiplier for `torch.mv(A, x)`, while `beta` is a multiplier for `y`.
- **数学定义**：y = alpha * torch.mv(A, x) + beta * y; result = torch.dot(y, x)
- **补充约束**：The function modifies the `y` vector in-place and calculates a dot product after the update.
- **题目算子链**：torch.mm, torch.mv, torch.exp, torch.min
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `matrix_vector_dot` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `matmul_linear` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, torch.mv, torch.exp, torch.min。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 122. openseek-8-f59fa2c7622c45649d2ebd96a1c9eef2 — `min_gelu`

- **任务类型**：matmul_linear
- **Wrapper**：`min_gelu(input, dim=None, keepdim=False, approximate='none', out=None) -> Tensor: input (Tensor): The input tensor. dim (int, optional): The dimension to reduce. If ``None``, returns the minimum of all elements. keepdim (bool, optional): Whether the output tensor retains :attr:`dim` as size 1. Default is ``False``. approximate (str, optional): The approximation method for GELU. Default is 'none'. out (Tensor, optional): The output tensor.`
- **功能描述**：Computes the Gaussian Error Linear Units (GELU) activation on the input tensor, then returns the minimum value along the specified dimension(s) or over all elements if no dimension is specified. The function supports two methods for computing GELU: exact and approximate using 'tanh'.
- **数学定义**：out = min(GELU(input)) GELU(x) = x * Φ(x) if approximate is 'none' GELU(x) = 0.5 * x * (1 + Tanh(√(2/π) * (x + 0.044715 * x^3))) if approximate is 'tanh'
- **补充约束**：Returns a namedtuple (values, indices) if dim is specified, otherwise returns the minimum value tensor.
- **题目算子链**：F.linear, torch.mm, F.gelu, torch.tanh, F.elu, torch.exp, torch.sin, torch.min
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `min_gelu` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `matmul_linear` 类，优先把自然语言描述映射到 PyTorch 算子链：F.linear, torch.mm, F.gelu, torch.tanh, F.elu, torch.exp, torch.sin, torch.min。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 123. openseek-8-aa17cdc9ea3b4b9692480d221ed2437b — `pow`

- **任务类型**：linalg
- **Wrapper**：`pow(input, exponent, *, out=None) -> Tensor; Args: input (Tensor): the input tensor. exponent (float or tensor): the exponent value; Keyword args: out (Tensor, optional): the output tensor.`
- **功能描述**：Takes the power of each element in input with exponent and returns a tensor with the result. exponent can be either a single float number or a Tensor with the same number of elements as input. If exponent is a scalar value, the operation applied is out_i = x_i ^ exponent. If exponent is a tensor, the operation applied is out_i = x_i ^ exponent_i. When exponent is a tensor, the shapes of input and exponent must be broadcastable.
- **数学定义**：out_i = x_i ^ exponent (for scalar exponent) out_i = x_i ^ exponent_i (for tensor exponent)
- **补充约束**：The operation supports both scalar and tensor exponents. When exponent is a tensor, its shape must be broadcastable with the input tensor.
- **题目算子链**：torch.mm, torch.exp, torch.sin, torch.min
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `pow` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `linalg` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, torch.exp, torch.sin, torch.min。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 124. openseek-8-fa93e89275484a3aa3306469ffc19232 — `relu_max_pool2d_conv2d`

- **任务类型**：conv_norm_pool
- **Wrapper**：`relu_max_pool2d_conv2d(input, weight, bias=None, conv_stride=1, conv_padding=0, conv_dilation=1, conv_groups=1, pool_kernel_size=2, pool_stride=None, pool_padding=0, pool_dilation=1, pool_ceil_mode=False, inplace=False) -> Tensor: input (Tensor): The input tensor of shape `(minibatch, in_channels, iH, iW)`. weight (Tensor): The convolution filters of shape `(out_channels, in_channels / groups, kH, kW)`. bias (Tensor, optional): Optional bias tensor of shape `(out_channels)`. Default: None. conv_`
- **功能描述**：Applies a 2D convolution over the input tensor, followed by max pooling and then applies the ReLU activation function element-wise to the pooled result. This combined operation is often used in convolutional neural networks (CNNs) for feature extraction, downsampling, and adding non-linearity.
- **数学定义**：\text{out} = \text{ReLU}(\text{MaxPool2D}(\text{conv2d}(\text{input}))) where the ReLU function is applied element-wise as: \text{ReLU}(x) = \max(0, x)
- **补充约束**：The function is typically used in CNNs.
- **题目算子链**：F.conv2d, F.linear, torch.mm, custom _rms_norm, F.max_pool2d, F.relu, F.elu, torch.exp, torch.max, torch.min, torch.where, torch.linalg.eig
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `relu_max_pool2d_conv2d` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `conv_norm_pool` 类，优先把自然语言描述映射到 PyTorch 算子链：F.conv2d, F.linear, torch.mm, custom _rms_norm, F.max_pool2d, F.relu, F.elu, torch.exp, torch.max, torch.min, torch.where, torch.linalg.eig。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 125. openseek-8-ba5e0d7afa334a6c9a9fb928e3e3a67b — `erf`

- **任务类型**：linalg
- **Wrapper**：`erf(input, *, out=None) -> Tensor; Args: input (Tensor): the input tensor. Keyword args: out (Tensor, optional): the output tensor.`
- **功能描述**：Computes the error function of the input tensor. The error function is used in probability, statistics, and partial differential equations describing diffusion.
- **数学定义**：\mathrm{erf}(x) = \frac{2}{\sqrt{\pi}} \int_{0}^{x} e^{-t^2} dt
- **补充约束**：The function outputs a tensor with values representing the error function of each element in the input tensor.
- **题目算子链**：torch.mm, torch.sqrt, torch.exp, torch.min, torch.linalg.qr
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `erf` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `linalg` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, torch.sqrt, torch.exp, torch.min, torch.linalg.qr。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 126. openseek-8-f30359dfb1514b54a0560bb570006024 — `sigmoid`

- **任务类型**：linalg
- **Wrapper**：`sigmoid(input, *, out=None) -> Tensor`
- **功能描述**：This function computes the sigmoid of the input tensor element-wise. The sigmoid function is a common activation function used in neural networks, which maps any real-valued number into the range (0, 1).
- **数学定义**：The sigmoid function is defined as: sigmoid(x) = 1 / (1 + exp(-x))
- **补充约束**：Alias for torch.special.expit.
- **题目算子链**：torch.mm, torch.sigmoid, torch.exp, torch.min
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `sigmoid` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `linalg` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, torch.sigmoid, torch.exp, torch.min。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 127. openseek-8-17a9f84522e74f8d980c07ebfc722a6b — `gelu`

- **任务类型**：matmul_linear
- **Wrapper**：`gelu(input, approximate='none') -> Tensor`
- **功能描述**：Applies the Gaussian Error Linear Unit (GELU) activation function element-wise to the input tensor. The function can be computed exactly or approximately using a tanh-based formula depending on the 'approximate' argument.
- **数学定义**：When approximate is 'none': GELU(x) = x * Φ(x), where Φ(x) is the Cumulative Distribution Function for Gaussian Distribution. When approximate is 'tanh': GELU(x) = 0.5 * x * (1 + Tanh(√(2/π) * (x + 0.044715 * x^3)))
- **补充约束**：See Gaussian Error Linear Units (GELUs) https://arxiv.org/abs/1606.08415
- **题目算子链**：F.linear, torch.mm, F.gelu, torch.tanh, F.elu, torch.exp, torch.sin, torch.min, torch.where
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `gelu` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `matmul_linear` 类，优先把自然语言描述映射到 PyTorch 算子链：F.linear, torch.mm, F.gelu, torch.tanh, F.elu, torch.exp, torch.sin, torch.min, torch.where。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 128. openseek-8-47ab9b2df14a4716a755b572550d005c — `det`

- **任务类型**：linalg
- **Wrapper**：`linalg.det(A, *, out=None) -> Tensor; A (Tensor): tensor of shape (*, n, n) where * is zero or more batch dimensions; out (Tensor, optional): output tensor. Ignored if None. Default: None.`
- **功能描述**：Computes the determinant of a square matrix. Supports input of float, double, cfloat and cdouble dtypes. Also supports batches of matrices, and if A is a batch of matrices then the output has the same batch dimensions.
- **补充约束**：:func:`torch.linalg.slogdet` computes the sign and natural logarithm of the absolute value of the determinant of square matrices.
- **题目算子链**：torch.mm, torch.exp, torch.log, torch.min, torch.where, torch.linalg.det
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `det` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `linalg` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, torch.exp, torch.log, torch.min, torch.where, torch.linalg.det。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 129. openseek-8-536ad43d80e44453b64fc5d527e231a1 — `fused_bmm_rmsnorm_gelu_dropout`

- **任务类型**：matmul_linear
- **Wrapper**：`fused_bmm_rmsnorm_gelu_dropout(input1, input2, normalized_shape, dropout_p=0.1, eps=1e-5, training=True, approximate='none', *, out=None) -> Tensor; input1 (Tensor): First input tensor for bmm, of shape (B, N, M), where B is the batch size; input2 (Tensor): Second input tensor for bmm, of shape (B, M, P); normalized_shape (int or list or torch.Size): Input shape from an expected input of size (B, N, P). This is the shape over which RMS normalization is applied; dropout_p (float, optional): Proba`
- **功能描述**：Performs a fused operation combining batch matrix multiplication, RMS normalization, GELU activation, and dropout.
- **数学定义**：Given two input tensors X and Y, this function computes: \[ \begin{align*} Z_1 &= \text{bmm}(X, Y) \\ Z_2 &= \text{RMSNorm}(Z_1, \epsilon) \\ Z_3 &= \text{GELU}(Z_2) \\ Z &= \text{Dropout}(Z_3, p) \end{align*} \] where: \- \text{bmm}(X, Y) performs batch matrix multiplication. \- \text{RMSNorm}(Z_1, \epsilon) = \frac{Z_1}{\sqrt{\text{mean}(Z_1^2, \text{dim}=\text{last}) + \epsilon}} \times \gamma, where \gamma is a learnable parameter (if `elementwise_affine=True`). \- \text{GELU}(Z_2) applies t
- **补充约束**：- The shapes of `input1` and `input2` must be compatible for batch matrix multiplication: `input1` of shape `(B, N, M)` and `input2` of shape `(B, M, P)` result in an output of shape `(B, N, P)`. - The `normalized_shape` argument for RMS normalization should match the shape of the last dimension(s) of the output tensor over which to compute the RMS. - The `GELU` activation is applied element-wise to the normalized output. - The `dropout` is applied during training when `training=True`. Set `trai
- **题目算子链**：F.linear, torch.bmm, torch.matmul, torch.mm, custom _rms_norm, F.dropout, F.gelu, torch.tanh, F.elu, torch.sqrt, torch.exp, torch.mean, torch.min, torch.linalg.vector_norm, torch.where, torch.linalg.qr
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `fused_bmm_rmsnorm_gelu_dropout` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `matmul_linear` 类，优先把自然语言描述映射到 PyTorch 算子链：F.linear, torch.bmm, torch.matmul, torch.mm, custom _rms_norm, F.dropout, F.gelu, torch.tanh, F.elu, torch.sqrt, torch.exp, torch.mean, torch.min, torch.linalg.vector_norm, torch.where, torch.linalg.qr。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 130. openseek-8-580130884817408d8c27ea57df9d733a — `floor`

- **任务类型**：reduction
- **Wrapper**：`floor(input, *, out=None) -> Tensor`
- **功能描述**：Returns a new tensor with the floor of the elements of the input, the largest integer less than or equal to each element. For integer inputs, follows the array-api convention of returning a copy of the input tensor.
- **数学定义**：\text{out}_{i} = \left\lfloor \text{input}_{i} \right\rfloor
- **补充约束**：For integer inputs, the function returns a copy of the input tensor.
- **题目算子链**：torch.mm, torch.exp, torch.min
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `floor` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `reduction` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, torch.exp, torch.min。
  1. 此类任务可短实现，重点处理 dim/keepdim/out/inplace/dtype 等参数。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 131. openseek-8-172f014718f34f869824a75fdb9b3094 — `rand`

- **任务类型**：indexing
- **Wrapper**：`rand(*size, *, generator=None, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False, pin_memory=False) -> Tensor`
- **功能描述**：Returns a tensor filled with random numbers from a uniform distribution on the interval [0, 1). The shape of the tensor is defined by the variable argument size.
- **补充约束**：The function can take a variable number of arguments to define the shape of the tensor. It supports optional parameters for generator, output tensor, data type, layout, device, autograd recording, and pinned memory.
- **题目算子链**：torch.mm, torch.exp, torch.var, torch.min
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `rand` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `indexing` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, torch.exp, torch.var, torch.min。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 132. openseek-8-a6cb6970cb9c4598aa966bc6942f93d8 — `cholesky_solve`

- **任务类型**：matmul_linear
- **Wrapper**：`cholesky_solve(B, L, upper=False, *, out=None) -> Tensor; B (Tensor): right-hand side tensor of shape (*, n, k) where * is zero or more batch dimensions; L (Tensor): tensor of shape (*, n, n) where * is zero or more batch dimensions consisting of lower or upper triangular Cholesky decompositions of symmetric or Hermitian positive-definite matrices; upper (bool, optional): flag that indicates whether L is lower triangular or upper triangular. Default: False; out (Tensor, optional): output tensor.`
- **功能描述**：Computes the solution of a system of linear equations with complex Hermitian or real symmetric positive-definite lhs given its Cholesky decomposition. Supports inputs of float, double, cfloat and cdouble dtypes. Also supports batches of matrices, and if :math:`A` or :math:`B` is a batch of matrices then the output has the same batch dimensions.
- **数学定义**：`A` or :math:`B` is a batch of matrices then the output has the same batch dimensions.
- **补充约束**：Supports float, double, cfloat, cdouble dtypes; Handles batches of matrices; Uses Cholesky decomposition
- **题目算子链**：F.linear, torch.mm, torch.exp, torch.min, torch.where, torch.linalg.cholesky, torch.linalg.solve
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `cholesky_solve` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `matmul_linear` 类，优先把自然语言描述映射到 PyTorch 算子链：F.linear, torch.mm, torch.exp, torch.min, torch.where, torch.linalg.cholesky, torch.linalg.solve。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 133. openseek-8-23d22a2a0b5949789ba007e8fd8e5f93 — `mul_sub`

- **任务类型**：reduction
- **Wrapper**：`def mul_sub(input, other_mul, other_sub, alpha=1, out=None) -> Tensor: input (Tensor): The input tensor to be multiplied. other_mul (Tensor or Number): The tensor or number to multiply with `input`. other_sub (Tensor or Number): The tensor or number to subtract from the multiplication result. alpha (Number, optional): The multiplier for :attr:`other_sub`. Default is 1. out (Tensor, optional): The output tensor.`
- **功能描述**：Multiplies the input tensor by another tensor or number, then subtracts another tensor or number from the result, scaled by a given alpha. This operation is performed element-wise.
- **数学定义**：\text{out}_i = (\text{input}_i \times \text{other\_mul}_i) - \text{alpha} \times \text{other\_sub}_i
- **补充约束**：The function allows for element-wise operations and supports both tensor and scalar inputs for multiplication and subtraction. The output can be stored in a specified tensor.
- **题目算子链**：torch.mm, torch.exp, torch.min
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `mul_sub` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `reduction` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, torch.exp, torch.min。
  1. 此类任务可短实现，重点处理 dim/keepdim/out/inplace/dtype 等参数。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 134. openseek-8-3a972c1556d2460ea5090fd8e1c73be6 — `ldl_factor`

- **任务类型**：matmul_linear
- **Wrapper**：`linalg.ldl_factor(A, *, hermitian=False, out=None) -> (Tensor, Tensor)`
- **功能描述**：Computes a compact representation of the LDL factorization of a Hermitian or symmetric (possibly indefinite) matrix. Supports input of float, double, cfloat and cdouble dtypes. Also supports batches of matrices, and if A is a batch of matrices then the output has the same batch dimensions. When A is complex valued it can be Hermitian (hermitian=True) or symmetric (hermitian=False). The factorization is of the form A = L D L^T. If hermitian is True then transpose operation is the conjugate transpose. L (or U) and D are stored in compact form in LD. They follow the format specified by LAPACK's sytrf function. These tensors may be used in torch.linalg.ldl_solve to solve linear systems.
- **数学定义**：A = L D L^T
- **补充约束**：When inputs are on a CUDA device, this function synchronizes that device with the CPU. For a version of this function that does not synchronize, see torch.linalg.ldl_factor_ex.
- **题目算子链**：F.linear, torch.mm, torch.exp, torch.min, torch.where, torch.linalg.solve
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `ldl_factor` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `matmul_linear` 类，优先把自然语言描述映射到 PyTorch 算子链：F.linear, torch.mm, torch.exp, torch.min, torch.where, torch.linalg.solve。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 135. openseek-8-b5d21b3de60a4ba4a05b2d523e1ecc8f — `abs`

- **任务类型**：linalg
- **Wrapper**：`abs(input, *, out=None) -> Tensor; Args: input (Tensor): the input tensor. Keyword args: out (Tensor, optional): the output tensor.`
- **功能描述**：Computes the absolute value of each element in the input tensor.
- **数学定义**：\text{out}_{i} = |\text{input}_{i}|
- **题目算子链**：torch.mm, torch.exp, torch.min
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `abs` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `linalg` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, torch.exp, torch.min。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 136. openseek-8-71baef9db7104be1819ab8f0c31187da — `mul`

- **任务类型**：reduction
- **Wrapper**：`mul(input, other, *, out=None) -> Tensor input (Tensor): the input tensor. other (Tensor or Number) - the tensor or number to multiply input by. out (Tensor, optional): the output tensor.`
- **功能描述**：Multiplies the input tensor by another tensor or a number, supporting broadcasting to a common shape, type promotion, and integer, float, and complex inputs.
- **数学定义**：\text{out}_i = \text{input}_i \times \text{other}_i
- **补充约束**：Supports broadcasting and type promotion.
- **题目算子链**：torch.mm, torch.exp, torch.min
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `mul` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `reduction` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, torch.exp, torch.min。
  1. 此类任务可短实现，重点处理 dim/keepdim/out/inplace/dtype 等参数。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 137. openseek-8-f51bac3ed24e40beb2f8d5041a140c84 — `softmax`

- **任务类型**：attention_softmax_loss
- **Wrapper**：`def softmax(input, dim, dtype=None) -> Tensor: input (Tensor): input; dim (int): A dimension along which softmax will be computed.; dtype (torch.dtype, optional): the desired data type of returned tensor. If specified, the input tensor is casted to dtype before the operation is performed. This is useful for preventing data type overflows. Default: None.`
- **功能描述**：Apply a softmax function to all slices along the specified dimension, re-scaling them so that the elements lie in the range [0, 1] and sum to 1.
- **数学定义**：Softmax(x_i) = exp(x_i) / sum_j exp(x_j)
- **补充约束**：This function doesn't work directly with NLLLoss, which expects the Log to be computed between the Softmax and itself. Use log_softmax instead (it's faster and has better numerical properties).
- **题目算子链**：torch.mm, F.log_softmax, F.softmax, torch.exp, torch.log, torch.sum, torch.max, torch.min
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `softmax` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `attention_softmax_loss` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, F.log_softmax, F.softmax, torch.exp, torch.log, torch.sum, torch.max, torch.min。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 138. openseek-8-1e86017637da48a7a9803d5bfda9c102 — `leaky_relu`

- **任务类型**：linalg
- **Wrapper**：`leaky_relu(input, negative_slope=0.01, inplace=False) -> Tensor`
- **功能描述**：Applies the Leaky ReLU activation function element-wise to the input tensor. The function is defined as LeakyReLU(x) = max(0, x) + negative_slope * min(0, x), where negative_slope is a small constant that allows a small, non-zero gradient when the unit is not active.
- **数学定义**：LeakyReLU(x) = max(0, x) + negative_slope * min(0, x)
- **补充约束**：See torch.nn.LeakyReLU for more details.
- **题目算子链**：torch.mm, F.leaky_relu, F.relu, F.elu, torch.exp, torch.max, torch.min, torch.where
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `leaky_relu` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `linalg` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, F.leaky_relu, F.relu, F.elu, torch.exp, torch.max, torch.min, torch.where。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 139. openseek-8-5f2629245b0141738177fbea44858a10 — `invert_matrix_lu`

- **任务类型**：matmul_linear
- **Wrapper**：`invert_matrix_lu(A, *, pivot=True, out=None) -> Tensor`
- **功能描述**：Computes the inverse of a square matrix using LU decomposition. Given a square invertible matrix A, it computes the inverse A^{-1} by performing LU decomposition and solving linear systems involving triangular matrices. Supports inputs of 'float', 'double', 'cfloat', and 'cdouble' dtypes, as well as batches of matrices.
- **数学定义**：A = P L U A^{-1} = U^{-1} L^{-1} P Y = L^{-1} P A^{-1} = U^{-1} Y
- **补充约束**：The function allows computing the inverse with or without pivoting (partial pivoting by default). It can handle batches of matrices, and an output tensor can be specified which will be ignored if set to None.
- **题目算子链**：F.linear, torch.mm, torch.exp, torch.sin, torch.min, torch.linalg.inv
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `invert_matrix_lu` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `matmul_linear` 类，优先把自然语言描述映射到 PyTorch 算子链：F.linear, torch.mm, torch.exp, torch.sin, torch.min, torch.linalg.inv。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 140. openseek-8-9e58a2371fd64537bd6d437d98e33fdb — `std`

- **任务类型**：linalg
- **Wrapper**：`def std(input, dim=None, *, correction=1, keepdim=False, out=None) -> Tensor: input (Tensor): the input tensor. dim (int or tuple of ints): the dimension or dimensions to reduce. correction (int): difference between the sample size and sample degrees of freedom. Defaults to `Bessel's correction`, correction=1. keepdim (bool): whether the output tensor has dim retained or not. out (Tensor, optional): the output tensor.`
- **功能描述**：Calculates the standard deviation over the specified dimensions of the input tensor. The dim argument can specify a single dimension, a list of dimensions, or None to reduce over all dimensions. If keepdim is set to True, the output tensor retains the reduced dimensions as size 1; otherwise, these dimensions are removed. The correction parameter adjusts the calculation for the difference between sample size and degrees of freedom, defaulting to Bessel's correction with correction=1.
- **数学定义**：\sigma = \sqrt{\frac{1}{\max(0,~N - \delta N)}\sum_{i=0}^{N-1}(x_i-\bar{x})^2}
- **补充约束**：The standard deviation function has undergone a change in version 2.0, where the argument previously called unbiased has been renamed to correction. Bessel's correction link: https://en.wikipedia.org/wiki/Bessel%27s_correction
- **题目算子链**：torch.mm, torch.sqrt, torch.exp, torch.sin, torch.sum, torch.std, torch.max, torch.min, torch.where, torch.linalg.qr
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `std` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `linalg` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, torch.sqrt, torch.exp, torch.sin, torch.sum, torch.std, torch.max, torch.min, torch.where, torch.linalg.qr。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 141. openseek-8-f359b4c150724486982dbf2f7f7bfee8 — `tril_mm_and_scale`

- **任务类型**：matmul_linear
- **Wrapper**：`def tril_mm_and_scale(A: torch.Tensor, B: torch.Tensor, alpha: float, beta: float) -> torch.Tensor: A (Tensor): A 2D matrix to be multiplied, of shape (n, n). B (Tensor): A matrix to be multiplied with the lower triangular part of A, of shape (n, p). alpha (float): Scaling factor for the initial matrix multiplication result. beta (float): Scaling factor for the final result.`
- **功能描述**：Performs a matrix multiplication of the lower triangular part of matrix `A` with matrix `B`, scales the result by `alpha`, and then scales the final output by `beta`. The operations are as follows: 1. Perform matrix multiplication between the lower triangular part of `A` (denoted as `torch.tril(A)`) and `B`, and scale the result by `alpha`. 2. Scale the resulting matrix from step 1 by `beta` to obtain the final result.
- **数学定义**：B = alpha * torch.mm(torch.tril(A), B) C = beta * B
- **题目算子链**：torch.matmul, torch.mm, custom _rms_norm, torch.exp, torch.min
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `tril_mm_and_scale` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `matmul_linear` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.matmul, torch.mm, custom _rms_norm, torch.exp, torch.min。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 142. openseek-8-6855d52e8dc4451dbdadc700c03a6746 — `A`

- **任务类型**：matmul_linear
- **Wrapper**：`A (Tensor), B (Tensor), *, left (bool, optional), out (Tensor, optional)`
- **功能描述**：Computes the solution of a square system of linear equations with a unique solution. Supports inputs of float, double, cfloat and cdouble dtypes. Also supports batches of matrices, and if the inputs are batches of matrices then the output has the same batch dimensions. Assumes that matrix A is invertible.
- **数学定义**：AX = B; XA = B
- **补充约束**：This function computes `X = A.inverse() @ B` in a faster and more numerically stable way than performing the computations separately. When inputs are on a CUDA device, this function synchronizes that device with the CPU. For a version of this function that does not synchronize, see `torch.linalg.solve_ex`.
- **题目算子链**：F.linear, torch.mm, torch.exp, torch.sum, torch.min, torch.linalg.solve, torch.linalg.inv
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `A` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `matmul_linear` 类，优先把自然语言描述映射到 PyTorch 算子链：F.linear, torch.mm, torch.exp, torch.sum, torch.min, torch.linalg.solve, torch.linalg.inv。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 143. openseek-8-8ea8849df9b24a91809cd8738fa3a5c9 — `airy_ai`

- **任务类型**：reduction
- **Wrapper**：`airy_ai(input, *, out=None) -> Tensor Args: input (Tensor): the input tensor. Keyword args: out (Tensor, optional): the output tensor.`
- **功能描述**：Computes the Airy function Ai for each element of the input tensor.
- **数学定义**：Airy function :math:`\text{Ai}\left(\text{input}\right)`.
- **题目算子链**：torch.mm, torch.exp, torch.min
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `airy_ai` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `reduction` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, torch.exp, torch.min。
  1. 此类任务可短实现，重点处理 dim/keepdim/out/inplace/dtype 等参数。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 144. openseek-8-aecb03abd3124ad49388b16605028005 — `signbit`

- **任务类型**：reduction
- **Wrapper**：`signbit(input, *, out=None) -> Tensor; Args: input (Tensor): the input tensor.; Keyword args: out (Tensor, optional): the output tensor.`
- **功能描述**：Tests if each element of the input tensor has its sign bit set or not. It handles signed zeros, so negative zero (-0) returns True.
- **补充约束**：signbit handles signed zeros, so negative zero (-0) returns True.
- **题目算子链**：torch.mm, torch.exp, torch.signbit, torch.min
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `signbit` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `reduction` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, torch.exp, torch.signbit, torch.min。
  1. 此类任务可短实现，重点处理 dim/keepdim/out/inplace/dtype 等参数。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 145. openseek-8-c0f63db0a8d84d1da24213d03b505974 — `matrix_multiply_and_row_dot`

- **任务类型**：matmul_linear
- **Wrapper**：`def matrix_multiply_and_row_dot(A: torch.Tensor, B: torch.Tensor, alpha: float, beta: float, C: torch.Tensor) -> torch.Tensor: A (Tensor): First input matrix of shape `(n, m)`. B (Tensor): Second input matrix of shape `(m, p)`. alpha (float): Scalar multiplier for the matrix-matrix product. beta (float): Scalar multiplier for the input matrix `C`. C (Tensor): Output matrix of shape `(n, p)` where the results are added.`
- **功能描述**：Computes a scaled matrix-matrix product, then calculates the dot product of the first two rows of the resulting matrix. First, it multiplies matrix A and B using the scalar alpha and then adds the scaled version of matrix C using scalar beta. Finally, it computes the dot product of the first two rows of the updated matrix C.
- **数学定义**：1. `C = alpha * torch.mm(A, B) + beta * C`; 2. `result = torch.dot(C[0], C[1])`
- **补充约束**：Assumes `C` has at least two rows for the dot product to be computed.
- **题目算子链**：torch.mm, torch.exp, torch.sin, torch.sum, torch.min, torch.where
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `matrix_multiply_and_row_dot` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `matmul_linear` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, torch.exp, torch.sin, torch.sum, torch.min, torch.where。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 146. openseek-8-02c5ca5a4ca444d6a8aac3278647e2be — `polygamma`

- **任务类型**：reduction
- **Wrapper**：`def polygamma(n, input, *, out=None) -> Tensor: n (int): the order of the polygamma function; input (Tensor): the input tensor.; out (Tensor, optional): the output tensor.`
- **功能描述**：Computes the n-th derivative of the digamma function on input. The function is implemented for nonnegative integers n >= 0.
- **数学定义**：\psi^{(n)}(x) = \frac{d^{(n)}}{dx^{(n)}} \psi(x)
- **补充约束**：Implemented only for nonnegative integers n >= 0.
- **题目算子链**：torch.mm, torch.exp, torch.min
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `polygamma` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `reduction` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, torch.exp, torch.min。
  1. 此类任务可短实现，重点处理 dim/keepdim/out/inplace/dtype 等参数。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 147. openseek-8-60b9ddf2ac9a4a34b1a2ae077afdf8f4 — `elu_linear`

- **任务类型**：matmul_linear
- **Wrapper**：`def elu_linear(input, weight, bias=None, alpha=1.0, inplace=False) -> Tensor: input (Tensor): The input tensor for the linear layer. weight (Tensor): The weight tensor for the linear transformation. bias (Tensor, optional): The bias tensor for the linear transformation. Default: None. alpha (float, optional): The \(\alpha\) parameter for the ELU function. Default: 1.0. inplace (bool, optional): Whether to apply ELU in-place. Default: False.`
- **功能描述**：Applies a linear transformation to the input tensor, followed by the Exponential Linear Unit (ELU) activation function applied element-wise. This combined operation first performs a linear transformation and then introduces non-linearity with ELU.
- **数学定义**：\text{out} = \text{ELU}(\text{Linear}(x)) \text{ELU}(x) = \begin{cases} x, & \text{ if } x > 0\\ \alpha * (\exp(x) - 1), & \text{ if } x \leq 0 \end{cases}
- **补充约束**：The function integrates linear transformation and ELU activation. The ELU activation applies element-wise to incorporate non-linearity after linear mapping.
- **题目算子链**：F.linear, torch.mm, custom _rms_norm, F.elu, torch.exp, torch.min, torch.linalg.eig
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `elu_linear` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `matmul_linear` 类，优先把自然语言描述映射到 PyTorch 算子链：F.linear, torch.mm, custom _rms_norm, F.elu, torch.exp, torch.min, torch.linalg.eig。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 148. openseek-8-514b7dabc27a48b097098f6986cfac12 — `fused_pairwise_distance_normalize`

- **任务类型**：linalg
- **Wrapper**：`def fused_pairwise_distance_normalize(x1: torch.Tensor, x2: torch.Tensor, p_norm: float = 2.0, eps_norm: float = 1e-12, eps_distance: float = 1e-6, keepdim: bool = False) -> torch.Tensor`
- **功能描述**：Computes the pairwise distance between two input tensors `x1` and `x2` after normalizing both tensors. Normalization is performed along the specified dimension, followed by pairwise distance calculation.
- **补充约束**：Normalization is performed along the specified dimension. Small values `eps_norm` and `eps_distance` are used to avoid division by zero during normalization and distance calculation, respectively.
- **题目算子链**：torch.mm, torch.exp, torch.min, torch.linalg.vector_norm
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `fused_pairwise_distance_normalize` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `linalg` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, torch.exp, torch.min, torch.linalg.vector_norm。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 149. openseek-8-e1e036a7a3c547bd8d5311a08a5b5997 — `Adam`

- **任务类型**：linalg
- **Wrapper**：`def Adam(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False, foreach=None, maximize=False, capturable=False, differentiable=False, fused=None) -> Optimizer`
- **功能描述**：Implements the Adam optimization algorithm, which is an adaptive learning rate optimization algorithm designed for training deep neural networks. It computes individual adaptive learning rates for different parameters from estimates of first and second moments of the gradients. The algorithm can optionally use the AMSGrad variant, apply weight decay, and maximize the objective function. It supports various implementation optimizations like foreach and fused implementations for performance improvements on CUDA.
- **数学定义**：m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t; v_t = \beta_2 v_{t-1} + (1-\beta_2) g^2_t; \widehat{m_t} = m_t/(1-\beta_1^t); \widehat{v_t} = v_t/(1-\beta_2^t); \theta_t = \theta_{t-1} - \gamma \widehat{m_t}/(\sqrt{\widehat{v_t}} + \epsilon)
- **补充约束**：The foreach and fused implementations are typically faster than the for-loop, single-tensor implementation. The algorithm is based on the paper 'Adam: A Method for Stochastic Optimization'.
- **题目算子链**：torch.mm, torch.sqrt, torch.exp, torch.sin, torch.var, torch.max, torch.min, torch.linalg.qr, torch.linalg.eig
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `Adam` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `linalg` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, torch.sqrt, torch.exp, torch.sin, torch.var, torch.max, torch.min, torch.linalg.qr, torch.linalg.eig。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 150. openseek-8-03bc8db2c11a462db455c5f133949ebb — `fused_hstack_div`

- **任务类型**：reduction
- **Wrapper**：`fused_hstack_div(tensors, divisor, *, rounding_mode=None, out=None) -> Tensor - **tensors** (sequence of Tensors): Sequence of tensors to be horizontally stacked. The tensors must have compatible shapes for stacking. - **divisor** (Tensor or Number): The tensor or number to divide the stacked tensor by. Must be broadcastable to the shape of the stacked tensor. - **rounding_mode** (str, optional): Type of rounding applied to the result: - `None`: Default behavior. Performs no rounding and, if bot`
- **功能描述**：Performs a fused operation combining horizontal stacking (hstack) and element-wise division. The function first horizontally stacks a sequence of tensors and then divides each element of the resulting tensor by the corresponding element of a divisor tensor, with optional rounding modes.
- **数学定义**：Given a sequence of tensors [X_1, X_2, \dots, X_n] and a divisor tensor D, the function computes: 1. **Horizontal Stacking:** \[ X = \text{hstack}(X_1, X_2, \dots, X_n) \] 2. **Element-wise Division:** \[ Y = \frac{X}{D} \]
- **补充约束**：- The tensors in `tensors` must have shapes that are compatible for horizontal stacking, i.e., the dimensions except for the stacking dimension must be the same. - The `divisor` tensor must be broadcastable to the shape of the stacked tensor. - The function supports autograd for gradient computation. - All operations are differentiable and support backpropagation.
- **题目算子链**：torch.mm, custom _rms_norm, torch.exp, torch.min
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `fused_hstack_div` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `reduction` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, custom _rms_norm, torch.exp, torch.min。
  1. 此类任务可短实现，重点处理 dim/keepdim/out/inplace/dtype 等参数。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 151. openseek-8-373c9833aa20491b8d0f164413265869 — `broadcast_tensors`

- **任务类型**：indexing
- **Wrapper**：`broadcast_tensors(*tensors) -> List of Tensors: *tensors (Args: any number of tensors of the same type) -> Example: x = torch.arange(3).view(1, 3), y = torch.arange(2).view(2, 1), a, b = torch.broadcast_tensors(x, y), a.size() == torch.Size([2, 3]), a == tensor([[0, 1, 2],[0, 1, 2]])`
- **功能描述**：Broadcasts the given tensors according to broadcasting semantics. This function takes multiple tensors as input and broadcasts them to have the same shape. Broadcasting refers to expanding the dimensions of tensors as necessary to make them compatible for element-wise operations. The broadcasted tensors share the same memory location for their elements, leading to potential issues with in-place operations.
- **补充约束**：More than one element of a broadcasted tensor may refer to a single memory location. In-place operations may result in incorrect behavior. If writing to tensors is needed, clone them first.
- **题目算子链**：torch.mm, torch.exp, torch.sin, torch.min
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `broadcast_tensors` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `indexing` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, torch.exp, torch.sin, torch.min。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 152. openseek-8-8050c0195af44fc39f4be117c5679de6 — `relu_conv2d`

- **任务类型**：conv_norm_pool
- **Wrapper**：`relu_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, inplace=False) -> Tensor: input (Tensor): The input tensor of shape (minibatch, in_channels, iH, iW). weight (Tensor): The convolution filters of shape (out_channels, in_channels / groups, kH, kW). bias (Tensor, optional): Optional bias tensor of shape (out_channels). Default: None. stride (int or tuple, optional): The stride of the convolution kernel. Default: 1. padding (int, tuple, or string, optional): Padding a`
- **功能描述**：Applies a 2D convolution over an input tensor, followed by applying the rectified linear unit (ReLU) activation function element-wise on the result. This operation first applies a 2D convolution over the input tensor using the specified filters, and then applies ReLU activation to the convolution result, setting all negative values to zero.
- **数学定义**：The operation is defined as: \text{out} = \text{ReLU}(\text{conv2d}(\text{input})), where \text{ReLU}(x) = \max(0, x).
- **补充约束**：Returns: Tensor: A tensor resulting from the 2D convolution followed by ReLU activation.
- **题目算子链**：F.conv2d, F.linear, torch.mm, F.relu, F.elu, torch.exp, torch.sin, torch.max, torch.min, torch.where, torch.linalg.eig
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `relu_conv2d` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `conv_norm_pool` 类，优先把自然语言描述映射到 PyTorch 算子链：F.conv2d, F.linear, torch.mm, F.relu, F.elu, torch.exp, torch.sin, torch.max, torch.min, torch.where, torch.linalg.eig。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 153. openseek-8-1ff3bd3bc01f4b7ea2acce581e682d0a — `log`

- **任务类型**：reduction
- **Wrapper**：`log(input, *, out=None) -> Tensor Args: input (Tensor): the input tensor. Keyword args: out (Tensor, optional): the output tensor.`
- **功能描述**：Returns a new tensor with the natural logarithm of the elements of the input tensor.
- **数学定义**：y_{i} = \log_{e} (x_{i})
- **补充约束**：The function computes the natural logarithm (base e) of each element in the input tensor.
- **题目算子链**：torch.mm, torch.exp, torch.log, torch.min
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `log` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `reduction` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, torch.exp, torch.log, torch.min。
  1. 此类任务可短实现，重点处理 dim/keepdim/out/inplace/dtype 等参数。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 154. openseek-8-e27f3597fb8242328141587a95027f22 — `adaptive_avg_pool2d`

- **任务类型**：conv_norm_pool
- **Wrapper**：`def adaptive_avg_pool2d(output_size) -> Tensor`
- **功能描述**：Apply a 2D adaptive average pooling over an input signal composed of several input planes. The output is of size H x W, for any input size. The number of output features is equal to the number of input planes. The target output size of the image can be a tuple (H, W) or a single H for a square image H x H. H and W can be either an int, or None which means the size will be the same as that of the input.
- **补充约束**：The target output size can be a single integer for square images or a tuple for rectangular dimensions. H and W can be None to retain input dimensions.
- **题目算子链**：torch.mm, F.avg_pool2d, F.adaptive_avg_pool2d, torch.exp, torch.sin, torch.mean, torch.min, torch.where
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `adaptive_avg_pool2d` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `conv_norm_pool` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, F.avg_pool2d, F.adaptive_avg_pool2d, torch.exp, torch.sin, torch.mean, torch.min, torch.where。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 155. openseek-8-8afb8e554ecf4aff97a0e4253c79e69c — `quantize_dynamic`

- **任务类型**：matmul_linear
- **Wrapper**：`quantize_dynamic(model, qconfig_spec=None, inplace=False, mapping=None) -> Model`
- **功能描述**：Converts a float model to a dynamic quantized model by replacing specified modules with their dynamic weight-only quantized versions. Provides simple usage with a dtype argument (either float16 or qint8), and fine-grained control with qconfig and mapping parameters. The process is performed in-place if specified, transforming the original model.
- **补充约束**：Dynamic quantization is typically performed on layers with large weight sizes such as Linear and RNN variants. The qconfig_spec can be a dictionary mapping submodule types or names to quantization configurations, or a set specifying which submodules to apply dynamic quantization to. If qconfig is provided, it overrides dtype.
- **题目算子链**：F.linear, torch.mm, torch.exp, torch.var, torch.min, torch.linalg.eig
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `quantize_dynamic` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `matmul_linear` 类，优先把自然语言描述映射到 PyTorch 算子链：F.linear, torch.mm, torch.exp, torch.var, torch.min, torch.linalg.eig。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 156. openseek-8-116e236fa2714fae998822e835c5c7a1 — `conv2d_add`

- **任务类型**：conv_norm_pool
- **Wrapper**：`conv2d_add(input, weight, bias=None, other=None, stride=1, padding=0, dilation=1, groups=1, alpha=1, out=None) -> Tensor: input (Tensor): The input tensor of shape (minibatch, in_channels, iH, iW). weight (Tensor): The convolution filters of shape (out_channels, in_channels / groups, kH, kW). bias (Tensor, optional): Optional bias tensor of shape (out_channels). Default: None. other (Tensor or Number, optional): The tensor or number to add to the convolution result. Default: None. stride (int or`
- **功能描述**：Applies a 2D convolution over an input image using specified filters and an optional bias, then adds another tensor or scalar to the convolution result, scaled by alpha. The input tensor shape is (minibatch, in_channels, iH, iW), and the weight tensor shape is (out_channels, in_channels / groups, kH, kW). The function also allows for setting the stride, padding, dilation, groups, and an optional output tensor.
- **数学定义**：\text{out} = \text{conv2d}(\text{input}, \text{weight}) + \alpha \times \text{other}
- **补充约束**：The 'groups' argument must divide both in_channels and out_channels. Padding can be specified as 'valid', 'same', a single number, or a tuple. The output tensor shape depends on convolution parameters.
- **题目算子链**：F.conv2d, torch.mm, torch.exp, torch.sin, torch.min, torch.linalg.eig
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `conv2d_add` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `conv_norm_pool` 类，优先把自然语言描述映射到 PyTorch 算子链：F.conv2d, torch.mm, torch.exp, torch.sin, torch.min, torch.linalg.eig。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 157. openseek-8-7bf6eff22256447782da374f4fb4ecd2 — `ifftshift`

- **任务类型**：linalg
- **Wrapper**：`ifftshift(input, dim=None) -> Tensor`
- **功能描述**：The function torch.fft.ifftshift is the inverse of torch.fft.fftshift. It rearranges the elements of the input tensor, which is in FFT order, such that the zero-frequency component is moved back to the original position. This is useful for preparing data for inverse FFT operations. The function can rearrange specified dimensions or all dimensions by default.
- **补充约束**：Inverse of torch.fft.fftshift.
- **题目算子链**：torch.mm, torch.exp, torch.min, torch.linalg.inv
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `ifftshift` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `linalg` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, torch.exp, torch.min, torch.linalg.inv。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 158. openseek-8-8bf05d5dc7404f4ea036130533e3578d — `signbit_bitwise_and`

- **任务类型**：linalg
- **Wrapper**：`def signbit_bitwise_and(input: torch.Tensor, other: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]: Args: input (Tensor): The input tensor. other (Tensor): The second tensor for bitwise AND, should be of integral or boolean types. Example: >>> a = torch.tensor([0.7, -1.2, 0., 2.3]) >>> b = torch.tensor([1, 0, 1, 1], dtype=torch.int8) >>> signbit_result, bitwise_and_result = signbit_bitwise_and(a, b) >>> signbit_result tensor([False, True, False, False]) >>> bitwise_and_result tensor([0, 0, 0`
- **功能描述**：Computes the sign bit check and the bitwise AND operation on the input tensors. `signbit` checks if the sign bit of each element in `input` is set, returning True for negative values, including -0. `bitwise_and` computes the bitwise AND between `input` and `other`, with the tensors needing to be of integral or boolean types.
- **补充约束**：torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]: Args: input (Tensor): The input tensor. other (Tensor): The second tensor for bitwise AND, should be of integral or boolean types. Example: >>> a = torch.tensor([0.7, -1.2, 0., 2.3]) >>> b = torch.tensor([1, 0, 1, 1], dtype=torch.int8) >>> signbit_result, bitwise_and_result = signbit_bitwise_and(a, b) >>> signbit_result tensor([False, True, False, False]) >>> bitwise_and_result tensor([0, 0, 0, 0], dtype=torch.int8)
- **题目算子链**：torch.mm, torch.exp, torch.signbit, torch.bitwise_and, torch.min
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `signbit_bitwise_and` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `linalg` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, torch.exp, torch.signbit, torch.bitwise_and, torch.min。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 159. openseek-8-5cd43fb7d32f415590ca4dfa123762b6 — `fused_repeat_interleave_log_softmax`

- **任务类型**：attention_softmax_loss
- **Wrapper**：`fused_repeat_interleave_log_softmax(input, repeats, dim=None, *, output_size=None, dtype=None, out=None) -> Tensor`
- **功能描述**：Performs a fused operation combining element-wise repeat interleave and log-softmax activation. First, the input tensor is repeated along the specified dimension according to the values in 'repeats'. Then, a log-softmax activation is applied to the repeated tensor along the specified dimension. This function is differentiable and supports autograd for gradient computation, making it useful for backpropagation in neural networks.
- **数学定义**：Given an input tensor X and repeats r, the function computes: 1. Repeat Interleave: The input tensor is repeated along the specified dimension: Y = repeat_interleave(X, r, dim). 2. Log-Softmax Activation: The log-softmax function is applied to the repeated tensor along the specified dimension: Z_i = log( exp(Y_i) / sum_j exp(Y_j) ) where the summation is over the specified dimension.
- **补充约束**：The 'repeats' parameter controls how many times each element is repeated along the specified dimension. The 'dim' parameter specifies the dimension along which to repeat and apply log-softmax. If 'dim' is None, the input is flattened before repeating. All operations are differentiable and support backpropagation.
- **题目算子链**：torch.mm, custom _rms_norm, F.log_softmax, F.softmax, torch.exp, torch.log, torch.sum, torch.max, torch.min, torch.repeat_interleave, torch.where
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `fused_repeat_interleave_log_softmax` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `attention_softmax_loss` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, custom _rms_norm, F.log_softmax, F.softmax, torch.exp, torch.log, torch.sum, torch.max, torch.min, torch.repeat_interleave, torch.where。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 160. openseek-8-c83a278744b94bd1a0ee2fdcf199989f — `cholesky`

- **任务类型**：linalg
- **Wrapper**：`def linalg.cholesky(A, *, upper=False, out=None) -> Tensor`
- **功能描述**：Computes the Cholesky decomposition of a complex Hermitian or real symmetric positive-definite matrix. Supports input of float, double, cfloat and cdouble dtypes. Also supports batches of matrices, and if A is a batch of matrices then the output has the same batch dimensions.
- **数学定义**：A = LL^{\text{H}} where L is a lower triangular matrix with real positive diagonal and L^{\text{H}} is the conjugate transpose when L is complex, and the transpose when L is real-valued.
- **补充约束**：When inputs are on a CUDA device, this function synchronizes that device with the CPU. For a version of this function that does not synchronize, see torch.linalg.cholesky_ex. Raises RuntimeError if the A matrix or any matrix in a batched A is not Hermitian (resp. symmetric) positive-definite.
- **题目算子链**：torch.mm, torch.exp, torch.min, torch.where, torch.linalg.cholesky
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `cholesky` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `linalg` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, torch.exp, torch.min, torch.where, torch.linalg.cholesky。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 161. openseek-8-aca437779ac84f62bd511eadb6202c94 — `ones_like`

- **任务类型**：linalg
- **Wrapper**：`ones_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format) -> Tensor; input (Tensor): the size of :attr:`input` will determine size of the output tensor.; dtype (torch.dtype, optional): the desired data type of returned Tensor. Default: if None, defaults to the dtype of :attr:`input`.; layout (torch.layout, optional): the desired layout of returned tensor. Default: if None, defaults to the layout of :attr:`input`.; device (torch.device, op`
- **功能描述**：Returns a tensor filled with the scalar value 1, with the same size as the input tensor. It mirrors the properties of the input in terms of dtype, layout, device, and memory format unless specified otherwise. The function does not support the 'out' keyword as of version 0.4, and equivalent operation needs an alternative approach.
- **补充约束**：Function does not support an 'out' keyword as of version 0.4. Use torch.ones for similar functionality if 'out' keyword is needed.
- **题目算子链**：torch.mm, custom _rms_norm, torch.exp, torch.min
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `ones_like` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `linalg` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, custom _rms_norm, torch.exp, torch.min。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 162. openseek-8-155a864dd7fb44d58adb6a1c60aa7949 — `autocast`

- **任务类型**：reduction
- **Wrapper**：`autocast(device_type, enabled=True, dtype=None, cache_enabled=True) -> ContextManager`
- **功能描述**：The function `torch.cuda.amp.autocast` is deprecated and replaced by `torch.amp.autocast("cuda", args...)`. It allows scripts to run in mixed precision, improving performance while maintaining accuracy. `autocast` serves as a context manager or decorator, wrapping the forward pass(es) of a network and any related loss computations. Tensors can be any type when entering an autocast region, and it is not necessary to manually cast models or inputs to `half()` or `bfloat16()`. The function selects op-specific data types for operations within an autocast region. Backward operations should not be run under autocast, as they execute in the same data type chosen for the corresponding forward operations.
- **补充约束**：Deprecated in favor of torch.amp.autocast("cuda"). Recommended to use for forward pass and loss computation only. Avoid using for backward passes. State is thread-local. Can be nested with `autocast(enabled=False)` to force a subregion to run in a specific dtype. The use of autocast in a new thread requires invoking the context manager or decorator in that thread.
- **题目算子链**：torch.mm, torch.exp, torch.sin, torch.min
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `autocast` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `reduction` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, torch.exp, torch.sin, torch.min。
  1. 此类任务可短实现，重点处理 dim/keepdim/out/inplace/dtype 等参数。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 163. openseek-8-8159829c87d0461bb1cbf060d61fe800 — `reciprocal`

- **任务类型**：reduction
- **Wrapper**：`reciprocal(input, *, out=None) -> Tensor; input (Tensor): the input tensor.; out (Tensor, optional): the output tensor.`
- **功能描述**：Returns a new tensor with the reciprocal of the elements of the input. Unlike NumPy's reciprocal, this function supports integral inputs by promoting them to the default scalar type.
- **数学定义**：\text{out}_{i} = \frac{1}{\text{input}_{i}}
- **补充约束**：Integral inputs to reciprocal are automatically promoted to the default scalar type.
- **题目算子链**：torch.mm, torch.exp, torch.min
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `reciprocal` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `reduction` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, torch.exp, torch.min。
  1. 此类任务可短实现，重点处理 dim/keepdim/out/inplace/dtype 等参数。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 164. openseek-8-bc963f2985a4493c9f0e747073930e5d — `cos_signbit`

- **任务类型**：reduction
- **Wrapper**：`def cos_signbit(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]: Args: input (Tensor): The input tensor for which the cosine and sign bit are computed.`
- **功能描述**：Computes the cosine of each element in the input tensor, followed by determining the sign bit for each cosine result, indicating if it is positive or negative.
- **数学定义**：\text{cos\_result} = \cos(\text{input}) \text{sign\_bit} = \text{signbit}(\text{cos\_result})
- **补充约束**：Returns a tuple containing the cosine of each element and a boolean tensor indicating the sign bit of each cosine result.
- **题目算子链**：torch.mm, torch.exp, torch.cos, torch.sin, torch.signbit, torch.min
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `cos_signbit` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `reduction` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, torch.exp, torch.cos, torch.sin, torch.signbit, torch.min。
  1. 此类任务可短实现，重点处理 dim/keepdim/out/inplace/dtype 等参数。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 165. openseek-8-7b43064c8d5e4260a988dddb31dcfa46 — `spectral_norm_eig`

- **任务类型**：linalg
- **Wrapper**：`spectral_norm_eig(A, *, out=None) -> Tensor A (Tensor): Tensor of shape `(*, n, n)` where `*` is zero or more batch dimensions consisting of square matrices. out (Tensor, optional): Output tensor. Ignored if `None`. Default: `None`.`
- **功能描述**：Computes the spectral norm (operator norm induced by the Euclidean vector norm) of a square matrix using its eigenvalues. The spectral norm is the largest absolute value among the eigenvalues of a matrix. It supports inputs of float, double, cfloat, and cdouble dtypes and handles batches of matrices.
- **数学定义**：\|A\|_2 = \max \{ |\lambda| : \lambda \text{ is an eigenvalue of } A \}
- **补充约束**：For normal matrices (where A A^{H} = A^{H} A), the spectral norm equals the largest absolute eigenvalue.
- **题目算子链**：torch.mm, torch.exp, torch.sin, torch.max, torch.min, torch.linalg.vector_norm, torch.where, torch.linalg.eig
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `spectral_norm_eig` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `linalg` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, torch.exp, torch.sin, torch.max, torch.min, torch.linalg.vector_norm, torch.where, torch.linalg.eig。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

## 166. openseek-8-54e0885a7e7a4f00bd14e3a05e53c090 — `fftn`

- **任务类型**：linalg
- **Wrapper**：`fftn(input, s=None, dim=None, norm=None, *, out=None) -> Tensor; input (Tensor): the input tensor; s (Tuple[int], optional): Signal size in the transformed dimensions. If given, each dimension dim[i] will either be zero-padded or trimmed to the length s[i] before computing the FFT. If a length -1 is specified, no padding is done in that dimension. Default: s = [input.size(d) for d in dim]; dim (Tuple[int], optional): Dimensions to be transformed. Default: all dimensions, or the last len(s) dimen`
- **功能描述**：Computes the N dimensional discrete Fourier transform of the input tensor. It returns all positive and negative frequency terms, even though for real inputs, half of these values are redundant. Supports torch.half and torch.chalf on CUDA with GPU Architecture SM53 or greater, but only for powers of 2 signal length in every transformed dimension.
- **补充约束**：The Fourier domain representation of any real signal satisfies the Hermitian property. torch.fft.rfftn returns the more compact one-sided representation where only the positive frequencies of the last dimension are returned.
- **题目算子链**：torch.mm, custom _rms_norm, torch.sqrt, torch.exp, torch.log, torch.min, torch.linalg.vector_norm, torch.where, torch.linalg.qr, torch.linalg.inv
- **答案中显式 API**：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d, F.adaptive_avg_pool2d, F.pixel_shuffle, F.relu, F.leaky_relu, torch.sigmoid, F.selu, torch.sqrt, torch.tanh, torch.exp, torch.log, torch.erfc, torch.rad2deg, torch.cos, torch.signbit, torch.bitwise_and, torch.argmax, F.softmax, F.log_softmax, torch.repeat_interleave, F.linear, F.softplus, F.elu, F.silu, F.hardsigmoid, torch.mv, torch.cholesky_solve, F.cosine_embedding_loss, F.normalize, F.cosine_similarity, F.pairwise_distance, F.embedding, torch.eq, torch.index_select, torch.gather, torch.masked_select, torch.div, torch.hstack, F.cross_entropy, F.layer_norm, torch.dot, torch.sum, torch.abs, torch.tril, torch.std, torch.min, F.affine_grid, F.grid_sample, torch.ones_like, torch.distributions, torch.quantization, torch.optim, torch.autocast, torch.linalg.solve, torch.linalg.cholesky, torch.linalg.lstsq, torch.linalg.pinv, torch.linalg.svd, torch.linalg.matrix_power, torch.linalg.det, torch.linalg.inv, torch.linalg.matrix_norm, torch.linalg.vector_norm
- **答案实现风格**：提供 _write_out，支持 out= 写回；使用 *args/**kwargs 增强参数兼容性；主要采用 PyTorch fallback；包含通用 torch/F API 分发；代码中显式使用：torch.nn, torch.rsqrt, torch.mean, torch.bmm, F.gelu, F.dropout, F.conv2d, F.batch_norm, F.instance_norm, F.max_pool2d。
- **拆解思路**：
  1. 先识别 wrapper `fftn` 与参数来源，保证最终代码定义同名函数。
  1. 题目属于 `linalg` 类，优先把自然语言描述映射到 PyTorch 算子链：torch.mm, custom _rms_norm, torch.sqrt, torch.exp, torch.log, torch.min, torch.linalg.vector_norm, torch.where, torch.linalg.qr, torch.linalg.inv。
  1. 此类任务容易因 Triton 维度/stride/mask 出错，稳定策略是先用 PyTorch fallback 实现语义。
  1. 答案风格通常包含 import torch、import torch.nn.functional as F、_write_out 辅助函数，以及 *args/**kwargs 兼容封装。

