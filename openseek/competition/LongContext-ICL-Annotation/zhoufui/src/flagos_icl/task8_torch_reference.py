from __future__ import annotations

REFERENCE_CODE = r'''
import math
from typing import *

import torch
import torch.nn.functional as F


def _out(y, out=None):
    if out is not None and torch.is_tensor(y):
        out.copy_(y)
        return out
    return y


def _norm_shape(x, normalized_shape=None):
    if normalized_shape is None:
        return (x.shape[-1],)
    if isinstance(normalized_shape, int):
        return (normalized_shape,)
    return tuple(normalized_shape)


def _rms_norm(x, normalized_shape=None, eps=1e-5):
    shape = _norm_shape(x, normalized_shape)
    dims = tuple(range(x.dim() - len(shape), x.dim()))
    scale = torch.rsqrt(torch.mean(x * x, dim=dims, keepdim=True) + eps)
    return x * scale


def _bn(x, running_mean=None, running_var=None, weight=None, bias=None, training=False, momentum=0.1, eps=1e-5):
    c = x.shape[1] if x.dim() > 1 else x.shape[0]
    if running_mean is None:
        running_mean = torch.zeros(c, device=x.device, dtype=x.dtype)
    if running_var is None:
        running_var = torch.ones(c, device=x.device, dtype=x.dtype)
    return F.batch_norm(x, running_mean, running_var, weight, bias, training, momentum, eps)


def _linear_last(x, weight, bias=None):
    return F.linear(x, weight, bias)


def _channel_layer_norm(x, weight=None, bias=None, eps=1e-5):
    if x.dim() >= 3 and weight is not None and x.shape[1] == weight.numel():
        order = [0] + list(range(2, x.dim())) + [1]
        inv = [0] * x.dim()
        for index, dim in enumerate(order):
            inv[dim] = index
        y = x.permute(order)
        y = F.layer_norm(y, (weight.numel(),), weight, bias, eps)
        return y.permute(inv)
    shape = tuple(weight.shape) if weight is not None else (x.shape[-1],)
    return F.layer_norm(x, shape, weight, bias, eps)


def _special(name, fallback):
    fn = getattr(torch.special, name, None)
    return fn if fn is not None else fallback


def add(input, other, *, alpha=1, out=None):
    return _out(torch.add(input, other, alpha=alpha), out)


def sub(input, other, *, alpha=1, out=None):
    return _out(torch.sub(input, other, alpha=alpha), out)


def mul(input, other, *, out=None):
    return _out(torch.mul(input, other), out)


def div(input, other, *, rounding_mode=None, out=None):
    return _out(torch.div(input, other, rounding_mode=rounding_mode), out)


def pow(input, exponent, *, out=None):
    return _out(torch.pow(input, exponent), out)


def abs(input, *, out=None):
    return _out(torch.abs(input), out)


def tanh(input, *, out=None):
    return _out(torch.tanh(input), out)


def sqrt(input, *, out=None):
    return _out(torch.sqrt(input), out)


def rsqrt(input, *, out=None):
    return _out(torch.rsqrt(input), out)


def reciprocal(input, *, out=None):
    return _out(torch.reciprocal(input), out)


def sigmoid(input, *, out=None):
    return _out(torch.sigmoid(input), out)


def relu(input, inplace=False):
    return F.relu(input, inplace=inplace)


def leaky_relu(input, negative_slope=0.01, inplace=False):
    return F.leaky_relu(input, negative_slope=negative_slope, inplace=inplace)


def gelu(input, approximate='none'):
    return F.gelu(input, approximate=approximate)


def selu(input, inplace=False):
    return F.selu(input, inplace=inplace)


def log(input, *, out=None):
    return _out(torch.log(input), out)


def log1p(input, *, out=None):
    return _out(torch.log1p(input), out)


def logit(input, eps=None, *, out=None):
    return _out(torch.logit(input, eps=eps), out)


def exp(input, *, out=None):
    return _out(torch.exp(input), out)


def floor(input, *, out=None):
    return _out(torch.floor(input), out)


def trunc(input, *, out=None):
    return _out(torch.trunc(input), out)


def cos(input, *, out=None):
    return _out(torch.cos(input), out)


def asin(input, *, out=None):
    return _out(torch.asin(input), out)


def erf(input, *, out=None):
    return _out(torch.erf(input), out)


def i0(input, *, out=None):
    return _out(torch.i0(input), out)


def erfc(input, *, out=None):
    return _out(torch.erfc(input), out)


def gammaln(input, *, out=None):
    return _out(torch.lgamma(input), out)


def digamma(input, *, out=None):
    return _out(torch.digamma(input), out)


def polygamma(n, input, *, out=None):
    return _out(torch.polygamma(n, input), out)


def zeta(input, other, *, out=None):
    return _out(_special("zeta", lambda x, q: torch.zeros_like(x))(input, other), out)


def bessel_j1(input, *, out=None):
    return _out(_special("bessel_j1", lambda x: torch.sin(x) / torch.clamp(x, min=1e-12))(input), out)


def airy_ai(input, *, out=None):
    return _out(_special("airy_ai", lambda x: torch.zeros_like(x))(input), out)


def signbit(input, *, out=None):
    return _out(torch.signbit(input), out)


def bitwise_and(input, other, *, out=None):
    return _out(torch.bitwise_and(input, other), out)


def mean(input, dim=None, keepdim=False, dtype=None, out=None):
    return _out(torch.mean(input, dim=dim, keepdim=keepdim, dtype=dtype), out)


def sum(input, dim=None, keepdim=False, *, dtype=None):
    return torch.sum(input, dim=dim, keepdim=keepdim, dtype=dtype)


def std(input, dim=None, *, correction=1, keepdim=False, out=None):
    return _out(torch.std(input, dim=dim, correction=correction, keepdim=keepdim), out)


def max(input, dim=None, keepdim=False, *, out=None):
    return torch.max(input) if dim is None else torch.max(input, dim=dim, keepdim=keepdim)


def min(input, dim=None, keepdim=False, *, out=None):
    return torch.min(input) if dim is None else torch.min(input, dim=dim, keepdim=keepdim)


def argmax(input, dim=None, keepdim=False):
    return torch.argmax(input, dim=dim, keepdim=keepdim)


def matmul(input, other, *, out=None):
    return _out(torch.matmul(input, other), out)


def addmm(input, mat1, mat2, *, beta=1, alpha=1, out=None):
    return _out(torch.addmm(input, mat1, mat2, beta=beta, alpha=alpha), out)


def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    return F.conv2d(input, weight, bias, stride, padding, dilation, groups)


def batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-5):
    return _bn(input, running_mean, running_var, weight, bias, training, momentum, eps)


def softmax(input, dim, dtype=None):
    return F.softmax(input, dim=dim, dtype=dtype)


def logsumexp(input, dim, keepdim=False, *, out=None):
    return _out(torch.logsumexp(input, dim=dim, keepdim=keepdim), out)


def log_softmax_linear(input, weight, bias=None, dim=-1, dtype=None):
    return F.log_softmax(_linear_last(input, weight, bias), dim=dim, dtype=dtype)


def softplus_linear(input, weight, bias=None, beta=1, threshold=20):
    return F.softplus(_linear_last(input, weight, bias), beta=beta, threshold=threshold)


def tanh_linear(input, weight, bias=None):
    return torch.tanh(_linear_last(input, weight, bias))


def elu_linear(input, weight, bias=None, alpha=1.0, inplace=False):
    return F.elu(_linear_last(input, weight, bias), alpha=alpha, inplace=inplace)


def dropout_sigmoid_linear(input, weight, bias=None, p=0.5, training=True, inplace=False):
    return F.dropout(torch.sigmoid(_linear_last(input, weight, bias)), p=p, training=training, inplace=inplace)


def sigmoid_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, out=None):
    return _out(torch.sigmoid(F.conv2d(input, weight, bias, stride, padding, dilation, groups)), out)


def relu_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, inplace=False):
    return F.relu(F.conv2d(input, weight, bias, stride, padding, dilation, groups), inplace=inplace)


def leaky_relu_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, negative_slope=0.01, inplace=False):
    return F.leaky_relu(F.conv2d(input, weight, bias, stride, padding, dilation, groups), negative_slope=negative_slope, inplace=inplace)


def gelu_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, approximate='none', out=None):
    return _out(F.gelu(F.conv2d(input, weight, bias, stride, padding, dilation, groups), approximate=approximate), out)


def conv2d_add(input, weight, bias=None, other=None, stride=1, padding=0, dilation=1, groups=1, alpha=1, out=None):
    y = F.conv2d(input, weight, bias, stride, padding, dilation, groups)
    if other is not None:
        y = torch.add(y, other, alpha=alpha)
    return _out(y, out)


def pixel_shuffle_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, upscale_factor=2):
    return F.pixel_shuffle(F.conv2d(input, weight, bias, stride, padding, dilation, groups), upscale_factor)


def relu_batch_norm_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, running_mean=None, running_var=None, bn_weight=None, bn_bias=None, training=False, momentum=0.1, eps=1e-5, inplace=False):
    return F.relu(_bn(F.conv2d(input, weight, bias, stride, padding, dilation, groups), running_mean, running_var, bn_weight, bn_bias, training, momentum, eps), inplace=inplace)


def dropout_relu_batch_norm_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, p=0.5, training=True, inplace=False):
    return F.dropout(F.relu(_bn(F.conv2d(input, weight, bias, stride, padding, dilation, groups), None, None, None, None, training), inplace=inplace), p=p, training=training)


def fused_instance_norm_selu_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, num_features=None, eps=1e-5, momentum=0.1, affine=False, track_running_stats=False):
    y = F.conv2d(input, weight, bias, stride, padding, dilation, groups)
    y = F.selu(y)
    return F.instance_norm(y, None, None, None, None, True, momentum, eps)


def relu_max_pool2d_conv2d(input, weight, bias=None, conv_stride=1, conv_padding=0, conv_dilation=1, conv_groups=1, pool_kernel_size=2, pool_stride=None, pool_padding=0, pool_dilation=1, pool_ceil_mode=False, inplace=False):
    y = F.conv2d(input, weight, bias, conv_stride, conv_padding, conv_dilation, conv_groups)
    y = F.max_pool2d(y, pool_kernel_size, pool_stride, pool_padding, pool_dilation, pool_ceil_mode)
    return F.relu(y, inplace=inplace)


def silu_batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-5):
    return F.silu(_bn(input, running_mean, running_var, weight, bias, training, momentum, eps))


def sigmoid_batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-5):
    return torch.sigmoid(_bn(input, running_mean, running_var, weight, bias, training, momentum, eps))


def fused_hardsigmoid_batch_norm(x, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-5, inplace=False):
    return F.hardsigmoid(_bn(x, running_mean, running_var, weight, bias, training, momentum, eps), inplace=inplace)


def fused_silu_layer_norm_conv2d(x, weight, conv_weight, conv_bias=None, conv_stride=1, conv_padding=0, conv_dilation=1, conv_groups=1, ln_eps=1e-5):
    y = F.conv2d(x, conv_weight, conv_bias, conv_stride, conv_padding, conv_dilation, conv_groups)
    y = _channel_layer_norm(y, weight, None, ln_eps)
    return F.silu(y)


def fused_layer_norm_relu_linear(input, weight, bias=None, normalized_shape=None, eps=1e-5, elementwise_affine=True):
    y = F.relu(F.linear(input, weight, bias))
    return F.layer_norm(y, _norm_shape(y, normalized_shape), eps=eps)


def fused_add_mul_groupnorm(input1, input2, weight, bias, num_groups, eps=1e-5, *, out=None):
    return _out(F.group_norm((input1 + input2) * input2, num_groups, weight, bias, eps), out)


def fused_bmm_rmsnorm_gelu_dropout(input1, input2, normalized_shape, dropout_p=0.1, eps=1e-5, training=True, approximate='none', *, out=None):
    y = torch.bmm(input1, input2)
    y = _rms_norm(y, normalized_shape, eps)
    y = F.gelu(y, approximate=approximate)
    y = F.dropout(y, p=dropout_p, training=training)
    return _out(y, out)


def fused_bmm_rmsnorm_gelu_dropout_sub(input1, input2, other, normalized_shape, dropout_p=0.5, training=True, approximate='none', eps=1e-5, *, out=None):
    y = fused_bmm_rmsnorm_gelu_dropout(input1, input2, normalized_shape, dropout_p, eps, training, approximate)
    return _out(y - other, out)


def fused_bmm_dropout_gelu(input1, input2, p=0.5, training=True, inplace=False, approximate='none', *, out=None):
    y = torch.bmm(input1, input2)
    y = F.dropout(y, p=p, training=training, inplace=inplace)
    return _out(F.gelu(y, approximate=approximate), out)


def fused_mv_logsoftmax_dropout(input, vec, p=0.5, training=True, inplace=False, dim=0, *, out=None):
    y = torch.mv(input, vec)
    y = F.log_softmax(y, dim=dim)
    return _out(F.dropout(y, p=p, training=training, inplace=inplace), out)


def fused_mv_sigmoid_sub(input, vec, other, alpha=1, *, out=None):
    return _out(torch.sigmoid(torch.mv(input, vec)) - alpha * other, out)


def fused_mul_add_logsoftmax_dropout_bmm(input1, input2, other, mat2, p=0.5, training=True, inplace=False, dim=-1, *, out=None):
    y = F.log_softmax(input1 * input2 + other, dim=dim)
    y = F.dropout(y, p=p, training=training, inplace=inplace)
    y = torch.bmm(y, mat2) if y.dim() == 3 else torch.matmul(y, mat2)
    return _out(y, out)


def fused_transformer_block(input, weight1, weight2, residual, dropout_p=0.1, eps=1e-5, *, out=None):
    y = F.softmax(torch.matmul(input, weight1), dim=-1)
    y = F.dropout(y, p=dropout_p, training=True)
    y = torch.matmul(y, weight2) + residual
    y = F.layer_norm(y, (y.shape[-1],), eps=eps)
    return _out(y, out)


def combined_activation(input, weight1, weight2, bias, *, out=None):
    return _out(torch.tanh(torch.sigmoid(torch.matmul(input, weight1))) * weight2 + bias, out)


def fused_index_select_eq(input, dim, index, other, *, out=None):
    return _out(torch.eq(torch.index_select(input, dim, index), other), out)


def fused_masked_select_add_gelu(input, mask, other, *, alpha=1, approximate='none', out=None):
    return _out(F.gelu(torch.masked_select(input, mask) + alpha * other, approximate=approximate), out)


def fused_gather_masked_fill(input, dim, index, mask, value, *, sparse_grad=False, out=None):
    y = torch.gather(input, dim, index, sparse_grad=sparse_grad).masked_fill(mask, value)
    return _out(y, out)


def fused_embedding_add_tanh(input_indices, weight, other, *, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False, out=None):
    y = F.embedding(input_indices, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse)
    return _out(torch.tanh(y + other), out)


def fused_cross_entropy_softmax_layernorm(logits, targets, normalized_shape, weight=None, ignore_index=-100, reduction='mean', label_smoothing=0.0, eps=1e-5, *, out=None):
    loss = F.cross_entropy(logits, targets, weight=weight, ignore_index=ignore_index, reduction=reduction, label_smoothing=label_smoothing)
    probs = F.softmax(logits, dim=1)
    normed = F.layer_norm(probs, _norm_shape(probs, normalized_shape), eps=eps)
    return loss, _out(normed, out)


def fused_cross_entropy_log_softmax(input, target, dim=1, weight=None, ignore_index=-100, reduction='mean', label_smoothing=0.0):
    return F.cross_entropy(input, target, weight=weight, ignore_index=ignore_index, reduction=reduction, label_smoothing=label_smoothing)


def fused_cosine_embedding_loss_with_normalization(input1, input2, target, margin=0, reduction='mean'):
    return F.cosine_embedding_loss(F.normalize(input1), F.normalize(input2), target, margin=margin, reduction=reduction)


def fused_pairwise_distance_normalize(x1, x2, p_norm=2.0, eps_norm=1e-12, eps_distance=1e-6, keepdim=False):
    return F.pairwise_distance(F.normalize(x1, p=p_norm, eps=eps_norm), F.normalize(x2, p=p_norm, eps=eps_norm), p=p_norm, eps=eps_distance, keepdim=keepdim)


def normalize_pairwise_distance(x1, x2, p_distance=2.0, eps_distance=1e-6, keepdim=False, p_norm=2, dim_norm=1, eps_norm=1e-12):
    y = F.pairwise_distance(x1, x2, p=p_distance, eps=eps_distance, keepdim=keepdim)
    dim = dim_norm if -y.dim() <= dim_norm < y.dim() else -1
    return F.normalize(y, p=p_norm, dim=dim, eps=eps_norm)


def normalized_cosine_similarity(x1, x2, dim=1, eps_similarity=1e-8, p_norm=2, eps_norm=1e-12):
    return F.cosine_similarity(F.normalize(x1, p=p_norm, dim=dim, eps=eps_norm), F.normalize(x2, p=p_norm, dim=dim, eps=eps_norm), dim=dim, eps=eps_similarity)


def fused_pairwise_distance_adaptive_avg_pool2d(x1, x2, output_size, p=2.0, eps=1e-6, keepdim=False):
    return F.pairwise_distance(F.adaptive_avg_pool2d(x1, output_size).flatten(1), F.adaptive_avg_pool2d(x2, output_size).flatten(1), p=p, eps=eps, keepdim=keepdim)


def fused_avg_pool2d_cosine_similarity(x1, x2, kernel_size, stride=None, padding=0, eps=1e-8):
    y = F.cosine_similarity(x1, x2, dim=1, eps=eps).unsqueeze(1)
    return F.avg_pool2d(y, kernel_size, stride, padding)


def sigmoid_argmax(input, dim=None, keepdim=False):
    return torch.argmax(torch.sigmoid(input), dim=dim, keepdim=keepdim)


def softmax_mul(input, other, dim, dtype=None, out=None):
    return _out(F.softmax(input, dim=dim, dtype=dtype) * other, out)


def softmax_log(input, dim=-1, dtype=None):
    return F.softmax(torch.log(input), dim=dim, dtype=dtype)


def add_gelu(input, other, alpha=1, approximate='none', out=None):
    return _out(F.gelu(input + alpha * other, approximate=approximate), out)


def sub_gelu(input, other, alpha=1, approximate='none', out=None):
    return _out(F.gelu(input - alpha * other, approximate=approximate), out)


def mul_relu(input, other, inplace=False, out=None):
    return _out(F.relu(input * other, inplace=inplace), out)


def mul_sub(input, other_mul, other_sub, alpha=1, out=None):
    return _out(input * other_mul - alpha * other_sub, out)


def add_mean(input, other, dim=None, alpha=1, keepdim=False, dtype=None, out=None):
    return _out(torch.mean(input + alpha * other, dim=dim, keepdim=keepdim, dtype=dtype), out)


def sum_std(input, dim=None, keepdim=False, dtype=None, correction=1, out=None):
    y = torch.sum(input, dim=dim, keepdim=keepdim, dtype=dtype) + torch.std(input, dim=dim, keepdim=keepdim, correction=correction)
    return _out(y, out)


def gelu_std(input, dim=None, keepdim=False, correction=1, approximate='none', out=None):
    return _out(torch.std(F.gelu(input, approximate=approximate), dim=dim, keepdim=keepdim, correction=correction), out)


def gelu_min(input, approximate='none', dim=None, keepdim=False, out=None):
    y = F.gelu(input, approximate=approximate)
    return torch.min(y) if dim is None else torch.min(y, dim=dim, keepdim=keepdim)


def min_gelu(input, dim=None, keepdim=False, approximate='none', out=None):
    y = F.gelu(input, approximate=approximate)
    return torch.min(y) if dim is None else torch.min(y, dim=dim, keepdim=keepdim)


def sqrt_tanh(input, out=None):
    return _out(torch.tanh(torch.sqrt(input)), out)


def sqrt_exp(input, out=None):
    return _out(torch.exp(torch.sqrt(input)), out)


def exp_sqrt(input, out=None):
    return _out(torch.sqrt(torch.exp(input)), out)


def exp_mean(input, dim=None, keepdim=False, dtype=None, out=None):
    return _out(torch.mean(torch.exp(input), dim=dim, keepdim=keepdim, dtype=dtype), out)


def log_tanh(input, out=None):
    return _out(torch.tanh(torch.log(input)), out)


def relu_sqrt(input, inplace=False, out=None):
    return _out(torch.sqrt(F.relu(input, inplace=inplace)), out)


def cos_avg_pool1d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True):
    return F.avg_pool1d(torch.cos(input), kernel_size, stride, padding, ceil_mode, count_include_pad)


def sigmoid_adaptive_avg_pool2d(input, output_size):
    return torch.sigmoid(F.adaptive_avg_pool2d(input, output_size))


def adaptive_avg_pool2d(input=None, output_size=None):
    if output_size is None:
        return torch.nn.AdaptiveAvgPool2d(input)
    return F.adaptive_avg_pool2d(input, output_size)


def fused_fractional_max_pool2d_with_relu(input, kernel_size, output_size=None, output_ratio=None, return_indices=False):
    y = F.fractional_max_pool2d(F.relu(input), kernel_size, output_size=output_size, output_ratio=output_ratio, return_indices=return_indices)
    if return_indices:
        return F.relu(y[0]), y[1]
    return F.relu(y)


def fused_hardshrink_dropout(input, p=0.5, training=True, inplace=False, lambd=0.5):
    return F.hardshrink(F.dropout(input, p=p, training=training, inplace=inplace), lambd=lambd)


def fused_repeat_interleave_log_softmax(input, repeats, dim=None, *, output_size=None, dtype=None, out=None):
    y = torch.repeat_interleave(input, repeats, dim=dim, output_size=output_size)
    return _out(F.log_softmax(y, dim=(dim if dim is not None else -1), dtype=dtype), out)


def fused_tile_exp(input, dims, *, out=None):
    return _out(torch.exp(input.repeat(dims)), out)


def fused_hstack_div(tensors, divisor, *, rounding_mode=None, out=None):
    return _out(torch.div(torch.hstack(tensors), divisor, rounding_mode=rounding_mode), out)


def erfc_sqrt(input):
    return torch.erfc(input), torch.sqrt(input)


def rad2deg_sqrt(input):
    return torch.rad2deg(input), torch.sqrt(input)


def cos_signbit(input):
    y = torch.cos(input)
    return y, torch.signbit(y)


def signbit_bitwise_and(input, other):
    return torch.signbit(input), torch.bitwise_and(input.to(other.dtype), other)


def bitwise_and_binomial(input, other, total_count, probs=None, logits=None):
    return torch.bitwise_and(input, other)


def chebyshev_polynomial_t(input, n, *, out=None):
    if torch.is_tensor(n):
        n_int = int(n.item())
    else:
        n_int = int(n)
    y = torch.cos(n_int * torch.acos(input))
    return _out(y, out)


def tensordot(a, b, dims):
    return torch.tensordot(a, b, dims=dims)


def tensordot_rsqrt(a, b, dims):
    return torch.rsqrt(torch.tensordot(a, b, dims=dims))


def qr(A, mode='reduced', *, out=None):
    return torch.linalg.qr(A, mode=mode)


def lu(A, *, pivot=True, out=None):
    return torch.linalg.lu(A, pivot=pivot)


def solve_multiple_lu(A, Bs, *, pivot=True, out=None):
    return _out(torch.linalg.solve(A, Bs), out)


def fused_lu_solve(A, b):
    return torch.linalg.solve(A, b)


def fused_qr_solve(A, b):
    return torch.linalg.lstsq(A, b).solution


def least_squares_qr(A, b, *, mode='reduced', out=None):
    return _out(torch.linalg.lstsq(A, b).solution, out)


def determinant_via_qr(A, *, mode='reduced', out=None):
    return _out(torch.linalg.det(A), out)


def determinant_lu(A, *, pivot=True, out=None):
    return _out(torch.linalg.det(A), out)


def invert_matrix_lu(A, *, pivot=True, out=None):
    return _out(torch.linalg.inv(A), out)


def solve_symmetric_ldl(A, b, *, hermitian=False, out=None):
    return _out(torch.linalg.solve(A, b), out)


def solve(A, B, *, left=True, out=None):
    y = torch.linalg.solve(A, B) if left else torch.linalg.solve(A.mT, B.mT).mT
    return _out(y, out)


def solve_and_add_scaled_vector(A, b, y, alpha):
    return torch.linalg.solve_triangular(A, b, upper=True) + alpha * y


def fused_cholesky_solve(A, b):
    return torch.cholesky_solve(b, torch.linalg.cholesky(A))


def cholesky_solve(B, L, upper=False, *, out=None):
    return _out(torch.cholesky_solve(B, L, upper=upper), out)


def pseudoinverse_svd(A, *, full_matrices=True, rcond=1e-15, out=None):
    return _out(torch.linalg.pinv(A, rtol=rcond), out)


def low_rank_svd_approximation(A, k, *, full_matrices=True, out=None):
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    y = (U[..., :, :k] * S[..., :k].unsqueeze(-2)) @ Vh[..., :k, :]
    return _out(y, out)


def fused_svd_reconstruct(A):
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    return (U * S.unsqueeze(-2)) @ Vh


def matrix_power_eig(A, k, *, out=None):
    return _out(torch.linalg.matrix_power(A, int(k)), out)


def spectral_norm_eig(A, *, out=None):
    return _out(torch.linalg.matrix_norm(A, ord=2), out)


def symmetric_matrix_vector_norm(A, x, alpha, beta, p=2.0):
    y = alpha * torch.mv(A, x) + beta
    return torch.norm(y, p=p)


def scaled_add_norm(y, x, alpha):
    y.add_(x, alpha=alpha)
    return torch.norm(y, p=2)


def scaled_add_dot(y, x, alpha):
    y.add_(x, alpha=alpha)
    return torch.dot(y, y)


def symmetric_mm_and_abs_sum(A, C, alpha, beta):
    y = alpha * torch.mm(A, A.T) + beta * C
    return torch.sum(torch.abs(y))


def matrix_multiply_symmetric(A, B, C, alpha, beta):
    C = alpha * torch.mm(A, B) + beta * C
    return alpha * torch.mm(C, C.T) + beta * C


def matrix_multiply_and_row_dot(A, B, alpha, beta, C):
    C = alpha * torch.mm(A, B) + beta * C
    return torch.dot(C[0], C[1])


def matrix_vector_dot(A, x, y, alpha, beta):
    y = alpha * torch.mv(A, x) + beta * y
    return torch.dot(y, x)


def tril_mm_and_scale(A, B, alpha, beta):
    return beta * (alpha * torch.mm(torch.tril(A), B))


def softmax_log(input, dim=-1, dtype=None):
    return torch.log(F.softmax(input, dim=dim, dtype=dtype))


def grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=False):
    return F.grid_sample(input, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)


def grid_sample_with_affine(input, theta, size, mode='bilinear', padding_mode='zeros', align_corners=False):
    return F.grid_sample(input, F.affine_grid(theta, size, align_corners=align_corners), mode=mode, padding_mode=padding_mode, align_corners=align_corners)


def index_fill_(input, dim, index, value):
    return input.index_fill_(dim, index, value)


def logspace(start, end, steps, base=10.0, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False):
    return _out(torch.logspace(start, end, steps, base=base, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad), out)


def rand(*size, generator=None, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False, pin_memory=False):
    return _out(torch.rand(*size, generator=generator, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad, pin_memory=pin_memory), out)


def broadcast_tensors(*tensors):
    return torch.broadcast_tensors(*tensors)


def ones_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format):
    return torch.ones_like(input, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad, memory_format=memory_format)


def ifftshift(input, dim=None):
    return torch.fft.ifftshift(input, dim=dim)


def fftn(input, s=None, dim=None, norm=None, *, out=None):
    return _out(torch.fft.fftn(input, s=s, dim=dim, norm=norm), out)


def autocast(device_type, enabled=True, dtype=None, cache_enabled=True):
    return torch.amp.autocast(device_type, enabled=enabled, dtype=dtype, cache_enabled=cache_enabled)


def quantize_dynamic(model, qconfig_spec=None, inplace=False, mapping=None):
    return torch.quantization.quantize_dynamic(model, qconfig_spec=qconfig_spec, inplace=inplace, mapping=mapping)


def SGD(params, lr=1e-3, momentum=0, weight_decay=0, dampening=0, nesterov=False, maximize=False, foreach=None, differentiable=False, fused=None):
    return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay, dampening=dampening, nesterov=nesterov, maximize=maximize, foreach=foreach, differentiable=differentiable, fused=fused)


def Adam(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False, foreach=None, maximize=False, capturable=False, differentiable=False, fused=None):
    return torch.optim.Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad, foreach=foreach, maximize=maximize, capturable=capturable, differentiable=differentiable, fused=fused)


def permute_copy(input, dims):
    return input.permute(dims).clone()


if not hasattr(torch, "permute_copy"):
    torch.permute_copy = permute_copy


class _LinalgNamespace:
    @staticmethod
    def svd(A, full_matrices=True, *, driver=None, out=None):
        return torch.linalg.svd(A, full_matrices=full_matrices, driver=driver)

    @staticmethod
    def eig(A, *, out=None):
        return torch.linalg.eig(A)

    @staticmethod
    def det(A, *, out=None):
        return _out(torch.linalg.det(A), out)

    @staticmethod
    def ldl_factor(A, *, hermitian=False, out=None):
        if hasattr(torch.linalg, "ldl_factor"):
            return torch.linalg.ldl_factor(A, hermitian=hermitian)
        return torch.linalg.lu_factor(A)

    @staticmethod
    def cholesky(A, *, upper=False, out=None):
        return _out(torch.linalg.cholesky(A, upper=upper), out)

    @staticmethod
    def solve(A, B, *, left=True, out=None):
        return solve(A, B, left=left, out=out)


linalg = _LinalgNamespace()
linalg_svd = linalg.svd
linalg_eig = linalg.eig
linalg_det = linalg.det
linalg_ldl_factor = linalg.ldl_factor
linalg_cholesky = linalg.cholesky
'''
