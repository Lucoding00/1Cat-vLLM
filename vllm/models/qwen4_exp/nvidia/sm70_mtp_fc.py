# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Preserve local FP16 residual bytes while sharing the draft FC gather."""

import torch

import vllm.envs as envs
from vllm.compilation.sm70_decode_graph import is_sm70_decode_graph_compiling
from vllm.distributed import tensor_model_parallel_all_gather
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    UnquantizedLinearMethod,
)
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.utils.torch_utils import direct_register_custom_op


@triton.jit
def _qwen38_mtp_fc_fused_gemv_add_kernel(
    e_ptr,
    h_ptr,
    w_e_ptr,
    w_h_ptr,
    out_ptr,
    K: tl.constexpr,
    N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    col = tl.program_id(0)
    row = tl.program_id(1)
    offsets = tl.arange(0, BLOCK_K)
    acc_e = tl.zeros((BLOCK_K,), dtype=tl.float32)
    acc_h0 = tl.zeros((BLOCK_K,), dtype=tl.float32)
    acc_h1 = tl.zeros((BLOCK_K,), dtype=tl.float32)
    acc_h2 = tl.zeros((BLOCK_K,), dtype=tl.float32)
    acc_h3 = tl.zeros((BLOCK_K,), dtype=tl.float32)
    e_row_ptr = e_ptr + row * K
    h_row_ptr = h_ptr + row * (4 * K)
    w_e_col_ptr = w_e_ptr + col * K
    w_h_col_ptr = w_h_ptr + col * K
    for block_start in tl.static_range(0, K, BLOCK_K):
        idx = block_start + offsets
        we = tl.load(w_e_col_ptr + idx).to(tl.float32)
        wh = tl.load(w_h_col_ptr + idx).to(tl.float32)
        ev = tl.load(e_row_ptr + idx).to(tl.float32)
        h0 = tl.load(h_row_ptr + 0 * K + idx).to(tl.float32)
        h1 = tl.load(h_row_ptr + 1 * K + idx).to(tl.float32)
        h2 = tl.load(h_row_ptr + 2 * K + idx).to(tl.float32)
        h3 = tl.load(h_row_ptr + 3 * K + idx).to(tl.float32)
        acc_e += ev * we
        acc_h0 += h0 * wh
        acc_h1 += h1 * wh
        acc_h2 += h2 * wh
        acc_h3 += h3 * wh
    el = tl.sum(acc_e, axis=0).to(tl.float16)
    hl0 = tl.sum(acc_h0, axis=0).to(tl.float16)
    hl1 = tl.sum(acc_h1, axis=0).to(tl.float16)
    hl2 = tl.sum(acc_h2, axis=0).to(tl.float16)
    hl3 = tl.sum(acc_h3, axis=0).to(tl.float16)
    out_base = out_ptr + row * (4 * N) + col
    tl.store(out_base + 0 * N, el + hl0)
    tl.store(out_base + 1 * N, el + hl1)
    tl.store(out_base + 2 * N, el + hl2)
    tl.store(out_base + 3 * N, el + hl3)


def _qwen38_sm70_mtp_fc_local_add(
    e: torch.Tensor,
    h: torch.Tensor,
    w_e: torch.Tensor,
    w_h: torch.Tensor,
) -> torch.Tensor:
    rows = e.shape[0]
    if (
        HAS_TRITON
        and e.device.type == "cuda"
        and 1 <= rows <= 4
        and e.shape == (rows, 2560)
        and h.shape == (rows, 4, 2560)
        and w_e.shape == (640, 2560)
        and w_h.shape == (640, 2560)
        and e.dtype == h.dtype == w_e.dtype == w_h.dtype == torch.float16
        and e.is_contiguous()
        and h.is_contiguous()
        and w_e.is_contiguous()
        and w_h.is_contiguous()
        and not getattr(torch.backends.cuda.matmul, "allow_fp16_accumulation", False)
    ):
        out = torch.empty((rows, 4, 640), dtype=e.dtype, device=e.device)
        _qwen38_mtp_fc_fused_gemv_add_kernel[(640, rows)](
            e,
            h,
            w_e,
            w_h,
            out,
            K=2560,
            N=640,
            BLOCK_K=512,
            num_warps=2,
        )
        return out
    el = torch.nn.functional.linear(e, w_e)
    hl = torch.nn.functional.linear(h, w_h)
    return el.unsqueeze(-2) + hl


def _qwen38_sm70_mtp_fc_local_add_fake(
    e: torch.Tensor,
    h: torch.Tensor,
    w_e: torch.Tensor,
    w_h: torch.Tensor,
) -> torch.Tensor:
    del h, w_h
    return e.new_empty((e.shape[0], 4, w_e.shape[0]))


direct_register_custom_op(
    op_name="qwen38_sm70_mtp_fc_local_add",
    op_func=_qwen38_sm70_mtp_fc_local_add,
    fake_impl=_qwen38_sm70_mtp_fc_local_add_fake,
)


@torch.compiler.assume_constant_result
def _sm70_device():
    return current_platform.is_device_capability(70)


def _can_combine_fc(model, embedding, hidden):
    if (
        envs.VLLM_BATCH_INVARIANT
        or embedding.ndim != 2
        or hidden.ndim != 2
        or embedding.shape[-1] != 2560
        or hidden.shape[-1] != 10240
        or embedding.dtype != torch.float16
        or hidden.dtype != torch.float16
        or not embedding.is_cuda
        or not hidden.is_cuda
        or not _sm70_device()
    ):
        return False
    return all(
        hasattr(model, attr)
        and type(layer := getattr(model, attr)) is ColumnParallelLinear
        and type(layer.quant_method) is UnquantizedLinearMethod
        and layer.tp_size == 4
        and layer.gather_output
        and not layer.return_bias
        and layer.bias is None
        and layer.weight.shape == (640, 2560)
        and layer.weight.dtype == torch.float16
        for attr in ("fc_embedding", "fc_hidden")
    )


def maybe_combine_fc(model, embedding, hidden):
    if not _can_combine_fc(model, embedding, hidden):
        return None
    rows = embedding.shape[0]
    e = model.pre_fc_norm_embedding(embedding)
    h = model.pre_fc_norm_hidden(hidden).view(rows, 4, 2560)
    if e.device.type == "cuda":
        local_sum = torch.ops.vllm.qwen38_sm70_mtp_fc_local_add(
            e, h, model.fc_embedding.weight, model.fc_hidden.weight
        )
    else:
        el = model.fc_embedding.quant_method.apply(model.fc_embedding, e)
        hl = model.fc_hidden.quant_method.apply(model.fc_hidden, h)
        local_sum = el.unsqueeze(-2) + hl
    # Addition commutes with the feature gather, with exactly the same FP16
    # rounding per element. Norms/projections and TP ownership are unchanged.
    return tensor_model_parallel_all_gather(local_sum).flatten(-2)
