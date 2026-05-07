"""
gpt-oss style SwiGLU Feed-Forward Network

Reference SwiGLU implementation:
https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/swiglu.py
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl

from liger_kernel.ops.utils import calculate_settings, ensure_contiguous


@triton.jit
def silu(x, alpha):
    return x * tl.sigmoid(x * alpha)


@triton.jit
def _swiglu_forward_kernel(a_ptr, b_ptr, c_ptr, alpha: tl.constexpr, limit: tl.constexpr, stride, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # a = gate, b = up
    program_id = tl.program_id(0).to(tl.int64)

    # locate start index
    a_ptr += program_id * stride
    b_ptr += program_id * stride
    c_ptr += program_id * stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # sigmoid requires type float32
    a_row = tl.load(a_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
    a_row = tl.minimum(a_row, limit)
    b_row = tl.load(b_ptr + col_offsets, mask=mask, other=0)
    b_row = tl.clamp(b_row, min=-limit, max=limit)
    c_row = silu(a_row, alpha).cast(b_row.dtype) * (b_row + 1)
    tl.store(c_ptr + col_offsets, c_row, mask=mask)


@triton.jit
def _swiglu_backward_kernel(dc_ptr, a_ptr, b_ptr, ALPHA: tl.constexpr, LIMIT: tl.constexpr,
                            stride, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    program_id = tl.program_id(0).to(tl.int64)

    # locate start index
    dc_ptr += program_id * stride
    a_ptr += program_id * stride
    b_ptr += program_id * stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    dc_row = tl.load(dc_ptr + col_offsets, mask=mask, other=0)
    a_row = tl.load(a_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
    mask_a = (a_row <= LIMIT)
    a_clamped = tl.minimum(a_row, LIMIT)
    
    b_row = tl.load(b_ptr + col_offsets, mask=mask, other=0)
    mask_b = (b_row >= -LIMIT) & (b_row <= LIMIT)
    b_clamped = tl.clamp(b_row, min=-LIMIT, max=LIMIT)
    
    sigm =  tl.sigmoid(a_clamped * ALPHA)
    silu = a_clamped * sigm

    # recomputation to save memory
    # TODO: Update backward pass for gpt-oss style implementation (formula will be different!)
    db_row = dc_row * silu * mask_b
    da_row = dc_row *  (b_clamped + 1) * (sigm + ALPHA * silu * (1 - sigm)) * mask_a

    tl.store(a_ptr + col_offsets, da_row, mask=mask)
    tl.store(b_ptr + col_offsets, db_row, mask=mask)
    # HINT: We're already recomputing values here.
    # Could a third store here help avoid saving something else?
    
    tl.store(dc_ptr + col_offsets, ((b_clamped + 1) * silu), mask=mask)


def swiglu_forward(a, b, alpha, limit):
    ori_shape = a.shape

    n_cols = ori_shape[-1]
    a = a.view(-1, n_cols)
    b = b.view(-1, n_cols)
    c = torch.empty_like(a)
    n_rows = a.shape[0]

    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    _swiglu_forward_kernel[(n_rows,)](
        a,
        b,
        c,
        alpha,
        limit,
        c.stride(-2),
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return c.view(*ori_shape)


def swiglu_backward(a, b, dc, alpha, limit):
    ori_shape = dc.shape
    n_cols = ori_shape[-1]
    dc = dc.view(-1, n_cols)
    n_rows = dc.shape[0]

    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    _swiglu_backward_kernel[(n_rows,)](
        dc,
        a,
        b,
        alpha,
        limit,
        dc.stride(-2),
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return a.view(*ori_shape), b.view(*ori_shape), dc.view(*ori_shape)


class MemoryEfficientSwiGLUMLP(torch.autograd.Function):
    """
    Memory-optimized SwiGLU MLP with selective recomputation.
    """
    
    @staticmethod
    @ensure_contiguous
    def forward(ctx, x, w_gate, w_up, w_down, alpha, limit):
        gate = x @ w_gate.T
        up = x @ w_up.T

        # TODO: Replace with fused swiglu_forward kernel
        activation_out = swiglu_forward(gate, up, alpha, limit)

        out = activation_out @ w_down.T

        # TODO: Save tensors for backward
        ctx.save_for_backward(x, gate, up, w_gate, w_up, w_down)  # TODO: fill this
        ctx.alpha = alpha
        ctx.limit = limit

        return out
    
    @staticmethod
    @ensure_contiguous
    def backward(ctx, grad_output):
        # TODO: Implement backward pass
        # ... unpack, recompute whats needed, d_activation ...
        x, gate, up, w_gate, w_up, w_down = ctx.saved_tensors
        alpha = ctx.alpha
        limit = ctx.limit
        
        x_shape = x.shape
        hidden_dim = x_shape[-1]
        x_flat = x.reshape(-1, hidden_dim)
        grad_output_flat = grad_output.reshape(-1, grad_output.shape[-1])
        intermediate_dim = gate.shape[-1]

        gate_flat = gate.reshape(-1, intermediate_dim)
        up_flat = up.reshape(-1, intermediate_dim)
        
        dc = grad_output_flat @ w_down

        grad_gate_flat, grad_up_flat, c_out_flat = swiglu_backward(
            gate_flat, up_flat, dc, alpha, limit
        )

        # Use c_out_flat immediately, then release it.
        grad_w_down = grad_output_flat.T @ c_out_flat
        del c_out_flat
        del dc

        # Avoid tmp1 + tmp2 allocation.
        grad_x_flat = grad_gate_flat @ w_gate
        grad_x_flat.addmm_(grad_up_flat, w_up)
        grad_x = grad_x_flat.reshape(x_shape)

        grad_w_gate = grad_gate_flat.T @ x_flat
        grad_w_up = grad_up_flat.T @ x_flat

        return grad_x, grad_w_gate, grad_w_up, grad_w_down, None, None




class SwiGLUFeedForward(nn.Module):
    """
    gpt-oss style SwiGLU.
    
    output = W_down @ ((up + 1) * gate * sigmoid(gate * alpha))
    """
    
    def __init__(self, hidden_dim: int, intermediate_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.alpha = 1.702
        self.limit = 7.0

        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return MemoryEfficientSwiGLUMLP.apply(
            x, self.gate_proj.weight, self.up_proj.weight, self.down_proj.weight, self.alpha, self.limit
        )
