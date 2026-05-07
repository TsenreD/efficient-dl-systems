"""
Zero-Centered RMSNorm
"""

import torch
import torch.nn as nn


@torch.compile
def rmsnorm_forward(x, weight, eps):
    """Zero-Centered RMSNorm forward.

    y = x / rms(x) * (1 + weight)
    """
    input_dtype = x.dtype

    x_float = x.float()
    weight_float = weight.float()

    mean_squared = (x_float * x_float).mean(dim=-1, keepdim=True)
    rsqrt = torch.rsqrt(mean_squared + eps)

    normalized = x_float * rsqrt
    scale = 1.0 + weight_float

    output = normalized * scale

    # Return rsqrt so backward does not need to recompute it.
    return output.to(input_dtype), rsqrt


@torch.compile
def rmsnorm_backward(grad_output, x, weight, rsqrt):
    """Zero-Centered RMSNorm backward."""
    input_dtype = x.dtype

    x_float = x.float()
    grad_output_float = grad_output.float()
    weight_float = weight.float()

    scale = 1.0 + weight_float

    # grad wrt normalized x
    grad_norm = grad_output_float * scale

    # Last dimension is the hidden dimension being normalized.
    hidden_dim = x.shape[-1]

    # dot = sum_j grad_norm_j * x_j
    dot = (grad_norm * x_float).sum(dim=-1, keepdim=True)

    # dx = r * grad_norm - x * r^3 * mean(grad_norm * x)
    grad_x = (
        rsqrt * grad_norm
        - x_float * (rsqrt ** 3) * dot / hidden_dim
    )

    # weight gradient:
    # output = normalized * (1 + weight)
    # doutput/dweight = normalized
    normalized = x_float * rsqrt
    grad_weight_unreduced = grad_output_float * normalized

    if grad_weight_unreduced.ndim > 1:
        reduce_dims = tuple(range(grad_weight_unreduced.ndim - 1))
        grad_weight = grad_weight_unreduced.sum(dim=reduce_dims)
    else:
        grad_weight = grad_weight_unreduced

    return grad_x.to(input_dtype), grad_weight.to(weight.dtype)


class RMSNormFunction(torch.autograd.Function):
    """
    Memory-efficient Zero-Centered RMSNorm autograd function.
    """

    @staticmethod
    def forward(ctx, x, weight, eps):
        output, rsqrt = rmsnorm_forward(x, weight, eps)

        # Save only what backward needs:
        # x      -> needed for grad_x and grad_weight
        # weight -> needed for scale = 1 + weight
        # rsqrt  -> avoids recomputing rms inverse
        ctx.save_for_backward(x, weight, rsqrt)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, rsqrt = ctx.saved_tensors

        grad_x, grad_weight = rmsnorm_backward(
            grad_output,
            x,
            weight,
            rsqrt,
        )

        if not ctx.needs_input_grad[0]:
            grad_x = None

        if not ctx.needs_input_grad[1]:
            grad_weight = None

        # One returned gradient per forward input:
        # x, weight, eps
        return grad_x, grad_weight, None


class RMSNorm(nn.Module):
    """
    Zero-Centered RMSNorm:
        y = x / rms(x) * (1 + weight)

    weight is initialized to zero, so initial scale is 1.
    """

    def __init__(self, hidden_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return RMSNormFunction.apply(x, self.weight, self.eps)