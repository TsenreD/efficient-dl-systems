import math
import os
from typing import Any

os.environ["TORCH_LOGS"] = "+output_code"
os.environ["TORCH_LOGS_OUT"] = "compiled_kernels.log"

import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.optimizer import _get_value


def linear_warmup_scheduler_scalar(
    step: float,
    *,
    alpha_end: float,
    alpha_start: float = 0.0,
    warmup: int | None = None,
) -> float:
    if warmup is None:
        return float(alpha_end)

    a = min(float(step) / float(warmup), 1.0)
    return (1.0 - a) * float(alpha_start) + a * float(alpha_end)


def linear_hl_warmup_scheduler_scalar(
    step: float,
    *,
    beta_end: float,
    beta_start: float = 0.0,
    warmup: int | None = None,
    eps: float = 1e-8,
) -> float:
    if warmup is None:
        return float(beta_end)

    a = min(float(step) / float(warmup), 1.0)

    log_half = math.log(0.5)

    def f(beta: float) -> float:
        return log_half / math.log(float(beta) + eps) - 1.0

    def f_inv(t: float) -> float:
        return 0.5 ** (1.0 / (t + 1.0))

    t_start = f(beta_start)
    t_end = f(beta_end)
    t = (1.0 - a) * t_start + a * t_end

    return f_inv(t)


@torch.compile(fullgraph=True)
def ademamix_foreach_fn(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    exp_avgs_slow: list[Tensor],
    exp_avg_sqs: list[Tensor],
    alphas: Tensor,
    beta3s: Tensor,
    one_minus_beta3s: Tensor,
    bias_1_cors: Tensor,
    sqrt_bias_2_cors: Tensor,
    *,
    lr: Tensor,
    beta1: float,
    beta2: float,
    eps: float,
    lmbda: float,
):
    device = params[0].device

    # These are CPU tensors outside the compiled region.
    # Moving them to CUDA inside the graph avoids Dynamo guards on changing Python floats.
    alphas_t = tuple(alphas.to(device).unbind())
    beta3s_t = tuple(beta3s.to(device).unbind())
    one_minus_beta3s_t = tuple(one_minus_beta3s.to(device).unbind())
    bias_1_cors_t = tuple(bias_1_cors.to(device).unbind())
    sqrt_bias_2_cors_t = tuple(sqrt_bias_2_cors.to(device).unbind())
    lr_t = lr.to(device)

    if beta1 != 0.0:
        torch._foreach_lerp_(exp_avgs, grads, 1.0 - beta1)
        fast = exp_avgs
    else:
        fast = grads

    # exp_avg_slow = beta3_i * exp_avg_slow + (1 - beta3_i) * grad
    torch._foreach_mul_(exp_avgs_slow, beta3s_t)
    scaled_grads = torch._foreach_mul(grads, one_minus_beta3s_t)
    torch._foreach_add_(exp_avgs_slow, scaled_grads)

    # exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad * grad
    torch._foreach_mul_(exp_avg_sqs, beta2)
    torch._foreach_addcmul_(exp_avg_sqs, grads, grads, value=1.0 - beta2)

    # denom_i = sqrt(exp_avg_sq) / sqrt(bias_correction2_i) + eps
    denoms = torch._foreach_sqrt(exp_avg_sqs)
    torch._foreach_div_(denoms, sqrt_bias_2_cors_t)
    torch._foreach_add_(denoms, eps)

    # update_i = fast_i / bias_correction1_i
    updates = torch._foreach_div(fast, bias_1_cors_t)

    # update_i += alpha_i * exp_avg_slow_i
    scaled_slow = torch._foreach_mul(exp_avgs_slow, alphas_t)
    torch._foreach_add_(updates, scaled_slow)

    # update_i /= denom_i
    torch._foreach_div_(updates, denoms)

    # update_i += weight_decay * param_i
    if lmbda != 0.0:
        torch._foreach_add_(updates, params, alpha=lmbda)

    # param_i -= lr * update_i
    for param, update in zip(params, updates):
        param.sub_(update * lr_t)


class AdEMAMix(Optimizer):
    r"""Implements the AdEMAMix algorithm."""

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float, float] = (0.9, 0.999, 0.9999),
        alpha: float = 2.0,
        beta3_warmup: int | None = None,
        alpha_warmup: int | None = None,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= betas[2] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 2: {betas[2]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= alpha:
            raise ValueError(f"Invalid alpha value: {alpha}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            alpha=alpha,
            beta3_warmup=beta3_warmup,
            alpha_warmup=alpha_warmup,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state: dict[str, Any]):
        super().__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            lmbda = group["weight_decay"]
            eps = group["eps"]
            beta1, beta2, beta3_final = group["betas"]
            beta3_warmup = group["beta3_warmup"]
            alpha_final = group["alpha"]
            alpha_warmup = group["alpha_warmup"]

            lr_t = group.get("lr_tensor")
            if lr_t is None:
                lr_t = torch.tensor(float(lr), dtype=torch.float32)
                group["lr_tensor"] = lr_t
            else:
                lr_t.fill_(float(lr))

            params: list[Tensor] = []
            grads: list[Tensor] = []
            exp_avgs: list[Tensor] = []
            exp_avgs_slow: list[Tensor] = []
            exp_avg_sqs: list[Tensor] = []

            alphas: list[float] = []
            beta3s: list[float] = []
            one_minus_beta3s: list[float] = []
            bias_1_cors: list[float] = []
            sqrt_bias_2_cors: list[float] = []

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdEMAMix does not support sparse gradients.")

                state = self.state[p]

                if len(state) == 0:
                    # CPU scalar tensor, like Adam's non-capturable foreach path.
                    # This avoids a CUDA kernel just for step += 1.
                    state["step"] = torch.tensor(0.0)

                    if beta1 != 0.0:
                        state["exp_avg_fast"] = torch.zeros_like(
                            p,
                            memory_format=torch.preserve_format,
                        )
                    else:
                        state["exp_avg_fast"] = None

                    state["exp_avg_slow"] = torch.zeros_like(
                        p,
                        memory_format=torch.preserve_format,
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        p,
                        memory_format=torch.preserve_format,
                    )

                # Convert older checkpoints / previous variants safely.
                if torch.is_tensor(state["step"]):
                    if not state["step"].is_cpu:
                        state["step"] = state["step"].detach().cpu()
                    if state["step"].dtype != torch.float32:
                        state["step"] = state["step"].to(dtype=torch.float32)
                else:
                    state["step"] = torch.tensor(float(state["step"]), dtype=torch.float32)

                state["step"] += 1
                step = float(_get_value(state["step"]))

                alpha_t = linear_warmup_scheduler_scalar(
                    step,
                    alpha_end=alpha_final,
                    alpha_start=0.0,
                    warmup=alpha_warmup,
                )

                beta3_t = linear_hl_warmup_scheduler_scalar(
                    step,
                    beta_end=beta3_final,
                    beta_start=beta1,
                    warmup=beta3_warmup,
                )

                if beta1 != 0.0:
                    bias_1_cor = 1.0 - beta1**step
                else:
                    bias_1_cor = 1.0

                sqrt_bias_2_cor = math.sqrt(1.0 - beta2**step)

                params.append(p)
                grads.append(grad)

                if beta1 != 0.0:
                    if state["exp_avg_fast"] is None:
                        state["exp_avg_fast"] = torch.zeros_like(
                            p,
                            memory_format=torch.preserve_format,
                        )
                    exp_avgs.append(state["exp_avg_fast"])

                exp_avgs_slow.append(state["exp_avg_slow"])
                exp_avg_sqs.append(state["exp_avg_sq"])

                alphas.append(alpha_t)
                beta3s.append(beta3_t)
                one_minus_beta3s.append(1.0 - beta3_t)
                bias_1_cors.append(bias_1_cor)
                sqrt_bias_2_cors.append(sqrt_bias_2_cor)

            if len(params) == 0:
                continue

            # Do NOT pass Python float lists directly into the compiled function.
            # Dynamo will guard on changing float values and recompile each step.
            #
            # Keep these CPU tensors here. The compiled function moves them to CUDA.
            alphas_t = torch.tensor(alphas, dtype=torch.float32)
            beta3s_t = torch.tensor(beta3s, dtype=torch.float32)
            one_minus_beta3s_t = torch.tensor(one_minus_beta3s, dtype=torch.float32)
            bias_1_cors_t = torch.tensor(bias_1_cors, dtype=torch.float32)
            sqrt_bias_2_cors_t = torch.tensor(sqrt_bias_2_cors, dtype=torch.float32)

            ademamix_foreach_fn(
                params=params,
                grads=grads,
                exp_avgs=exp_avgs,
                exp_avgs_slow=exp_avgs_slow,
                exp_avg_sqs=exp_avg_sqs,
                alphas=alphas_t,
                beta3s=beta3s_t,
                one_minus_beta3s=one_minus_beta3s_t,
                bias_1_cors=bias_1_cors_t,
                sqrt_bias_2_cors=sqrt_bias_2_cors_t,
                lr=lr_t,
                beta1=beta1,
                beta2=beta2,
                eps=eps,
                lmbda=lmbda,
            )

        return loss
