import math

import os
os.environ["TORCH_LOGS"] = "+output_code"
os.environ["TORCH_LOGS_OUT"] = "compiled_kernels.log"
 
import torch
from torch import Tensor
from torch.distributed.tensor import DTensor
from torch.optim import Optimizer
 

def linear_warmup_scheduler_tensor(
    steps: Tensor,
    *,
    alpha_end: float,
    alpha_start: float = 0.0,
    warmup: int | None = None,
) -> Tensor:
    steps_f = steps.to(torch.float32)

    if warmup is None:
        return torch.full_like(steps_f, float(alpha_end))

    a = torch.clamp(steps_f / float(warmup), max=1.0)
    return (1.0 - a) * float(alpha_start) + a * float(alpha_end)


def linear_hl_warmup_scheduler_tensor(
    steps: Tensor,
    *,
    beta_end: float,
    beta_start: float = 0.0,
    warmup: int | None = None,
    eps: float = 1e-8,
) -> Tensor:
    steps_f = steps.to(torch.float32)

    if warmup is None:
        return torch.full_like(steps_f, float(beta_end))

    a = torch.clamp(steps_f / float(warmup), max=1.0)

    beta_start_t = torch.full_like(steps_f, float(beta_start))
    beta_end_t = torch.full_like(steps_f, float(beta_end))

    log_half = math.log(0.5)

    def f(beta: Tensor) -> Tensor:
        return log_half / torch.log(beta + eps) - 1.0

    def f_inv(t: Tensor) -> Tensor:
        return torch.pow(torch.full_like(t, 0.5), 1.0 / (t + 1.0))

    t_start = f(beta_start_t)
    t_end = f(beta_end_t)
    t = (1.0 - a) * t_start + a * t_end

    return f_inv(t)
 
 
@torch.compile(fullgraph=True) # you can comment out this line for subtask 1
def ademamix_foreach_fn(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    exp_avgs_slow: list[Tensor],
    exp_avg_sqs: list[Tensor],
    steps: Tensor,
    *,
    lr: float,
    beta1: float,
    beta2: float,
    beta3_final: float,
    beta3_warmup: int | None,
    alpha_final: float,
    alpha_warmup: int | None,
    eps: float,
    lmbda: float,
):
    # TODO: Replace per-tensor ops with torch._foreach_* equivalents:
    # torch._foreach_lerp_(exp_avgs, grads, 1 - beta1)
    # torch._foreach_mul_(exp_avg_sqs, beta2)
    # torch._foreach_addcmul_(exp_avg_sqs, grads, grads, 1 - beta2)
    # etc.
    # Decay the first and second moment running average coefficient
    if beta1 != 0.0:
        torch._foreach_lerp_(exp_avgs, grads, 1.0 - beta1)
    else:
        exp_avgs = grads
        
    steps_f = steps.to(torch.float32)

    alphas = linear_warmup_scheduler_tensor(
        steps_f,
        alpha_end=alpha_final,
        alpha_start=0.0,
        warmup=alpha_warmup,
    )

    beta3s = linear_hl_warmup_scheduler_tensor(
        steps_f,
        beta_end=beta3_final,
        beta_start=beta1,
        warmup=beta3_warmup,
    )

    bias_1_cors = 1.0 - torch.pow(
        torch.full_like(steps_f, float(beta1)),
        steps_f,
    )

    bias_2_cors = 1.0 - torch.pow(
        torch.full_like(steps_f, float(beta2)),
        steps_f,
    )
    
    # Convert Tensor[N] into tuple of scalar tensors for foreach scalar-list overloads.
    beta3s_t = tuple(beta3s.unbind())
    one_minus_beta3s_t = tuple((1.0 - beta3s).unbind())
    alphas_t = tuple(alphas.unbind())
    bias_1_cors_t = tuple(bias_1_cors.unbind())
    sqrt_bias_2_cors_t = tuple(torch.sqrt(bias_2_cors).unbind())
    
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

    # update_i = exp_avg_fast_i / bias_correction1_i
    updates = torch._foreach_div(exp_avgs, bias_1_cors_t)

    # update_i += alpha_i * exp_avg_slow_i
    scaled_slow = torch._foreach_mul(exp_avgs_slow, alphas_t)
    torch._foreach_add_(updates, scaled_slow)

    # update_i /= denom_i
    torch._foreach_div_(updates, denoms)

    # update_i += weight_decay * param_i
    if lmbda != 0.0:
        torch._foreach_add_(updates, params, alpha=lmbda)

    # param_i -= lr * update_i
    torch._foreach_add_(params, updates, alpha=-lr)
 
class AdEMAMix(Optimizer):
    r"""Implements the AdEMAMix algorithm.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999, 0.9999)) 
            corresponding to beta_1, beta_2, beta_3 in AdEMAMix
        alpha (float): AdEMAMix alpha coeficient mixing the slow and fast EMAs (default: 2)
        beta3_warmup (int, optional): number of warmup steps used to increase beta3 (default: None)
        alpha_warmup: (int, optional): number of warmup steps used to increase alpha (default: None)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay as in AdamW (default: 0)
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999, 0.9999), alpha=2.0, 
                 beta3_warmup=None, alpha_warmup=None,  eps=1e-8,
                 weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= betas[2] < 1.0:
            raise ValueError("Invalid beta parameter at index 2: {}".format(betas[2]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= alpha:
            raise ValueError("Invalid alpha value: {}".format(alpha))
        defaults = dict(lr=lr, betas=betas, eps=eps, alpha=alpha, beta3_warmup=beta3_warmup,
                        alpha_warmup=alpha_warmup, weight_decay=weight_decay)
        super(AdEMAMix, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdEMAMix, self).__setstate__(state)

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
 
            params: list[Tensor] = []
            grads: list[Tensor] = []
            exp_avgs: list[Tensor] = []
            exp_avgs_slow: list[Tensor] = []
            exp_avg_sqs: list[Tensor] = []
            
            steps: list[int] = []

            # TODO: prepare lists of tensors for 'foreach' step
            # as well as all other necessary inputs
            # HINT: (think about input types)
            
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('AdEMAMix does not support sparse gradients.')
                
                params.append(p)
                grads.append(grad)
                
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    if beta1 != 0.0: # save memory in case beta1 is 0.0
                        state['exp_avg_fast'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    else: 
                        state['exp_avg_fast'] = None
                    state['exp_avg_slow'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avgs.append(state['exp_avg_fast'])
                exp_avgs_slow.append(state['exp_avg_slow'])
                exp_avg_sqs.append(state['exp_avg_sq'])
                
                state['step'] += 1
                steps.append(state["step"])
                # bias_1_cors.append(1 - beta1 ** state['step'])
                # bias_2_cors.append(1 - beta2 ** state['step'])
                
                # # Compute the effective alpha and beta3 in case warmup is used 
                # if alpha_warmup is not None:
                #     alphas.append(linear_warmup_scheduler(state["step"], alpha_end=alpha_final, alpha_start=0, warmup=alpha_warmup))
                # else:
                #     alphas.append(alpha_final)
                
                # if beta3_warmup is not None:
                #     beta3s.append(linear_hl_warmup_scheduler(state["step"], beta_end=beta3_final, beta_start=beta1, warmup=beta3_warmup))
                # else:
                #     beta3s.append(beta3_final)
            
            
            if len(params) > 0:
                steps_t = torch.tensor(
                    steps,
                    device=params[0].device,
                    dtype=torch.float32,
                )

                ademamix_foreach_fn(
                    params=params,
                    grads=grads,
                    exp_avgs=exp_avgs,
                    exp_avgs_slow=exp_avgs_slow,
                    exp_avg_sqs=exp_avg_sqs,
                    steps=steps_t,
                    lr=lr,
                    beta1=beta1,
                    beta2=beta2,
                    beta3_final=beta3_final,
                    beta3_warmup=beta3_warmup,
                    alpha_final=alpha_final,
                    alpha_warmup=alpha_warmup,
                    eps=eps,
                    lmbda=lmbda,
                )
        return loss