import os
import torch
from torch.profiler import profile, ProfilerActivity
from torch._inductor import metrics

# Import your torch implementation here
from ademamix import AdEMAMix


DEVICE = "cuda"
DTYPE = torch.float32
N = 1_048_576


def make_optimizer_only_case():
    p = torch.nn.Parameter(torch.randn(N, device=DEVICE, dtype=DTYPE))

    opt = AdEMAMix(
        [p],
        lr=1e-3,
        betas=(0.9, 0.999, 0.9999),
        alpha=2.0,
        eps=1e-8,
        weight_decay=0.01,
    )

    grad_buf = torch.randn_like(p)

    # Important: initialize optimizer state eagerly before compiling.
    p.grad = grad_buf
    opt.step()
    opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    def step_fn():
        p.grad = grad_buf
        opt.step()
        opt.zero_grad(set_to_none=True)

    return step_fn


def profile_one_call(fn):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        with_stack=False,
    ) as prof:
        fn()
        torch.cuda.synchronize()

    print("\nCUDA kernel-like events:")
    total = 0

    for evt in prof.key_averages():
        key = evt.key.lower()
        cuda_time = getattr(evt, "self_cuda_time_total", 0.0)

        if cuda_time > 0 and (
            "triton" in key
            or "inductor" in key
            or "cuda" in key
            or "void" in key
            or "foreach" in key
        ):
            print(f"{evt.key:100s} count={evt.count}")
            total += evt.count

    print(f"\nTotal counted CUDA launches: {total}")
    return total


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    torch.manual_seed(0)

    torch._dynamo.reset()
    metrics.reset()

    step_fn = make_optimizer_only_case()

    compiled_step = torch.compile(
        step_fn,
        fullgraph=False,
        dynamic=False,
    )

    # First call triggers compilation.
    compiled_step()
    torch.cuda.synchronize()

    print("\nInductor generated kernel count:")
    print(metrics.generated_kernel_count)

    launched = profile_one_call(compiled_step)

    print("\nSummary:")
    print("Generated Inductor kernels:", metrics.generated_kernel_count)
    print("Launched CUDA kernels:", launched)


if __name__ == "__main__":
    main()
    
# rm -rf torch_compile_debug
# rm -rf /tmp/torchinductor_*

# TORCH_COMPILE_DEBUG=1 \
# TORCH_LOGS="+output_code,graph_breaks,recompiles" \
# TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1 \
# python tests/test_optimizer.py 