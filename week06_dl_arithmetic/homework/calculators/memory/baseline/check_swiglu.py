import os
import sys
from collections import defaultdict

import torch
from torch.autograd.graph import saved_tensors_hooks

#ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = "/home/jovyan/shares/SR008.fs2/eadyagin/sandbox/efficient-dl-systems/week06_dl_arithmetic/homework"
sys.path.insert(0, ROOT)

from model.swiglu import SwiGLUFeedForward


def inspect_saved_tensors(dtype):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    swiglu = SwiGLUFeedForward(hidden_dim=16, intermediate_dim=64).to(
        device=device,
        dtype=dtype,
    )
    x = torch.randn(3, 12, 16, device=device, dtype=dtype, requires_grad=True)

    saved = []

    def pack(tensor):
        saved.append((tuple(tensor.shape), tensor.dtype, tensor.numel() * tensor.element_size()))
        return tensor

    def unpack(tensor):
        return tensor

    with saved_tensors_hooks(pack, unpack):
        y = swiglu(x)
        y.sum().backward()

    return saved


def print_report(name, saved):
    print(f"\n{name}")
    print("-" * len(name))
    for shape, dtype, size in saved:
        print(f"{shape}, {dtype}, {size} bytes")

    grouped = defaultdict(int)
    for shape, dtype, size in saved:
        grouped[(shape, dtype)] += size

    print("\ngrouped:")
    for (shape, dtype), size in grouped.items():
        print(f"{shape}, {dtype}: {size} bytes")

    print("total bytes:", sum(size for _, _, size in saved))


for dtype in (torch.float32, torch.bfloat16):
    print_report(f"swiglu saved tensors, dtype={dtype}", inspect_saved_tensors(dtype))
