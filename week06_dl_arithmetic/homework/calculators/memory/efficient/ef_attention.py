import os
import sys
from collections import defaultdict

import torch
from torch.autograd.graph import saved_tensors_hooks

#ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = "/home/jovyan/shares/SR008.fs2/eadyagin/sandbox/efficient-dl-systems/week06_dl_arithmetic/homework"
sys.path.insert(0, ROOT)

from config import TransformerConfig
from efficient_model.attention import MultiHeadAttention


def inspect_saved_tensors(dtype):
    config = TransformerConfig(
        vocab_size=128,
        hidden_dim=64,
        num_heads=4,
        num_layers=2,
        intermediate_dim=32,
        max_seq_len=32,
        dropout=0.0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    attention = MultiHeadAttention(config).to(device=device, dtype=dtype)
    x = torch.randn(3, 8, 64, device=device, dtype=dtype, requires_grad=True)

    saved = []

    def pack(tensor):
        saved.append((tuple(tensor.shape), tensor.dtype, tensor.numel() * tensor.element_size()))
        return tensor

    def unpack(tensor):
        return tensor

    with saved_tensors_hooks(pack, unpack):
        y = attention(x)
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
    print_report(f"attention saved tensors, dtype={dtype}", inspect_saved_tensors(dtype))
