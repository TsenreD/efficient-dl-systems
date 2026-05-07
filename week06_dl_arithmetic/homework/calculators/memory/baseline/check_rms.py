import torch
from torch.autograd.graph import saved_tensors_hooks

import sys
ROOT = "/home/jovyan/shares/SR008.fs2/eadyagin/sandbox/efficient-dl-systems/week06_dl_arithmetic/homework"
sys.path.insert(0, ROOT)

from model.norm import RMSNorm

# norm = RMSNorm(hidden_dim=16)
# x = torch.randn(2, 8, 16, requires_grad=True)

norm = RMSNorm(16).to(dtype=torch.bfloat16)
x = torch.randn(2, 8, 16, dtype=torch.bfloat16, requires_grad=True)


saved = []

def pack(t):
    saved.append((tuple(t.shape), t.dtype, t.numel() * t.element_size()))
    return t

def unpack(t):
    return t

with saved_tensors_hooks(pack, unpack):
    y = norm(x)
    y.sum().backward()

print(saved)
print("total bytes:", sum(size for _, _, size in saved))
