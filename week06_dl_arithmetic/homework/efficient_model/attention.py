"""
Attention with RoPE
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import TransformerConfig
from flash_attn import flash_attn_func
from flash_attn.layers.rotary import apply_rotary_emb


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE).
    """

    def __init__(self, head_dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len: int):
        """Build sin/cos cache up to seq_len."""
        positions = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(positions, self.inv_freq)  # (S, D/2)
        
        self.register_buffer("cos", freqs.cos(), persistent=False)  # (S, D/2)
        self.register_buffer("sin", freqs.sin(), persistent=False)  # (S, D/2)

        self.max_seq_len = seq_len
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary positional embedding to q and k.
        
        Args:
            q: (B, num_heads, S, head_dim)
            k: (B, num_heads, S, head_dim)
            seq_len: sequence length (must be <= max_seq_len)
            
        Returns:
            q_rotated, k_rotated with same shapes
        """
        assert seq_len <= self.max_seq_len, \
            f"seq_len ({seq_len}) exceeds max_seq_len ({self.max_seq_len})"
        
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)

        cos = self.cos[:seq_len].to(device=q.device, dtype=q.dtype)
        sin = self.sin[:seq_len].to(device=q.device, dtype=q.dtype)

        q_r = apply_rotary_emb(q, cos, sin)
        k_r = apply_rotary_emb(k, cos, sin)
        
        return q_r, k_r


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention with vanilla implementation and RoPE.
    """
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_dim // config.num_heads

        # TODO: Replace with fused QKV projection
        self.qkv_proj = nn.Linear(config.hidden_dim, 3 * config.hidden_dim, bias=False)
        self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)

        self.rope = RotaryPositionalEmbedding(
            head_dim=self.head_dim,
            max_seq_len=config.max_seq_len,
            theta=config.rope_theta,
        )

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self, 
        x: torch.Tensor, 
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, S, H = x.shape

        qkv = self.qkv_proj(x)
        q = qkv[:, :, :H]
        k = qkv[:, :, H:2*H]
        v = qkv[:, :, 2*H:3*H]

        q = q.view(B, S, self.num_heads, self.head_dim)#.transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim)#.transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim)# .transpose(1, 2)

        q, k = self.rope(q, k, S)
        # q = q.transpose(1, 2)  # back to (B, H, S, D)
        # k = k.transpose(1, 2)

        out = flash_attn_func(
            q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16), causal=True
        ).contiguous().view(B, S, H).to(x.dtype)
        # scale = 1.0 / math.sqrt(self.head_dim)
        # attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

        # if attention_mask is None:
        #     causal_mask = torch.triu(
        #         torch.ones(S, S, dtype=torch.bool, device=x.device), 
        #         diagonal=1
        #     )
        #     attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
        # else:
        #     attn_weights = attn_weights + attention_mask

        # attn_weights = F.softmax(attn_weights, dim=-1)
        # attn_weights = self.dropout(attn_weights)

        # out = torch.matmul(attn_weights, v)

        # out = out.transpose(1, 2).contiguous().view(B, S, H)
        out = self.out_proj(out)

        return out
