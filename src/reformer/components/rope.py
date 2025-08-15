import torch
import torch.nn as nn
from typing import Tuple

class RotaryEmbedding(nn.Module):
    """Helper module for applying RoPE in attention layers"""
    def __init__(self, dim: int, seq_len: int, base: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0, f"RoPE dimension must be even, got {dim}"
        self.dim = dim
        self.seq_len = seq_len
        self.base = base
        
        evens_i = torch.arange(0, dim, 2).float()
        inv_freq = 1.0 / (base ** (evens_i / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # cache for performance
        self._cached_seq_len = 0
        self._cached_cos = None
        self._cached_sin = None
    
    def _compute_cached_rotations(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """Compute and cache rotation values"""
        if seq_len <= self._cached_seq_len and self._cached_cos is not None:
            return self._cached_cos[:seq_len], self._cached_sin[:seq_len] # type: ignore
        
        t = torch.arange(seq_len, device=device, dtype=dtype)
        freqs = torch.outer(t, self.inv_freq.to(dtype))
        
        cos_vals = torch.cos(freqs)
        sin_vals = torch.sin(freqs)
        
        # cache the computed values
        self._cached_seq_len = seq_len
        self._cached_cos = cos_vals
        self._cached_sin = sin_vals
        
        return cos_vals, sin_vals
    
    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims"""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    
    def apply_rotary_pos_emb(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary positional embedding to query and key tensors"""
        q_len = q.size(dim = -2)
        k_len = k.size(dim=-2) 

        cos_q, sin_q = self._compute_cached_rotations(q_len, q.device, q.dtype)
        # reshpae for broadcasting: (seq_len, head_dim/2) -> (1, 1, seq_len, head_dim/2)
        cos_q = cos_q.unsqueeze(0).unsqueeze(1).repeat_interleave(2, dim=-1)
        sin_q = sin_q.unsqueeze(0).unsqueeze(1).repeat_interleave(2, dim=-1)

        # apply rotation to query 
        q_embed = q * cos_q + self.rotate_half(q) * sin_q

        cos_k, sin_k = self._compute_cached_rotations(k_len, k.device, k.dtype)
        cos_k = cos_k.unsqueeze(0).unsqueeze(1).repeat_interleave(2, dim=-1)
        sin_k = sin_k.unsqueeze(0).unsqueeze(1).repeat_interleave(2, dim=-1)

        # apply rotation to key
        k_embed = k * cos_k + self.rotate_half(k) * sin_k
        
        return q_embed, k_embed