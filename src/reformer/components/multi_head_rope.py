import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
from transformer.components.base.attention import BaseAttention
from .rope import RotaryEmbedding


class MultiHeadAttentionRoPE(BaseAttention):
    """Multi-Head Attention with RoPE"""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float,
        src_max_len: int,
        tgt_max_len: int,
        base: float,
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.head_dim = d_model // num_heads
        assert (
            self.head_dim % 2 == 0
        ), f"head_dim must be even for RoPE, got {self.head_dim}"

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        max_seq_len = max(src_max_len, tgt_max_len)
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len, base)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, kv: torch.Tensor, mask: Union[torch.Tensor, None] = None
    ) -> torch.Tensor:
        batch_size, x_seq_len, _ = x.shape
        _, kv_seq_len, _ = kv.shape

        query = self.q_proj(x) # shape: (batch_size, src_seq_len, d_model)
        key = self.k_proj(kv) # shape: (batch_size, kv_seq_len, d_model)
        value = self.v_proj(kv) # shape: (batch_size, kv_seq_len, d_model)
        
        # reshape q, k, and v to multi-head (decompose the d_model -> (num_heads, head_dim))
        query = query.view(
            batch_size, x_seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key = key.view(batch_size, kv_seq_len, self.num_heads, self.head_dim).transpose(
            1, 2
        )
        value = value.view(
            batch_size, kv_seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        # apply RoPE 
        query, key = self.rope.apply_rotary_pos_emb(query, key)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = attn @ value 
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, x_seq_len, self.d_model)
        )

        output = self.out_proj(context)
        return output
