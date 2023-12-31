import torch
import torch.nn as nn
from torch.nn import functional as F

from encoder_decoder.classes.SingleHeadAttention import SingleHeadAttention

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(
        self, 
        n_heads: int, 
        head_size: int,
        embedding_dim: int,
        block_size: int,
        dropout: float,
        masked: bool
    ):
        super().__init__()

        self.n_heads = n_heads
        self.head_size = head_size
        self.embedding_dim = embedding_dim
        self.block_size = block_size
        self.dropout = dropout
        self.masked = masked

        # combine all single heads
        self.heads = nn.ModuleList([SingleHeadAttention(self.embedding_dim, self.head_size, self.block_size, self.dropout, self.masked) for _ in range(n_heads)])

        # creates a linear layer after the multi-head attention; combines results of all single head attentions essentially and creates a meaningful representation for it
        self.proj = nn.Linear(self.head_size * self.n_heads, self.embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor, 
        k: torch.Tensor, 
        q: torch.Tensor, 
        v: torch.Tensor
    ):
        out = torch.cat([h(x, k, q, v) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out