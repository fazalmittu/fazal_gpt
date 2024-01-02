import torch
import torch.nn as nn
from torch.nn import functional as F

from decoder.classes.SingleHeadAttention import SingleHeadAttention

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(
        self, 
        n_heads: int, 
        head_size: int,
        embedding_dim: int,
        block_size: int,
        dropout: int
    ):
        super().__init__()

        self.n_heads = n_heads
        self.head_size = head_size
        self.embedding_dim = embedding_dim
        self.block_size = block_size
        self.dropout = dropout

        # combine all single heads
        self.heads = nn.ModuleList([SingleHeadAttention(self.embedding_dim, self.head_size, self.block_size, self.dropout) for _ in range(n_heads)])  # combine all single heads together

        # creates a linear layer after the multi-head attention; combines results of all single head attentions essentially and creates a meaningful representation for it
        self.proj = nn.Linear(self.head_size * self.n_heads, self.embedding_dim)
        self.dropout = nn.Dropout(dropout)  # dropout layer essentially makes it so that only some of the neurons are activated (only some of the parameters of the overall network are used) to help prevent overfitting

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # take the outputs of all the single heads and combine them into one big vector
        out = self.dropout(self.proj(out))  # first runs the output of the multi-head attention through a linear layer and then uses a dropout layer to help prevent overfitting
        return out