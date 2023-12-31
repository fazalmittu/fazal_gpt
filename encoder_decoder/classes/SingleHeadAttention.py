import torch
import torch.nn as nn
from torch.nn import functional as F

class SingleHeadAttention(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
        head_size: int,
        block_size: int,
        dropout: float,
        masked: bool
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.head_size = head_size
        self.block_size = block_size
        self.masked = masked

        self.key = nn.Linear(self.embedding_dim, self.head_size, bias=False)    # key represents what the embedding has to offer
        self.query = nn.Linear(self.embedding_dim, self.head_size, bias=False)  # query represents what the current embedding is looking for
        self.value = nn.Linear(self.embedding_dim, self.head_size, bias=False)  # value is the embedding that will be given 

        if self.masked:
            self.register_buffer('tril', torch.tril(torch.ones(self.block_size, self.block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,  # (B, T, C)
        k: torch.Tensor,
        q: torch.Tensor,
        v: torch.Tensor
    ):
        
        B, T, C = x.shape
                
        k: torch.Tensor = self.key(k)    # (B, T, head_size)
        q: torch.Tensor = self.query(q)  # (B, T, head_size)
        v: torch.Tensor = self.query(v)  # (B, T, head_size)

        # need the attentions matrix (dot products between q and k)
        weights = q @ k.transpose(-2, -1) * self.head_size**-0.5  # (B, T, head_size) @ (B, T, head_size) --> (B, T, T); divide by sqrt of head_size in order to make numbers closer to 0 (less extreme improves the softmax outcome)

        # now have to make it so that the matrix is in triangular format so that each token only gets to see the tokens that precede it
        # weights = torch.masked_fill(self.tril[:self.block_size, :self.block_size] == 0, float('-inf'))
        if self.masked:
            weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)

        out = weights @ v

        return out