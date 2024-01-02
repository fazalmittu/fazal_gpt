import torch
import torch.nn as nn
from torch.nn import functional as F

from decoder.classes.FeedForward import FeedFoward
from decoder.classes.MultiHeadAttention import MultiHeadAttention

class Block(nn.Module):
    """ Overall decoder block """

    def __init__(
        self, 
        embedding_dim: int, 
        n_heads: int, 
        block_size: int, 
        dropout: float
    ):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        # essentially just the decoder block; first multi_head_attention and then feed forward to add depth to the model
        super().__init__()
        head_size = embedding_dim // n_heads
        self.sa = MultiHeadAttention(n_heads, head_size, embedding_dim, block_size, dropout)
        self.ffwd = FeedFoward(embedding_dim, dropout)
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)

    def forward(self, x: torch.Tensor):

        # residual connections (not entirely sure why we need this but helps with optimization a lot)
        x = x + self.sa(self.ln1(x))    # layer normalization applied to input before it is passed into the multi_head_attention layer
        x = x + self.ffwd(self.ln2(x))  # layer normalization applied to input before it is passed into the feedforwardlayer

        return x