import torch
import torch.nn as nn
from torch.nn import functional as F

from encoder_decoder.classes.FeedForward import FeedFoward
from encoder_decoder.classes.MultiHeadAttention import MultiHeadAttention

class DecoderBlock(nn.Module):
    """ Decoder Block: masked attention + attention + ffwd (with layer_norm) """

    def __init__(
        self, 
        embedding_dim: int, 
        n_heads: int, 
        block_size: int, 
        dropout: float
    ):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = embedding_dim // n_heads
        self.sa_masked = MultiHeadAttention(n_heads, head_size, embedding_dim, block_size, dropout, True)
        self.sa = MultiHeadAttention(n_heads, head_size, embedding_dim, block_size, dropout, False)
        self.ffwd = FeedFoward(embedding_dim, dropout)
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)
        self.ln3 = nn.LayerNorm(embedding_dim)

    def forward(self, x: torch.Tensor, k: torch.Tensor, q: torch.Tensor, v: torch.Tensor):

        # residual connections (not entirely sure why we need this but helps with optimization a lot)
        x = x + self.sa_masked(self.ln1(x), self.ln1(x), self.ln1(x), self.ln1(x))    # layer normalization applied to input before it is passed into the multi_head_attention layer
        x = x + self.sa(self.ln2(x), k, q, v)
        x = x + self.ffwd(self.ln3(x))  # layer normalization applied to input before it is passed into the feedforwardlayer

        return x
    
class EncoderBlock(nn.Module):
    """ Encoder block: attention + ffwd (with layer_norm) """

    def __init__(
        self, 
        embedding_dim: int, 
        n_heads: int, 
        block_size: int, 
        dropout: float
    ):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = embedding_dim // n_heads
        self.sa = MultiHeadAttention(n_heads, head_size, embedding_dim, block_size, dropout, False)
        self.ffwd = FeedFoward(embedding_dim, dropout)
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)

    def forward(
        self, 
        x: torch.Tensor,
        k: torch.Tensor, 
        q: torch.Tensor, 
        v: torch.Tensor
    ):

        # residual connections (not entirely sure why we need this but helps with optimization a lot)
        x = x + self.sa(self.ln1(x), k, q, v)    # layer normalization applied to input before it is passed into the multi_head_attention layer
        x = x + self.ffwd(self.ln2(x))  # layer normalization applied to input before it is passed into the feedforwardlayer

        return x