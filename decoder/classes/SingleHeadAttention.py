import torch
import torch.nn as nn
from torch.nn import functional as F

class SingleHeadAttention(nn.Module):
    """ Implementing a single head of masked self attention """
    def __init__(
        self,
        embedding_dim: int,
        head_size: int,
        block_size: int,
        dropout: float
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.head_size = head_size
        self.block_size = block_size

        self.key = nn.Linear(self.embedding_dim, self.head_size, bias=False)    # key represents what the embedding has to offer
        self.query = nn.Linear(self.embedding_dim, self.head_size, bias=False)  # query represents what the current embedding is looking for
        self.value = nn.Linear(self.embedding_dim, self.head_size, bias=False)  # value is the embedding that will be given 

        self.register_buffer('tril', torch.tril(torch.ones(self.block_size, self.block_size)))  # creating the mask which is just a matrix essentially 

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor  # (B, T, C); the input tensor
    ):
        
        B, T, C = x.shape
                
        k: torch.Tensor = self.key(x)    # (B, T, head_size) k represents the key (what the token has to offer)
        q: torch.Tensor = self.query(x)  # (B, T, head_size) q represents the query (what the token is looking for)
        v: torch.Tensor = self.query(x)  # (B, T, head_size) v represents the value (what the token will actually give)

        # need the attentions matrix (dot products between q and k)
        weights = q @ k.transpose(-2, -1) * self.head_size**-0.5  # (B, T, head_size) @ (B, T, head_size) --> (B, T, T); divide by sqrt of head_size in order to make numbers closer to 0 (less extreme improves the softmax outcome)

        # now have to make it so that the matrix is in triangular format so that each token only gets to see the tokens that precede it
        # weights = torch.masked_fill(self.tril[:self.block_size, :self.block_size] == 0, float('-inf'))
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T); this essentially creates the masked matrix to use
        weights = F.softmax(weights, dim=-1)  # softmax basically normalizes all the values in the matrix so that all the rows add up to 1
        weights = self.dropout(weights)  # dropout layer essentially makes it so that only some of the neurons are activated (only some of the parameters of the overall network are used) to help prevent overfitting

        out = weights @ v  # this is the final output which is essentially a linear combination (weighted sum) of all the tokens in the input sequence

        return out