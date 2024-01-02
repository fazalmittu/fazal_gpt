import torch
import torch.nn as nn
from torch.nn import functional as F

from decoder.classes.Blocks import Block

class LanguageModel(nn.Module):

    def __init__(
        self, 
        vocab_size: int, 
        embedding_dim: int, 
        block_size: int,
        n_layers: int,
        n_heads: int,
        dropout: float,
        device
    ):

        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.block_size = block_size
        self.n_layer = n_layers
        self.n_heads = n_heads
        self.dropout = dropout
        self.device = device

        # create a lookup table of size (vocab_size, vocab_size) where each row corresponds to a character and contains the likelihood of the each character being next
        self.token_embedding_table = nn.Embedding(self.vocab_size, self.embedding_dim)

        # create a table of size (block_size, embedding_dim) to keep track of the position of a character in a sequence
        self.position_embedding_table = nn.Embedding(self.block_size, self.embedding_dim)  

        # the blocks that are used for attention
        self.blocks = nn.Sequential(*[Block(self.embedding_dim, self.n_heads, self.block_size, self.dropout) for _ in range(n_layers)])

        # layer normalization that occurs at the end (makes sure each row has a mean ~ 0 and a standard deviation ~ 1)
        self.ln_f = nn.LayerNorm(self.embedding_dim)  

        # final linear layer that comes after the attention blocks (idea is that after the attention, the model is able to use what it learned by looking at itself to tweak weights)
        self.lm_head = nn.Linear(self.embedding_dim, self.vocab_size)

        # initializes weights for the model
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)  # make the weights have a mean of 0 and a standard deviation of 0.02 
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)  # if there is an added bias, initialize it to be 0's
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, 
        x: torch.Tensor, 
        y: torch.Tensor = None # don't need targets if we are doing inference
    ):  
        
        B, T = x.shape  # where B is batch_size and T is block_size

        tok_embeddings: torch.Tensor = self.token_embedding_table(x)  # (B, T, C) where C is embedding_dim; each value (character) in the tensor has an embedding vector
        pos_emb: torch.Tensor = self.position_embedding_table(torch.arange(T, device=self.device))  # tensor 1 -> block_size to keep track of position of each character in sequence

        x: torch.Tensor = tok_embeddings + pos_emb  # so each tensor in the batch also has its position kept in mind

        x = self.blocks(x)  # run inputs through attention blocks
        x = self.ln_f(x)  # run inputs through layer normalization layer
        logits: torch.Tensor = self.lm_head(x)  # gets final logits by running through final linear layer

        if y is None:
           loss = None  # if we are just doing inference, we don't need to keep track of loss (also no way to calculate it without y)
        else:
            # have to reshape logits/y so it can go into cross_entropy function
            logits = logits.view(B*T, logits.shape[-1])  # make 2D; get rid of the batch dimension essentially and make all samples in 1 large batch
            y = y.view(B*T)  # basically just flatten the matrix
            loss = F.cross_entropy(logits, y)  # calculate the loss

        return logits, loss
    
    def generate(
        self, 
        x, 
        max_new_tokens
    ):
        # x is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            x_cond = x[:, -self.block_size:]   # crop x to the last block_size tokens
            logits, loss = self(x_cond)        # get the predictions
            
            logits = logits[:, -1, :]          # becomes (B, C); focus only on the last time step
            probs = F.softmax(logits, dim=-1)  # (B, C); apply softmax to get probabilities
            
            x_next = torch.multinomial(probs, num_samples=1) # (B, 1); sample from the distribution
            x = torch.cat((x, x_next), dim=1) # (B, T+1); append sampled index to the running sequence
            
        return x