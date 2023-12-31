import torch
import torch.nn as nn
from torch.nn import functional as F

from encoder_decoder.classes.Blocks import DecoderBlock, EncoderBlock

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

        # # the decoder blocks
        # self.decoder_blocks = nn.Sequential(*[DecoderBlock(self.embedding_dim, self.n_heads, self.block_size, self.dropout) for _ in range(n_layers)])

        # # the encoder blocks
        # self.encoder_blocks = nn.Sequential(*[EncoderBlock(self.embedding_dim, self.n_heads, self.block_size, self.dropout) for _ in range(n_layers)])

        self.encoder_blocks = nn.ModuleList([EncoderBlock(self.embedding_dim, self.n_heads, self.block_size, self.dropout) for _ in range(n_layers)])
        self.decoder_blocks = nn.ModuleList([DecoderBlock(self.embedding_dim, self.n_heads, self.block_size, self.dropout) for _ in range(n_layers)])


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

        input_tok_embeddings: torch.Tensor = self.token_embedding_table(x)  # (B, T, C) where C is embedding_dim; each value (character) in the tensor has an embedding vector
        input_pos_emb: torch.Tensor = self.position_embedding_table(torch.arange(T, device=self.device))  # tensor 1 -> block_size to keep track of position of each character in sequence

        output_tok_embeddings: torch.Tensor = self.token_embedding_table(y)
        output_pos_emb: torch.Tensor = self.position_embedding_table(torch.arange(T, device=self.device))

        encoder_out: torch.Tensor = input_tok_embeddings + input_pos_emb  # so each tensor in the batch also has its position kept in mind
        decoder_out: torch.Tensor = output_tok_embeddings + output_pos_emb

        for enc in self.encoder_blocks:
            encoder_out = enc(x=encoder_out, k=encoder_out, q=encoder_out, v=encoder_out)

        for dec in self.decoder_blocks:
            decoder_out = dec(x=decoder_out, k=encoder_out, q=decoder_out, v=encoder_out)

        # encoder_out = self.encoder_blocks(x=x, k=x, q=x, v=x)  # run inputs through encoder blocks
        # decoder_out = self.decoder_blocks(x=y, k=encoder_out, q=y, v=encoder_out)  # run inputs through decoder blocks

        # x = self.blocks(x)  # run inputs through attention blocks
        norm_decoder_out = self.ln_f(decoder_out)  # run inputs through layer normalization layer
        logits: torch.Tensor = self.lm_head(norm_decoder_out)  # gets final logits by running through final linear layer

        # have to reshape logits/y so it can go into cross_entropy function
        logits = logits.view(B*T, logits.shape[-1])  # make 2D; get rid of the batch dimension essentially and make all samples in 1 large batch
        y = y.view(B*T)  # basically just flatten the matrix
        loss = F.cross_entropy(logits, y)

        # else:
        #    loss = None  # if we are just doing inference, we don't need to keep track of loss (also no way to calculate it without y)

        return logits, loss
    
    def generate(
        self, 
        x, 
        max_new_tokens
    ):
        # x is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop x to the last block_size tokens
            x_cond = x[:, -self.block_size:]
            # get the predictions
            logits, loss = self(x_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            x_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            x = torch.cat((x, x_next), dim=1) # (B, T+1)
        return x