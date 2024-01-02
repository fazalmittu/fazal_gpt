import torch
import torch.nn as nn
from torch.nn import functional as F

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),  # linear layer that has bias (matrix multipliation followed by an addede bias vector); this is where the learning of the model happens after attention
            nn.ReLU(),  # non-linear activation function that allows our model to gain a deeper understanding of the inputs
            nn.Linear(4 * n_embd, n_embd),  # output layer that outputs a final embedding for the output (remember that the embedding is a deeper understanding of some sequence)
            nn.Dropout(dropout),  # dropout layer essentially makes it so that only some of the neurons are activated (only some of the parameters of the overall network are used) to help prevent overfitting
        )

    def forward(self, x):
        return self.net(x)