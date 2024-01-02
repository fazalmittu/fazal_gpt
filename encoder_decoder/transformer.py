"""
File: transformer.py
Author: Fazal Mittu
Code modified from Andrej Karpathy's example

Implementation of Simple Transformer
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

from encoder_decoder.classes.LanguageModel import LanguageModel

# hyperparameters
batch_size: int = 32         # we randomly group together training samples in batches and feed those the network rather than sending all examples in at once
block_size: int = 8          # each entry in a batch is a vector of size 8
embedding_dim: int = 96      # dimensions for our embeddings
n_heads: int = 3             # how many heads we have for multi head attention
n_layers: int = 6            # number of attention + ff blocks we have
epochs: int = 200            # how many epochs we are training for (how many times we do forward/back prop)
learning_rate: float = 1e-3  # affects how big of a step we take in adjusting the weights of the network based on the gradients
eval_iters: int = 200        # how many times we evaluate the model on our validation set
dropout: float = 0.1         # what percentage of weights we randomly switch off during training
eval_interval: int = 10      # how often we evaluate loss during training

device = 'cuda' if torch.cuda.is_available() else 'cpu'  # setting device to gpu if available
print(device)

# loading in dataset (text file with shakespeare text)
with open('input.txt', 'r', encoding='utf-8') as f:
    text: str = f.read()

# build vocabulary for our transformer (for now, just made up of all the different characters found in the dataset)
chars: list = sorted(list(set(text)))  # gets all the unique characters found in the text (use set because it removes duplicates)
vocab_size: int = len(chars)  # how many different characters our model has access to (all possible predictions it can make essentially)

# encoding the characters so we can vectorize them; very simple encoding that just assigns a number to each character using index only
char_to_num: dict[int, str] = {val : index for index, val in enumerate(chars)}
num_to_char: dict[str, int] = {index : val for index, val in enumerate(chars)}

# encode/decode functions so we can go from word --> vector and vector --> word
def encode(text: str):
    encoded = []
    for character in text:
        encoded.append(char_to_num[character])
    return encoded

def decode(vector):
    text = ''
    for number in vector:
        text += num_to_char[number]
    return text

# split data into train/eval
data = torch.tensor(encode(text), dtype=torch.long)  # we convert all the text into a large tensor with a bunch of numbers (longs are just ints)
n = int(0.9*len(data))  # first 90% of data will be used for training (cast to int in case of a decimal number)
train_data = data[:n]
eval_data = data[n:]

# function that takes in all data and creates a random batch, and then loads it onto the gpu if possible
def get_batch(train: bool): 
    if train:
        data = train_data
    else:
        data = eval_data
    
    # batch has to be of size (block_size,)
    start_indices = torch.randint(low=0, high=len(data) - block_size, size=(batch_size,))
    x = torch.stack([data[i: i+block_size] for i in start_indices])    # x represents the input vectors 
    y = torch.stack([data[i+1: i+block_size+1] for i in start_indices])  # y represents the expected output vectors (offset x by 1)
    
    x, y = x.to(device), y.to(device)  # move the x, y to the device being used (either cpu or gpu)

    return x, y  # both have size (batch_size, block_size)

@torch.no_grad()  # tells pytorch that we are not going to be adjusting these parameters so it doesn't keep track of gradients
def estimate_loss(model: LanguageModel):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            train = True if split == "train" else False
            X, Y = get_batch(train)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def train():
    model = LanguageModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        block_size=block_size,
        n_layers=n_layers,
        n_heads=n_heads,
        dropout=dropout,
        device=device
    )
    m = model.to(device)
    # print the number of parameters in the model
    print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in range(epochs):

        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == epochs - 1:
            losses = estimate_loss(m)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # sample a batch of data
        xb, yb = get_batch(True)

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    return m

def generate(model: LanguageModel):
    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))

if __name__ == "__main__":
    model = train()
    generate(model)

    # python3 -m transformer










