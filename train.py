import os
import time
import math
import pickle
from contextlib import nullcontext

# import sys

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT
from tokenizer.basic_bpe import BasicTokenizer

# hyperparameters
block_size = 32
batch_size = 16
max_iters = 2000
eval_interval = 100
learning_rate = 1e-3
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = int(max_iters * 0.05)
vocab_size = 512 # Parameterize this

out_dir = 'output'
os.makedirs(out_dir, exist_ok=True)
print(f"using device: {device}")
print(f"output directory: {out_dir}")

# model
n_layer = 8
n_head = 8
n_embd = 512
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?

# import sys; sys.exit(0)


torch.manual_seed(327)


### unneeded tokenization stuff

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
# with open('/Users/samswitz/GitHub/micro-research/transformer/data/input.txt', 'r', encoding='utf-8') as f:
#     text = f.read()
# print('big load of data')

# tokenizer = BasicTokenizer()
# tokenizer.train(text, vocab_size)

# # Train and test splits
# data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
# print('successfully trained tokenizer')
# import sys; sys.exit(0)

# n = int(0.9*len(data)) # first 90% will be train, rest val
# train_data = data[:n]
# val_data = data[n:]

###


data_dir = os.path.join('data')
# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    # data = train_data if split == 'train' else val_data
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+block_size+1]).astype(np.int64)) for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=1024, dropout=dropout) # start with model_args from command line

config = GPTConfig(**model_args)

model = GPT(config)
m = model.to(device)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# import sys; sys.exit(0)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

checkpoint_step = max_iters / 4

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
    if iter % checkpoint_step == 0 or iter == max_iters - 1:
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'model_args': model_args,
            'iter_num': iter,
            # 'best_val_loss': best_val_loss,
            'config': config,
        }
        print(f"saving checkpoint to {out_dir}")
        torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()