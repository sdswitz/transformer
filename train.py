import os
import time
import math
import pickle
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT
from tokenizer.basic_bpe import BasicTokenizer

# hyperparameters
max_iters = 500
eval_interval = 50

learning_rate = 1e-3
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

eval_iters = int(max_iters * 0.05)

enc = BasicTokenizer()

out_dir = 'output/vocab256_block512'
os.makedirs(out_dir, exist_ok=True)
print(f"using device: {device}")
print(f"output directory: {out_dir}")

# model hyperparams — smaller prefix-LM config
batch_size = 64

block_size: int = 512
vocab_size: int = len(enc.vocab)
n_layer: int = 6
n_head: int = 6
n_embd: int = 384
dropout: float = 0.0
bias: bool = False
n_causal_layers: int = 4  # 4 causal + 2 bidirectional

model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=vocab_size, dropout=dropout,
                  n_causal_layers=n_causal_layers)

config = GPTConfig(**model_args)


data_dir = os.path.join('data')
# data loading
def get_batch(split):
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+block_size+1]).astype(np.int64)) for i in ix])

    # Random prefix length per example: between 10% and 50% of block_size
    min_prefix = int(block_size * 0.1)
    max_prefix = int(block_size * 0.5)
    prefix_lens = torch.randint(min_prefix, max_prefix + 1, (batch_size,))

    x, y = x.to(device), y.to(device)
    prefix_lens = prefix_lens.to(device)
    return x, y, prefix_lens

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y, prefix_lens = get_batch(split)
            logits, loss = model(X, Y, prefix_len=prefix_lens)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

model = GPT(config)
m = model.to(device)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

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
            'config': config,
        }
        print(f"saving checkpoint to {out_dir}")
        torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

    # sample a batch of data
    xb, yb, prefix_lens = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb, prefix_len=prefix_lens)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
