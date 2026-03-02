import os
import argparse
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

# argument parsing
default_tokenizer_model_path = os.path.join(os.path.dirname(__file__), 'tokenizer.model')
parser = argparse.ArgumentParser()
parser.add_argument('--tokenizer-model-path', default=default_tokenizer_model_path)
parser.add_argument('--compile', dest='use_compile', action='store_true')
parser.add_argument('--no-compile', dest='use_compile', action='store_false')
parser.set_defaults(use_compile=False)

# training hyperparameters
parser.add_argument('--max-iters', type=int, default=500)
parser.add_argument('--eval-interval', type=int, default=50)
parser.add_argument('--learning-rate', type=float, default=1e-3)
parser.add_argument('--batch-size', type=int, default=64)

# model hyperparameters
parser.add_argument('--block-size', type=int, default=512)
parser.add_argument('--vocab-size', type=int, default=None)
parser.add_argument('--n-layer', type=int, default=6)
parser.add_argument('--n-head', type=int, default=6)
parser.add_argument('--n-embd', type=int, default=384)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--bias', action='store_true')
parser.add_argument('--n-causal-layers', type=int, default=4)

# paths
parser.add_argument('--data-dir', type=str, default='data')
parser.add_argument('--out-dir', type=str, default='output')

# logging
parser.add_argument('--wandb', action='store_true', help='Log to Weights & Biases')
parser.add_argument('--wandb-project', type=str, default='transformer')
parser.add_argument('--wandb-run-name', type=str, default=None)

args = parser.parse_args()

# load tokenizer
enc = BasicTokenizer()
enc.load(args.tokenizer_model_path)

# resolve settings
max_iters = args.max_iters
eval_interval = args.eval_interval
learning_rate = args.learning_rate
batch_size = args.batch_size
block_size = args.block_size
vocab_size = args.vocab_size if args.vocab_size is not None else len(enc.vocab)

if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

eval_iters = max(1, int(max_iters * 0.05))

out_dir = args.out_dir
os.makedirs(out_dir, exist_ok=True)
print(f"using device: {device}")
print(f"output directory: {out_dir}")

model_args = dict(
    block_size=block_size, vocab_size=vocab_size,
    n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
    dropout=args.dropout, bias=args.bias,
    n_causal_layers=args.n_causal_layers,
)
config = GPTConfig(**model_args)

data_dir = args.data_dir
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
max_data_id = int(max(train_data.max(), val_data.max()))
if max_data_id >= vocab_size:
    raise ValueError(
        f"Data token id range is [0, {max_data_id}] but model/tokenizer vocab_size is {vocab_size}. "
        "Rebuild data bins with the same tokenizer model used for training."
    )

# data loading
def get_batch(split):
    if split == 'train':
        data = train_data
    else:
        data = val_data
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
model = model.to(device)
if args.use_compile:
    model = torch.compile(model)
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

if args.wandb:
    import wandb
    wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=model_args)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

checkpoint_step = max_iters / 4

train_start = time.time()

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if args.wandb:
            wandb.log({'train/loss': losses['train'], 'val/loss': losses['val']}, step=iter)

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

elapsed = time.time() - train_start
minutes, seconds = divmod(elapsed, 60)
print(f"training complete: {int(minutes)}m {seconds:.1f}s")

if args.wandb:
    wandb.finish()
