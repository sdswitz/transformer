import os
import requests
# import tiktoken
import numpy as np

# --- add these lines at the very top of data/prepare.py ---
from pathlib import Path
import sys

# Make the repo root importable when running this file directly
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# --- end shim ---

from tokenizer.basic_bpe import BasicTokenizer

# print(os.getcwd())
# import sys; sys.exit(0)

with open('data/input.txt', 'r', encoding='utf-8') as f:
    data = f.read()
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode with tiktoken gpt2 bpe
enc = BasicTokenizer()
enc.load('/Users/samswitz/GitHub/transformer/tokenizer.model')


train_ids = enc.encode(train_data)
val_ids = enc.encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# train.bin has 301,966 tokens
# val.bin has 36,059 tokens