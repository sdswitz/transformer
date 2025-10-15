import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT

from tokenizer.basic_bpe import BasicTokenizer


out_dir = 'output'
start = "\n"
num_samples = 10
max_new_tokens = 500
temperature = 1.0
top_k = 200

# if torch.cuda.is_available():
#     device = 'cuda'
# elif torch.backends.mps.is_available():
#     device = 'mps'
# else:
#     device = 'cpu'

device = 'cpu'
device_type = 'cpu'

ctx = nullcontext()

ckpt_path = os.path.join(out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
state_dict = checkpoint['model']
model.load_state_dict(state_dict)

model.eval()
model.to(device)

enc = BasicTokenizer()

## TODO Parameterize this
enc.load('/Users/samswitz/GitHub/transformer/tokenizer.model')

start_ids = enc.encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(enc.decode(y[0].tolist()))
            print('---------------')