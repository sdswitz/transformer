import os
import argparse
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT

from tokenizer.basic_bpe import BasicTokenizer


out_dir = 'output/vocab256_block512'
start = "Hello"
num_samples = 1
max_new_tokens = 500
temperature = 0.9
top_k = 200
default_tokenizer_model_path = '/Users/samswitz/GitHub/transformer/tokenizer.model'

parser = argparse.ArgumentParser()
parser.add_argument(
    '--tokenizer-model-path',
    default=default_tokenizer_model_path,
    help='Path to tokenizer model file',
)
args = parser.parse_args()

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


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
enc.load(args.tokenizer_model_path)
tokenizer_vocab_size = len(enc.vocab)
if tokenizer_vocab_size != model.config.vocab_size:
    raise ValueError(
        f"Tokenizer vocab size ({tokenizer_vocab_size}) does not match model vocab size ({model.config.vocab_size}). "
        "Use matching tokenizer/checkpoint pair or retrain with aligned vocab_size."
    )

start_ids = enc.encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# Pass prompt length as prefix_len so bidirectional layers see the full prompt
prefix_len = x.size(1)

with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k, prefix_len=prefix_len)
            print(enc.decode(y[0].tolist()))
            print('---------------')
