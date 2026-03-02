import os
import argparse
from contextlib import nullcontext
import torch
from model import GPTConfig, GPT
from tokenizer.basic_bpe import BasicTokenizer

default_tokenizer_model_path = os.path.join(os.path.dirname(__file__), 'tokenizer.model')

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, default='output/ckpt.pt', help='Path to checkpoint file')
parser.add_argument('--tokenizer-model-path', default=default_tokenizer_model_path)
parser.add_argument('--prompt', type=str, default='Hello')
parser.add_argument('--num-samples', type=int, default=1)
parser.add_argument('--max-new-tokens', type=int, default=500)
parser.add_argument('--temperature', type=float, default=0.9)
parser.add_argument('--top-k', type=int, default=200)
parser.add_argument('--byte-level', action='store_true', help='Use raw byte encoding instead of tokenizer')
args = parser.parse_args()

if torch.cuda.is_available():
    device = 'cuda'
# elif torch.backends.mps.is_available():
#     device = 'mps'
else:
    device = 'cpu'

checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)

# handle checkpoints saved with/without RoPE buffers
state_dict = checkpoint['model']
state_dict = {k: v for k, v in state_dict.items() if k not in ('cos', 'sin')}
model.load_state_dict(state_dict, strict=False)

model.eval()
model.to(device)

if args.byte_level:
    encode = lambda s: list(s.encode('utf-8'))
    decode = lambda t: bytes(t).decode('utf-8', errors='replace')
else:
    enc = BasicTokenizer()
    enc.load(args.tokenizer_model_path)
    if len(enc.vocab) != model.config.vocab_size:
        raise ValueError(
            f"Tokenizer vocab size ({len(enc.vocab)}) != model vocab size ({model.config.vocab_size}). "
            "Use --byte-level for byte-level models or provide matching tokenizer."
        )
    encode = enc.encode
    decode = enc.decode

start_ids = encode(args.prompt)
x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
prefix_len = x.size(1)

with torch.no_grad():
    for k in range(args.num_samples):
        y = model.generate(x, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k, prefix_len=prefix_len)
        print(decode(y[0].tolist()))
        print('---------------')
