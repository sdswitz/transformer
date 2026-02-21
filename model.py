import torch
import torch.nn as nn
from torch.nn import functional as F

from dataclasses import dataclass
import math
import inspect

## Borrowed heavily from nanoGPT

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class SelfAttn(nn.Module):

    def __init__(self, config, is_causal=True):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.is_causal = is_causal

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x, prefix_len=None):
        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if self.is_causal:
            # Pure causal attention — same as original CausalSelfAttn
            if self.flash:
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
            else:
                att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
                att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float("-inf"))
                att = F.softmax(att, dim=-1)
                att = self.attn_dropout(att)
                y = att @ v
        else:
            # Bidirectional layer: prefix gets full attention, generation tokens get causal
            if prefix_len is None:
                # Full bidirectional attention (e.g., encoding the prompt at inference)
                if self.flash:
                    y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=False)
                else:
                    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
                    att = F.softmax(att, dim=-1)
                    att = self.attn_dropout(att)
                    y = att @ v
            else:
                # Build prefix-LM mask: prefix tokens attend to all prefix tokens,
                # generation tokens use causal masking
                # mask shape: (T, T), True = allowed to attend
                if isinstance(prefix_len, int):
                    # Single prefix_len for the whole batch
                    causal_mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
                    # Allow prefix tokens to attend to all other prefix tokens
                    causal_mask[:prefix_len, :prefix_len] = True
                    attn_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)
                else:
                    # Per-example prefix_len: prefix_len is (B,) tensor
                    causal_mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
                    causal_mask = causal_mask.unsqueeze(0).expand(B, -1, -1).clone()  # (B, T, T)
                    for i in range(B):
                        pl = prefix_len[i].item()
                        causal_mask[i, :pl, :pl] = True
                    attn_mask = causal_mask.unsqueeze(1)  # (B, 1, T, T)

                if self.flash:
                    y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0, is_causal=False)
                else:
                    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
                    att = att.masked_fill(~attn_mask, float("-inf"))
                    att = F.softmax(att, dim=-1)
                    att = self.attn_dropout(att)
                    y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config, is_causal=True):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = SelfAttn(config, is_causal=is_causal)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, prefix_len=None):
        x = x + self.attn(self.ln_1(x), prefix_len=prefix_len)
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0
    bias: bool = True
    n_causal_layers: int = 4  # first N layers are causal, remaining are bidirectional

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        # Build blocks: first n_causal_layers are causal, rest are bidirectional
        blocks = []
        for i in range(config.n_layer):
            is_causal = (i < config.n_causal_layers)
            blocks.append(Block(config, is_causal=is_causal))

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList(blocks),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer['wte'].weight = self.lm_head.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        n_causal = config.n_causal_layers
        n_bidir = config.n_layer - config.n_causal_layers
        print(f"number of params: {self.get_num_params()/1e6:.2f}M ({n_causal} causal + {n_bidir} bidirectional layers)")

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer['wpe'].weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, prefix_len=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of len {t}, block size only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x, prefix_len=prefix_len)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            if prefix_len is not None:
                # Only compute loss on generation tokens (after the prefix)
                if isinstance(prefix_len, int):
                    # Mask out prefix positions in targets
                    masked_targets = targets.clone()
                    masked_targets[:, :prefix_len] = -1
                else:
                    # Per-example prefix lengths
                    masked_targets = targets.clone()
                    for i in range(b):
                        masked_targets[i, :prefix_len[i]] = -1
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), masked_targets.view(-1), ignore_index=-1)
            else:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, prefix_len=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.

        prefix_len: length of the initial prompt, so bidirectional layers see the full prompt.
        During generation, all tokens are "past" so causal masking applies naturally.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond, prefix_len=prefix_len)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
