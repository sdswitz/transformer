# Hybrid Causal + Bidirectional Transformer (Prefix-LM)

## Overview

This project implements a **prefix-LM** transformer architecture that combines causal attention in early layers with bidirectional attention in later layers. The idea: early layers learn local grammar and patterns with standard left-to-right attention, while later layers can "look back" across the full prompt prefix to build richer representations before generating.

## Architecture

### Layer Split

The model has 6 layers total, configured via `n_causal_layers` in `GPTConfig`:

```
Layer 0-3: Causal attention (standard GPT-style, left-to-right only)
Layer 4-5: Bidirectional attention (prefix-LM masking)
```

### How Bidirectional Layers Work

Bidirectional layers use a **prefix-LM attention mask** — not fully bidirectional like BERT, but a hybrid:

- **Prefix tokens** (positions `0` to `prefix_len-1`): attend to all other prefix tokens freely (full bidirectional attention within the prefix)
- **Generation tokens** (positions `prefix_len` onward): use standard causal masking (can only attend to earlier positions)

This means during training, the model sees the prompt with full context before it has to predict generation tokens. During inference/generation, all previously generated tokens are "past" so causal masking applies naturally.

### Model Size

~11M parameters with the current defaults:

| Parameter | Value |
|-----------|-------|
| `n_layer` | 6 |
| `n_head` | 6 |
| `n_embd` | 384 |
| `block_size` | 256 |
| `n_causal_layers` | 4 |
| `vocab_size` | 1024 |

## Files Changed

### `model.py`

**`SelfAttn`** (replaced `CausalSelfAttn`):
- Takes `is_causal` flag at init time to set the layer's default behavior
- `forward(x, prefix_len=None)`:
  - Causal layers (`is_causal=True`): identical to the old `CausalSelfAttn` — always uses a causal mask
  - Bidirectional layers (`is_causal=False`):
    - `prefix_len=None` → full bidirectional attention (used during generation when the whole sequence is "past")
    - `prefix_len` as int or per-batch tensor → builds the prefix-LM mask described above
- Supports both flash attention (`scaled_dot_product_attention`) and manual attention paths

**`Block`**: accepts `is_causal` and forwards `prefix_len` to `SelfAttn`.

**`GPTConfig`**: added `n_causal_layers: int = 4`. Updated defaults to smaller model.

**`GPT`**:
- Constructor builds the first `n_causal_layers` blocks as causal and the rest as bidirectional
- `forward(idx, targets, prefix_len)`: passes `prefix_len` to all blocks. When computing loss with targets, masks out prefix positions (sets them to `-1` / `ignore_index`) so loss is only on generation tokens.
- `generate(idx, ..., prefix_len)`: passes `prefix_len` through so bidirectional layers can use full attention on the prompt

### `train.py`

**`get_batch`**: now returns `(x, y, prefix_lens)` where `prefix_lens` is a `(batch_size,)` tensor with random per-example prefix lengths sampled uniformly between 10% and 50% of `block_size`.

**Training loop**: passes `prefix_len=prefix_lens` to `model()` on every step.

**`estimate_loss`**: updated to use the new 3-return `get_batch` and pass prefix lengths.

**Config**: updated to `n_layer=6, n_head=6, n_embd=384, block_size=256, n_causal_layers=4`.

### `sample.py`

Computes `prefix_len = x.size(1)` (the encoded prompt length) and passes it to `model.generate()`.

## Training Details

- Loss is only computed on **generation tokens** (positions after the prefix). The prefix tokens have their targets masked to `-1` so `cross_entropy` ignores them.
- Each training example gets a different random prefix length, so the model learns to handle varying prompt/generation splits.
- Optimizer: AdamW, lr=1e-3
- 5000 iterations, eval every 200 steps

## Why This Design

Standard causal LMs waste capacity: every layer processes the prompt left-to-right even though the full prompt is available. A prefix-LM lets later layers attend bidirectionally over the prompt, which should help the model:

1. Build better prompt representations before it starts generating
2. Still learn local sequential patterns in the early causal layers
3. Maintain autoregressive generation capability (causal masking on generation tokens)

The configurable `n_causal_layers` split makes it easy to experiment with different ratios.
