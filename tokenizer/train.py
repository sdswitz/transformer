import os
import time

from basic_bpe import BasicTokenizer

with open('/Users/samswitz/GitHub/micro-research/transformer/data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
print('big load of data')

vocab_size = 1024
tokenizer = BasicTokenizer()

t0 = time.time()

tokenizer.train(text, vocab_size)
prefix = '/Users/samswitz/GitHub/micro-research/transformer/tokenizer'
tokenizer.save(prefix)

t1 = time.time()
delta = t1 - t0

print(f"Training took {delta:.2f} seconds, an average of {(vocab_size/(delta)):.2f} tokens per second")