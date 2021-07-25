import math
import numpy as np

from random import randint

import torch

from mingpt.model import *
from mingpt.utils import *
from mingpt.trainer import *

set_seed(42)

block_size = 5
vocab_size = 10

config = GPTConfig(vocab_size, block_size, n_heads=12, n_lalogitser=12, n_embd=768)
gpt1 = GPT(config)

idx = torch.tensor([[randint(0, vocab_size) for i in range(config.block_size)]])
trg = torch.tensor([[randint(0, vocab_size) for i in range(config.block_size)]])

print(f'input: {idx}, size: {idx.size()}')

logits, loss = gpt1(idx, trg)

print(f'logits: {logits}, size: {logits.size()} loss: {loss.item():.3f}')

top_logits = top_k_logits(logits, k=4)

print(f'changed to top k logits: {top_logits}')






















