import math
import numpy as np

from random import randint

import torch

from mingpt.model import *
from mingpt.utils import *
from mingpt.trainer import *

set_seed(42)

config = GPTConfig(10, 5, n_heads=12, n_layer=12, n_embd=768)
gpt1 = GPT(config)

idx = torch.tensor([[randint(0, config.vocab_size-1) for i in range(config.block_size)]])
trg = torch.tensor([[randint(0, config.vocab_size-1) for i in range(config.block_size)]])

print(f'input: {idx}, size: {idx.size()}')

logits, loss = gpt1(idx, trg)

print(f'logits: {logits}, size: {logits.size()} loss: {loss.item():.3f}')

sampled_idxs = sample(gpt1, idx, steps=30, top_k=5)

print(f'all the sampled indexes:  {sampled_idxs}')

train_cfg = TrainerConfig()
gpt1.configure_optimizers(train_cfg)






















