import math
import numpy as np

from random import randint

import torch

from mingpt.model import *
from mingpt.utils import *

set_seed(42)

seq_len = 5
vocab_size = 60

config = GPTConfig(vocab_size, seq_len, n_heads=12, n_layer=12, n_embd=768)
gpt1 = GPT(config)

idx = torch.tensor([[randint(0, vocab_size) for i in range(config.block_size)]])
trg = torch.tensor([[randint(0, vocab_size) for i in range(config.block_size)]])

print(idx.size(), idx)

y, loss = gpt1(idx, trg)

print(y, y.size(), loss.item())























