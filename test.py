import math
import numpy as np

from random import randint

import torch

from mingpt.model import *
from mingpt.utils import *

set_seed(42)

config = GPTConfig(1000, 5, n_heads=12, n_layer=12, n_embd=768)
gpt1 = GPT(config)

idx = torch.tensor([[randint(0, 1000) for i in range(config.block_size)]])
y = gpt1(idx)

print(y, y.size())






















