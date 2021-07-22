import torch

from mingpt.model import *
from mingpt.utils import set_seed

config = GPTConfig(1000, 3, n_heads=12, n_layer=12, n_embd=768)

self_attn = MHSelfAttention(config)

x = torch.rand(1, 3, 768)
attn = self_attn(x)

print(attn, attn.size())






















