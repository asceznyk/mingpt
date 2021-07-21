import math

import torch
import torch.nn as nn

from torch.nn import functional as F

class GPTConfig:
    embd_drop = 0.1
    resid_drop = 0.1
    attn_drop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size

        for k, v in kwargs.items():
            setattr(self, k, v)

class GPT1Config(GPTConfig):
    n_layer = 12
    n_head = 12
    n_embd = 768

class SelfAttention(nn.Module):
    def __init__(config):
        assert config.n_embd % config.n_head == 0
