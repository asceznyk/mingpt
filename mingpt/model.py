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

class MHSelfAttention(nn.Module):
    def __init__(config):
        assert config.n_embd % config.n_head == 0

        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)

        self.proj = nn.Linear(config.n_embd, config.n_embd)

        self.attn_drop = nn.Dropout(config.attn_drop)
        self.resid_drop = nn.Dropout(config.resid_drop)

        self.register_buffer('mask', torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

        self.n_head = config.n_head


    def forward(self, x):
        B, S, E = x.size()

        #k, q, v (B, nh, S, hs) 
        k = self.key(x).view(B, S, -1, E // self.n_head).transpose(1,2)
        q = self.query(x).view(B, S, -1, E // self.n_head).transpose(1,2)
        v = self.value(x).view(B, S, -1, E // self.n_head).transpose(1,2)

        #(B, nh S, hs) * (B, nh, hs, S) -> (B, nh, S, S)
        attn = (q @ k.transpose(-2,-1) / math.sqrt(k.size(-1)))
        attn = attn.masked_fill(self.mask[:,:,:S,:S] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        attn = attn @ v # (B, nh, S, S) * (B, nh, S, hs) -> (B, nh, S, hs)

        y = attn.transpose(1,2).contiguous().view(B, S, E)

        return self.resid_drop(self.proj(y))

