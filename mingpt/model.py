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
    n_heads = 12
    n_embd = 768

class MHSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_heads == 0

        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)

        self.proj = nn.Linear(config.n_embd, config.n_embd)

        self.attn_drop = nn.Dropout(config.attn_drop)
        self.resid_drop = nn.Dropout(config.resid_drop)

        self.register_buffer(
            'mask',
            torch.tril(
                torch.ones(config.block_size, config.block_size)
            ).view(1, 1, config.block_size, config.block_size)
        )

        self.n_heads = config.n_heads

    def forward(self, x):
        B, S, E = x.size()

        #k, q, v (B, nh, S, hs) 
        k = self.key(x).view(B, S, -1, E // self.n_heads).transpose(1,2)
        q = self.query(x).view(B, S, -1, E // self.n_heads).transpose(1,2)
        v = self.value(x).view(B, S, -1, E // self.n_heads).transpose(1,2)

        #(B, nh S, hs) * (B, nh, hs, S) -> (B, nh, S, S)
        attn = (q @ k.transpose(-2,-1) / math.sqrt(k.size(-1)))
        attn = attn.masked_fill(self.mask[:,:,:S,:S] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        attn = attn @ v # (B, nh, S, S) * (B, nh, S, hs) -> (B, nh, S, hs)

        y = attn.transpose(1,2).contiguous().view(B, S, E)
        return self.resid_drop(self.proj(y))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mha = MHSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_drop)
        )

    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        return x + self.mlp(self.ln2(x))

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.tok_emb = nn.Linear(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))

        self.drop = nn.Dropout(config.embd_drop)

        self.block_size = config.block_size

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None):
        b, s = idx.size()
        assert s <= self.block_size, 'Cannot forward, model block size exceeded!'

        token_emeddings = self.tok_emb(idx)
        position_embeddings = self.pos_emb[:, :t, :]

        return (token_emeddings + position_embeddings)


