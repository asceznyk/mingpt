import math
import numpy as np

from random import randint

import torch

from torch.utils.data import Dataset

from mingpt.model import *
from mingpt.utils import *
from mingpt.trainer import *

set_seed(42)

class CharDataset(Dataset):
    def __init__(self, data, block_size):
        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))

        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = data

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.block_size + 1]
        dix = [self.stoi[s] for s in chunk]
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)

        return x, y

block_size = 128
text = open('shakespeare.txt', 'r').read()
train_dataset = CharDataset(text, block_size)

mcfg = GPTConfig(train_dataset.vocab_size, train_dataset.block_size, n_layer=8, n_heads=8, n_embd=512)
model = GPT(mcfg)

tcfg = TrainerConfig(max_epochs=2, batch_size=128, learning_rate=6e-4, lr_decay=True, warmup_tokens=128*20, final_tokens=2*len(train_dataset)*block_size, num_workers=2)
trainer = Trainer(model, train_dataset, None, tcfg)
trainer.train()



