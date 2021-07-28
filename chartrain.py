import sys
import math

import numpy as np

from random import randint

import torch

from torch.utils.data import Dataset

from mingpt.model import *
from mingpt.utils import *
from mingpt.trainer import *

from datasets import *

set_seed(42)

block_size = 128
text = open(sys.argv[1], 'r').read()
train_dataset = CharDataset(text, block_size)

mcfg = GPTConfig(train_dataset.vocab_size, train_dataset.block_size, n_layer=8, n_heads=8, n_embd=512)
model = GPT(mcfg)

tcfg = TrainerConfig(max_epochs=2, batch_size=128, learning_rate=6e-4, lr_decay=True, warmup_tokens=128*20, final_tokens=2*len(train_dataset)*block_size, num_workers=2, ckpt_path='char.model.ckpt')
trainer = Trainer(model, train_dataset, None, tcfg)
trainer.train()

