import sys
import math
import argparse

import numpy as np

from random import randint

import torch

from torch.utils.data import Dataset

from mingpt.model import *
from mingpt.utils import *
from mingpt.trainer import *

from datasets import *

def train_char_lvl(options):
    set_seed(42)

    block_size = options.sequence_length
    train_dataset = CharDataset(open(options.txt, 'r').read(), block_size)

    mcfg = GPTConfig(train_dataset.vocab_size, train_dataset.block_size, n_layer=8, n_heads=8, n_embd=512)
    model = GPT(mcfg)
    if options.pretrained_weights is not None:
        model.load_state_dict(torch.load(options.pretrained_weights))

    torch.save(train_dataset, 'char.dataset.pt')
    train_dataset = torch.load('char.dataset.pt')

    tcfg = TrainerConfig(max_epochs=2, batch_size=128, learning_rate=6e-4, lr_decay=True, warmup_tokens=128*20, final_tokens=2*len(train_dataset)*block_size, num_workers=2, ckpt_path=options.ckpt_path)
    trainer = Trainer(model, train_dataset, None, tcfg)
    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--txt', type=str, help='path to text file for training data')
    parser.add_argument('--sequence_length', type=int, help='max. length of sequence', default=128)
    parser.add_argument('--ckpt', type=str, help='path for saving model weights', default='char.model.ckpt')
    parser.add_argument('--pretrained_weights', type=str, help='path to pre-trained weights file', default=None)

    options = parser.parse_args()

    print(options)

    train_char_lvl(options)

