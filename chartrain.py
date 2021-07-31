import sys
import math
import argparse

import numpy as np

import torch

from mingpt.model import *
from mingpt.utils import *
from mingpt.trainer import *

from datasets import *

def train_char_lvl(options):
    set_seed(42)

    block_size = options.sequence_length
    batch_size = options.batch_size
    train_dataset = CharDataset(open(options.txt, 'r').read(), block_size)

    mcfg = GPTConfig(train_dataset.vocab_size, train_dataset.block_size, n_layer=8, n_heads=8, n_embd=512)
    model = GPT(mcfg)

    torch.save(train_dataset, options.char_data)
    print('saved train char dataset..')

    tcfg = TrainerConfig(
        max_epochs=options.n_epochs,
        batch_size=batch_size,
        learning_rate=6e-4,
        lr_decay=True,
        warmup_tokens=batch_size*20,
        final_tokens=2*len(train_dataset)*block_size,
        num_workers=options.n_workers,
        ckpt_path=options.ckpt
    )
    trainer = Trainer(model, train_dataset, None, tcfg)
    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--txt', type=str, help='path to text file for training data')
    parser.add_argument('--sequence_length', type=int, help='max length of sequence', default=128)
    parser.add_argument('--ckpt', type=str, help='path for saving model weights', default='char.model.ckpt')
    parser.add_argument('--char_data', type=str, help='path for saving the character level dataset which has string-to-index mapping required for the model', default='char.dataset.pt')
    parser.add_argument('--n_epochs', type=int, help='number of epochs to train the model', default=2)
    parser.add_argument('--batch_size', type=int, help='batch size', default=128)
    parser.add_argument('--n_workers', type=int, help='num workers', default=2)

    options = parser.parse_args()

    print(options)

    train_char_lvl(options)

