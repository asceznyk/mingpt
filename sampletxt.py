import sys
import math
import argparse

import numpy as np

from tqdm import tqdm

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader

from mingpt.model import *
from mingpt.utils import *
from mingpt.trainer import *

from datasets import *

def sample_text(options):
    test_dataset = CharDataset(open(options.txt, 'r').read(), 128)

    mcfg = GPTConfig(test_dataset.vocab_size, test_dataset.block_size, n_layer=8, n_heads=8, n_embd=512)
    model = GPT(mcfg)
    model.load_state_dict(torch.load(options.weights_path))

    context = sys.argv[3]
    x = torch.tensor([test_dataset.stoi[s] for s in context])[None, :].to(device)
    y = sample(model, x, 2500, temp=1.0, top_k=10)[0]
    completion = ''.join([test_dataset.itos[int(i)] for i in y])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--txt', type=str, help='path to text file for initailzing test data')
    parser.add_argument('--weights_path', type=str, help='path to model weights file')


    options = parser.parse_args()

    print(options)


