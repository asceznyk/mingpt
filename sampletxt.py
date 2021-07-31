import sys
import math
import argparse

import numpy as np

import torch

from mingpt.model import *
from mingpt.utils import *

from datasets import *

def sample_text(options):
    assert options.weights is not None, 'model weights file path not given!'
    assert options.char_data is not None, 'char_data path is not given!'

    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

    char_data = torch.load(options.char_data)
    mcfg = GPTConfig(char_data.vocab_size, char_data.block_size, n_layer=8, n_heads=8, n_embd=512)
    model = GPT(mcfg)
    model.load_state_dict(torch.load(options.weights))
    model = model.to(device)

    context = options.context
    x = torch.tensor([char_data.stoi[s] for s in context])[None, :].to(device)
    y = sample(model, x, options.gen_len, temp=options.temperature, top_k=options.top_k)[0]
    completion = ''.join([char_data.itos[int(i)] for i in y])

    print('')
    print('='*40)
    print(completion)
    print('='*40)
    print('')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--context', type=str, help='context for model to predict on', default='fuck you for not giving context!')
    parser.add_argument('--weights', type=str, help='path to trained model weights file (required)', default=None)
    parser.add_argument('--char_data', type=str, help='path to dataset containing string-to-index mapping (required)', default=None)
    parser.add_argument('--gen_len', type=int, help='generated sequence length', default=2500)
    parser.add_argument('--top_k', type=int, help='top k chars to select while predicting', default=10)
    parser.add_argument('--temperature', type=int, help='logits enchancement!', default=1.0)

    options = parser.parse_args()

    print(options)

    sample_text(options)


