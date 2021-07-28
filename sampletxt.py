import sys
import math

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

device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
test_dataset = CharDataset(open(sys.argv[1], 'r').read(), 128)

mcfg = GPTConfig(test_dataset.vocab_size, test_dataset.block_size, n_layer=8, n_heads=8, n_embd=512)
model = GPT(mcfg)
model.load_state_dict(torch.load(sys.argv[2]))
model = model.to(device)

context = sys.argv[3]
x = torch.tensor([test_dataset.stoi[s] for s in context])[None, :].to(device)
y = sample(model, x, 2500, temp=1.0, top_k=10)[0]
completion = ''.join([test_dataset.itos[int(i)] for i in y])

print(completion)

