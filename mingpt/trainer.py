import math

import numpy as np

from tqdm import tqdm

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader

class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights

    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6
    final_tokens = 260e9 # (at what point we reach 10% of original LR)

    # checkpoint settings
    ckpt_path = None
    num_workers = 0 # for DataLoader

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class Trainer:
    def __init__(self, model, train_dataset, test_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config

        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

    def save_checkpoint(self):
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        logger.info("saving %s", self.config.ckpt_path)
        torch.save(raw_model.state_dict(), self.config.ckpt_path)

    def train(self):
        model, config = self.model, self.config
        raw_model = self.model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)

        #def run_epoch()

        best_loss = float('inf')
        self.tokens = 0

        for e in range(self.config.max_epochs):
            run_epoch('train')

            if self.test_dataset is not None:
                test_loss = run_epoch('test')

        good_model = self.test_dataset is None or test_loss < best_loss
        if self.config.ckpt_path is not None and good_model:
            best_loss = test_loss
            self.save_checkpoint()
