import torch

from torch.utils.data import Dataset

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
        chunk = self.data[idx : idx+self.block_size+1]
        idx = [self.stoi[s] for s in chunk]
        x = torch.tensor(idx[:-1], dtype=torch.long)
        y = torch.tensor(idx[1:], dtype=torch.long)

        return x, y


