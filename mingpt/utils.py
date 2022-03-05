import random
import numpy as np

import torch
import torch.nn as nn

from torch.nn import functional as F

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def top_k_logits(logits, k):
    v, _ = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out

@torch.no_grad()
def sample(model, x, steps, temp=1.0,  sample=False, top_k=None):
    model.eval()

    block_size = model.get_block_size()
    for i in range(steps):
        x_ctx = x if x.size(1) <= block_size else x[:, -block_size:]
        logits, _ = model(x_ctx)
        logits = logits[:, -1, :] / max(0.1, temp)

        if top_k is not None:
            logits = top_k_logits(logits, top_k)

        probs = F.softmax(logits, dim=-1)

        if sample:
            idx = torch.multinomial(probs, num_samples=1)
        else:
            _, idx = torch.topk(probs, k=1, dim=-1)

        x = torch.cat((x, idx), dim=1)

    return x




