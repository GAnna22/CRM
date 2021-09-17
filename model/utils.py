import torch

def get_batches(t_0, t_n, batch_size):
    n_samples = t_n + 1 - t_0
    # Shuffle at the start of epoch
    indices = torch.randperm(n_samples)
    for ind, start in enumerate(range(0, n_samples, batch_size)):
        end = min(start + batch_size, n_samples)
        batch_idx = indices[start:end]
        yield ind, batch_idx
