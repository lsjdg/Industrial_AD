import torch
import random


def coreset_sampling(mem_bank: torch.Tensor, M: int) -> torch.Tensor:
    """
    Greedy k-center algorithm to select M representative points from mem_bank (N', D).
    Returns indices of selected centroids.
    """
    N, D = mem_bank.shape
    centroids = [random.randrange(N)]
    distances = torch.cdist(mem_bank, mem_bank[centroids], p=2).squeeze(1)
    M = int(M)
    for _ in range(1, M):
        idx = int(torch.argmax(distances))
        centroids.append(idx)
        new_dist = torch.cdist(mem_bank, mem_bank[[idx]], p=2).squeeze(1)
        distances = torch.min(distances, new_dist)

    idxs = torch.tensor(centroids, dtype=torch.long)
    return mem_bank[idxs]
