import torch
import numpy as np
from utils import Timer

grads = [
        torch.tensor([0,0]).float(),
        torch.tensor([0,1]).float(),
        torch.tensor([1,0]).float(),
        torch.tensor([1,1]).float(),
        torch.tensor([1,2]).float(),
        torch.tensor([2,1]).float(),
        torch.tensor([2,2]).float(),]
grads = [torch.randn(6656840) for _ in range(100)]

def _krum(grads):
    grads = torch.stack(grads).cuda().float()
    n_workers = len(grads)
    n_models = n_workers - 3
    distances = {i: {j: None for j in range(n_workers) if i != j} for i in range(n_workers)}
    closest_sums = torch.zeros(n_workers)
    for idx, g in enumerate(grads):
        for jdx, j in enumerate(grads):
            if idx != jdx:
                if distances[jdx][idx] is not None:
                    distances[idx][jdx] = distances[jdx][idx]
                else:
                    distances[idx][jdx] = (g - j).norm(p=2)
        dist_array = torch.tensor([val for key, val in distances[idx].items()])
        dist_array = torch.sort(dist_array)[0]
        closest_dists = dist_array[:-3]
        closest_sums[idx] = closest_dists.sum()
    closest_model = grads[torch.sort(closest_sums)[1][0]]
    return closest_model

def _bulyan_krum(distances, n_workers):
    # keep an array of the sum of the distances of all other models except for the 3 furthest to this worker
    closest_sums = torch.ones(n_workers) * float('inf')
    for idx, dists in distances.items():
        dist_array = torch.tensor([val for key, val in dists.items()])
        dist_array = torch.sort(dist_array)[0]
        closest_dists = dist_array[:-3]
        closest_sums[idx] = closest_dists.sum()
    # return the model that is "overall closer" to all the other models"
    argmin_closest = torch.sort(closest_sums)[1][0]
    return argmin_closest

def _bulyan(grads, f):
    #f = 24
    f = int(f)
    grads = torch.stack(grads).float()
    n_workers = len(grads)
    theta = len(grads) - 2 * f
    s = []
    # compute distances between all models
    distances = {i: {j: None for j in range(n_workers) if i != j} for i in range(n_workers)}
    for idx, g in enumerate(grads):
        for jdx, j in enumerate(grads):
            if idx != jdx:
                if distances[jdx][idx] is not None:
                    distances[idx][jdx] = distances[jdx][idx]
                else:
                    distances[idx][jdx] = (g - j).norm(p=2)
    grads = grads.cpu()
    while len(s) < theta:
        # get new candidate based on the output of krum
        model_idx = _bulyan_krum(distances, n_workers)
        # remove candidate from distances for recursion
        distances = {key_outer: {key_inner: val_inner for key_inner, val_inner in val_outer.items() if key_inner != model_idx} for key_outer, val_outer in distances.items() if key_outer != model_idx}
        # add candidate to s
        grad = grads[model_idx].cpu()
        s.append(grad)
    del grads
    # return the trimmed mean of the candidate set
    return torch.stack(s).sort()[0][f:-f].mean()
    #return _trimmed_mean(s, f)

def _trimmed_mean(grads, f):
    n_workers = len(grads)
    gs = torch.stack(grads).permute(1,0).sort(axis=1)[0][:,f:-f]
    trimmed_sum = gs.sum(axis=1)
    trimmed_mean = trimmed_sum / (n_workers - 2*f)
    return trimmed_mean


def _coordinate_median(grads):
    return torch.stack(grads).cuda().permute(1,0).median(axis=1)[0]

if __name__ == "__main__":
    timer = Timer()
    print(_bulyan(grads, 10))
