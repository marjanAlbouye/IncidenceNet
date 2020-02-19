from collections import defaultdict
import numpy as np
import torch
import networkx as nx
import shutil
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# this function flattens the batch and adjusts the indices needed for future operations
def get_indices(adjacency_list, edge_neighbor=None):
    idx = defaultdict(list)
    shift_all, shift_N1, shift_N2 = 0, -1, -1
    for i, adj in enumerate(adjacency_list):
        N1, N2 = adj.shape[0], adj.shape[1]
        r, c = adj.nonzero().numpy().transpose()

        # indices
        idx['row'].extend(r + (shift_N1 + 1))
        idx['col'].extend(c + (shift_N2 + 1))
        idx['all'].extend(np.zeros_like(r) + i)
        idx['n1'].append(N1)
        idx['n2'].append(N2)
        if edge_neighbor is not None:
            idx['edge_neigh'].extend(edge_neighbor[i] + (shift_N2 + 1))

        t = np.array([], dtype=np.int32)
        for j in range(adj.shape[0]):
            t = np.append(t, np.where(c == j)[0])
        # trans = np.argsort(c)
        idx['trans'].extend(t + shift_all)

        d = np.array(np.where(r == c)[0])
        idx['diag'].extend(d + shift_all)
        idx['diag_broad'].extend(np.zeros_like(d) + i)

        # normalization
        row_idx, row_norm = np.unique(r, return_counts=True)
        if len(row_idx) != N1:
            missed_row_idx = np.setdiff1d(np.arange(N1), row_idx)
            row_norm = np.insert(row_norm, missed_row_idx, 1)
        idx['row_norm'].extend(row_norm.astype(np.float32))

        col_idx, col_norm = np.unique(c, return_counts=True)
        if len(col_idx) != N2:
            col_norm_out = np.zeros(N2)
            missed_col_idx = np.setdiff1d(np.arange(N2), col_idx)
            np.put(col_norm_out, col_idx, col_norm)
            np.put(col_norm_out, missed_col_idx, 1)
            idx['col_norm'].extend(col_norm_out.astype(np.float32))
        else:
            idx['col_norm'].extend(col_norm.astype(np.float32))
        idx['diag_norm'].append(np.float32(len(d)))
        idx['all_norm'].append(np.float32(len(r)))
        shift_all += len(r)
        shift_N1 += N1
        shift_N2 += N2
    idx_tensor = {}
    for k, v in idx.items():
        idx_tensor[k] = torch.tensor(np.asarray(v))
    return idx_tensor


def save_checkpoint(state, is_best, directory):

    if not os.path.isdir(directory):
        os.makedirs(directory)
    checkpoint_file = os.path.join(directory, 'checkpoint.pth')
    best_model_file = os.path.join(directory, 'model_best.pth')
    torch.save(state, checkpoint_file)
    if is_best:
        shutil.copyfile(checkpoint_file, best_model_file)


def rotate_z(theta, x):
    theta = torch.tensor(theta).to(device).unsqueeze(0)
    outz = x[:, 2].unsqueeze(1)
    sin_t = torch.sin(theta)
    cos_t = torch.cos(theta)
    xx = x[:, 0].unsqueeze(1)
    yy = x[:, 1].unsqueeze(1)
    outx = cos_t * xx - sin_t * yy
    outy = sin_t * xx + cos_t * yy
    return torch.stack([outx, outy, outz], dim=1).squeeze()
