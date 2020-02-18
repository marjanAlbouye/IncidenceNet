import networkx as nx
import numpy as np
import torch
from torch.utils.data import DataLoader
from functools import partial
import utils
cuda = torch.device("cuda")
import gc



class CustomDataSet(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
       return self.data[index]


def my_collate(batch, mode, is_linear):
    adj_list = []
    feature_list = []
    labels = []
    edge_neighbor_list = []
    for item in batch:
        if mode == 0:
            adj_list.append(item['adjacency'])
            feature_list.extend(item['adj_features'])
        elif mode == 1:
            adj_list.append(item['incident'])
            feature_list.extend(item['incident_features'])
            if is_linear:
                edge_neighbor_list.append(item['edge_neigh'])

        labels.append(item['labels'])
    feature_tensor = torch.stack(feature_list, dim=0)
    labels_tensor = torch.stack(labels)
    if is_linear:
        idx = utils.get_indices(adj_list, edge_neighbor=edge_neighbor_list)
    else:
        idx = utils.get_indices(adj_list)

    return feature_tensor,idx, labels_tensor


def batch_preprocessing(dataloader):
    batch_processed_data = []
    for i, (feature_tensor, idx, target) in enumerate(dataloader):
        feature_tensor = feature_tensor.cuda()
        for k, v in idx.items():
            idx[k] = v.cuda()
        target = target.cuda()
        batch_processed_data.append((feature_tensor, idx, target))
    return batch_processed_data


def setup_data_loader(data_path, batch_size=32, mode=0, is_linear=False):
    train_molecules = np.load(data_path+'/train.npy')
    val_molecules = np.load(data_path+'/valid.npy')
    test_molecules = np.load(data_path+'/test.npy')
    info = np.load(data_path + '/info.npy')
    train_dataset = CustomDataSet(train_molecules)
    valid_dataset = CustomDataSet(val_molecules)
    test_dataset = CustomDataSet(test_molecules)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=5, collate_fn=partial(my_collate, mode=mode, is_linear=is_linear))
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=5, collate_fn=partial(my_collate, mode=mode, is_linear=is_linear))
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=5, collate_fn=partial(my_collate, mode=mode, is_linear=is_linear))

    # for batch handeling, we flatten the batch dimension and adjust the indices of adjacency matrices accordingly
    train_batch_data = batch_preprocessing(train_dataloader)
    valid_batch_data = batch_preprocessing(valid_dataloader)
    test_batch_data = batch_preprocessing(test_dataloader)
    return train_batch_data, valid_batch_data, test_batch_data, torch.tensor(info, device=cuda)
