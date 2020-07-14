import logging
from rdkit import RDLogger
from chainer_chemistry import datasets
from chainer_chemistry.dataset.preprocessors.ggnn_preprocessor import GGNNPreprocessor
from chainer_chemistry.links.scaler.standard_scaler import StandardScaler
from chainer_chemistry.datasets import NumpyTupleDataset
from sklearn import preprocessing
import numpy as np
import networkx as nx


class MoleculeObj(object):
    def __init__(self, original_adj, node_features, edge_features):
        self.original_adj = original_adj
        self.edge_features = edge_features
        self.node_features = node_features
        self.graph = nx.from_numpy_matrix(self.original_adj)
        self.edge_pairs = self.get_edge_pairs()
        self.node_adj, self.adj_features = self.get_adj_features()
        self.incident, self.incident_features = self.get_incident_features()


    def get_adj_features(self):
        full_adj = self.original_adj.copy()
        np.fill_diagonal(full_adj, 1)

        r, c = full_adj.nonzero()
        adj_features = np.zeros((len(r), self.node_features.shape[1]+self.edge_features.shape[2]))
        diag_idx = np.array(np.where(r == c)[0])
        adj_features[diag_idx, :self.node_features.shape[1]] = self.node_features

        non_diag_idx = np.array(np.where(r != c)[0])
        if len(non_diag_idx) != 0:
            adj_features[non_diag_idx, self.node_features.shape[1]:] = self.edge_features[self.edge_features.nonzero()[0], self.edge_features.nonzero()[1]]

        return full_adj, adj_features

    def get_incident_features(self):
        m = self.edge_pairs.shape[0]
        n = self.node_features.shape[0]

        incident = np.zeros((m, n))
        incident_features = []

        for i, edge in enumerate(self.edge_pairs):
            incident[i, edge[0]] = 1
            incident[i, edge[1]] = 1
            incident_features.append(np.concatenate((self.node_features[edge[0]], self.edge_features[edge[0], edge[1]])))
            incident_features.append(np.concatenate((self.node_features[edge[1]], self.edge_features[edge[0], edge[1]])))

        return incident, np.array(incident_features)

    def get_edge_pairs(self):
        if len(self.graph.edges()) == 0:
            return []
        x, y = zip(*self.graph.edges())
        num_edges = len(x)
        edge_pairs = np.ndarray(shape=(num_edges, 2), dtype=np.int32)
        edge_pairs[:, 0] = x
        edge_pairs[:, 1] = y
        return edge_pairs


# # Disable errors by RDKit occurred in preprocessing QM9 dataset.
# lg = RDLogger.logger()
# lg.setLevel(RDLogger.CRITICAL)
#
# # show INFO level log from chainer chemistry
# logging.basicConfig(level=logging.INFO)
#
# dataset_filepath = datasets.get_qm9_filepath()
#
# print('dataset_filepath =', dataset_filepath)
#
label_names = datasets.get_qm9_label_names()
print('QM9 label_names =', label_names)
#
preprocessor = GGNNPreprocessor()
dataset, dataset_smiles = datasets.get_qm9(preprocessor, labels=None, return_smiles=True)
# scaler = StandardScaler()
scaled_t = preprocessing.scale(dataset.get_datasets()[-1])
dataset = NumpyTupleDataset(*(dataset.get_datasets()[:-1]
                              + (scaled_t,)))

atom_types = 4
edge_types = 4
atom_types_dict = {6: 0,
                  7: 1,
                  8: 2,
                  9: 3}

molecule_lists = []
for index, [atom, adj, labels] in enumerate(dataset):
    print(index)
    if len(atom) == 1:
        continue
    full_adj = np.any(adj, axis=0).astype(int)
    N = adj.shape[1]
    bond_dim = adj.shape[0]
    edge_features = np.zeros((N, N, bond_dim))
    for i, bond_adj in enumerate(adj):
        edge_idx = np.nonzero(bond_adj)
        edge_features[edge_idx[0], edge_idx[1], i] = 1

    node_labels = [atom_types_dict[a] for a in atom]
    node_features = np.zeros((N, atom_types))
    node_features[np.arange(N), node_labels] = 1
    mol = MoleculeObj(full_adj, node_features, edge_features)
    molecule_dict = {
        'adjacency': mol.node_adj,
        'adj_features': mol.adj_features,
        'incident': mol.incident,
        'incident_features': mol.incident_features,
        'labels': labels
    }
    molecule_lists.append(molecule_dict)
#
np.save('qm9_processed', molecule_lists)
#
#
