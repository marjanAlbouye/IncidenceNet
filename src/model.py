import torch.nn as nn
import torch
import geometric_layer
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ChannelDropout(nn.Module):
    def __init__(self, p=0.5):
        super(ChannelDropout, self).__init__()
        self.p = p
        self.D = nn.Dropout2d(p)

    def forward(self, x):
        if self.training:
            (S, K) = x.shape
            u = np.random.binomial(1, 1-self.p, size=K)
            u_tensor = torch.tensor(np.repeat(u.reshape(1, K), S, axis=0).reshape(S, K), dtype=torch.float).to(device)
            return (x*u_tensor) / (1-self.p)
        else:
            return x


class Regressor(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, activation=nn.ReLU(), mode=0, is_linear=False, is_sym=False, dropout=0.6):
        super(Regressor, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.out_dim = out_dim
        self.mode = mode
        self.dropout = dropout
        self.num_layers = num_layers
        self.is_linear = is_linear
        self.is_sym = is_sym
        print('is_linear: ', self.is_linear)
        print('is_symmetric:', self.is_sym)


        # self.L_in = geometric_layer.GeometricLayer(self.in_dim, self.hidden_dim, mode=self.mode)
        # self.L_rec = geometric_layer.GeometricLayer(self.hidden_dim, self.hidden_dim, mode=self.mode)
        # self.L_out = geometric_layer.GeometricLayer(self.hidden_dim, self.out_dim, mode=self.mode)
        # #self.Linear = nn.Linear(self.hidden_dim, self.out_dim)
        # self.BN = nn.BatchNorm1d(self.hidden_dim, track_running_stats=False)

        # L = geometric_layer.GeometricLayer(self.in_dim, self.hidden_dim, mode=self.mode)
        # self.layers = nn.ModuleList([L])
        # for i in range(1,12):
        #     L = geometric_layer.GeometricLayer(self.hidden_dim*(i) + self.in_dim, self.hidden_dim, mode=self.mode)
        #     self.layers.append(L)

        # for i in range(5):
        #     L = geometric_layer.GeometricLayer(self.hidden_dim*(i+1) + self.in_dim, self.hidden_dim, mode=self.mode)
        #     self.layers.append(L)

        # for i in range(4):
        #     L = geometric_layer.GeometricLayer(self.hidden_dim*(i+3) + self.in_dim, self.hidden_dim, mode=self.mode)
        #     self.layers.append(L)

        # L = geometric_layer.GeometricLayer(self.hidden_dim*(12) + self.in_dim, self.out_dim, mode=self.mode)
        # self.layers.append(L)

        self.layers = nn.ModuleList([])
        self.bns = nn.ModuleList([])
        L = geometric_layer.GeometricLayer(self.in_dim, self.hidden_dim, mode=self.mode, is_linear=self.is_linear, is_sym=self.is_sym)
        self.layers.append(L)
        for i in range(self.num_layers-2):
            L2 = geometric_layer.GeometricLayer(self.hidden_dim, self.hidden_dim, mode=self.mode, is_linear=self.is_linear, is_sym=self.is_sym)
            BN = nn.BatchNorm1d(self.hidden_dim, track_running_stats=True)
            self.bns.append(BN)
            self.layers.append(L2)
        L3 = geometric_layer.GeometricLayer(self.hidden_dim, self.out_dim, mode=self.mode, is_linear=self.is_linear, is_sym=self.is_sym)
        self.layers.append(L3)

        print(self)

    def forward(self, X_input):
        (input_layer, idx) = X_input

        inp = input_layer
        for l, lay in enumerate(self.layers):
            inp = lay((inp,idx), last_layer = (l==(len(self.layers)-1)))
            if l < len(self.layers) - 1:
                if l > 0:
                    inp = self.bns[(l-1)](inp)
                inp = self.activation(inp)

        return inp



