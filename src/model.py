import torch.nn as nn
import geometric_layer


class Regressor(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, device, activation=nn.ReLU(), mode=0, is_linear=False, is_sym=False, dropout=0.6):
        super(Regressor, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.activation = activation
        self.out_dim = out_dim
        self.mode = mode
        self.dropout = dropout
        self.num_layers = num_layers
        self.is_linear = is_linear
        self.is_sym = is_sym

        self.layers = nn.ModuleList([])
        self.bns = nn.ModuleList([])
        L = geometric_layer.GeometricLayer(self.in_dim, self.hidden_dim, self.device, mode=self.mode, is_linear=self.is_linear, is_sym=self.is_sym)
        self.layers.append(L)
        for i in range(self.num_layers-2):
            L2 = geometric_layer.GeometricLayer(self.hidden_dim, self.hidden_dim, self.device, mode=self.mode, is_linear=self.is_linear, is_sym=self.is_sym)
            BN = nn.BatchNorm1d(self.hidden_dim, track_running_stats=True)
            self.bns.append(BN)
            self.layers.append(L2)
        L3 = geometric_layer.GeometricLayer(self.hidden_dim, self.out_dim, self.device, mode=self.mode, is_linear=self.is_linear, is_sym=self.is_sym)
        self.layers.append(L3)

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



