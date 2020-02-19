import torch
import torch.nn as nn
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GeometricLayer(nn.Module):
    def __init__(self, in_dim, out_dim, activation=nn.Tanh, mode=0, adj_op=9, incident_op=4, bias_num=2, is_linear=False, is_sym=False):
        super(GeometricLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.activation = activation
        self.mode = mode
        self.adj_op = adj_op
        self.incident_op = incident_op
        self.bias_num = bias_num
        self.is_linear = is_linear
        self.is_sym = is_sym
        self.h_params_sym = 9
        self.h_params_asym = 15
        self.i_params_linear = 5
        self.i_params_nonlinear = 4
        stdv = 1. / math.sqrt(self.in_dim)

        # node_node_checkpoints case
        if mode == 1:
            if self.is_linear:
                self.weights = nn.Parameter(torch.Tensor(self.i_params_linear, self.in_dim, self.out_dim).data.uniform_(-stdv, stdv))
            else:
                self.weights = nn.Parameter(torch.Tensor(self.i_params_nonlinear, self.in_dim, self.out_dim).data.uniform_(-stdv, stdv))
        else:
            if self.is_sym:
                self.weights = nn.Parameter(torch.Tensor(self.h_params_sym, self.in_dim, self.out_dim).data.uniform_(-stdv, stdv))
            else:
                self.weights = nn.Parameter(torch.Tensor(self.h_params_asym, self.in_dim, self.out_dim).data.uniform_(-stdv, stdv))

    def forward(self, X, last_layer=False):
        (input_layer, idx) = X
        (S, k) = input_layer.shape

        output = torch.zeros(S, self.out_dim).to(device)

        if self.mode == 1:

            # OP 1: no pooling
            output += torch.matmul(input_layer, self.weights[0])

            # OP 2: row pool, row broadcast
            row_pool = self.pool(input_layer, idx['col'], idx['col_norm'])
            output += self.broadcast(torch.matmul(row_pool, self.weights[1]), idx['col'])

            # OP 3: col pool, col broadcast
            col_pool = self.pool(input_layer, idx['row'], idx['row_norm'])
            output += self.broadcast(torch.matmul(col_pool, self.weights[2]), idx['row'])

            # OP 4: pool all, broadcast all
            all_pool = self.pool(input_layer, idx['all'], idx['all_norm'])
            output += self.broadcast(torch.matmul(all_pool, self.weights[3]), idx['all'])

            if self.is_linear:
                edge_pool = self.gather_idx(row_pool, idx['edge_neigh'].to(torch.long))
                output += torch.matmul(edge_pool, self.weights[4])

        else:
            if self.is_sym:
                # OP 1: transpose + identity
                output += torch.matmul(input_layer, self.weights[0])

                # OP 2: Diagonal
                diagonal = self.gather_idx(input_layer, idx['diag'])
                output += self.broadcast_diag(torch.matmul(diagonal, self.weights[1]), idx['diag'], output.shape)

                # OP 3: row pool, broadcast col + row pool, broadcast row +
                # col pool, broadcast row + col pool, broadcast row
                row_pool = self.pool(input_layer, idx['col'], idx['col_norm'])
                row_pool_b_row = self.broadcast(torch.matmul(row_pool, self.weights[2]), idx['col'])
                row_pool_b_col = self.broadcast(torch.matmul(row_pool, self.weights[2]), idx['row'])
                output += row_pool_b_col + row_pool_b_row

                # OP 4: pool all, broadcast all
                all_pool = self.pool(input_layer, idx['all'], idx['all_norm'])
                output += self.broadcast(torch.matmul(all_pool, self.weights[3]), idx['all'])

                # OP 5: pool diag, broadcast diag
                diag_pool = self.pool(diagonal, idx['diag_broad'], idx['diag_norm'])
                expand_diag_pool = self.broadcast(torch.matmul(diag_pool, self.weights[4]), idx['diag_broad'])
                output += self.broadcast_diag(expand_diag_pool, idx['diag'], output.shape)

                # OP 6: pool all, broadcast diag
                expand_all_pool = self.broadcast(torch.matmul(all_pool, self.weights[5]), idx['diag_broad'])
                output += self.broadcast_diag(expand_all_pool, idx['diag'], output.shape)

                # OP 7: row pool, broadcast diag + col pool, broadcast diag
                output += self.broadcast_diag(torch.matmul(row_pool, self.weights[6]), idx['diag'], output.shape)

                # OP 8: pool diag, broadcast all
                output += self.broadcast(torch.matmul(diag_pool, self.weights[7]), idx['all'])

                # OP 9: broadcast diag to row + broadcast diag to col
                output += self.broadcast(torch.matmul(diagonal, self.weights[8]), idx['col'])

            else:
                # OP 1: transpose + identity
                transpose = self.gather_idx(input_layer, idx['trans'])
                output += torch.matmul(input_layer, self.weights[0])
                output += torch.matmul(transpose, self.weights[1])

                # OP 2: Diagonal
                diagonal = self.gather_idx(input_layer, idx['diag'])
                output += self.broadcast_diag(torch.matmul(diagonal, self.weights[2]), idx['diag'], output.shape)

                # OP 3: row pool, broadcast col + row pool, broadcast row +
                # col pool, broadcast row + col pool, broadcast row
                row_pool = self.pool(input_layer, idx['col'], idx['col_norm'])
                row_pool_b_row = self.broadcast(torch.matmul(row_pool, self.weights[3]), idx['col'])
                row_pool_b_col = self.broadcast(torch.matmul(row_pool, self.weights[4]), idx['row'])

                col_pool = self.pool(input_layer, idx['row'], idx['row_norm'])
                col_pool_b_col = self.broadcast(torch.matmul(col_pool, self.weights[5]), idx['row'])
                col_pool_b_row = self.broadcast(torch.matmul(col_pool, self.weights[6]), idx['col'])
                output += row_pool_b_col + row_pool_b_row
                output += col_pool_b_col + col_pool_b_row

                # OP 4: pool all, broadcast all
                all_pool = self.pool(input_layer, idx['all'], idx['all_norm'])
                output += self.broadcast(torch.matmul(all_pool, self.weights[7]), idx['all'])

                # OP 5: pool diag, broadcast diag
                diag_pool = self.pool(diagonal, idx['diag_broad'], idx['diag_norm'])
                expand_diag_pool = self.broadcast(torch.matmul(diag_pool, self.weights[8]), idx['diag_broad'])
                output += self.broadcast_diag(expand_diag_pool, idx['diag'], output.shape)

                # OP 6: pool all, broadcast diag
                expand_all_pool = self.broadcast(torch.matmul(all_pool, self.weights[9]), idx['diag_broad'])
                output += self.broadcast_diag(expand_all_pool, idx['diag'], output.shape)

                # OP 7: row pool, broadcast diag + col pool, broadcast diag
                output += self.broadcast_diag(torch.matmul(row_pool, self.weights[10]), idx['diag'], output.shape)
                output += self.broadcast_diag(torch.matmul(col_pool, self.weights[11]), idx['diag'], output.shape)

                # OP 8: pool diag, broadcast all
                output += self.broadcast(torch.matmul(diag_pool, self.weights[12]), idx['all'])

                # OP 14: broadcast diag to row + broadcast diag to col
                output += self.broadcast(torch.matmul(diagonal, self.weights[13]), idx['col'])
                output += self.broadcast(torch.matmul(diagonal, self.weights[14]), idx['row'])

        # expand_bias0 = self.bias[0].view(1, self.out_dim).expand(S, self.out_dim)
        # bias_diag = self.broadcast_diag(expand_bias0, idx['diag'], output.shape)

        if last_layer:
            return self.pool(output, idx['all'], idx['all_norm'])

        return output

    def gather_idx(self, X, idx):
        (_, k) = X.shape
        idx_0 = idx.shape[0]
        idx_tensor_expanded = idx.view(idx_0, 1).expand(idx_0, k)
        out = torch.gather(X, 0, idx_tensor_expanded)
        return out

    def pool(self, X, idx, norm_idx):
        (_, k) = X.shape
        n_segments = int((torch.max(idx) + 1).item())
        pooled_output = torch.zeros(n_segments, k).to(device)
        pooled_output = pooled_output.index_add(0, idx, X)
        try:
            mean_pooled_output = torch.div(pooled_output, norm_idx.view(norm_idx.shape[0], 1))
        except RuntimeError:
            print('hi')

        return mean_pooled_output

    def broadcast(self, X, idx):
        (_, k) = X.shape
        idx_0 = idx.shape[0]
        idx_tensor_expanded = idx.view(idx_0, 1).expand(idx_0, k)
        out = torch.gather(X, 0, idx_tensor_expanded)
        return out

    def broadcast_diag(self, X, idx, shape):
        (_, k) = X.shape
        out = torch.zeros(shape).to(device)
        idx_0 = idx.shape[0]
        broadcast_idx_tensor_expanded = idx.view(idx_0, 1).expand(idx_0, k)

        out = out.scatter_(0, broadcast_idx_tensor_expanded, X)
        return out

