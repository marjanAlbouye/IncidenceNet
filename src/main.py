#!/usr/bin/env python
import math
import torch
import torch.nn as nn
import torch.optim as optim
import os
import model
import data_loader
from LogMetric import AverageMeter
import utils

from tqdm import tqdm
from random import shuffle
import random
import numpy as np
import time
import datetime
import argparse



validation_freq = 3
save_freq = 5
in_dim = 21
out_dim = 1
cuda = torch.device("cuda")


class Qm9Trainer(object):
    def __init__(self, num_layers, batch_size, log_path, data_path):
        self.learning_rate = args.lr
        self.hidden_dim = args.hidden
        self.num_epochs = args.epochs
        self.reg = args.reg
        self.mode = args.mode
        self.is_linear = args.is_linear
        self.is_sym = args.is_sym
        self.target_index = args.target_index

        self.batch_size = batch_size
        self.num_layers = num_layers
        self.log_path = log_path
        self.data_path = data_path

        # preparing data that contains the node/edge features + indices from adjacency matrix
        self.train_data_loader, self.valid_data_loader, self.test_data_loader, self.std = data_loader.setup_data_loader(self.data_path,
                                                                                                              batch_size=self.batch_size,
                                                                                                              mode=self.mode, is_linear=self.is_linear)

        # Setup the model
        self.model = model.Regressor(in_dim=in_dim, hidden_dim=self.hidden_dim, out_dim=out_dim,
                                     num_layers=self.num_layers, mode=self.mode, is_linear=self.is_linear, is_sym=self.is_sym).cuda()
        self.model = nn.DataParallel(self.model).to(cuda)

        self.L = nn.MSELoss().cuda()
        self.optimizer = optim.Adam([{'params': self.model.parameters()}], lr=self.learning_rate, weight_decay=self.reg)

        self.evaluation = lambda output, target: torch.mean(torch.abs(output - target))

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=5, cooldown=0,
                                                                    min_lr=1e-7, factor=.5)

    def main(self, f):
        best_error = math.inf
        best_epoch = 0

        for i in range(self.num_epochs):
            print('#' * 20 + ' epoch {0} '.format(i + 1) + '#' * 20)
            f.write('\n' + '#' * 20 + ' epoch {0} '.format(i + 1) + '#' * 20 + '\n')
            start_time = time.time()
            train_loss, train_error, new_loss = self.train()
            f.write('Train : Loss {0: .5f}, MAE {1: .5f}, New_loss {2: .5f} \n'.format(train_loss, train_error, new_loss))
            val_loss, val_error, new_val_error = self.validate()
            end_time = time.time()-start_time
            f.write('Val : Loss {0: .5f}, MAE {1: .5f}, new_MAE {2:.5f} \n'.format(val_loss, val_error, new_val_error))
            f.write('Time: {0}'.format(end_time))
            is_best = new_val_error < best_error
            if is_best:
                best_epoch = i
            best_error = min(new_val_error, best_error)
            utils.save_checkpoint({'epoch': i + 1, 'state_dict': self.model.state_dict(), 'best_er1': best_error,
                                  'optimizer': self.optimizer.state_dict(), }, is_best=is_best, directory=self.log_path)
            f.flush()
        f.write('************** Final Results ******************* \n')
        f.write('best validation error:{} , best epoch: {}\n'.format(best_error, best_epoch))
        f.flush()

    def train(self):
        self.model.train()

        data_counter = 0

        losses = AverageMeter()
        error_ratio = AverageMeter()
        loss_all = 0.

        for j, (feature_tensor, idx, target) in enumerate(self.train_data_loader):
            theta = random.uniform(-np.pi, np.pi)
            feature_tensor[:, 0:3] = utils.rotate_z(theta, feature_tensor[:, 0:3])
            data_counter += target.shape[0]
            self.optimizer.zero_grad()

            model_output = self.model((feature_tensor, idx)).squeeze()
            target = target.squeeze()[:, self.target_index]

            train_loss = self.L(model_output, target)
            loss_all += train_loss * target.shape[0]
            losses.update(train_loss, target.shape[0])
            error_ratio.update(self.evaluation(model_output, target), target.shape[0])

            train_loss.backward()
            self.optimizer.step()

            del train_loss

        loss_all = loss_all / data_counter

        print('Loss {loss.avg: .5f}, MAE {error.avg: .5f}, Loss all {loss_all:.5f}'.format(loss=losses, error=error_ratio, loss_all=loss_all))

        self.scheduler.step(losses.avg.item())
        return losses.avg, error_ratio.avg, loss_all

    def validate(self):
        print(' *Validation*')
        self.model.eval()
        error = 0
        data_counter = 0
        with torch.no_grad():

            val_losses = AverageMeter()
            val_error_ratio = AverageMeter()

            for j, (feature_tensor, idx, target) in enumerate(self.valid_data_loader):
                data_counter += target.shape[0]

                model_output = self.model((feature_tensor, idx)).squeeze()
                target = target.squeeze()[:, self.target_index]

                error += ((model_output * self.std[self.target_index]) -
                  (target* self.std[self.target_index])).abs().sum().item()

                val_loss = self.L(model_output, target)
                val_losses.update(val_loss, target.shape[0])
                val_error_ratio.update(self.evaluation(model_output, target),
                                        target.shape[0])

                del val_loss
            error = error / data_counter

            print(' Val Loss {loss.avg: .5f}, Val MAE {error.avg: .5f}, New MAE {n_error: .5f}, New MAE Norm {n_norm: .5f}'
                  .format(loss=val_losses, error=val_error_ratio, n_error=error, n_norm= error/self.std[self.target_index]))
            return val_losses.avg, val_error_ratio.avg, error

    def test(self):

        self.model.eval()
        error = 0
        data_counter = 0
        with torch.no_grad():

            test_losses = AverageMeter()
            test_error_ratio = AverageMeter()

            for j, (feature_tensor, idx, target) in enumerate(self.test_data_loader):
                data_counter += target.shape[0]

                model_output = self.model((feature_tensor, idx)).squeeze()
                target = target.squeeze()[:, self.target_index]
                error += ((model_output * self.std[self.target_index]) -
                          (target * self.std[self.target_index])).abs().sum().item()

                test_loss = self.L(model_output, target)
                test_losses.update(test_loss, target.shape[0])
                test_error_ratio.update(self.evaluation(model_output, target),
                                        target.shape[0])

                del test_loss
        error = error / data_counter
        print('test Loss {loss.avg: .5f}, test MAE {error.avg: .5f}, New MAE {n_error: .5f}, New MAE Norm {n_norm: .5f}'
              .format(loss=test_losses, error=test_error_ratio, n_error=error, n_norm= error/self.std[self.target_index]))
        return test_losses.avg, test_error_ratio.avg, error


if __name__ == '__main__':

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden', type=int, default=30)
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--mode', type=int, default=0)
    parser.add_argument('--reg', type=float, default=1e-5)
    parser.add_argument('--target_index', type=int, default=0)
    parser.add_argument('--log_path', type=str, default='../results/homo_checkpoints/t0_mu/')
    parser.add_argument('--data_path', type=str, default='../data/qm9_dense_h')
    parser.add_argument('--is_linear', type=int, default=0)
    parser.add_argument('--is_sym', type=int, default=0)
    parser.add_argument('--graph_type', type=str, default='dense')

    args = parser.parse_args()

    # create log file
    now = datetime.datetime.now()
    sym = 'symm' if args.is_sym else 'non-symm'
    lin = 'linear' if args.is_linear else 'non-linear'
    log_file_name = args.graph_type + '_' + sym + '_' +lin + '_' + str(now)
    print('log_file_name: ', log_file_name)
    log_path = args.log_path + log_file_name
    if not os.path.isdir(log_path):
        os.makedirs(log_path)
    file_name = log_path + '/' + 'log.txt'
    f = open(file_name, "w+")
    for arg in vars(args):
        f.write('- {} : {}\n'.format(arg, getattr(args, arg)))
        print('- {} : {}'.format(arg, getattr(args, arg)))
    f.flush()
    print('*' * 20 + 'Start Running' + '*' * 20)
    print('-----\n num_layers: {} - batch_size: {} - hidden: {}\n'.format(args.layers, args.batch_size, args.hidden))

    t = Qm9Trainer(args.layers, args.batch_size, log_path, args.data_path)

    t.main(f)

    print('*' * 20 + 'Finish Running' + '*' * 20)
    f.write('====== Finish Training =======\n')
    best_model_file = os.path.join(t.log_path, 'model_best.pth')
    checkpoint = torch.load(best_model_file)
    t.model.load_state_dict(checkpoint['state_dict'])
    t.optimizer.load_state_dict(checkpoint['optimizer'])
    test_loss, test_error, new_test_error = t.test()
    f.write('test Loss {0: .5f}, test MAE {1: .5f}, new_test MAE {2: .5f}, new_test_norm {3:.5f} \n'
            .format(test_loss, test_error, new_test_error, new_test_error/t.std[t.target_index]))
    f.close()
