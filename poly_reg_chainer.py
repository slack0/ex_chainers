'''
Polynomial regression / Function Approximation with Chainer
'''

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

import time
import argparse

from blessings import Terminal

import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

from sklearn.cross_validation import train_test_split

class display_train(object):
    def __init__(self):
        self.train_log = []
        self.header = ''
        self.term = Terminal()
        self.W = self.term.width
        self.H = self.term.height - 7

        ''' enter fullscreen here '''
        print(self.term.enter_fullscreen)

    def display(self):
        print(self.term.clear)
        ''' get the length of a sample string in train_log for formatting header '''

        print(self.term.bold_red + '='*self.W + self.term.normal)
        print(self.term.bold + self.header + self.term.normal)
        print(self.term.bold_red + '='*self.W + self.term.normal)
        for row in self.train_log:
            print(row)

        time.sleep(3)

        ''' reset train_log data after printing'''
        self.train_log = []

class QuadChain(Chain):
    def __init__(self, n_hidden_units):
        super(QuadChain, self).__init__(
                l1 = L.Linear(1,n_hidden_units),
                l2 = L.Linear(n_hidden_units,n_hidden_units),
                l3 = L.Linear(n_hidden_units,1)
                )

    def __call__(self,x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return F.relu(self.l3(h2))


def main():
    parser = argparse.ArgumentParser(description='Second order polynomial approximation with MLP')
    parser.add_argument('--n_hidden_units', '-n', type=int, default=3, help='Number of hidden units per hidden layer')
    parser.add_argument('--batch_size', '-b', type=int, default=25, help='Batch size for each epoch')
    parser.add_argument('--n_epochs', '-e', type=int, default=1000, help='Number of epochs to run')
    args = parser.parse_args()

    print('Number of hidden units: {}'.format(args.n_hidden_units))
    print('Number of epocsh: {}'.format(args.n_epochs))
    print('Batch size: {}'.format(args.batch_size))
    print('')

    ''' generate data '''
    x = np.random.rand(1000).astype(np.float32)
    y = 7*x**2 - 4*x +10
    y += 27*np.random.rand(1000).astype(np.float32)

    N = args.batch_size * 30
    x_train, x_test = np.split(x, [N])
    y_train, y_test = np.split(y, [N])

    ''' instantiate the model and setup optimizer '''
    model = QuadChain(args.n_hidden_units)
    # optimizer = optimizers.AdaDelta(rho=0.9)
    optimizer = optimizers.MomentumSGD()
    # optimizer = optimizers.Adam()
    optimizer.use_cleargrads() 
    optimizer.setup(model) 

    ''' 
    -- prepare test data here
    -- train data needs to be shuffled for each epoch; so we will deal with it there
    '''
    test_data = Variable(x_test.reshape(x_test.shape[0],-1))
    test_target = Variable(y_test.reshape(y_test.shape[0],-1))

    ''' 
    - start training 
    - for each epoch, iterate over each mini batches and perform model update
    - at the end of each mini batch, calculate test loss
    '''

    dt = display_train()
    dt.header = 'Epoch\tMini Batch\tTraining Loss\tTest Loss'

    for each_epoch in xrange(1, args.n_epochs+1):
        permuted_ordering = np.random.permutation(N)

        for mini_batch_index in xrange(0, N, args.batch_size):
            x_batch = x_train[permuted_ordering[mini_batch_index : mini_batch_index + args.batch_size]]
            y_batch = y_train[permuted_ordering[mini_batch_index : mini_batch_index + args.batch_size]]

            train_data = Variable(x_batch.reshape(x_batch.shape[0],-1))
            train_target = Variable(y_batch.reshape(y_batch.shape[0],-1))

            train_pred = model(train_data)
            train_loss = F.mean_squared_error(train_target, train_pred) 

            model.cleargrads()
            train_loss.backward()
            optimizer.update() 


            ''' calculate test loss after this mini batch optimizer/network update '''
            test_pred = model(test_data)
            test_loss = F.mean_squared_error(test_target, test_pred)

            logstr = str("{}\t{}\t{}\t{}".format(each_epoch, mini_batch_index, train_loss.data, test_loss.data))

            dt.train_log.append(logstr)
            if (len(dt.train_log) == dt.H):
                dt.display()


if __name__ == "__main__":
    main()
