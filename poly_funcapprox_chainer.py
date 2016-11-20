'''
Polynomial regression / Function Approximation with Chainer
'''

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

import time
from blessings import Terminal

import argparse

from sklearn.cross_validation import train_test_split

import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from chainer.functions.loss import mean_squared_error
from chainer import reporter


class QuadChain(chainer.Chain):

    def __init__(self, n_hidden_units):
        super(QuadChain, self).__init__(
                l1 = L.Linear(1, n_hidden_units),
                l2 = L.Linear(n_hidden_units, n_hidden_units),
                l3 = L.Linear(n_hidden_units, 1)
        )

    def __call__(self, x):
        h1 = F.sigmoid(self.l1(x))
        h2 = F.sigmoid(self.l2(h1))
        return self.l3(h2)


class FuncApproxer(chainer.Chain):

    def __init__(self, predictor, lossfun=mean_squared_error.mean_squared_error):
        super(FuncApproxer, self).__init__(predictor=predictor)
        self.lossfun = lossfun
        self.y = None
        self.loss = None

    def __call__(self, *args):
        assert len(args) >= 2
        x = args[:-1]
        t = args[-1]
        print('\nX = {}'.format(x[0].data))
        print('\nt = {}'.format(t.data))
        self.y = None
        self.loss = None
        self.y = self.predictor(*x)
        self.loss = self.lossfun(self.y, t)
        reporter.report({'loss': self.loss}, self)

        return self.loss


def main():
    parser = argparse.ArgumentParser(description='Second order polynomial approximation with MLP')
    parser.add_argument('--n_hidden_units', '-n', type=int, default=3, 
            help='Number of hidden units per hidden layer')
    parser.add_argument('--batch_size', '-b', type=int, default=1,
            help='Batch size for each epoch')
    parser.add_argument('--n_epochs', '-e', type=int, default=1000,
            help='Number of epochs to run')
    args = parser.parse_args()

    print('Number of hidden units: {}'.format(args.n_hidden_units))
    print('Number of epocsh: {}'.format(args.n_epochs))
    print('Batch size: {}'.format(args.batch_size))
    print('')

    model = FuncApproxer(QuadChain(args.n_hidden_units))
    optimizer = chainer.optimizers.Adam()
    optimizer.use_cleargrads()
    optimizer.setup(model)

    ''' create train/test data '''
    x = np.random.rand(1000).astype(np.float32)
    y = 7*x**2 - 8*x +10
    y += 6*np.random.rand(1000).astype(np.float32)
    X_tr, X_te, y_tr, y_te = train_test_split(x, y)
    train = [(X_tr[i], y_tr[i]) for i in range(X_tr.shape[0])]
    test  = [(X_te[i], y_te[i]) for i in range(X_te.shape[0])]

    ''' create train and test iterators for chainer optimizer '''
    train_iter = chainer.iterators.SerialIterator(train, args.batch_size)
    test_iter = chainer.iterators.SerialIterator(test, args.batch_size, repeat=False, shuffle=False)

    ''' setup trainer '''
    updater = training.StandardUpdater(train_iter, optimizer)
    trainer = training.Trainer(updater, (args.n_epochs, 'epoch'))

    ''' evaluate the model with test data for each epoch '''
    trainer.extend(extensions.Evaluator(test_iter, model))

    ''' dump loss '''
    trainer.extend(extensions.dump_graph('main/loss'))

    trainer.extend(extensions.LogReport())

    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

    trainer.extend(extensions.ProgressBar())

    ''' Run training '''
    trainer.run()

if __name__ == "__main__":
    main()
