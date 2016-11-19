'''
Polynomial regression / Function Approximation with Chainer
'''

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

import time
from blessings import Terminal

import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

class display_train(object):
    def __init__(self):
        self.train_log = []
        self.header = ''
        self.term = Terminal()
        self.W = self.term.width
        self.H = self.term.height - 7 ### because we are using up three lines already

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

        time.sleep(0.1)

        ''' reset train_log data after printing'''
        self.train_log = []

class QuadChain(Chain):
    def __init__(self):
        super(QuadChain, self).__init__(
                l1 = L.Linear(1,3),
                l2 = L.Linear(3,3),
                l3 = L.Linear(3,1)
                # l4 = L.Linear(4,2),
                # l5 = L.Linear(2,1)
                )

    def __call__(self,x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        # h1 = F.relu(self.l1(x))
        # h2 = F.relu(self.l2(h1))
        # h3 = F.relu(self.l3(h2))
        # h4 = F.relu(self.l4(h3))
        # return F.relu(self.l2(h1))
        return self.l3(h2)

def linear_train(train_data, train_target, n_epochs=1000):
    dt = display_train()
    dt.header = 'Epoch\tLoss'

    for each_epoch in range(n_epochs):
        out_lin_reg = model(train_data)
        reg_loss = F.mean_squared_error(train_target,out_lin_reg)
        model.cleargrads()
        reg_loss.backward()
        optimizer.update()

        logstr = str("{}\t{}".format(each_epoch,reg_loss.data))

        dt.train_log.append(logstr)
        if (len(dt.train_log) == dt.H):
            dt.display()

if __name__ == "__main__":
    ### simple linear regression
    x = np.random.rand(1000).astype(np.float32)
    y = 7*x**2 - 8*x +10
    # y += 27*np.random.rand(1000).astype(np.float32)

    model = QuadChain()
    optimizer = optimizers.AdaDelta(rho=0.9)
    # optimizer = optimizers.MomentumSGD()
    # optimizer = optimizers.Adam()
    optimizer.use_cleargrads()
    optimizer.setup(model) 

    try: 
        x_var = Variable(x.reshape(1000,-1))
        y_var = Variable(y.reshape(1000,-1))
        linear_train(x_var,y_var)
    finally:
        x_test = np.random.rand(1000).astype(np.float32)
        x_test_var = Variable(x_test.reshape(1000,-1))
        y_actual = 7*x_test**2 - 8*x_test +10
        y_actual_var = Variable(y_actual.reshape(1000,-1))
        y_predicted = model(x_test_var)
        test_loss = F.mean_squared_error(y_actual_var, y_predicted)
        print("\nMSE (y_actual, y_predicted) = {}".format(test_loss.debug_print()))

