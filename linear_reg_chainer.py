'''
Trying linear regression with Chainer!

--  Added blessings (curses wrapper) based printing of
    loss and weight updates as the training progresses

--  Turn-off sleep timer to make the run go fast
'''

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt


from chainer import FunctionSet, Variable, optimizers, serializers
import chainer.functions as F
import chainer.links as L


import time
from blessings import Terminal


class display_train(object):
    def __init__(self):
        self.train_log = []
        self.header = ''
        self.term = Terminal()
        self.W = self.term.width
        self.H = self.term.height-7

        ''' enter fullscreen here '''
        print(self.term.enter_fullscreen)

    def display(self):
        print(self.term.clear)
        ''' get the length of a sample string in train_log '''
        num_chars = len(self.train_log[0])

        print(self.term.bold_red + '='*2*num_chars + self.term.normal)
        print(self.term.bold + self.header + self.term.normal)
        print(self.term.bold_red + '='*2*num_chars + self.term.normal)
        for row in self.train_log:
            print(row)

        time.sleep(0.5)

        ''' reset train_log data after printing'''
        self.train_log = []


def linear_forward(data):
    return linear_function(data)


def linear_train(train_data, train_target, n_epochs=1000):
    dt = display_train()
    dt.header = 'Epoch\tLoss\tW\tb'
    for each_epoch in range(n_epochs):
        out_lin_reg = linear_forward(train_data)
        reg_loss = F.mean_squared_error(train_target, out_lin_reg)
        linear_function.cleargrads()
        reg_loss.backward()
        optimizer.update()
        logstr = str("{:d}\t{:f}\t{:f}\t{:f}".format(
            each_epoch,
            np.float32(reg_loss.data),
            np.float32(linear_function.W.data[0, 0]),
            np.float32(linear_function.b.data[0]))
        )
        dt.train_log.append(logstr)
        if (len(dt.train_log) == dt.H):
            dt.display()


if __name__ == "__main__":
    ''' simple linear regression '''
    x = 30*np.random.rand(1000).astype(np.float32)
    y = 7*x+10
    y += 5*np.random.rand(1000).astype(np.float32)

    linear_function = L.Linear(1, 1)

    x_var = Variable(x.reshape(1000, -1))
    y_var = Variable(y.reshape(1000, -1))

    optimizer = optimizers.MomentumSGD(lr=0.002)
    optimizer.setup(linear_function)

    try:
        linear_train(x_var, y_var)
    finally:
        slope = linear_function.W.data[0, 0]
        intercept = linear_function.b.data[0]
        print "y = {} * x + {}".format(slope, intercept)
