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

# class display_train(object):
#     def __init__(self):
#         self.train_log = []
#         self.header = ''
#         self.term = Terminal()
#         self.W = self.term.width
#         self.H = self.term.height - 7 ### because we are using up three lines already

#         ''' enter fullscreen here '''
#         print(self.term.enter_fullscreen)

#     def display(self):
#         print(self.term.clear)
#         ''' get the length of a sample string in train_log for formatting header '''
#         num_chars = len(self.train_log[0])

#         print(self.term.bold_red + '='*num_chars + self.term.normal)
#         print(self.term.bold + self.header + self.term.normal)
#         print(self.term.bold_red + '='*num_chars + self.term.normal)
#         for row in self.train_log:
#             print(row)

#         time.sleep(0.5)

#         ''' reset train_log data after printing'''
#         self.train_log = []

class QuadChain(Chain):
    def __init__(self):
        super(QuadChain, self).__init__(
                l1 = L.Linear(1,1),
                l2 = L.Linear(1,1),
                l3 = L.Linear(1,1),
                l4 = L.Linear(1,1)
                )

    def __call__(self,x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        h3 = F.relu(self.l3(h2))
        return F.relu(self.l4(h3))

def linear_train(train_data, train_target, n_epochs=10000):
    # dt = display_train()
    # dt.header = 'Epoch\tLoss\tW\tb'

    print "received train_data..."
    print(train_data.data)

    for each_epoch in range(n_epochs):
        ''' shuffle train and test '''
        shuffle_input = np.concatenate(train_data.data, train_target.data)
        np.random.shuffle(shuffle_input)
        train_data.data = shuffle_input[:,0]
        train_target.data = shuffle_input[:,1]
        out_lin_reg = model(train_data)
        # print "output of model..."
        # print(out_lin_reg.data)

        reg_loss = F.mean_squared_error(train_target,out_lin_reg)
        # print "mean_squared_error ..."
        # print(reg_loss.data)

        # print "updating model ..."
        model.cleargrads()
        reg_loss.backward()
        optimizer.update()

        logstr = str("{}\t{}\t{}\t{}".format(each_epoch,np.float32(reg_loss.data),np.float32(model.l1.W.data[0,0]),np.float32(model.l1.b.data[0])))
        print(logstr)
        # dt.train_log.append(logstr)
        # if (len(dt.train_log) == dt.H):
        #     dt.display()

if __name__ == "__main__":
    ### simple linear regression
    x = np.random.rand(1000).astype(np.float32)
    y = 7*x**2 - 8*x +10
    #y += 50*np.random.rand(1000).astype(np.float32)

    model = QuadChain()
    optimizer = optimizers.AdaDelta(rho=0.9)
    optimizer.use_cleargrads()
    optimizer.setup(model) 

    try: 
        x_var = Variable(x.reshape(1000,-1))
        y_var = Variable(y.reshape(1000,-1))

        print(x_var.data)
        print "calling linear_train....\n\n"

        linear_train(x_var,y_var)
    finally:
        slope = model.l1.W.data[0,0]
        intercept = model.l1.b.data[0]

