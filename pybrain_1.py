# -*- coding: utf-8 -*-
"""
Spyder Editor

pybrain website example.
"""

from pybrain.tools.shortcuts import buildNetwork

#more algorithms
from pybrain.structure import TanhLayer
from pybrain.structure import SoftmaxLayer


#########################################################
#                Building a Network                     #
#########################################################

#2 i/p neurons, 3 hidden neurons, 1 o/p neuron
#net = buildNetwork(2, 3, 1)
net = buildNetwork(2,3,1,outclass=SoftmaxLayer)
#net is initialized with random values, generate output for a random input
op1 = net.activate([300, 350])

#print what we got
print type(op1)
print op1

print net['in']
print net['hidden0']
print net['out']

#########################################################
#                Creating a Dataset                     #
#########################################################

from pybrain.datasets import SupervisedDataSet

# 2 dimensional input and 1 dimensional target
ds = SupervisedDataSet(2,1)

#data for XOR
ds.addSample((0,0),(0,))
ds.addSample((1,1),(0,))
ds.addSample((0,1),(1,))
ds.addSample((1,0),(1,))

print len(ds)

for inpt, target in ds:
    print inpt, target

print ds['input']
print ds['target']

#########################################################
#                Train the network with the dataset     #
#########################################################
from pybrain.supervised.trainers import BackpropTrainer

trainer = BackpropTrainer(net, ds)

trainer.train()