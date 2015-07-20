# -*- coding: utf-8 -*-
"""
Spyder Editor

pybrain website example.
"""

from pybrain.tools.shortcuts import buildNetwork

#more algorithms
#from pybrain.structure import TanhLayer
#from pybrain.structure import SoftmaxLayer
from pybrain.structure import LinearLayer

#########################################################
#                Building a Network                     #
#########################################################

#2 i/p neurons, 3 hidden neurons, 1 o/p neuron
#net = buildNetwork(2, 3, 1)
net = buildNetwork(2,4,1,outclass=LinearLayer)

print net['in']
print net['hidden0']
print net['out']

net.reset()
#########################################################
#                Creating a Dataset                     #
#########################################################

from pybrain.datasets import SupervisedDataSet

# 2 dimensional input and 1 dimensional target
ds = SupervisedDataSet(2,1)

#dummy data, output constant
ds.addSample((0,0),(0,))
ds.addSample((1,1),(0,))
ds.addSample((0,1),(0,))
ds.addSample((1,0),(0,))

ds.addSample((4,4),(0,))
ds.addSample((2,2),(0,))
ds.addSample((1,4),(0,))
ds.addSample((7,6),(0,))

ds.addSample((3,4),(0,))
ds.addSample((8,2),(0,))
ds.addSample((9,4),(0,))
ds.addSample((7,7),(0,))

ds.addSample((4,77),(0,))
ds.addSample((2,88),(0,))
ds.addSample((90,4),(0,))
ds.addSample((7,16),(0,))

#########################################################
#                Train the network with the dataset     #
#########################################################
from pybrain.supervised.trainers import BackpropTrainer

trainer = BackpropTrainer(net, ds)

#trainer.trainUntilConvergence()
trainer.train()

print net.activate([1, 0])
print net.activate([0, 1])
print net.activate([1, 1])
print net.activate([0, 0])

#########################################################
#                Save the model, load and test again    #
#########################################################

from pybrain.tools.customxml import NetworkWriter
from pybrain.tools.customxml import NetworkReader

NetworkWriter.writeToFile(net, "savedModel.xml")
net2 = NetworkReader.readFrom("savedModel.xml")

print net2.activate([1, 0])
print net2.activate([0, 1])
print net2.activate([1, 1])
print net2.activate([0, 0])