import numpy as np
import cPickle
import matplotlib.pylab as pylab
import rbfnetwork
import utils
from normalizer import Normalizer
from rbfclassifier import RbfClassifier

#import utils

print "Regression 1d"

x = np.linspace(-np.pi, np.pi, 100)
x = np.array(x, ndmin=2).T
y = np.sin(x)

normalizer = Normalizer()
normalizer.calculate_from_input(x)
norm_input = normalizer.normalize(x)
print "NORM SHAPE:", norm_input.shape


net = rbfnetwork.RbfNetwork(1,1,0.2)
net.select_random_kernels(norm_input, 7)

err = net.lsqtrain(norm_input,y)
print "Err: ", err

pylab.figure()
x = np.linspace(-np.pi, np.pi, 1000)
x = np.array(x, ndmin=2).T
y = np.sin(x)
norm_input = normalizer.normalize(x)

pylab.plot(x, y)
pylab.plot(x, net.output(norm_input))
pylab.show()

print "Regression 2d"
input1 = np.linspace(0, np.pi, 100)
input2 = 0.5 + np.linspace(-np.pi/2, np.pi/2, 100)

output = np.sin(input1) * np.cos(input2)
output = np.array(output, ndmin=2).T

tot_input = np.vstack( (input1, input2) ).T

normalizer = Normalizer()
normalizer.calculate_from_input(tot_input)
norm_input =  normalizer.normalize(tot_input)

for sgh in xrange(1):
    net = rbfnetwork.RbfNetwork(2, 1, 0.1)
    utils.brute_force_training(net, (norm_input, 
                                     output+np.random.randn(*output.shape)*0.1), 
                                     (norm_input,output), 500, classifier=False)
    print "Net sigma: ", net.sigma
    print
    
netout = net(norm_input)
print "Final err: ", np.mean(np.abs(netout-output) )
 
denorm_input = normalizer.denormalize(norm_input)
print "Inputs All close? ", np.allclose(denorm_input.ravel(), tot_input.ravel())

s = cPickle.dumps(net)
net = cPickle.loads(s)
print "Pickle OK"

print "\nClassification"
inputs = np.random.rand(500,2)
outs = inputs[:,0] + inputs[:,1] > 0.5
for sgh in xrange(5):
    print "Trial: ", sgh
    classifier = RbfClassifier(2,2,0.1)
    utils.brute_force_training(classifier, (inputs,outs), (inputs,outs), 500, classifier=True,
                               )
#    print "Net sigma: ", classifier.sigma
#    print "Net kernels: ", classifier.kernels
#    print "Net weights: ", classifier.weights
    print

trials = 1000
res = 0.0
for sgh in xrange(trials):
    vals = np.random.rand(1,2)
    outcome = np.sum(vals) > 0.5
    netout = classifier.output(vals)
    res += np.abs(netout - outcome)

print "Classification success: ", 1.0 - (res / trials)

print "Done"
