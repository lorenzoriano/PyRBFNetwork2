import numpy
import random
from rbfnetwork import RbfNetwork

def classes2matrix(classes):
    """
    Convert a vector of classes to a matrix suitable for
    RBF use.

    Each class is an integer number, including zero.
    The number N of classes is taken as max(classes) + 1.
    The output is a MxN matrix, where M is len(classes)
    """

    classes = numpy.asarray(classes,  dtype=numpy.int).flatten()
    M = len(classes)
    N = numpy.max(classes) + 1
    out = numpy.zeros( (M,  N),  dtype=numpy.float32)
    out[ numpy.arange(M),  classes] = 1
    return out

def matrix2classes(mat):
    return numpy.argmax(mat, 1).reshape((mat.shape[0], 1))

class RbfClassifier(RbfNetwork):

    def __init__(self,  input_size, num_classes, sigma):
        if num_classes <= 1:
            raise ValueError("num_classes must be > 1")
        super(RbfClassifier,self).__init__(input_size,  num_classes, sigma)
        self.__indeces = []

    def select_random_kernels(self,  input,  size):
        newinput = numpy.asarray(input)
        if newinput.ndim != 2:
            raise ValueError("input has to be a matrix")
        if newinput.shape[1] != self.input_size:
            raise ValueError("input dimension differs from the RBF one")
        
        indeces = random.sample(xrange(newinput.shape[0]), size) 

        self.__indeces = list(indeces)
        self.kernels = newinput[self.__indeces,  :]

    def lsqtrain(self,  input,  output):

        newinput = numpy.asarray(input)
        newoutput = classes2matrix(output)

        if newinput.ndim != newoutput.ndim:
            raise ValueError("input and output must have the same shape")

        if newinput.ndim != 2:
            raise ValueError("input has to be a matrix")

        if newinput.shape[1] != self.input_size:
            raise ValueError("input dimension differs from the RBF one")
        if newoutput.shape[1] != self.output_size:
            raise ValueError("output dimension differs from the RBF one")

        if newoutput.shape[0] != newinput.shape[0]:
            raise ValueError(
                "input and output must have the same number of rows ")

        if newoutput.shape[1] <=0:
            raise ValueError("output has <=0 columns")
        if newinput.shape[1] <=0:
            raise ValueError("input has <=0 columns")

        newweights = numpy.vstack( (numpy.zeros((1,newoutput.shape[1])), newoutput[self.__indeces,  :]) )
        self.weights = newweights

        netout = self.output(newinput)
        return netout != self.matrix2classes(newoutput)

    def output(self,  input):
        output = RbfNetwork.output(self,  input)
        return self.matrix2classes(output)

    def raw_output(self,  input):
        return RbfNetwork.output(self,  input)

    def __call__(self, input):
        return self.output(input)

    def output_conf(self, input):
        output, conf = RbfNetwork.output_conf(self,  input)
        return self.matrix2classes(output), conf

    def safe_output(self, input, default_class = 0, thr = 0.5):
        out, conf = self.output_conf(input)
        out[conf<thr] = default_class
        return out
        

    @staticmethod
    def matrix2classes(m):
        return matrix2classes(m)

    @staticmethod
    def classes2matrix(c):
        return classes2matrix