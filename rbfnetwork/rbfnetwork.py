import numpy as np
from exceptions import ValueError
import random
import libfunctions

class RbfNetwork(object):
    def __init__(self, input_size, output_size, sigma):
        self.input_size = input_size
        self.output_size = output_size
        self.sigma = sigma
        
        self.kernels = None
        self.weights = None
        
    def select_random_kernels(self, input, size):
        '''
        Select a number of rows from the input matrix to be the kernels for 
        the RBFNetwork. The weights matrix will be reset.
        
        @param input: The matrix the kernels will be taken from. Its number of
        columns must be equal to the network input_size
        @param size: The number of rows to take from the input matrix. It must
        be between one and the number of rows of input
        '''
        
        newinput = np.asarray(input)
        if newinput.ndim == 1: #dealing with a vector
            newinput = newinput.reshape( (newinput.shape[0], 1))
        elif newinput.ndim != 2: #matrix
            raise ValueError("input has to be either a vector or a matrix")

        if size > newinput.shape[0]:
            raise IndexError("asking for more elements that in input")

        if newinput.shape[1] != self.input_size:
            print newinput.shape
            raise ValueError("input dimension differs from the RBF one")
        
        self.kernels = np.empty(shape = (size, self.input_size), dtype=np.double,
                                order='C')
        self.weights = np.empty(shape= (size + 1, self.output_size), dtype=np.double,
                                order='C')
        
        draws = random.sample(xrange(newinput.shape[0]), size)
        
        self.kernels = input[draws, :].copy()
        
    def first_layer_output(self, input):
        
        num_inputs = input.shape[0]
        num_kernels = self.kernels.shape[0]
        res = np.ndarray(shape=(num_inputs, 1+self.kernels.shape[0]))
        
        libfunctions.first_layer_output(input, self.kernels, res,
                                        num_kernels, num_inputs, 
                                        self.input_size, self.sigma)
        return res
    
    def output(self, input):
        newinput = np.asarray(input)
        if newinput.ndim == 1: #dealing with a vector
            newinput = newinput.reshape( (newinput.shape[0], 1))
        elif newinput.ndim != 2: #matrix
            raise ValueError("input has to be either a vector or a matrix")
        if newinput.shape[1] != self.input_size:
            raise ValueError("input dimension differs from the RBF one")
        
        res = self.first_layer_output(newinput)
        return np.dot(res, self.weights)
    
    def __call__(self, input):
        return self.output(input)
    
    def lsqtrain(self, input, output):
        """
        Perform least sqare training over input/outputs

        input/output has to be a 2d ndvector, and every row should be a  multi-dimensional variable
        Returns an ndarray of the same size of input/output
        """
        
        newinput = np.asarray(input)
        if newinput.ndim == 1: #dealing with a vector
            newinput = newinput.reshape( (newinput.shape[0], 1))
        elif newinput.ndim != 2: #matrix
            raise ValueError("input has to be either a vector or a matrix")
        if newinput.shape[1] != self.input_size:
            raise ValueError("input dimension differs from the RBF one")
        
        newoutput = np.asarray(output)
        if newoutput.ndim == 1: #dealing with a vector
            newoutput = newoutput.reshape( (newoutput.shape[0], 1))
        elif newoutput.ndim != 2: #matrix
            raise ValueError("input has to be either a vector or a matrix")
        if newoutput.shape[1] != self.output_size:
            raise ValueError("input dimension differs from the RBF one")

        if newinput.ndim != newoutput.ndim:
            raise ValueError("input and output must have the same shape")
        if newoutput.shape[0] != newinput.shape[0]:
            raise ValueError("input and output must have the same number of rows ")
        
        A = self.first_layer_output(input)
        b = output
        
        self.weights, errs, _, _ = np.linalg.lstsq(A, b)
        
        return errs
        
    def output_conf(self, input):
        newinput = np.asarray(input)
        if newinput.ndim == 1: #dealing with a vector
            newinput = newinput.reshape( (newinput.shape[0], 1))
        elif newinput.ndim != 2: #matrix
            raise ValueError("input has to be either a vector or a matrix")
        if newinput.shape[1] != self.input_size:
            raise ValueError("input dimension differs from the RBF one")
        
        firsto = self.first_layer_output(newinput)
        out = np.dot(firsto, self.weights)
        
        conf = np.max(firsto[:,1:], 1)
        return out, conf
    
    def sample_inputs(self, n_samples):
#        out = np.empty((n_samples, self.input_size) )
#        for i in xrange(n_samples):
#            kernel_index = np.random.randint(self.kernels.shape[0])
#            out[i,:] = np.random.multivariate_normal(self.kernels[kernel_index,:],
#                                                     self.sigma*np.eye(self.input_size) )
#        
#        return out
        out = np.empty((n_samples, self.input_size) )
        libfunctions.sample_inputs(n_samples, out, self.kernels,
                                   self.kernels.shape[0],
                                   self.input_size,
                                   self.sigma
                                   )
        return out
        