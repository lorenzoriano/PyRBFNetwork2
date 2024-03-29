import numpy as np

class Normalizer(object):
    
    def __init__(self, min = None, max = None):
        self.min = min
        self.max = max
    
    def calculate_from_input(self, input):
        input = np.asarray(input)
        if input.ndim == 1:
            input = np.array(input, ndmin=2)
        self.min = np.min(input, 0)
#        self.min = np.array(self.min, ndmin=2).T
        self.max = np.max(input, 0)
#        self.max = np.array(self.max, ndmin=2).T
        
        
    def normalize(self, input):
        input = np.asarray(input)
        if input.ndim == 1:            
            if input.shape[0] != self.max.shape[0]:
                print input.shape
                print self.max.shape
                raise ValueError("input dimension differs from the Normalizer one")
        elif input.ndim == 2:
            if input.shape[1] != self.max.shape[0]:
                raise ValueError("input dimension differs from the Normalizer one")
        else:
            raise ValueError("An input matrix is expected")

        return (input - self.min) / (self.max - self.min)
    
    def denormalize(self, input):
        input = np.asarray(input)
        if input.ndim == 1:
            if input.shape[0] != self.max.shape[0]:
                raise ValueError("input dimension differs from the Normalizer one")
        elif input.ndim == 2:
            if input.shape[1] != self.max.shape[0]:
                raise ValueError("input dimension differs from the Normalizer one")
        else:
            raise ValueError("An input matrix is expected")
        return input * (self.max - self.min) + self.min