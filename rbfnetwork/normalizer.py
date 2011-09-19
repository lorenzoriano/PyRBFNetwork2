import numpy as np

class Normalizer(object):
    
    def __init__(self, min = None, max = None):
        self.min = min
        self.max = max
    
    def calculate_from_input(self, input):
        input = np.asarray(input)
        if input.ndim == 1:
            input = np.array(input, ndmin=2).T
        self.min = np.min(input, 0)
#        self.min = np.array(self.min, ndmin=2).T
        self.max = np.max(input, 0)
#        self.max = np.array(self.max, ndmin=2).T
        
        
    def normalize(self, input):
        input = np.asarray(input)
        if input.ndim == 1:
            input = np.array(input, ndmin=2).T
        return (input - self.min) / (self.max - self.min)
    
    def denormalize(self, input):
        input = np.asarray(input)
        return input * (self.max - self.min) + self.min