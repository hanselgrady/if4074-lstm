import numpy as np
from functions import Functions

class LSTMLayer:
    def __init__(self, size, n_cell):
        self.size = size
        self.n_cell = n_cell
        self.parameter = {}
        self.parameter['bf'] = np.zeros((self.size, 1))
        self.parameter['bi'] = np.zeros((self.size, 1))
        self.parameter['bc'] = np.zeros((self.size, 1))
        self.parameter['bo'] = np.zeros((self.size, 1))
    
    def cellState(self):
        print('cellState')
    
    def forgetGate(self):
        print('forgetGate')
    
    def inputGate(self):
        print('inputGate')
    
    def outputGate(self):
        print('outputGate')