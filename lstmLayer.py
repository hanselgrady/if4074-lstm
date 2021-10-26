import numpy as np
from functions import Functions

class LSTMLayer:
    def __init__(self, size, n_cell, x, target):
        self.size = size
        self.n_cell = n_cell
        self.c_prev = 0
        self.h_prev = 0
        self.x = x # array of array
        self.target = target # array of number

        self.parameter = {}
        self.parameter['bf'] = np.zeros((self.size, 1))
        self.parameter['bi'] = np.zeros((self.size, 1))
        self.parameter['bc'] = np.zeros((self.size, 1))
        self.parameter['bo'] = np.zeros((self.size, 1))
    
    def forgetGate(self, timestep):
        self.parameter['ft'] = Functions.sigmoid(np.dot(self.parameter['Uf'], self.x[timestep]) + np.dot(self.parameter['Wf'], self.h_prev) + self.parameter['bf'])
    
    def inputGate(self):
        print('inputGate')
    
    def cellState(self):
        print('cellState')
    
    def outputGate(self):
        print('outputGate')