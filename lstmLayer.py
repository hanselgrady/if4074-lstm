import numpy as np
from functions import Functions

class LSTMLayer:
    def __init__(self, size, n_cell, x, target):
        self.size = size
        self.n_cell = n_cell
        self.c_prev = np.zeros((self.n_cell, 1))
        self.h_prev = np.zeros((self.n_cell, 1))
        self.x = x # array of array
        self.target = target # array of number
        self.sigmoid = np.vectorize(Functions.sigmoid) # allow function to receive input in form of vector

        self.parameter = {}
        self.parameter['uf'] = np.random.rand(self.n_cell, self.size)
        self.parameter['ui'] = np.random.rand(self.n_cell, self.size)
        self.parameter['uc'] = np.random.rand(self.n_cell, self.size)
        self.parameter['uo'] = np.random.rand(self.n_cell, self.size)

        self.parameter['wi'] = np.random.rand(self.n_cell, self.n_cell)
        self.parameter['wf'] = np.random.rand(self.n_cell, self.n_cell)
        self.parameter['wc'] = np.random.rand(self.n_cell, self.n_cell)
        self.parameter['wo'] = np.random.rand(self.n_cell, self.n_cell)
        
        self.parameter['bf'] = np.zeros((self.n_cell, 1))
        self.parameter['bi'] = np.zeros((self.n_cell, 1))
        self.parameter['bc'] = np.zeros((self.n_cell, 1))
        self.parameter['bo'] = np.zeros((self.n_cell, 1))
    
    def forgetGate(self, timestep):
        # print('Uf : ')
        # print(self.parameter['uf'], end='\n\n')
        # print('xt : ')
        # print(self.x[timestep], end='\n\n')
        # print('Wf : ')
        # print(self.parameter['wf'], end='\n\n')
        # print('ht-1 : ')
        # print(self.h_prev, end='\n\n')
        # print('bf : ')
        # print(self.parameter['bf'], end='\n\n')
        # print('Uf.xt : ')
        # print(np.dot(self.parameter['uf'], self.x[timestep]), end='\n\n')
        # print('Wf.ht-1 : ')
        # print(np.dot(self.parameter['wf'], self.h_prev), end='\n\n')
        # print('Uf.xt + Wf.ht-1 + bf : ')
        # print(np.dot(self.parameter['uf'], self.x[timestep]) + np.dot(self.parameter['wf'], self.h_prev) + self.parameter['bf'], end='\n\n')
        self.parameter['f0'] = self.sigmoid(np.dot(self.parameter['uf'], self.x[timestep]) + np.dot(self.parameter['wf'], self.h_prev) + self.parameter['bf'])
    
    def inputGate(self):
        print('inputGate')
    
    def cellState(self):
        print('cellState')
    
    def outputGate(self):
        print('outputGate')