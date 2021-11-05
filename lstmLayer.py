from typing import ForwardRef
import numpy as np
from functions import Functions

class LSTMLayer:
    # def __init__(self, size, n_cell, vocab_size, x, target):
    def __init__(self, size, n_cell, x, target):
        self.size = size
        self.n_cell = n_cell
        # self.vocab_size = vocab_size
        self.c_prev = np.zeros((self.n_cell, 1))
        self.h_prev = np.zeros((self.n_cell, 1))
        self.x = x # array of array
        self.target = target # array of number
        self.sigmoid = np.vectorize(Functions.sigmoid) # allow function to receive input in form of vector
        self.softmax = np.vectorize(Functions.softmax) # allow function to receive input in form of vector

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

        # self.parameter["wv"] = np.random.randn(self.vocab_size, self.n_cell) * (1.0/np.sqrt(self.vocab_size))
        # self.parameter["bv"] = np.zeros((self.vocab_size, 1))
    
    def forgetGate(self, t):
        self.parameter['f'+str(t)] = self.sigmoid(np.dot(self.parameter['uf'], self.x[t]) + np.dot(self.parameter['wf'], self.h_prev) + self.parameter['bf'])
    
    def inputGate(self, t):
        self.parameter['i'+str(t)] = self.sigmoid(np.dot(self.parameter['ui'], self.x[t]) + np.dot(self.parameter['wi'], self.h_prev) + self.parameter['bi'])
        self.parameter['CFlag'+str(t)] = np.tanh(np.dot(self.parameter['uc'], self.x[t]) + np.dot(self.parameter['wc'], self.h_prev) + self.parameter['bc'])
    
    def cellState(self, t):
        self.parameter['C'+str(t)] = np.multiply(self.parameter['f'+str(t)], self.c_prev) + np.multiply(self.parameter['i'+str(t)], self.parameter['CFlag'+str(t)])
    
    def outputGate(self, t):
        self.parameter['o'+str(t)] = self.sigmoid(np.dot(self.parameter['uo'], self.x[t]) + np.dot(self.parameter['wo'], self.h_prev) + self.parameter['bo'])
        self.parameter['h'+str(t)] = np.multiply(self.parameter['o'+str(t)], np.tanh(self.parameter['C'+str(t)]))

    def forwardProp(self, t):
        self.forgetGate(t)
        self.inputGate(t)
        self.cellState(t)
        self.outputGate(t)

        h = self.parameter['h'+str(t)]
        # v = np.dot(self.parameter["wv"], h) + self.parameter["bv"]
        # y_hat = self.softmax(v)

        # return y_hat
        return h

    def predict(self, t):
        return self.forwardProp(t)

    def getLayerType(self, id):
        #id: id layer >= 0
        return 'lstm'+str(id)+' (LSTM)'