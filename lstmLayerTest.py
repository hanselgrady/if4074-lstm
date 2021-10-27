from lstmLayer import LSTMLayer
import numpy as np

x = np.array([[[0.1], [0.2], [0.3], [0.4]], [[0.25], [0.25], [0.3], [0.2]]])
size = 4
n_cell = 2
target = np.array([4, 3])
lstm = LSTMLayer(size, n_cell, x, target)
lstm.forgetGate(0)
print(lstm.parameter['f0'], end='\n\n')

x = np.array([[[0.1], [0.2], [0.3], [0.4]]])
size = 4
n_cell = 1
target = np.array([4])
lstm = LSTMLayer(size, n_cell, x, target)
lstm.forgetGate(0)
print(lstm.parameter['f0'])