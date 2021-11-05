import numpy as np
from sequentialModel import SequentialModel
from lstmLayer import LSTMLayer
from flatten import Flatten
from denseLayer import DenseLayer

model = SequentialModel(
    LSTMLayers = [
        LSTMLayer(
            size = 4,
            n_cell = 2
        )
    ],
    flatten = Flatten(),
    denseLayers = [
        DenseLayer(n = 10)
    ]
)

input = np.array([[[0.1], [0.2], [0.3], [0.4]], [[0.25], [0.25], [0.3], [0.2]]])
predicted = model.predict(input)
model.printSummary()