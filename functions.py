import math
import numpy as np
from scipy.special import expit

class Functions:
  def relu(x):
    return max(0, x)
  
  def sigmoid(x):
    return 1 / (1 + math.exp(-x))

  def softmax(x):
    return expit(x) / np.sum(expit(x), axis=0)

  def getOutputDimension(inputSize, filterSize, paddingSize, strideSize):
    return int((inputSize - filterSize + 2 * paddingSize) / strideSize + 1)

  def getErrorValue(value):
    return -1 * np.exp(value)
  
  def getOutputGradient(outputValue, realValue, output):
    return -1 * (1 - output) if outputValue == realValue else output