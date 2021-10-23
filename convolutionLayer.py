import numpy as np
from scipy import signal
from functions import Functions

class ConvolutionLayer:
  def __init__(self, inputSize, paddingSize, filterAmount, filterSize, strideSize, plainData, inputFilter):
    self.inputSize = inputSize
    self.paddingSize = paddingSize
    self.filterAmount = filterAmount
    self.filterSize = filterSize
    self.strideSize = strideSize
    self.inputMatrix = self.getFinalMatrix(plainData)
    self.inputFilter = np.array(inputFilter)
    self.bias = np.zeros((len(plainData), Functions.getOutputDimension(self.inputSize, self.filterSize, self.paddingSize, self.strideSize), Functions.getOutputDimension(self.inputSize, self.filterSize, self.paddingSize, self.strideSize)))
    self.convolutionResult = []
    self.detectionResult = []
    self.poolingResult = []
  
  def getFinalMatrix(self, plainData):
    # Get final matix for computation
    finalMatrixSize = self.inputSize + self.paddingSize * 2
    finalMatrix = [[[0 for i in range (0, finalMatrixSize)] for j in range (0, finalMatrixSize)] for k in range (0, len(plainData))]
    for k in range (0, len(plainData)):
      for i in range (self.paddingSize, self.inputSize + self.paddingSize):
        for j in range (self.paddingSize, self.inputSize + self.paddingSize):
          finalMatrix[k][i][j] = plainData[k][i - self.paddingSize][j - self.paddingSize]
    return np.array(finalMatrix)

  def convolution(self):
    outputDimension = Functions.getOutputDimension(self.inputSize, self.filterSize, self.paddingSize, self.strideSize)
    result = []
    for m in range (0, self.filterAmount):
      # Use Output Dimension as a base for building matrix result
      # outputData stores final result data for a single kernel
      outputData = [[0 for i in range (0, outputDimension)] for j in range (0, outputDimension)]
      # Use stride value and output dimension as a base to decide which column and row would be the start of iteration
      for n in range (0, len(self.inputMatrix)):
        # tempData stores multiplication result for a single input AND single kernel
        tempData = [[0 for i in range (0, outputDimension)] for j in range (0, outputDimension)]
        for i in range (0, outputDimension * self.strideSize, self.strideSize):
          for j in range (0, outputDimension * self.strideSize, self.strideSize):
            multiplicationResult = 0
            for k in range (0, self.filterSize):
              for l in range (0, self.filterSize):
                multiplicationResult = multiplicationResult + self.inputMatrix[n][i+k][j+l] * self.inputFilter[m][n][k][l]
            tempData[i // self.strideSize][j // self.strideSize] = multiplicationResult
        for i in range (0, outputDimension):
          for j in range (0, outputDimension):
            outputData[i][j] = outputData[i][j] + tempData[i][j]
      result.append(outputData)
    self.convolutionResult = result
    return result
  
  def backward(self, outputGradient, learningRate):
    filterGradient = np.array([[0 for i in range (0, self.filterSize)] for j in range (0, self.filterSize)])
    inputGradient = np.array([[0 for i in range (0, self.inputSize)] for j in range (0, self.inputSize)])
    self.bias = self.bias - outputGradient * learningRate
    for i in range (self.filterAmount):
      for j in range (len(self.inputMatrix)):
        filterGradient[i, j] = signal.correlate2d(self.inputMatrix[j], outputGradient[i], "valid")
        inputGradient[j] = inputGradient[j] + signal.convolve2d(outputGradient[i], self.inputFilter[i, j], "full")
    self.inputFilter = self.inputFilter - learningRate * filterGradient
    self.bias = self.bias - learningRate * outputGradient
    return inputGradient
  
  def detector(self, method):
    result = []
    for i in range (0, len(self.convolutionResult)):
      output = [[0 for j in range (0, len(self.convolutionResult[i]))] for k in range (0, len(self.convolutionResult[i]))]
      for j in range (0, len(self.convolutionResult[i])):
        for k in range (0, len(self.convolutionResult[i])):
          if (method == 'relu'):
            output[j][k] = Functions.relu(self.convolutionResult[i][j][k])
          else:
            output[j][k] = Functions.sigmoid(self.convolutionResult[i][j][k])
      result.append(output)
    self.detectionResult = result
    return result
  
  def pooling(self, poolingFilterSize, poolingStrideSize, poolingMode):
    result = []
    for i in range (0, len(self.detectionResult)):
      outputDimension = Functions.getOutputDimension(len(self.detectionResult[i]), poolingFilterSize, 0, poolingStrideSize)
      outputData = [[0 for i in range (0, outputDimension)] for j in range (0, outputDimension)]
      for j in range (0, outputDimension * poolingStrideSize, poolingStrideSize):
        for k in range (0, outputDimension * poolingStrideSize, poolingStrideSize):
          if (poolingMode == 'max'):
            maxValue = -100
            for l in range (0, poolingFilterSize):
              for m in range (0, poolingFilterSize):
                maxValue = max(maxValue, self.detectionResult[i][j+l][k+m])
            outputData[j // poolingFilterSize][k // poolingFilterSize] = maxValue
          else:
            sumValue = 0
            for l in range (0, poolingFilterSize):
              for m in range (0, poolingFilterSize):
                sumValue = sumValue + self.detectionResult[i][j+l][k+m]
            sumValue = sumValue / (poolingFilterSize ** 2)
            outputData[j // poolingFilterSize][k // poolingFilterSize] = sumValue
      result.append(outputData)
    self.poolingResult = result
    return result

  def getLayerType(self):
    return 'conv2d (Conv2D)'

  def getPoolingType(self, method):
    return (str(method)+'_pooling ('+str(method)+'Pooling)')

  def getDetectorType(self, method):
    return ('Leaky_'+str(method))