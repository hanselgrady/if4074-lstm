import numpy as np
from functools import reduce

class SequentialModel:
  """Neural network model.
    Attributes
    ----------
    convolutionlayers, denseLayers : list
        Layers used in the model.
    w_grads : dict
        Weights' gradients during backpropagation.
    b_grads : dict
        Biases' gradients during backpropagation.
    cost_function : CostFunction
        Cost function to be minimized.
    optimizer : Optimizer
        Optimizer used to update trainable parameters (weights and biases).
    l2_lambda : float
        L2 regularization parameter.
    trainable_layers: list
        Trainable layers(those that have trainable parameters) used in the model.
  """

  # def __init__(self, LSTMLayers, denseLayers, flatten, cost_function, optimizer, l2_lambda=0):
  def __init__(self, LSTMLayers, denseLayers, flatten):
    self.LSTMLayers = LSTMLayers
    self.denseLayers = denseLayers
    self.flatten = flatten
    self.summary = []
  
  def addLSTMLayer(self, lstmLayer):
    """
        Add the convo layer to model, parameter ConvolutionLayer Object
    """
    self.LSTMLayers.append(lstmLayer)
  
  def addDenseLayer(self, denseLayer):
    """
        Add dense layer to model, parameter DenseLayer Object
    """
    self.denseLayers.append(denseLayer)

  def setFlatten(self, flatten):
    """
        Set flatten object between convo and dense layers
    """
    self.flatten = flatten

  def pop(self):
    """
        Performs popping out the last layer from model
    """
    if (len(self.denseLayers) != 0):
      self.denseLayers.pop()
    else:
      self.LSTMLayers.pop()
  
  def forwardPropagate(self, x):
    """
        Performs a forward propagation pass.
        Parameters
        ----------
        x : numpy.ndarray
            Input that is fed to the first layer.
        training : bool
            Whether the model is training.
        Returns
        -------
        numpy.ndarray
            Model's output, corresponding to the last layer's activations.
    """
    res = x
    # Compute LSTM Layers
    for idx, eachLSTMLayer in enumerate(self.LSTMLayers):
      print("Layer Convolusi")
      eachLSTMLayer.x = res
      res = eachLSTMLayer.predict(0)
      self.summary.append([eachLSTMLayer.getLayerType(idx), np.array(res).shape, eachLSTMLayer.getNumOfParam(len(res.shape))])
      print("LSTM:")
      print(np.array(res))
    # Compute Dimension Flattening
    print("Layer Flatten")
    self.flatten.init(np.array(res).shape)
    res = self.flatten.forward(np.array(res))
    self.summary.append([self.flatten.getLayerType(), np.array(res).shape, 0])
    print(np.array(res))
    # Compute Dense Layers
    for idx, eachDenseLayer in enumerate(self.denseLayers):
      print("Layer Dense")
      eachDenseLayer.set_input(np.array(res))
      eachDenseLayer.set_weights(np.random.rand(np.array(res).shape[-1]).tolist())
      res = eachDenseLayer.compute_output()
      self.summary.append([eachDenseLayer.getLayerType(idx), np.array(res).shape, (self.summary[-1][1][-1]+1)*np.array(res).shape[-1]])
      print(np.array(res))

    return res
  
  def predict(self, x):
    """
        Calculates the output of the model for the input.
        Parameters
        ----------
        x : numpy.ndarray
            Input.
        Returns
        -------
        numpy.ndarray
            Prediction of the model, corresponding to the last layer's activations.
    """
    res_last = self.forwardPropagate(x)
    print("\nOUTPUT")
    print(res_last)
    return res_last
  
  def printSummary(self):
    tot = 0
    print('---------------------------------------------------------------')
    print('Layer (type)    Output Shape    Param #')
    print('===============================================================')
    for eachSummary in self.summary:
      tot += int(eachSummary[2])
      print((str(eachSummary[0])+'{between}'+str(eachSummary[1])+'{between}'+str(eachSummary[2])).format(between=' '*8))
      print('---------------------------------------------------------------')
    print('===============================================================')
    print('Total params: '+str(tot))