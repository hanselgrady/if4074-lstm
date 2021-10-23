import numpy as np
from functions import Functions
import random

class DenseLayer:
  def __init__(self, n, func='sigmoid'):
    self.neuron = n
    self.func = func
    if (func == 'relu'):
      self.activation = Functions.relu
    elif (func == 'softmax'):
      self.activation = Functions.softmax
    else:
      self.activation = Functions.sigmoid
    self.input = []
    self.weight = []
    self.output = []
    self.predict = []
    self.delta_weight = []

  def getLayerType(self):
    return 'dense (Dense)'

  def get_func(self):
    return self.func

  def get_neuron(self):
    return self.neuron

  def set_input(self, i):
    self.input = i

  def set_weights(self, weight):
    self.weight = weight
    
  def set_predict(self, predict):
    self.predict = predict
      
  def randomize_weights(self, count_input):
    for i in range(self.neuron):
        list = []
        for j in range(count_input):
            list.append(random.uniform(0,1))
        self.weight.append(list)

  def reset_delta_weight(self):
    reset = []
    for i in self.weight:
        temp = []
        for value in i:
            temp.append(0)
        reset.append(temp)
    self.delta_weight = reset
      
  # derivative net/w
  def derivative_net(self):
    return self.input
      
  # derivative o/net
  def derivative_output(self):
    result = []
    for i in range(len(self.output)):
        if(self.func == 'sigmoid'):
            result.append(self.output[i][0]*(1-self.output[i][0]))
        elif(self.func == 'linear'):
            result.append(1)
        elif(self.func == 'relu'):
            if(self.output[i][0] > 0):
                result.append(1)
            else:
                result.append(0)
    return result
  
  # derivative error/o
  def derivative_error(self):
    result = []
    for i in range(len(self.output)):
        result.append(-1*(self.output[i][0] - self.predict))
    return result
      
  def gradien(self):
    result = []
    d1 = self.derivative_net()
    d2 = self.derivative_output()
    d3 = self.derivative_error()
    
    for k in range(self.neuron):
      batch = []
      for i in d1: 
          for j in i:  
              batch.append(j*d2[k]*d3[k])
      result.append(batch)
    return result
  
  def compute_output(self):
    result = []
    if not len(self.input):
        print("Cant make output, input not defined")
        return []
    if not len(self.weight):
      print("Cant make output, weights not defined")
      return []
    else:
        dot_product = np.dot(np.array(self.weight).reshape((1,np.array(self.weight).shape[0])), np.array(self.input).reshape(-1,1))
        for x in dot_product:
            for y in x:
                result.append([self.activation(y)])
        self.output = result
        return result
  
  def compute_delta_weight(self, eta):
    delta_weight = []
    gradien = self.gradien()
    for i in gradien:
        batch = []
        for value in i:
            if(self.delta_weight == []):
              batch.append(-1*value*eta)
            else:
              batch.append(self.delta_weight[gradien.index(i)][i.index(value)] + (-1*value*eta))
        delta_weight.append(batch)
    self.delta_weight = delta_weight
      
  def update_weight(self):
    weight = []
    for i in range(len(self.weight)):
        temp = []
        for j in range(len(self.weight[i])):
            temp.append(self.weight[i][j] + self.delta_weight[i][j])
        weight.append(temp)
    self.weight = weight
    self.reset_delta_weight()
