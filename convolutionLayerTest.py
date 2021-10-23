from convolutionLayer import ConvolutionLayer

inputSize = 3
paddingSize = 0
filterAmount = 2
filterSize = 2
strideSize = 1
# plainData = [[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7], [4, 5, 6, 7, 8], [5, 6, 7, 8, 9]]
# inputFilter = [[[1, 1, 1], [1, 1, 1], [1, 1, 1]], [[-1, -1, -1], [-1, -1,-1], [-1, -1, -1]]]
# plainData dimension : amount x inputSize x inputSize
plainData = [[[16, 24, 32], [47, 18, 26], [68, 12, 9]], [[26, 57, 43], [24, 21, 12], [2, 11, 19]], [[18, 47, 21], [4, 6, 12], [81, 22, 13]]]
# inputFilter dimension : filterAmount x inputSize x filterSize x filterSize
inputFilter = [[[[0, -1], [1, 0]], [[5, 4], [3, 2]], [[16, 24], [68, -2]]], [[[60, 22], [32, 18]], [[35, 46], [7, 23]], [[78, 81], [20, 42]]]]
convoLayer = ConvolutionLayer(inputSize, paddingSize, filterAmount, filterSize, strideSize, plainData, inputFilter)
print(convoLayer.inputMatrix)
print('')
print(convoLayer.inputFilter)
print('')
print(convoLayer.convolution())
print('')
print(convoLayer.detector('sigmoid'))
print('')
print(convoLayer.detector('relu'))
print('')
print(convoLayer.pooling(2, 2, 'max'))
print('')
print(convoLayer.pooling(2, 2, 'average'))