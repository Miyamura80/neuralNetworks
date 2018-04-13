import numpy as np
import random as r
from numba import jit
import math
from NeuralNetFunctions import importTextFileDataSet
import matplotlib.pyplot as plt

STEP_SIZE = 0.001
DX = 0.001

@jit
def createOneNeuralNet(struc):
    iSize = struc[0]
    hSize = struc[1]
    oSize = struc[2]
    hNum = struc[3]
    if hSize < 1:
        print("Error: No hidden layers!")
    inputL = np.zeros((iSize,1),dtype=np.float64)
    np.random.seed(0)
    WeightLayers = []
    biasLayers = []
    iToH = np.float64(np.random.rand(hSize,iSize))-1
    iToHBias = np.zeros((hSize,1),dtype=np.float64)
    WeightLayers.append(iToH)
    biasLayers.append(iToHBias)

    for i in range(hNum-1):
        weights = np.float64(np.random.rand(hSize,hSize))-1
        WeightLayers.append(weights)
        bias = np.zeros((hSize,1),dtype=np.float64)
        biasLayers.append(bias)

    hToO = np.float64(np.random.rand(oSize,hSize))-1
    hToOBias = np.zeros((oSize,1),dtype=np.float64)
    WeightLayers.append(hToO)
    biasLayers.append(hToOBias)


    return inputL,WeightLayers, biasLayers

@jit
def forwardProp(wLayers,bl,inputs):
    stats = analyzeNNStats(wLayers)
    if stats[0]!=len(inputs):
        print("Invalid input size")
        return -1
    currentLayer = inputs
    for i in range(len(wLayers)):
        currentLayer = activationFunction2(wLayers[i].dot(currentLayer)+bl[i])
    return currentLayer

@jit
def activationFunction(layer):
    for x in range(len(layer)):
        layer[x][0] = max(layer[x][0],0)
    return layer

def activationFunction2(layer):
    layer = 1/(1+np.exp(-layer))
    return layer

def analyzeNNStats(WeightLayers):
    front = WeightLayers[0].shape
    back = WeightLayers[len(WeightLayers)-1].shape
    stats = []
    stats.append(front[1])
    stats.append(front[0])
    stats.append(back[0])
    stats.append(len(WeightLayers)-1)
    return stats

@jit
def numericalBackPropagation(wl,bl,inputs,expectedOutput):
    copyWeights = copyWL(wl)
    copyBias = copyBL(wl,bl)

    actualOutput = forwardProp(wl,bl,inputs)
    cost = costFunction(actualOutput,expectedOutput)

    for l in range(len(wl)):
        currentLayerStruc = wl[l].shape
        height = currentLayerStruc[0]
        width = currentLayerStruc[1]
        for i in range(height):
            for j in range(width):
                copyWeights[l][i][j] += DX
                xPlusDxOutput = forwardProp(copyWeights,bl,inputs)
                cost2 = costFunction(xPlusDxOutput,expectedOutput)
                derivative = (cost2-cost)/DX
                wl[l][i][j] -= derivative*STEP_SIZE
                copyWeights[l][i][j] -= DX
            copyBias[l][i][0] += DX
            xPlusDxOutput = forwardProp(wl, bl, inputs)
            cost2 = costFunction(xPlusDxOutput,expectedOutput)
            derivative = (cost2-cost)/DX
            bl[l][i][0] -= derivative*STEP_SIZE
            copyBias[l][i][0] -= DX


    return wl,bl,cost

@jit
def copyWL(wl):
    struc = analyzeNNStats(wl)
    inputs, newCopy, biasLayers = createOneNeuralNet(struc)

    for l in range(len(wl)):
        currentLayerStruc = wl[l].shape
        height = currentLayerStruc[0]
        width = currentLayerStruc[1]
        for i in range(height):
            for j in range(width):
                newCopy[l][i][j] = wl[l][i][j]

    return newCopy

@jit
def copyBL(wl,bl):
    struc = analyzeNNStats(wl)
    inputs,newCopy,biasLayers = createOneNeuralNet(struc)

    for l in range(len(bl)):
        if l==len(bl)-1:
            k = struc[2]
        else:
            k = struc[1]
        for i in range(k):
            biasLayers[l][i][0] = bl[l][i][0]
    return biasLayers



def costFunction(actualOutputs,expectedOutputs):
    height = actualOutputs.shape[0]
    differences = expectedOutputs-actualOutputs
    total = 0
    for i in range(height):
        total += differences[i][0]*differences[i][0]
    return total

def costFunction2(actualOutputs,expectedOutputs):
    expOut = convertListToArrayInput(expectedOutputs)
    dif = expOut-actualOutputs
    dif = dif*dif*0.5
    return dif


@jit
def trainBot(num,WeightLayers,biasLayers,trainingData):
    plotRecord = []
    dataLength = len(trainingData)
    for i in range(num):
        #Convert dataset to input the neural network can process
        currentInput = convertListToArrayInput(trainingData[i%dataLength][0])
        expOutp = convertListToArrayInput(trainingData[i % dataLength][1])

        #Carry out forward propagation + backpropagation
        WeightLayers,biasLayers,cost = numericalBackPropagation(WeightLayers,biasLayers,currentInput,expOutp)
        plotRecord.append(cost)
        print(i,":",cost)
    plt.plot(plotRecord)
    plt.show()


def convertListToArrayInput(listInput):
    length = len(listInput)
    k = np.zeros((length,1))
    for i in range(length):
        k[i][0] = listInput[i]
    return k



if __name__ == '__main__':
    data = [[[1, 1], [0]], [[0, 1], [1]], [[1, 0], [1]], [[0, 0], [0]]]
    data = importTextFileDataSet("testDataSet.txt")
    inputLayer,wl, biasLayers = createOneNeuralNet([2,4,1,4])



    trainBot(50000,wl,biasLayers,data)
    #TODO: FORGOT BIAS





4
