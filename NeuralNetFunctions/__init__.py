import random as ran
import tkinter
import math
import matplotlib.pyplot as plt
from numba import jit
import winsound
from GenerallyUsefulFunctions import *

class Neuron:
    def __init__(self, holdValue):
        self.value = holdValue

    def updateNeuronAttributes(self, sum):
        self.summedValue = sum
        self.sig = sigmoidActF(self.summedValue)
        self.gradient_wrt_plSummation = self.value * (1 - self.value)

    def calcGradient(self):
        self.err_wrt_nodeValue = 0

class Connections:
    def __init__(self, holdValue):
        self.value = holdValue

    def updateGradient(self, new):
        self.err_wrt_gradient = new

# GLOBAL VARIABLES
stepSize = 1
deltaX = 0.001
randWeightRange = [-4, 4]
fillWith = 0.5


# Returns a new neural net with random weights and each neuron filled with 0
# NOTE: THIS IS UNDER AN ASSUMPTION THAT THERE IS AT LEAST 1 HIDDEN LAYER
# NOTE: ADJUST RANDOM INTEGER RANGE AND FILLWITH VALUE HERE
@jit
def createRandomNeuralNet(inputNum, HlayerNum, HlayerSize, outputNum):
    nn = [[], []]
    # Index references:
    # nn[x][y][z]
    # x-Input, Hidden or Output layer Index
    # y-Neuron list index
    # z=0-The value stored within the neuron


    # NOTE: BIASES ARE ALWAYS AT THE LAST INDEX OF SUB-SUB-LISTS

    if HlayerNum <= 0 or HlayerSize <= 0:
        print("Invalid neural net parameters")
        return -1

    # Making hidden neuron layers
    for i in range(HlayerNum):
        nn.append([])

    # Adding node/weight lists into input layer
    for i in range(inputNum):
        nn[0].append([])
        # Creating Node values
        nn[0][i].append(Neuron(fillWith))
        # Creating random weight list values for each node
        for j in range(HlayerSize):
            nn[0][i].append(Connections(ran.uniform(randWeightRange[0], randWeightRange[1])))
    # [[inputs nn[0]] [hidden nn[1]] [outputs nn[2]]]



    # Adding nodes/weight lists into hidden layer
    for l in range(1, HlayerNum + 1):

        for i in range(HlayerSize):
            nn[l].append([])
            # Generate node values
            nn[l][i].append(Neuron(fillWith))
            # Generate weight values connected to the node for inbetween layers
            if l != (HlayerNum):
                for j in range(HlayerSize):
                    nn[l][i].append(Connections(ran.uniform(randWeightRange[0], randWeightRange[1])))
            # Generate weight values connected to the output nodes
            else:
                for j in range(outputNum):
                    nn[l][i].append(Connections(ran.uniform(randWeightRange[0], randWeightRange[1])))
            # BIAS
            nn[l][i].append(ran.uniform(randWeightRange[0], randWeightRange[1]))

    # Filling output layer with filler
    for o in range(outputNum):
        nn[HlayerNum + 1].append([])
        nn[HlayerNum + 1][o].append(Neuron(fillWith))
        # BIAS
        nn[HlayerNum + 1][o].append(ran.uniform(randWeightRange[0], randWeightRange[1]))

    return nn


# Puts a list of integer/float values into the input neuron of nn
def insertInputs(inputs, nn):
    length = len(inputs)

    # Safety precaution: In case length of inputs is different from configured input neuron length
    if length != len(nn[0]):
        print("ERR: Invalid number of inputs")
        return

    # Replacing node values with input list values
    for i in range(length):
        nn[0][i][0].value = inputs[i]
    return nn

# Takes a list of inputs and the pre-generated neural network data structure
# Returns a list of outputs
@jit
def forwardProp(inputs, nn: list):
    # Insert inputs into nn
    nn = insertInputs(inputs, nn)

    outputs = []

    layerNum = int(len(nn))
    inputLayerSize = len(nn[0])
    outputLayerSize = len(nn[layerNum - 1])
    hLayerSize = len(nn[1])

    # Iterating from 1st hidden layer to output layer
    for i in range(1, layerNum):
        # Decides if the pointed layer is output layer or hidden layer and changes loopCount based on it
        if i == layerNum - 1:
            currentLayerLoopCount = outputLayerSize
        else:
            currentLayerLoopCount = hLayerSize
        # Decides the number of neurons in the previous layer
        if i == 1:
            previousLayerNeuronNum = inputLayerSize
        else:
            previousLayerNeuronNum = hLayerSize
        # Decides number of neurons in next layer
        if i < layerNum - 2:
            nextLoopNum = hLayerSize
        else:
            nextLoopNum = outputLayerSize

        # Iterating through all the neurons in the current layer i
        for j in range(currentLayerLoopCount):

            # INIT
            total = 0
            # Sum of all previous layer neurons multiplied by their corresponding weight
            for k in range(previousLayerNeuronNum):
                previousNeuron = nn[i - 1][k][0].value
                weightCorresponding = nn[i - 1][k][j + 1].value
                total += previousNeuron * weightCorresponding

            # Addition of biases
            if i == layerNum - 1:
                total += nn[i][j][nextLoopNum]
            else:
                total += nn[i][j][nextLoopNum + 1]
            # Working out:
            # [N, w,w,w,w,b]
            # NLP = 4

            sigmoidedValue = sigmoidActF(total)
            # Changes the values based on if it is an output or not
            # Order matters a lot here -> value update then neuron attribute update next
            nn[i][j][0].value = sigmoidedValue
            nn[i][j][0].updateNeuronAttributes(total)

    for i in range(outputLayerSize):
        outputs.append(nn[layerNum - 1][i][0].value)
    return outputs, nn

# Using RELU
def activationFunctionRelu(num: int) -> object:
    return max(0,num)

def breedWeights(w1, w2):
    if w1 > 0 > w2:
        return w1
    elif w1 < 0 < w2:
        return w1
    else:
        return w1 + w2 / 2

def breedNeuralNets(nn1, nn2):
    layerNum = int(len(nn1))
    inputLayerSize = len(nn1[0])
    outputLayerSize = len(nn1[layerNum - 1])
    hLayerSize = len(nn1[1])

    # Breeding them by taking an average for input layer
    for i in range(inputLayerSize):
        for j in range(hLayerSize):
            w1 = nn1[0][i][j].value
            w2 = nn2[0][i][j].value
            nn1[0][i][j].value = breedWeights(w1, w2)
    # Breeding weights for the hidden layer
    for l in range(1, layerNum - 1):
        if l == layerNum - 2:
            weightNum = outputLayerSize
        else:
            weightNum = hLayerSize
        for i in range(hLayerSize):
            for j in range(1, weightNum + 1):
                w1 = nn1[l][i][j].value
                w2 = nn2[l][i][j].value
                nn1[l][i][j].value = breedWeights(w1, w2)

    return nn1

#TODO: Fix this for values where the len() is more than 1
# Takes in file name as input, returns list of input-output pairs
def importTextFileDataSet(fileName):
    f = open(fileName, "r")
    m = f.readlines()

    for i in range(len(m)):
        k = m[i].strip("\n").split(":")
        #k -> len2 0->inputs 1-> ouputs
        m[i] = k
        inputs = m[i][0].split(",")
        # if len(inputs)!=20:
        #     print(str(i),len(inputs))
        outputs = m[i][1].split(",")
        m[i][0] = inputs
        m[i][1] = outputs
    # This is under the assumption the number of data sets remain constant!
    for i in range(len(m)):
        for j in range(len(m[0][0])):
            m[i][0][j] = float(m[i][0][j])
        for k in range(len(m[0][1])):
            m[i][1][k] = float(m[i][0][j])
    f.close()
    return m

# Returns the 2 best neural nets out of the tests, as well as the generation graph
def randomGeneration(bestNN, num, p1, p2, p3, p4, Tinputs, expectedOutputs, allowMutation=True, mutChance=30):
    errorValues = [0, 0]
    pyPlotList = []
    minErr1 = 100
    minErr2 = 100
    for i in range(len(expectedOutputs)):
        errorValues.append(0)

    for i in range(num):
        if i == 0:
            nn = bestNN
        else:
            nn = createRandomNeuralNet(p1, p2, p3, p4)
        if allowMutation:
            nn = mutationBreeding(bestNN, nn, mutChance)
        Toutputs, forwardedNN = forwardProp(Tinputs, nn)
        errorValues[0] = expectedOutputs[0] - Toutputs[0]
        errorValues[1] = expectedOutputs[1] - Toutputs[1]
        averageErr = (abs(errorValues[0]) + abs(errorValues[1])) / 2
        pyPlotList.append(averageErr)
        if averageErr < minErr1:
            minErr1 = averageErr
            meNN1 = nn
            continue
        if averageErr < minErr2:
            minErr2 = averageErr
            meNN2 = nn
    return meNN1, meNN2, pyPlotList

def mutationBreeding(bestNN, newNN, mutationChancePerWeight):
    layerNum = int(len(bestNN))
    inputLayerSize = len(bestNN[0])
    outputLayerSize = len(bestNN[layerNum - 1])
    hLayerSize = len(bestNN[1])

    # Breeding them by taking an average for input layer
    for i in range(inputLayerSize):
        for j in range(hLayerSize):
            w1 = bestNN[0][i][j].value
            w2 = newNN[0][i][j].value
            mut = ran.randint(0, 100)
            if mut < mutationChancePerWeight:
                newNN[0][i][j].value = breedWeights(w1, w2)

    # Breeding weights for the hidden layer
    for l in range(1, layerNum - 1):
        if l == layerNum - 2:
            weightNum = outputLayerSize
        else:
            weightNum = hLayerSize
        for i in range(hLayerSize):
            for j in range(1, weightNum + 1):
                w1 = bestNN[l][i][j].value
                w2 = newNN[l][i][j].value
                mut = ran.randint(0, 100)
                if mut < mutationChancePerWeight:
                    newNN[0][i][j].value = breedWeights(w1, w2)

    return newNN

def sigmoidActF(x):
    return 1 / (1 + math.exp(-x))


def inverseSigmoid(y):
    return -1 * math.log((1 / y) - 1)


# TODO: Do stochastic gradient descent, where the forward/back propagation is done in batches
# TODO: Only need to adjust the gradient wrt total error!
# TODO: The weights which lead to multiple outputs needs the gradients to be added
def backpropagationAttempt1(inputs, expectedOutputs, forwardedNN, stepSize):
    layerNum = int(len(forwardedNN))
    inputLayerSize = len(forwardedNN[0])
    outputLayerSize = len(forwardedNN[layerNum - 1])
    hLayerSize = len(forwardedNN[1])

    actualOutputs, forwardedNN = forwardProp(inputs, forwardedNN)
    actualError = calculateTotalError(expectedOutputs, actualOutputs)

    for l in range(layerNum - 1, -1, -1):
        if l == layerNum - 1:
            loopCount = outputLayerSize

        for i in range(hLayerSize):
            u = inverseSigmoid(forwardedNN[l][i][0].value)
            sig = sigmoidActF(u)
            for j in range(outputLayerSize):
                du_wrt_weight = forwardedNN[l - 1][j][0]
                gradient = du_wrt_weight * sig * (1 - sig)
                forwardedNN[l - 1][j][i] -= gradient * stepSize

    return forwardedNN


# Naiive moving it back and forth way
# FAIL
# FAIL
# FAIL
def backpropagationAttemptNaive(inputs, expectedOutputs, forwardedNN, stepSize):
    layerNum = int(len(forwardedNN))
    inputLayerSize = len(forwardedNN[0])
    outputLayerSize = len(forwardedNN[layerNum - 1])
    hLayerSize = len(forwardedNN[1])

    for l in range(layerNum - 1):
        if l == layerNum - 2:
            wSize = outputLayerSize
        else:
            wSize = hLayerSize
        if l == 0:
            loopCount = inputLayerSize
        else:
            loopCount = hLayerSize

        for i in range(loopCount):

            for j in range(1, wSize + 1):
                aNN = copyNN(forwardedNN)
                aNN[l][i][j].value += stepSize
                a, fNNa = forwardProp(inputs, aNN)
                errA = calculateTotalError(expectedOutputs, a)
                bNN = copyNN(forwardedNN)
                bNN[l][i][j].value -= stepSize
                b, fNNb = forwardProp(inputs, bNN)
                errB = calculateTotalError(expectedOutputs, b)
                if errA < errB:
                    forwardedNN[l][i][j].value += stepSize
                else:
                    forwardedNN[l][i][j].value -= stepSize

            if l != 0:

                aNN = copyNN(forwardedNN)
                aNN[l][i][wSize + 1] += stepSize
                a, fNNa = forwardProp(inputs, aNN)
                errA = calculateTotalError(expectedOutputs, a)
                bNN = copyNN(forwardedNN)
                bNN[l][i][wSize + 1] -= stepSize
                b, fNNb = forwardProp(inputs, bNN)
                errB = calculateTotalError(expectedOutputs, b)
                if errA < errB:
                    forwardedNN[l][i][wSize + 1] += stepSize
                else:
                    forwardedNN[l][i][wSize + 1] -= stepSize
    for i in range(outputLayerSize):
        aNN = copyNN(forwardedNN)
        aNN[layerNum - 1][i][1] += stepSize
        a, fNNa = forwardProp(inputs, aNN)
        errA = calculateTotalError(expectedOutputs, a)
        bNN = copyNN(forwardedNN)
        bNN[layerNum - 1][i][1] -= stepSize
        b, fNNb = forwardProp(inputs, bNN)
        errB = calculateTotalError(expectedOutputs, b)
        if errA < errB:
            forwardedNN[layerNum - 1][i][1] += stepSize
        else:
            forwardedNN[layerNum - 1][i][1] -= stepSize

    return forwardedNN, calculateTotalError(expectedOutputs, a)

#FOR LANGUAGE PROCESSING
def convertBack(inputList):
    out = ""
    abc = "abcdefghijklmnopqrstuvwxyz"
    for i in inputList:
        k = int(round(i*26))
        if k==0:
            out += ""
        else:
            out += abc[k-1]

    return out

def tryBackProp(dataSet, num, a, b, c, d,nn=None):
    pList = []
    count = 0
    if nn==None:
        nn = createRandomNeuralNet(a, b, c, d)
    for i in range(num):
        inputs = dataSet[count][0]
        expectedOutputs = dataSet[count][1]
        count = (count + 1) % len(dataSet)
        nn, err, output = backpropagationAttemptGradient1(inputs, expectedOutputs, nn, stepSize)
        pList.append(err)
        print(i, ":", convertBack(inputs),"->",str(output))

    winsound.Beep(2500, 1000)
    plt.plot(pList)
    plt.show()

    saveChoice = input("Do you wish to save the bot? Press y to save: ")
    if saveChoice == "y":
        botName = input("Enter the bot's series name to be saved as: ")
        saveNN(nn, botName)

def copyNN(nn):
    layerNum = int(len(nn))
    inputLayerSize = len(nn[0])
    outputLayerSize = len(nn[layerNum - 1])
    hLayerSize = len(nn[1])

    newNN = createRandomNeuralNet(inputLayerSize, layerNum - 2, hLayerSize, outputLayerSize)
    for l in range(layerNum - 1):
        if l == 0:
            nodeCount = inputLayerSize
        else:
            nodeCount = hLayerSize
        if l < layerNum - 2:
            nextWeightCount = hLayerSize
        else:
            nextWeightCount = outputLayerSize

        for i in range(nodeCount):
            for j in range(nextWeightCount):
                newNN[l][i][j + 1].value = nn[l][i][j + 1].value
            # Replacing biases
            if l != 0:
                newNN[l][i][nextWeightCount + 1] = nn[l][i][nextWeightCount + 1]
    for i in range(outputLayerSize):
        newNN[layerNum - 1][i][1] = nn[layerNum - 1][i][1]

    return newNN

def backpropagationAttemptGradient1(inputs, expectedOutputs, inputNN, stepSize):
    layerNum = int(len(inputNN))
    inputLayerSize = len(inputNN[0])
    outputLayerSize = len(inputNN[layerNum - 1])
    hLayerSize = len(inputNN[1])

    initOutput, inputNN = forwardProp(inputs, inputNN)
    currentErr = calculateTotalError(expectedOutputs, initOutput)
    copy = copyNN(inputNN)

    for l in range(layerNum - 1):
        if l == layerNum - 2:
            wSize = outputLayerSize
        else:
            wSize = hLayerSize
        if l == 0:
            loopCount = inputLayerSize
        else:
            loopCount = hLayerSize

        for i in range(loopCount):

            # Adjusting weights
            for j in range(1, wSize + 1):
                aNN = copyNN(copy)
                aNN[l][i][j].value += deltaX
                a, fNNa = forwardProp(inputs, aNN)
                newErr = calculateTotalError(expectedOutputs, a)
                gradient = (newErr - currentErr) / deltaX
                inputNN[l][i][j].value -= gradient * stepSize

            # Adjust biases besides input layer
            if l != 0:
                aNN = copyNN(copy)
                aNN[l][i][wSize + 1] += deltaX
                a, fNNa = forwardProp(inputs, aNN)
                newErr = calculateTotalError(expectedOutputs, a)
                gradient = (newErr - currentErr) / deltaX
                inputNN[l][i][wSize + 1] -= gradient * stepSize

    # Adjust output layer biases
    for i in range(outputLayerSize):
        aNN = copyNN(copy)
        aNN[layerNum - 1][i][1] += deltaX
        a, fNNa = forwardProp(inputs, aNN)
        newErr = calculateTotalError(expectedOutputs, a)
        gradient = (newErr - currentErr) / deltaX
        inputNN[layerNum - 1][i][1] -= gradient * stepSize

    finalResult, inputNN = forwardProp(inputs, inputNN)

    return inputNN, calculateTotalError2(expectedOutputs, finalResult), finalResult

# Cost function
def calculateTotalError(expected, actual):
    error = 0
    for i in range(len(expected)):
        error += 0.5 * ((abs(expected[i] - actual[i])) ** 2)
    return error

#Mean eror function
def calculateTotalError2(expected, actual):
    error = 0
    for i in range(len(expected)):
        error += ((abs(expected[i] - actual[i])))
    error = error/len(expected)
    return error

# TODO: Allow importing data sets and updating the weight lines accordingly
# TODO: Display the numbers in each neurons
# TODO: Fix how it only currently displays random hold values
def trainingGUI(nn):
    layerNum = int(len(nn))
    inputLayerSize = len(nn[0])
    outputLayerSize = len(nn[layerNum - 1])
    hLayerSize = len(nn[1])

    cWidth = 1200
    cHeight = 600
    radius = 25

    # Calculated stuff
    diameter = 2 * radius
    xGap = (cWidth - layerNum * diameter) // (layerNum + 1)

    top = tkinter.Tk()
    canvas = tkinter.Canvas(top, width=cWidth, height=cHeight)
    coolBut2 = tkinter.Button(top, text="Make it do cool stuff!!")
    canvas.grid(row=2, column=0)
    coolBut2.grid(row=1, column=10)

    topLeftCords = [[] for i in range(layerNum)]
    for l in range(layerNum):
        x = xGap * (l + 1) + diameter * l
        if l == 0:
            layerSize = inputLayerSize
            color = "#10F111"
        elif l == layerNum - 1:
            layerSize = outputLayerSize
            color = "blue"
        else:
            layerSize = hLayerSize
            color = "yellow"

        ygap = (cHeight - layerSize * diameter) // (layerSize + 1)
        for i in range(layerSize):
            y = ygap * (i + 1) + diameter * i
            # In above expression, coefficient of i determines the gap
            # +3 is done to prevent it glitching off the window
            canvas.create_oval(x, y, x + diameter, y + diameter, fill=color)
            r = str(ran.uniform(0, 1))[:5]
            canvas.create_text(x + radius, y + radius, text=r, font="Arial")
            topLeftCords[l].append([x, y])

    for l in range(layerNum - 1):
        if l == 0:
            layerSize = inputLayerSize
        else:
            layerSize = hLayerSize

        nextLayerLen = len(topLeftCords[l + 1])
        for i in range(layerSize):
            x = topLeftCords[l][i][0] + diameter
            y = topLeftCords[l][i][1] + radius
            for j in range(nextLayerLen):
                x2 = topLeftCords[l + 1][j][0]
                y2 = topLeftCords[l + 1][j][1] + radius
                canvas.create_line(x, y, x2, y2, fill="blue", width=1.5)
    coolBut = tkinter.Button(top, text="Make it do cool stuff!",
                             command=lambda: spazStuff(canvas, topLeftCords, layerNum, inputLayerSize, radius,
                                                       hLayerSize))
    coolBut.grid(row=1, column=1)

    top.mainloop()

# Actually no meaning :/
def spazStuff(canvas, corrds, layerNum, inputLayerSize, radius, hLayerSize):
    diameter = radius * 2
    colours = ["red", "cyan", "black"]
    for l in range(layerNum - 1):
        if l == 0:
            layerSize = inputLayerSize
        else:
            layerSize = hLayerSize

        nextLayerLen = len(corrds[l + 1])
        for i in range(layerSize):
            x = corrds[l][i][0] + diameter
            y = corrds[l][i][1] + radius
            for j in range(nextLayerLen):
                x2 = corrds[l + 1][j][0]
                y2 = corrds[l + 1][j][1] + radius
                canvas.create_line(x, y, x2, y2, fill=colours[ran.randint(0, 2)], width=1.5)

# Returns a nn based on the text file
def loadNN(fileName):
    pathToBotFolder = getCurrentDir() + "\\Saved Bots\\"
    content = readFile(pathToBotFolder + fileName)
    structure = content[0].strip("\n").split(",")
    nn = createRandomNeuralNet(int(structure[0]),int(structure[1]),int(structure[2]),int(structure[3]))
    layerNum = int(structure[1])+2
    inputLayerSize = int(structure[0])
    outputLayerSize = int(structure[3])
    hLayerSize = int(structure[2])

    count = 1


    for l in range(layerNum - 1):
        if l == layerNum - 2:
            wSize = outputLayerSize
        else:
            wSize = hLayerSize
        if l == 0:
            loopCount = inputLayerSize
        else:
            loopCount = hLayerSize

        for i in range(loopCount):

            # All weights in input and hidden layer
            for j in range(1, wSize + 1):
                nn[l][i][j].value = float(content[count].strip("\n"))
                count += 1

            # Biases in hidden layer
            if l != 0:
                nn[l][i][wSize+1] = float(content[count].strip("\n"))
                count += 1

                # Output layer biases
    for i in range(outputLayerSize):
        nn[layerNum-1][i][1] = float(content[count].strip("\n"))
        count += 1
    return nn


def saveNN(inputNN, fileNameFirstPart):
    """
    :rtype None
    :parameter inputNN NN data structure
    :parameter fileNameFirstPart first part of the name
    Save the bot as a new text file
    """
    layerNum = int(len(inputNN))
    inputLayerSize = len(inputNN[0])
    outputLayerSize = len(inputNN[layerNum - 1])
    hLayerSize = len(inputNN[1])

    structure = str(inputLayerSize) + "," + str(layerNum - 2) + "," + str(hLayerSize) + "," + str(outputLayerSize)
    content = []
    content.append(structure)

    for l in range(layerNum - 1):
        if l == layerNum - 2:
            wSize = outputLayerSize
        else:
            wSize = hLayerSize
        if l == 0:
            loopCount = inputLayerSize
        else:
            loopCount = hLayerSize

        for i in range(loopCount):

            # All weights in input and hidden layer
            for j in range(1, wSize + 1):
                k = str(inputNN[l][i][j].value)
                content.append(k)

            # Biases in hidden layer
            if l != 0:
                k = str(inputNN[l][i][wSize + 1])
                content.append(k)

    # Output layer biases
    for i in range(outputLayerSize):
        k = str(inputNN[layerNum - 1][i][1])
        content.append(k)

    path = getCurrentDir() + "\\Saved Bots\\"
    saveNextTextFile(fileNameFirstPart, path, content)


#TODO: make a function that allows loading of the NN + entering data inputs manually

if __name__ == "__main__":
    # data = importTextFileDataSet("testDataSet.txt")
    data = [[[1, 1], [0]], [[0, 1], [1]], [[1, 0], [1]], [[0, 0], [0]]]
    # data = importTextFileDataSet("xorGateDataSet.txt")
    # tryBackProp(data, 10000, 2, 4, 4, 1)
    nn = createRandomNeuralNet(2,2,4,1)
    randomGeneration(nn,10000,2,2,4,1,data[0],data[i][1])


    # nn = loadNN("xorGate1.txt")
    # tryBackProp(data,300000,2,1,4,1,nn)
    # respone = ""
    # while respone!="x":
    #     ask1 = int(input("input 1"))
    #     ask2 = int(input("input 1"))
    #     nnIn = [ask1,ask2]
    #     out, nn = forwardProp(nnIn,nn)
    #     print(out)
    #
    #     respone = input("Press enter to go again")


