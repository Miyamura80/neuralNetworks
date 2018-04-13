import matplotlib.pyplot as plt
from NeuralNetFunctions import *

def plotData(list,gen):
    plt.plot(list)
    plt.ylabel("Average error value")
    plt.xlabel("Generation")
    plt.title("Error values over generations")
    plt.show()

def generationAverage(lst):
    total = 0
    length = len(lst)
    for i in range(length):
        k = lst[i]
        total += k
    return total/length

genNum = 900
mutationChance = 100



Tinputs = [0.5,0.10]
expectedOutput = [0.25,0.5]
errorValues = [0,0]
averagePerformanceList = []

bestnn = createRandomNeuralNet(2,2,2,2)

for i in range(genNum):
    if i==0:
        mNN1,mNN2,pPList = randomGeneration(bestnn,100,2,2,2,2,Tinputs,expectedOutput, False,mutationChance)
    else:
        mNN1, mNN2, pPList = randomGeneration(bestnn, 100, 2, 2, 2, 2, Tinputs, expectedOutput, True,mutationChance)
    averagePerformanceList.append(generationAverage(pPList))

plotData(averagePerformanceList,0)

finalOut = forwardProp(Tinputs,bestnn)
print(finalOut)



