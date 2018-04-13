from NeuralNetFunctions import *

# data = importTextFileDataSet("testDataSet.txt")
data = [[[1, 1], [0]], [[0, 1], [1]], [[1, 0], [1]], [[0, 0], [0]]]
# data = importTextFileDataSet("xorGateDataSet.txt")
# tryBackProp(data, 10000, 2, 4, 4, 1)


nn = loadNN("xorGate4.txt")
# tryBackProp(data,300000,2,1,4,1,nn)
respone = ""
while respone != "x":
    ask1 = int(input("input 1"))
    ask2 = int(input("input 1"))
    nnIn = [ask1, ask2]
    out, nn = forwardProp(nnIn, nn)
    print(out)

    respone = input("Press enter to go again")