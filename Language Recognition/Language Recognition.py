from NeuralNetFunctions import *
from GenerallyUsefulFunctions import getCurrentDir
def writeWordAsBit(word):
    abc = "abcdefghijklmnopqrstuvwxyz"
    bitWord = ""
    for i in word:
        k = abc.find(i)
        bitWord += str((k+1)/26)
        bitWord += ","
    return bitWord



# data = importTextFileDataSet("compiledDataSet.txt")
# tryBackProp(data, 10000, 20, 2, 10, 1)

nn = loadNN("lanTest6.txt")
# tryBackProp(data,200,20,2,30,1,nn)

respone = ""
while respone != "x":
    ask1 = input("input 1")
    ask = writeWordAsBit(ask1)
    ask = ask[:(len(ask)-1)]
    ask = ask.split(",")
    l = len(ask)
    for i in range(20):
        if i < l:
            ask[i] = float(ask[i])
        else:
            ask.append(0)
    out, nn = forwardProp(ask, nn)
    print(out)

    respone = input("Press enter to go again")

