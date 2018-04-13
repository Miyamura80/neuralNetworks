from GenerallyUsefulFunctions import readFile, getCurrentDir
from random import randint
import gym

#Change manually
HIGHEST_LENGTH_WORD = 20

def writeWordAsBit(word):
    abc = "abcdefghijklmnopqrstuvwxyz"
    bitWord = ""
    for i in word:
        k = abc.find(i)
        bitWord += str((k+1)/26)
        bitWord += ","
    return bitWord

def compileLangages():
    english = readFile(getCurrentDir()+"\englishShort.txt")
    notEnglish = readFile(getCurrentDir() + "\spanish.txt")
    eLen = len(english)
    jLen = len(notEnglish)

    compileF = open("compiledDataSet.txt","w")
    jCount = 0
    eCount = 0
    # highestLen = 0
    for i in range(max(eLen,jLen)):
        if jCount < jLen:
            # if len(notEnglish[jCount]) > highestLen:
            #     highestLen = len(notEnglish[jCount])
            wordL = HIGHEST_LENGTH_WORD-len(notEnglish[jCount])
            compileF.write(writeWordAsBit(notEnglish[jCount]))
            for j in range(wordL-1):
                compileF.write("0,")
            compileF.write("0")
            compileF.write(":0\n")
            jCount += 1
        if eCount < eLen:
            # if len(english[eCount]) > highestLen:
            #     highestLen = len(english[eCount])
            wordL = HIGHEST_LENGTH_WORD-len(english[eCount])
            compileF.write(writeWordAsBit(english[eCount]))
            for j in range(wordL-1):
                compileF.write("0,")
            compileF.write("0")
            compileF.write(":1\n")
            eCount += 1
    # print(highestLen)
    compileF.close()

def removeNonAlphabetFromFile(textFile):
    spanish = readFile(getCurrentDir()+"\\"+textFile)
    f = open(textFile+"BadRemoved.txt","w")
    abc = "abcdefghijklmnopqrstuvwxyz"
    for i in spanish:
        k = i.find(" ")
        writeT = i[0:k]
        output = ""
        for j in writeT:
            if j in abc:
                output += j
        f.write(output+"\n")
    f.close()

def shuffleFile(textFile):
    shuffleList = readFile(getCurrentDir() + "\\"+textFile)
    for i in range(len(shuffleList)):
        ran = randint(0, len(shuffleList) - 1)
        shuffleList[i] = shuffleList[ran]
    f = open(textFile+"Shuffled.txt", "w")
    for i in range(len(shuffleList)):
        f.write(shuffleList[i])
    f.close()


compileLangages()
"""
Max is 18 -> put up to 20
"""