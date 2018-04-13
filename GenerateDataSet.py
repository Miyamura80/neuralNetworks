#TODO: Generalize this script so it generates any of these given a function with compression to 1 and 0


f = open("additionDataSet.txt","w")

for i in range(1,11):
    for j in range(1,11):
        f.write(str(i/20)+","+str(j/20)+":"+str((i+j)/20)+"\n")

f.close()



