import os

import shutil

source_dir = "./fcbh_raw/Training_data/"

destPath = "C:/DS/Machine Learning/lingua-franca/fcbh/"

labelsDict = {}
labelsFile = open("fcbh_trainingData.csv", "r+")

for line in labelsFile:
    splt = line.split(",")
    labelsDict[splt[0]] = splt[1][:-1]
    # dest = destPath + splt[1][:-1]

    print("dest: " + dest)

    if not os.path.isdir(dest):
            os.makedirs(dest)
            print('Directory created at: ' + dest)

    oldLoc = source_dir + splt[0]
    if os.path.isfile(oldLoc):
        newLoc = destPath + splt[1][:-1] + "/" + splt[0]

        shutil.move(oldLoc, newLoc)