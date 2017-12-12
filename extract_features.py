#!/usr/bin/env python3

import numpy as np
import os
import readData
import pickle

def serialize(obj, fName):
    with open(fName, 'wb') as f:
        pickle.dump(obj, f)
directory = 'train/'
files = sorted(os.listdir(directory))
dataSize = len(files)
trainingSet = 50
y = np.array([1 if 'M' in list(x) else 0 for x in files[0:trainingSet]])
serialize(y, 'y')
X = []
for i, f in enumerate(files):
    X.append(readData.getDistribution(directory + f)[0])
    print(f, "has been read")
    if i == trainingSet - 1:
        break

with open('X', 'wb') as f:
    pickle.dump(X, f)
