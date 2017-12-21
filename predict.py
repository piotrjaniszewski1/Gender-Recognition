#!/usr/bin/env python3

import readData
import pickle
import os
import numpy as np

def load(fName):
   with open(fName, 'rb') as f:
       x = pickle.load(f)
   return x

directory = 'test/'
logregFile = 'logreg_object'
logreg = load(logregFile)
failureCounter = 0
files = sorted(os.listdir(directory))
for f in files:
    features = np.array(readData.getDistribution(directory + f)).reshape(1,-1)
    print(f + ': ', end='')
    print(logreg.predict(features), end='')
    if logreg.predict(features) == [1] and 'K' in list(f):
        print('Failure')
        failureCounter += 1
    elif logreg.predict(features) == [0] and 'M' in list(f):
        failureCounter += 1
        print('Failure')
    else:
        print('Succes')
setSize = len(files)
print('Accuracy:', (setSize - failureCounter)/setSize)

