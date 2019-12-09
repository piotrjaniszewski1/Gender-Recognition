#!/usr/bin/env python3

import read_data
import pickle
import os
import numpy as np

def load(fName):
   with open(fName, 'rb') as f:
       x = pickle.load(f)
   return x


def predict(files, directory):
    logreg_file = 'logreg_object'
    logreg = load(logreg_file)

    for f in files:
        features = np.array(read_data.get_distribution(directory + f)).reshape(1,-1)
        print(f + ': ', end='')
        print(logreg.predict(features), end='')
        if logreg.predict(features) == [1] and 'K' in list(f):
            print('Failure')
            failure_counter += 1
        elif logreg.predict(features) == [0] and 'M' in list(f):
            failure_counter += 1
            print('Failure')
        else:
            print('Succes')


def main():
    failure_counter = 0
    files = sorted(os.listdir(directory))

    predict(files, 'test/')

    set_size = len(files)
    print('Accuracy:', (set_size - failure_counter)/set_size)

    
if __name__== "__main__" :
    main()

