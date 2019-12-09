#!/usr/bin/env python3

import numpy as np
import os
import read_data
import pickle

def serialize(obj, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(obj, f)


def prepare_x(files, directory, training_set_count):
    X = []

    for i, f in enumerate(files):
        X.append(read_data.get_distribution(directory + f))
        print(f, "has been read")
        if i == training_set_count - 1:
            break
    
    return X


def prepare_y(files, training_set_count):
    return np.array([1 if 'M' in list(x) else 0 for x in files[0:training_set_count]])


def main():
    training_set_count = 50
    files = sorted(os.listdir(directory))

    y = prepare_y(files, training_set_count)
    serialize(y, 'y')

    X = prepare_x(files, 'train/', training_set_count)

    with open('X', 'wb') as f:
        pickle.dump(X, f)
    
    
if __name__== "__main__" :
    main()
