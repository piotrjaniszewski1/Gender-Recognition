#!/usr/bin/env python3

from sklearn import linear_model
import pickle
import numpy as np


def load(file_name):
    with open(file_name, 'rb') as f:
        x = pickle.load(f)
    return x


def save(file_name, obj):
    with open(file_name, 'wb') as f:
        pickle.dump(obj, f)


def main():
    train_data_file = 'X'
    y_data_file = 'y'

    X = load(train_data_file)
    y = load(y_data_file)
    print('Loaded', train_data_file, 'and', y_data_file)

    logreg = linear_model.LogisticRegression()

    logreg.fit(X, y)
    tmp = logreg.predict(X)
    result = [x == y for (x, y) in zip(tmp, y)]
    positive = sum(result)

    print('How many ', len(result), 'Positive', positive)
    save('logreg_object', logreg)

    
if __name__== "__main__" :
    main()