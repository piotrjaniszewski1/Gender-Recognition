#!/usr/bin/env python3

from sklearn import linear_model
import pickle

def load(fName):
    with open(fName, 'rb') as f:
        x = pickle.load(f)
    return x

def save(fName, obj):
    with open(fName, 'wb') as f:
        pickle.dump(obj, f)

trainDataFile = 'X'
yDataFile = 'y'

X = load(trainDataFile)
y = load(yDataFile)

print('Loaded', trainDataFile, 'and', yDataFile)

logreg = linear_model.LogisticRegression()

logreg.fit(X, y)
tmp = logreg.predict(X)
result = [x==y for (x, y) in zip(tmp, y)]
positive = sum(result)
print('How many ', len(result), 'Positive', positive)
save('logreg_object', logreg)

