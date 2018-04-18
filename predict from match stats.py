#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 08:10:42 2018

@author: zac
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

y=[]

dataset = pd.read_csv('1.csv')
result = dataset.iloc[4, 0:3:2].values
result[0] = int(result[0].strip("()"))
result[1] = int(result[1].strip("()"))
result = int(result[0] > result[1])
y.append(result)
X = dataset.iloc[5:, 0:3:2].values
X = X.flatten()
X[18] = X[18].strip("%")
X[19] = X[19].strip("%")
x = [float(numeric_string) for numeric_string in X]
x = np.asarray(x)

    

dataset = pd.read_csv('2.csv')
result = dataset.iloc[4, 0:3:2].values
result[0] = int(result[0].strip("()"))
result[1] = int(result[1].strip("()"))
result = int(result[0] > result[1])
y.append(result)
X = dataset.iloc[5:, 0:3:2].values
X = X.flatten()
X[18] = X[18].strip("%")
X[19] = X[19].strip("%")
z = [float(numeric_string) for numeric_string in X]
z = np.asarray(z)

a = np.vstack((x,z))

for i in range(1,176):
    string = 'zac('+str(i)+').csv'
    dataset = pd.read_csv(string)
    result = dataset.iloc[4, 0:3:2].values
    result[0] = int(result[0].strip("()"))
    result[1] = int(result[1].strip("()"))
    result = int(result[0] > result[1])
    y.append(result)
    X = dataset.iloc[5:, 0:3:2].values
    X = X.flatten()
    X[18] = X[18].strip("%")
    X[19] = X[19].strip("%")
    x = [float(numeric_string) for numeric_string in X]
    x = np.asarray(x)
    a = np.vstack((a,x))
    
y = np.asarray(y)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
a_scaled = sc.fit_transform(a)

import keras
from keras.models import Sequential
from keras.layers import Dense


# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 26))

# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(a_scaled, y, batch_size = 10, epochs = 100)

y_test=[]

dataset = pd.read_csv('TEST.csv')
result = dataset.iloc[4, 0:3:2].values
result[0] = int(result[0].strip("()"))
result[1] = int(result[1].strip("()"))
result = int(result[0] > result[1])
y_test.append(result)
X = dataset.iloc[5:, 0:3:2].values
X = X.flatten()
X[18] = X[18].strip("%")
X[19] = X[19].strip("%")
x = [float(numeric_string) for numeric_string in X]
x = np.asarray(x)

    

dataset = pd.read_csv('TEST2.csv')
result = dataset.iloc[4, 0:3:2].values
result[0] = int(result[0].strip("()"))
result[1] = int(result[1].strip("()"))
result = int(result[0] > result[1])
y_test.append(result)
X = dataset.iloc[5:, 0:3:2].values
X = X.flatten()
X[18] = X[18].strip("%")
X[19] = X[19].strip("%")
z = [float(numeric_string) for numeric_string in X]
z = np.asarray(z)

a = np.vstack((x,z))

for i in range(1,25):
    string = 'test_set('+str(i)+').csv'
    dataset = pd.read_csv(string)
    result = dataset.iloc[4, 0:3:2].values
    result[0] = int(result[0].strip("()"))
    result[1] = int(result[1].strip("()"))
    result = int(result[0] > result[1])
    y_test.append(result)
    X = dataset.iloc[5:, 0:3:2].values
    X = X.flatten()
    X[18] = X[18].strip("%")
    X[19] = X[19].strip("%")
    x = [float(numeric_string) for numeric_string in X]
    x = np.asarray(x)
    a = np.vstack((a,x))
    
y_test = np.asarray(y_test)




from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
a = sc.fit_transform(a)

y_pred = classifier.predict(a)
test = y_pred
y_pred = (y_pred > 0.5)
right = 0
wrong = 0
for i in range(0,26):

    
    if int(y_pred[i][0]) == y_test[i]:
        right += 1
    else:
        wrong+=1
    
    
    if int(y_pred[i][0]) == 1 :
        ai = "Win"
    else:
        ai = "loss"

    if y_test[i] == 1 :
        actual = "win"
    else:
        actual = "loss"
    print("Game Number: " + str(i))
    print(test[i][0])
    print("**********")
    print("AI predicted: " + ai)
    print("Actual result: " + actual)
    print("******\n")

accuracy = right/26.0
print("***** Accuracy: " + str(accuracy) + "*****")

y_test=[]

dataset = pd.read_csv('CARLTvCOLL.csv')
result = dataset.iloc[4, 0:3:2].values
result[0] = int(result[0].strip("()"))
result[1] = int(result[1].strip("()"))
result = int(result[0] > result[1])
y_test.append(result)
X = dataset.iloc[5:, 0:3:2].values
X = X.flatten()
X[18] = X[18].strip("%")
X[19] = X[19].strip("%")
x = [float(numeric_string) for numeric_string in X]
x = np.asarray(x)

    

dataset = pd.read_csv('TEST2.csv')
result = dataset.iloc[4, 0:3:2].values
result[0] = int(result[0].strip("()"))
result[1] = int(result[1].strip("()"))
result = int(result[0] > result[1])
y_test.append(result)
X = dataset.iloc[5:, 0:3:2].values
X = X.flatten()
X[18] = X[18].strip("%")
X[19] = X[19].strip("%")
z = [float(numeric_string) for numeric_string in X]
z = np.asarray(z)

a = np.vstack((x,z))

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
a = sc.fit_transform(a)

y_pred = classifier.predict(a)


perc = y_pred[0][0]
print("%1.9f"%perc)







