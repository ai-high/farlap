#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 14:26:02 2018

@author: zac
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
predictions = [0,0,0,0,0,0,0,0,0]
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


for h in range(0,20):
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
    
    # [scoring schots, goals]
    
    extra = []
    extra.append([15,13,12.3,8])
    extra.append([15.7,11,12.7,11.3])
    extra.append([14.7,13.7,12,10.7])
    extra.append([10,15,14.7,8.3])
    extra.append([10.3,11.3,11.7,9.3])
    extra.append([15,10.7,10.7,10.7])
    extra.append([13.3,10.7,14.7,12.3])
    extra.append([15,12.7,15,15.7])
    extra.append([14.3,12,9.3,12.3])
    
    i = 0
    string = 'game'+str(1)+'.csv'
    dataset = pd.read_csv(string)
    X = dataset.values
    X = np.delete(X,0,1)
    X = np.delete(X,1,1)
    X = np.delete(X,2,1)
    X = np.delete(X,3,1)
    X = np.delete(X,3,1)
    X = np.delete(X,3,1)
    X = np.delete(X,3,1)
    X = np.delete(X,3,1)
    X = np.delete(X,3,1)
    X = np.delete(X,3,1)
    X = np.delete(X,3,1)
    X = np.delete(X,4,1)
    X = np.delete(X,4,1)
    X = np.delete(X,4,1)
    X = np.delete(X,4,1)
    X = np.delete(X,4,1)
    X = np.delete(X,5,1)
    X = np.delete(X,5,1)
    X = np.delete(X,5,1)
    X = np.delete(X,5,1)
    X = np.delete(X,5,1)
    X = np.delete(X,5,1)
    X = np.delete(X,5,1)
    X = np.delete(X,5,1)
    X[0][5] = X[0][10]
    X[1][5] = X[1][10]
    X = np.delete(X,7,1)
    X[0][8] = extra[i][0] + extra[i][1]
    X[1][8] = extra[i][2] + extra[i][3]
    X = np.delete(X,9,1)
    X = np.delete(X,9,1)
    X = np.delete(X,9,1)
    X = np.delete(X,9,1)
    X = np.delete(X,9,1)
    X = np.delete(X,9,1)
    X = np.delete(X,9,1)
    X = np.insert(X, [10], [[1],[1]], axis = 1)
    X = np.insert(X, [11], [[0],[0]], axis = 1)
    X = np.insert(X, [12], [[0],[0]], axis = 1)
    X[0][11] = X[0][2]/float(extra[i][0])
    X[1][11] = X[1][2]/float(extra[i][2])
    X[0][12] = X[0][2]/float(extra[i][0] + extra[i][1])
    X[1][12] = X[1][2]/float(extra[i][2] + extra[i][3])
    x = np.zeros(26)
    q = 0
    j = 0
    for g in range(0,26):
        if g%2 == 0 :
            x[g] = X[0][q]
            q+=1
        else:
            x[g] = X[1][j]
            j+=1
    
    
    i = 1
    string = 'game'+str(2)+'.csv'
    dataset = pd.read_csv(string)
    X = dataset.values
    X = np.delete(X,0,1)
    X = np.delete(X,1,1)
    X = np.delete(X,2,1)
    X = np.delete(X,3,1)
    X = np.delete(X,3,1)
    X = np.delete(X,3,1)
    X = np.delete(X,3,1)
    X = np.delete(X,3,1)
    X = np.delete(X,3,1)
    X = np.delete(X,3,1)
    X = np.delete(X,3,1)
    X = np.delete(X,4,1)
    X = np.delete(X,4,1)
    X = np.delete(X,4,1)
    X = np.delete(X,4,1)
    X = np.delete(X,4,1)
    X = np.delete(X,5,1)
    X = np.delete(X,5,1)
    X = np.delete(X,5,1)
    X = np.delete(X,5,1)
    X = np.delete(X,5,1)
    X = np.delete(X,5,1)
    X = np.delete(X,5,1)
    X = np.delete(X,5,1)
    X[0][5] = X[0][10]
    X[1][5] = X[1][10]
    X = np.delete(X,7,1)
    X[0][8] = extra[i][0] + extra[i][1]
    X[1][8] = extra[i][2] + extra[i][3]
    X = np.delete(X,9,1)
    X = np.delete(X,9,1)
    X = np.delete(X,9,1)
    X = np.delete(X,9,1)
    X = np.delete(X,9,1)
    X = np.delete(X,9,1)
    X = np.delete(X,9,1)
    X = np.insert(X, [10], [[1],[1]], axis = 1)
    X = np.insert(X, [11], [[0],[0]], axis = 1)
    X = np.insert(X, [12], [[0],[0]], axis = 1)
    X[0][11] = X[0][2]/float(extra[i][0])
    X[1][11] = X[1][2]/float(extra[i][2])
    X[0][12] = X[0][2]/float(extra[i][0] + extra[i][1])
    X[1][12] = X[1][2]/float(extra[i][2] + extra[i][3])
    z = np.zeros(26)
    q = 0
    j = 0
    for g in range(0,26):
        if g%2 == 0 :
            z[g] = X[0][q]
            q+=1
        else:
            z[g] = X[1][j]
            j+=1
    
    
    a = np.vstack((x,z))
    
    for i in range(2,9):
        string = 'game'+str(i+1)+'.csv'
        dataset = pd.read_csv(string)
        X = dataset.values
        X = np.delete(X,0,1)
        X = np.delete(X,1,1)
        X = np.delete(X,2,1)
        X = np.delete(X,3,1)
        X = np.delete(X,3,1)
        X = np.delete(X,3,1)
        X = np.delete(X,3,1)
        X = np.delete(X,3,1)
        X = np.delete(X,3,1)
        X = np.delete(X,3,1)
        X = np.delete(X,3,1)
        X = np.delete(X,4,1)
        X = np.delete(X,4,1)
        X = np.delete(X,4,1)
        X = np.delete(X,4,1)
        X = np.delete(X,4,1)
        X = np.delete(X,5,1)
        X = np.delete(X,5,1)
        X = np.delete(X,5,1)
        X = np.delete(X,5,1)
        X = np.delete(X,5,1)
        X = np.delete(X,5,1)
        X = np.delete(X,5,1)
        X = np.delete(X,5,1)
        X[0][5] = X[0][10]
        X[1][5] = X[1][10]
        X = np.delete(X,7,1)
        X[0][8] = extra[i][0] + extra[i][1]
        X[1][8] = extra[i][2] + extra[i][3]
        X = np.delete(X,9,1)
        X = np.delete(X,9,1)
        X = np.delete(X,9,1)
        X = np.delete(X,9,1)
        X = np.delete(X,9,1)
        X = np.delete(X,9,1)
        X = np.delete(X,9,1)
        X = np.insert(X, [10], [[1],[1]], axis = 1)
        X = np.insert(X, [11], [[0],[0]], axis = 1)
        X = np.insert(X, [12], [[0],[0]], axis = 1)
        X[0][11] = X[0][2]/float(extra[i][0])
        X[1][11] = X[1][2]/float(extra[i][2])
        X[0][12] = X[0][2]/float(extra[i][0] + extra[i][1])
        X[1][12] = X[1][2]/float(extra[i][2] + extra[i][3])
        z = np.zeros(26)
        q = 0
        j = 0
        for g in range(0,26):
            if g%2 == 0 :
                z[g] = X[0][q]
                q+=1
            else:
                z[g] = X[1][j]
                j+=1
        a = np.vstack((a,z))
        
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    a = sc.fit_transform(a)
    y_pred = classifier.predict(a)
    y_pred = (y_pred > 0.5)
    for i in range(0,9):
        predictions[i] += int(y_pred[i])
            
            
print(predictions)  
