# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 18:36:55 2020

@author: mznid
"""

######

import sklearn
import tensorflow as tf
import random
import pandas as pd
import numpy as np
from apyori import apriori
from sklearn.preprocessing import MinMaxScaler

from sklearn.naive_bayes import GaussianNB , MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


from tensorflow import keras   
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Activation, Dropout, AveragePooling2D, Conv1D, MaxPooling1D
from tensorflow.keras import Sequential
import matplotlib.pyplot as plt


rawsplice = pd.read_csv("D:\\splicedata\\splice.data" , header = None)

rawsplice.shape


stripped = []
for index, row in rawsplice.iterrows():
    tempstring = row[2]
    stripped.append(tempstring.strip())



a = np.zeros(shape=(len(stripped),len(stripped[0])))

emptydf = pd.DataFrame(a, columns = list(range(0,60)))


arrays = []
for index, row in emptydf.iterrows():
    array = tf.keras.utils.to_categorical(np.array(row), dtype = object, num_classes = 8)
    print(array)
    break


valdict = {'A': 0, 'C': 1, 'D': 2, 'G': 3, 'N': 4, 'R': 5, 'S': 6, 'T': 7}

rawlabels = rawsplice.iloc[:,0]
labelset = set(rawlabels)
labellist = list(labelset)
labeldict = {'IE': 0,'N': 1,'EI': 2}
numberlabels = []
for each in rawlabels:
    numberlabels.append(labeldict[each])

rowcounter = 0
for each in stripped:
    colcounter = 0
    for letter in each:
        emptydf.iat[rowcounter,colcounter] = valdict[letter]   
        colcounter += 1
    rowcounter += 1



index = list(range(0,len(rawlabels)))
trainindex = random.sample(index, round(0.7 * len(rawlabels)))
testindex = []
for each in index:
    if each not in trainindex:
        testindex.append(each)   
random.shuffle(testindex)




emptydfhot = tf.keras.utils.to_categorical(emptydf, num_classes = len(valdict))
emptydfhot[0].shape
type(emptydfhot[0]) 

labelshot = tf.keras.utils.to_categorical(numberlabels)




#from sklearn.preprocessing import LabelBinarizer
#lb = LabelBinarizer()
#df['new'] = lb.fit_transform(df['ABC']).tolist()


train = emptydfhot[trainindex,] 
test = emptydfhot[testindex,] 

train.shape
test.shape


labeltrain = []
labeltest = []
for each in trainindex:
    labeltrain.append(labelshot[each,])
for each in testindex:
    labeltest.append(labelshot[each,])
labeltrain = np.array(labeltrain)
labeltest = np.array(labeltest)





##########################

nn = Sequential()

nn.add(Flatten(input_shape = train[0].shape))

#nn.add(Dense(128, activation = 'relu'))
#nn.add(Dropout(0.1))

nn.add(Dense(64, activation = 'relu'))
nn.add(Dropout(0.1))

nn.add(Dense(32, activation = 'relu'))
nn.add(Dropout(0.1))

nn.add(Dense(16, activation = 'relu'))

nn.add(Dense(len(labeldict), activation = 'softmax'))

nn.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

####################################




nn.fit(train, labeltrain, epochs=10, validation_data=(test, labeltest))

nnpredictions = nn.predict(test)

labeldictreverse = dict(map(reversed, labeldict.items()))


testlabellist = []
for each in labeltest:
    if each[0] == 1:
        testlabellist.append('IE')
    elif each [1] == 1:
        testlabellist.append('N')
    else:
        testlabellist.append('EI')
        



preds = []
for each in nnpredictions:
    index = np.argmax(each)
    preds.append(labeldictreverse[index])
    
    
correct = []    
for each in range(0,len(preds)):
    if preds[each] == testlabellist[each]:
        correct.append(1)
    else:
        correct.append(0)
        
accuracy = sum(correct)/len(correct)
print(accuracy)        

     

cf = pd.crosstab(np.array(preds), np.array(testlabellist))



confusedrows = []
counter = 0
for index, row in emptydf.iloc[testindex,:].iterrows():
    if correct[counter] == 0:
        confusedrows.append(row)
    counter += 1




len(confusedrows)
confusedrows[0]

diags = []
for each in confusedrows:
    diag = []
    for those in each:
        if those == 2:
            diag.append(2)
        if those == 4:
            diag.append(4)
        if those == 5:
            diag.append(5)
        if those == 6:
            diag.append(6)
        
    diags.append(diag)
        









