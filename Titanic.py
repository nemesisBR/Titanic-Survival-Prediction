# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset = pd.read_csv("F:\Kaggle\Titanic\\train.csv")

X = dataset.iloc[:, 1:11].values
Y = dataset.iloc[:, 11].values

names = dataset.iloc[:,2]
temp_title = []
title = []

for i in range(0,len(names)):
    temp_name = names[i]
    space = temp_name.find(" ")
    end = temp_name.find(".")
    if space != -1 and end != -1:
        temp_title.append(temp_name[space+1:end])

for i in range(0,len(temp_title)):
    temp_name = temp_title[i]
    space = temp_name.find(" ")
    if space != -1:
        title.append(temp_name[space+1:])
    else:
        title.append(temp_name)

X[:, 1] = title

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
imputer = imputer.fit(X[:, 3:4])
X[:, 3:4] = imputer.transform(X[:, 3:4])

temp_string = X[:, 8:10]
temp_string = pd.DataFrame(temp_string)
temp_string = temp_string.ffill().bfill()
X[:, 8:10] = temp_string.iloc[:, :].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 1] = labelencoder.fit_transform(X[:, 1])
X[:, 2] = labelencoder.fit_transform(X[:, 2])
X[:, 8] = labelencoder.fit_transform(X[:, 8])
X[:, 9] = labelencoder.fit_transform(X[:, 9])

X = np.delete(X, (6), axis=1)

X_Copy = pd.DataFrame(X)

'''-----------------------------------Data Cleansing Done - Extracted Titles - Removed Ticket Column -----------------

onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()
test = X[:, 1]
split after changing features'''
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.3, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
'''--------------------LOGISTICS REGRESSION----------------'''
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=60, criterion='entropy', random_state=0)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

'''------Training Done----------'''

test_dataset = pd.read_csv("F:\Kaggle\Titanic\\test.csv")
passengerID = test_dataset.iloc[:, 0:1].values
X2 = test_dataset.iloc[:,1:11].values

names2 = test_dataset.iloc[:,2]
temp_title2 = []
title2 = []

for i in range(0,len(names2)):
    temp_name2 = names2[i]
    space2 = temp_name2.find(" ")
    end2 = temp_name2.find(".")
    if space2 != -1 and end2 != -1:
        temp_title2.append(temp_name2[space2+1:end2])

for i in range(0,len(temp_title2)):
    temp_name2 = temp_title2[i]
    space2 = temp_name2.find(" ")
    if space2 != -1:
        title2.append(temp_name2[space+1:])
    else:
        title2.append(temp_name2)

X2[:, 1] = title2

from sklearn.preprocessing import Imputer
imputer2 = Imputer(missing_values="NaN", strategy="mean", axis=0)
imputer2 = imputer2.fit(X2[:, 3:4])
X2[:, 3:4] = imputer2.transform(X2[:, 3:4])

temp_string2 = X2[:,:]
temp_string2 = pd.DataFrame(temp_string2)
temp_string2 = temp_string2.ffill().bfill()
X2[:, :] = temp_string2.iloc[:, :].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder2 = LabelEncoder()
X2[:, 1] = labelencoder2.fit_transform(X2[:, 1])
X2[:, 2] = labelencoder2.fit_transform(X2[:, 2])
X2[:, 8] = labelencoder2.fit_transform(X2[:, 8])
X2[:, 9] = labelencoder2.fit_transform(X2[:, 9])

X2 = np.delete(X2, (6), axis=1)
X2_Copy = pd.DataFrame(X2)

X2_Copy.isnull().sum()



y_pred2 = classifier.predict(X2)
y_pred2 = y_pred2.reshape((418,1))

passengerID = pd.DataFrame(passengerID)
y_pred2 = pd.DataFrame(y_pred2)

result = pd.concat([passengerID, y_pred2], axis =1, names = ['PassengerId', 'Survived'])
result = pd.DataFrame(result)
result.to_csv('output3.csv', index=False )