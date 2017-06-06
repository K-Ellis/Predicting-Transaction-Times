"""****************************************************************************
UCD MSc Business Analytics Capstone Project - Predicting Transactions Times
*******************************************************************************
Iteration 2
Data modelling program
*******************************************************************************
Eoin Carroll
Kieron Ellis
*******************************************************************************
Working on dataset from Cosmic launch (6th Feb) to End March
****************************************************************************"""


"""****************************************************************************
Import libraries
****************************************************************************"""


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression


np.random.seed(12345)
df = pd.read_csv("../../../Data/preprocessed_data.csv", encoding='latin-1', low_memory=False)
trainData, testData = train_test_split(df, test_size=0.2)

trainData_X = trainData["TimeTaken"]
trainData_y = trainData.ix[:, trainData.columns != 'TimeTaken']
testData_X = testData["TimeTaken"]
testData_y = testData.ix[:, testData.columns != 'TimeTaken']

classifier = LinearRegression()
classifier = classifier.fit(trainData_X, trainData_y)
print ("Accuracy:", accuracy_score(testData_y, y_pred))