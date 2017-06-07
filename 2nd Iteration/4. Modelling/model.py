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
import matplotlib.pyplot as plt
from pylab import polyfit
from sklearn.linear_model import LinearRegression
from math import sqrt


def split_data(df):
    trainData_y = pd.DataFrame()
    trainData_y["TimeTaken"] = trainData["TimeTaken"]
    trainData_x = trainData.loc[:, trainData.columns != 'TimeTaken']
    testData_y = pd.DataFrame()
    testData_y["TimeTaken"] = testData["TimeTaken"]
    testData_x = testData.loc[:, testData.columns != 'TimeTaken']

    # trainData_X.to_csv("../../../Data/trainData_X.csv", index = False)  # export file
    # trainData_y.to_csv("../../../Data/trainData_y.csv", index = False)  # export file
    return trainData_x, trainData_y, testData_x, testData_y


def linear_regression(trainData_x, trainData_y, testData_x, testData_y):
    classifier = LinearRegression()
    classifier = classifier.fit(trainData_x, trainData_y)
    y_pred = classifier.predict(testData_x)

    plt.plot(testData_y, y_pred, 'ro')
    plt.xlabel('testData_y')
    plt.ylabel('y_pred')
    plt.title('LinearRegression')
    plt.show()

    # print(rmse(testData_y, y_pred))


def rmse(testData_y, y_pred):
    error = []
    sq_error = []
    # for i in range(len(df)):
    #    error[i] = (testData_y[i] - y_pred[i])
    #    sq_error[i] = (df["Error"][i]) ^ 2
    print(type(testData_y))
    print(testData_y)
    print(np.shape(testData_y))
    print(type(y_pred))
    print(y_pred[2])
    print(np.shape(y_pred))
    # return sqrt(sum(sq_error)/len(sq_error))


if __name__ == "__main__":  # Run program
    np.random.seed(12345)
    df = pd.read_csv("../../../Data/preprocessed_data.csv", encoding='latin-1', low_memory=False)  # Read in csv file
    trainData, testData = train_test_split(df, test_size=0.2)  # Split data 80:20 randomly

    trainData_X, trainData_y, testData_X, testData_y = split_data(df)
    linear_regression(trainData_X, trainData_y, testData_X, testData_y)
