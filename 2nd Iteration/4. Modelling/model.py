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
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from math import sqrt
from pylab import polyfit


def split_data(df):
    trainData, testData = train_test_split(df, test_size=0.2)  # Split data 80:20 randomly
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
    plt.axis('equal')
    plt.ylim(0, 2000000)
    plt.xlim(0, 2000000)
    plt.show()

    print("rmse:", sqrt(mean_squared_error(testData_y, y_pred)))  # Print Root Mean Squared Error
    # More tools in sklearn metrics


if __name__ == "__main__":  # Run program
    np.random.seed(12345)  # Set seed
    df = pd.read_csv("../../../Data/preprocessed_data.csv", encoding='latin-1', low_memory=False)  # Read in csv file

    trainData_x, trainData_y, testData_x, testData_y = split_data(df)  # Split data


    # if data is not well spaced out
    # compare mean
    # compare standard deviation
    # if test is not within 10% of train, redo

    # compare standard deviation
    # print(mean(trainData_y))
    # print(mean(testData_y))
        # print("Data not well stratified, retrying . . .")
        # trainData, testData = train_test_split(df, test_size=0.2)  # Split data 80:20 randomly


    linear_regression(trainData_x, trainData_y, testData_x, testData_y)  # Linear Regression