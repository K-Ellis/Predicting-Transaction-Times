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
from math import sqrt


def split_data(df):  # Split data into training and test data x, y.
    distribution = 0
    i = 0
    while distribution == 0:
        trainData, testData = train_test_split(df, test_size=0.2)  # Split data 80:20 randomly
        trainData_y = pd.DataFrame()
        trainData_y["TimeTaken"] = trainData["TimeTaken"]
        trainData_x = trainData.loc[:, trainData.columns != 'TimeTaken']
        testData_y = pd.DataFrame()
        testData_y["TimeTaken"] = testData["TimeTaken"]
        testData_x = testData.loc[:, testData.columns != 'TimeTaken']
        mean_train = sum(trainData_y["TimeTaken"].tolist())/len(trainData_y)
        mean_test = sum(testData_y["TimeTaken"].tolist()) / len(testData_y)
        std_train = np.std(trainData_y["TimeTaken"].tolist())
        std_test = np.std(testData_y["TimeTaken"].tolist())
        print(i, mean_train, mean_test, std_train, std_test)
        # Only accept a split with test data mean and std that is within 5% of train data mean and stc (stratification?)
        if (mean_train - mean_test) ** 2 < (mean_train * 0.05) ** 2:
            if (std_train - std_test) ** 2 < (std_train * 0.05) ** 2:
                distribution = 1
        i = i+1

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

    print("LinearRegression rmse:", sqrt(mean_squared_error(testData_y, y_pred)))  # Print Root Mean Squared Error
    # More tools in sklearn metrics or https://stackoverflow.com/questions/19068862/how-to-overplot-a-line-on-a-scatter-plot-in-python


if __name__ == "__main__":  # Run program
    np.random.seed(12345)  # Set seed
    df = pd.read_csv("../../../Data/vw_Incident_cleaned.csv", encoding='latin-1', low_memory=False)  # Read in csv file

    trainData_x, trainData_y, testData_x, testData_y = split_data(df)  # Split data

    linear_regression(trainData_x, trainData_y, testData_x, testData_y)  # Linear Regression
