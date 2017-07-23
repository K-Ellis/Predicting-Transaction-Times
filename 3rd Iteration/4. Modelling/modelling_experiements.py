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
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from math import sqrt
from sklearn import preprocessing


def histogram(df, column):  # Create histogram of preprocessed data
    plt.figure()  # Plot all data
    plt.hist(df[column], bins='auto')
    plt.xlabel('TimeTaken (Seconds)')
    plt.ylabel('Frequency')
    plt.title(column + " all data")
    plt.savefig("../../../Logs/" + column + "_all.png")

    # plt.figure()  # Plot times under 500,000 seconds
    # plt.hist(df[df.TimeTaken < 500000][column], bins='auto')
    # plt.xlabel('TimeTaken (Seconds)')
    # plt.ylabel('Frequency')
    # plt.title(column + " < 500000s data")
    # plt.savefig("../../../Logs/" + column + "_500000.png")

    # plt.figure()  # Plot times under 100,000 seconds
    # plt.hist(df[df.TimeTaken < 100000][column], bins='auto')
    # plt.xlabel('TimeTaken (Seconds)')
    # plt.ylabel('Frequency')
    # plt.title(column + " < 100000s data")
    # plt.savefig("../../../Logs/" + column + "_100000.png")

    # plt.figure()  # Plot all data
    # plt.hist(np.log(df[column]), bins='auto')
    # plt.xlabel('Log of TimeTaken (Seconds)')
    # plt.ylabel('Frequency')
    # plt.title(column + " Log of all data")
    # plt.savefig("../../../Logs/" + column + "_log_all.png")

    # plt.figure()  # Plot times under 500,000 seconds
    # plt.hist(np.log(df[df.TimeTaken < 500000][column]), bins='auto')
    # plt.xlabel('Log of TimeTaken (Seconds)')
    # plt.ylabel('Frequency')
    # plt.title(column + " Log of < 500000s data")
    # plt.savefig("../../../Logs/" + column + "_log_500000.png")

    # plt.figure()  # Plot times under 100,000 seconds
    # plt.hist(np.log(df[df.TimeTaken < 100000][column]), bins='auto')
    # plt.xlabel('Log of TimeTaken (Seconds)')
    # plt.ylabel('Frequency')
    # plt.title(column + " Log of < 100000s data")
    # plt.savefig("../../../Logs/" + column + "_log_100000.png")


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
        # Only accept a split with test data mean and std that is within 5% of train data mean and stc (stratification?)
        if (mean_train - mean_test) ** 2 < (mean_train * 0.05) ** 2:
            if (std_train - std_test) ** 2 < (std_train * 0.05) ** 2:
                distribution = 1
        i = i+1
    print("Number of iterations taken to get good data split:", i)
    print("Mean value of Train Y:", mean_train)
    print("Mean value of Test Y:", mean_test)
    print("Standard deviation of train Y:", std_train)
    print("Standard deviation of test Y:", std_test, "\n")

    # trainData_X.to_csv("../../../Data/trainData_X.csv", index = False)  # export file
    # trainData_y.to_csv("../../../Data/trainData_y.csv", index = False)  # export file
    return trainData_x, trainData_y, testData_x, testData_y


def linear_regression(trainData_x, trainData_y, testData_x, testData_y, scaler):
    classifier = LinearRegression()
    classifier = classifier.fit(trainData_x, trainData_y)
    y_pred = classifier.predict(testData_x)
    y_train_pred = classifier.predict(trainData_x)
    results(testData_y, y_pred, trainData_y, y_train_pred, "LinearRegression")


def elastic_net(trainData_x, trainData_y, testData_x, testData_y):  # Elastic Net
    classifier = ElasticNet()
    classifier = classifier.fit(trainData_x, trainData_y)
    # print(regressor.coef_)
    y_pred = classifier.predict(testData_x)
    y_train_pred = classifier.predict(trainData_x)
    results(testData_y, y_pred, trainData_y, y_train_pred, "ElasticNet")


def kernel_ridge(trainData_x, trainData_y, testData_x, testData_y):  # Kernel ridge regression
    classifier = KernelRidge()
    classifier = classifier.fit(trainData_x, trainData_y)
    y_pred = classifier.predict(testData_x)
    y_train_pred = classifier.predict(trainData_x)
    results(testData_y, y_pred, trainData_y, y_train_pred, "KernelRidge")


def results(testData_y, y_pred, trainData_y, y_train_pred, alg):
    y_pred = (scaler.inverse_transform(y_pred)) ** 2
    y_train_pred = (scaler.inverse_transform(y_train_pred)) ** 2

    trainData_y = (scaler.inverse_transform(trainData_y)) ** 2
    testData_y = (scaler.inverse_transform(testData_y)) ** 2

    plt.figure()
    plt.plot(trainData_y, y_train_pred, 'ro')
    plt.xlabel('trainData_y')
    plt.ylabel('y_train_pred')
    plt.title(alg + " - Train Data")
    plt.axis('equal')
    plt.ylim(0, 2000000)
    plt.xlim(0, 2000000)
    plt.savefig("../../../Logs/" + alg + "train.png")

    plt.figure()
    plt.plot(testData_y, y_pred, 'ro')
    plt.xlabel('testData_y')
    plt.ylabel('y_pred')
    plt.title(alg + " - Test Data")
    plt.axis('equal')
    plt.ylim(0, 2000000)
    plt.xlim(0, 2000000)
    plt.savefig("../../../Logs/" + alg + "test.png")

    print(alg, "Train rmse:", sqrt(mean_squared_error(trainData_y, y_train_pred)))  # Print Root Mean Squared Error
    print(alg, "Test rmse:", sqrt(mean_squared_error(testData_y, y_pred)))  # Print Root Mean Squared Error

    out_file_name = "../../../Logs/" + time.strftime("%Y%m%d-%H%M%S") + "_" + alg + ".txt"  # Log file name
    out_file = open(out_file_name, "w")  # Open log file
    out_file.write(alg + " " + time.strftime("%Y%m%d-%H%M%S") + "\n")
    out_file.write(alg + " Train RMSE: " + str(sqrt(mean_squared_error(trainData_y, y_train_pred))) + "\n")
    out_file.write(alg + " Test RMSE: " + str(sqrt(mean_squared_error(testData_y, y_pred))) + "\n")
    out_file.write(alg + " Train R^2 scoree: " + str(r2_score(trainData_y, y_train_pred)) + "\n")
    out_file.write(alg + " Test R^2 score: " + str(r2_score(testData_y, y_pred)) + "\n")
    out_file.close()

    print(alg, "Train R^2 score:", r2_score(trainData_y, y_train_pred))  # Print R Squared
    print(alg, "Test R^2 score:", r2_score(testData_y, y_pred), "\n")  # Print R Squared
    # http://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html


if __name__ == "__main__":  # Run program
    np.random.seed(12345)  # Set seed
    df = pd.read_csv("../../../Data/vw_Incident_cleaned.csv", encoding='latin-1', low_memory=False)  # Read in csv file

    # histogram(df, "TimeTaken")  # Save histogram plots of TimeTaken

    scaler = preprocessing.StandardScaler()

    X = df.drop('TimeTaken', axis=1)
    y = scaler.fit_transform(np.sqrt(df.TimeTaken))

    plt.hist(y)
    plt.xlabel('TimeTaken (Scaled Seconds)')
    plt.ylabel('Frequency')
    plt.title("TimeTaken Scaled Histogram" + " all data")
    plt.savefig("../../../Logs/TimeTaken_scaled" + "_all.png")
    # plt.show()

    trainData_x, testData_x, trainData_y, testData_y = train_test_split(X, y, test_size=0.2)


    linear_regression(trainData_x, trainData_y, testData_x, testData_y, scaler)  # Linear Regression
    # elastic_net(trainData_x, trainData_y, testData_x, testData_y)  # elastic net
    # kernel_ridge(trainData_x, trainData_y, testData_x, testData_y)  # Kernel ridge regression
