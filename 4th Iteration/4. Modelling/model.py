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
import os  # Used to create folders
import getpass  # Used to check PC name
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from math import sqrt
# from ask_user_which_file import ask_user


def histogram(df, column, newpath):  # Create histogram of preprocessed data
    plt.figure()  # Plot all data
    plt.hist(df[column], bins='auto')
    plt.xlabel('TimeTaken (Seconds)')
    plt.ylabel('Frequency')
    plt.title(column + " all data")
    plt.savefig(newpath + column + "_all.png")

    plt.figure()  # Plot times under 500,000 seconds
    plt.hist(df[df.TimeTaken < 500000][column], bins='auto')
    plt.xlabel('TimeTaken (Seconds)')
    plt.ylabel('Frequency')
    plt.title(column + " < 500000s data")
    plt.savefig(newpath + column + "_500000.png")

    plt.figure()  # Plot times under 100,000 seconds
    plt.hist(df[df.TimeTaken < 100000][column], bins='auto')
    plt.xlabel('TimeTaken (Seconds)')
    plt.ylabel('Frequency')
    plt.title(column + " < 100000s data")
    plt.savefig(newpath + column + "_100000.png")

    plt.figure()  # Plot all data
    plt.hist(np.log(df[column]), bins='auto')
    plt.xlabel('Log of TimeTaken (Seconds)')
    plt.ylabel('Frequency')
    plt.title(column + " Log of all data")
    plt.savefig(newpath + column + "_log_all.png")

    plt.figure()  # Plot times under 500,000 seconds
    plt.hist(np.log(df[df.TimeTaken < 500000][column]), bins='auto')
    plt.xlabel('Log of TimeTaken (Seconds)')
    plt.ylabel('Frequency')
    plt.title(column + " Log of < 500000s data")
    plt.savefig(newpath + column + "_log_500000.png")

    plt.figure()  # Plot times under 100,000 seconds
    plt.hist(np.log(df[df.TimeTaken < 100000][column]), bins='auto')
    plt.xlabel('Log of TimeTaken (Seconds)')
    plt.ylabel('Frequency')
    plt.title(column + " Log of < 100000s data")
    plt.savefig(newpath + column + "_log_100000.png")


def split_data(df, newpath):  # Split data into training and test data x, y.
    out_file_name = newpath + time.strftime("%Y%m%d-%H%M%S") + "_split_data.txt"  # Log file name
    out_file = open(out_file_name, "w")  # Open log file
    out_file.write("Date and time: " + time.strftime("%Y%m%d-%H%M%S") + "\n")

    distribution = 0
    i = 0
    while distribution == 0:  # Loop until the data is split well
        trainData, testData = train_test_split(df)  # Split data 75:25 randomly
        trainData_y = pd.DataFrame()
        trainData_y["TimeTaken"] = trainData["TimeTaken"]
        trainData_x = trainData.loc[:, trainData.columns != 'TimeTaken']
        testData_y = pd.DataFrame()
        testData_y["TimeTaken"] = testData["TimeTaken"]
        testData_x = testData.loc[:, testData.columns != 'TimeTaken']
        mean_train = sum(trainData_y["TimeTaken"].tolist()) / len(trainData_y)
        mean_test = sum(testData_y["TimeTaken"].tolist()) / len(testData_y)
        std_train = np.std(trainData_y["TimeTaken"].tolist())
        std_test = np.std(testData_y["TimeTaken"].tolist())
        # Only accept a split with test data mean and std that is within 5% of train data mean and stc (stratification?)
        if (mean_train - mean_test) ** 2 < (mean_train * 0.05) ** 2:
            if (std_train - std_test) ** 2 < (std_train * 0.05) ** 2:
                distribution = 1
        i = i + 1

    out_file.write("Number of iterations taken to get good data split: " + str(i) + "\n\n")
    out_file.write("Mean value of Train Y: " + str(mean_train) + "\n")
    out_file.write("Mean value of Test Y: " + str(mean_test) + "\n\n")
    out_file.write("Standard deviation of train Y: " + str(std_train) + "\n")
    out_file.write("Standard deviation of test Y: " + str(std_test) + "\n")

    # trainData_X.to_csv("../../../Data/trainData_X.csv", index = False)  # export file
    # trainData_y.to_csv("../../../Data/trainData_y.csv", index = False)  # export file
    out_file.close()
    return trainData_x, testData_x, trainData_y, testData_y


def linear_regression(trainData_x, trainData_y, testData_x, testData_y, newpath):
    classifier = LinearRegression()
    classifier = classifier.fit(trainData_x, trainData_y)
    y_pred = classifier.predict(testData_x)
    y_train_pred = classifier.predict(trainData_x)
    results(testData_y, y_pred, trainData_y, y_train_pred, "LinearRegression", newpath)


def elastic_net(trainData_x, trainData_y, testData_x, testData_y, newpath):  # Elastic Net
    classifier = ElasticNet(alpha=0.01, l1_ratio=0.9, max_iter=100000)
    classifier = classifier.fit(trainData_x, trainData_y)
    # print(classifier.coef_)
    y_pred = classifier.predict(testData_x)
    y_train_pred = classifier.predict(trainData_x)
    results(testData_y, y_pred, trainData_y, y_train_pred, "ElasticNet", newpath)


def kernel_ridge(trainData_x, trainData_y, testData_x, testData_y, newpath):  # Kernel ridge regression
    classifier = KernelRidge(alpha=0.1)
    classifier = classifier.fit(trainData_x, trainData_y)
    y_pred = classifier.predict(testData_x)
    y_train_pred = classifier.predict(trainData_x)
    results(testData_y, y_pred, trainData_y, y_train_pred, "KernelRidge", newpath)


def results(testData_y, y_pred, trainData_y, y_train_pred, alg, newpath):
    out_file_name = newpath + time.strftime("%Y%m%d-%H%M%S") + "_" + alg + ".txt"  # Log file name
    out_file = open(out_file_name, "w")  # Open log file
    out_file.write(alg + " " + time.strftime("%Y%m%d-%H%M%S") + "\n\n")

    number_close = 0  # Use to track number of close estimations
    for i in range(len(y_pred)):
        if y_pred[i] < 0:  # Convert all negative predictions to 0
            y_pred[i] = 0
            # print(y_pred[i], ": 0 found")
        if abs(y_pred[i] - testData_y.iloc[i, 0]) <= 3600:  # Within 1 hour
            number_close += 1

    out_file.write(alg + " Number predictions within 1 hour: " + str(number_close) + " / " + str(len(y_pred)) + "\n")
    out_file.write(
        alg + " % predictions within 1 hour: " + str(round(((number_close / len(y_pred)) * 100), 2)) + "%\n\n")

    plt.figure()
    plt.plot(trainData_y, y_train_pred, 'ro')
    plt.xlabel('trainData_y')
    plt.ylabel('y_train_pred')
    plt.title(alg + " - Train Data")
    plt.axis('equal')
    plt.ylim(0, 2000000)
    plt.xlim(0, 2000000)
    plt.savefig(newpath + time.strftime("%Y%m%d-%H%M%S") + "_" + alg + "_" + "train.png")

    plt.figure()
    plt.plot(testData_y, y_pred, 'ro')
    plt.xlabel('testData_y')
    plt.ylabel('y_pred')
    plt.title(alg + " - Test Data")
    plt.axis('equal')
    plt.ylim(0, 2000000)
    plt.xlim(0, 2000000)
    plt.savefig(newpath + time.strftime("%Y%m%d-%H%M%S") + "_" + alg + "_" + "test.png")

    out_file.write(alg + " Train RMSE: " + str(sqrt(mean_squared_error(trainData_y, y_train_pred))) + "\n")
    out_file.write(alg + " Test RMSE: " + str(sqrt(mean_squared_error(testData_y, y_pred))) + "\n\n")
    out_file.write(alg + " Train R^2 scoree: " + str(r2_score(trainData_y, y_train_pred)) + "\n")
    out_file.write(alg + " Test R^2 score: " + str(r2_score(testData_y, y_pred)) + "\n")
    out_file.close()

    print(alg, "Train rmse:", sqrt(mean_squared_error(trainData_y, y_train_pred)))  # Print Root Mean Squared Error
    print(alg, "Test rmse:", sqrt(mean_squared_error(testData_y, y_pred)))  # Print Root Mean Squared Error
    print(alg, "Train R^2 score:", r2_score(trainData_y, y_train_pred))  # Print R Squared
    print(alg, "Test R^2 score:", r2_score(testData_y, y_pred), "\n")  # Print R Squared
    # http://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html


def write_parameters(newpath, d):
    with open(newpath + "model_parameters.txt", "w") as f:
        keys = ["User", "Infile", "COSMIC_Number","linear_regression", "elastic_net", "kernel_ridge", "log_of_y",
                "seed", "histogram"]
        for key in keys:
            f.write(key + ": " + d[key] + "\n")

if __name__ == "__main__":  # Run program
    d = {}
    with open("../../../Data/model_inputs.txt", "r") as f:
        for line in f:
            line = line.replace(":", "")
            (key, val) = line.split()
            d[key] = val

    COSMIC_num = str(d["COSMIC_Number"])  # Use the Second COSMIC dataset
    newpath = r"../0. Results/" + str(getpass.getuser()) + "/model/" + "COSMIC_%s" % COSMIC_num + time.strftime(
        "/%Y.%m.%d-%H.%M.%S/")
    if not os.path.exists(newpath):
        os.makedirs(newpath)  # Make folder for storing results if it does not exist

    write_parameters(newpath, d)

    np.random.seed(int(d["seed"]))  # Set seed

    df = pd.read_csv("../../../Data/%s.csv" % d["Infile"], encoding='latin-1', low_memory=False)

    if d["histogram"] == "y":
        histogram(df, "TimeTaken", newpath)  # Save histogram plots of TimeTaken

    if d["log_of_y"] == "y":
        # Take log of y values
        print("Y has been transformed by log . . . comment out in model code if needed\n")
        df["TimeTaken"] = df["TimeTaken"].apply(lambda x: math.log(x))

    trainData_x, testData_x, trainData_y, testData_y = split_data(df, newpath)  # Split data

    if d["linear_regression"] == "y":
        linear_regression(trainData_x, trainData_y, testData_x, testData_y, newpath)  # Linear Regression
    if d["elastic_net"] == "y":
        elastic_net(trainData_x, trainData_y, testData_x, testData_y, newpath)  # elastic net
    if d["kernel_ridge"] == "y":
        kernel_ridge(trainData_x, trainData_y, testData_x, testData_y, newpath)  # Kernel ridge regression


"""
User:

Infile:
- vw_Incident_cleaned
    - default iteration 3 file
- vw_Incident_cleaned(collinearity_thresh_0.9)
    - columns with correlation >0.9 deleted from df
- ABT_Incident_HoldDuration_cleaned(3std)
    - Incident merged with summed HoldDuration from HoldActivity df and std cut off increased to 3 std
        - LinearRegression Test rmse: 343551.1532271481
        - LinearRegression Test R^2 score: 0.4419826483
        - ElasticNet Test rmse: 343659.9254466486
        - ElasticNet Test R^2 score: 0.441629243027
- ABT_Incident_HoldDuration_cleaned(2std)
    - Incident merged with summed HoldDuration from HoldActivity df and std cut off increased to 2 std

COSMIC_Number:
- 1
- 2

linear_regression:
- y
- n

elastic_net:
- y
- n

kernel_ridge:
- y
- n

log_of_y:
- y
- n

seed:
- 12345

histogram:
- y
- n
"""