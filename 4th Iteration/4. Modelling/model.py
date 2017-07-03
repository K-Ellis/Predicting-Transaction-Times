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
from ask_user_which_file import ask_user


def histogram(df, column, COSMIC_num):  # Create histogram of preprocessed data
    plt.figure()  # Plot all data
    plt.hist(df[column], bins='auto')
    plt.xlabel('TimeTaken (Seconds)')
    plt.ylabel('Frequency')
    plt.title(column + " all data")
    plt.savefig("../0. Results/COSMIC_%s/" % COSMIC_num + str(getpass.getuser()) + "_" + time.strftime("%Y%m%d") +
                "/model/" +
                time.strftime("%Y%m%d-%H%M%S") + "_" + column + "_all.png")

    plt.figure()  # Plot times under 500,000 seconds
    plt.hist(df[df.TimeTaken < 500000][column], bins='auto')
    plt.xlabel('TimeTaken (Seconds)')
    plt.ylabel('Frequency')
    plt.title(column + " < 500000s data")
    plt.savefig("../0. Results/COSMIC_%s/" % COSMIC_num + str(getpass.getuser()) + "_" + time.strftime("%Y%m%d") +
                "/model/" +
                time.strftime("%Y%m%d-%H%M%S") + "_" + column + "_500000.png")

    plt.figure()  # Plot times under 100,000 seconds
    plt.hist(df[df.TimeTaken < 100000][column], bins='auto')
    plt.xlabel('TimeTaken (Seconds)')
    plt.ylabel('Frequency')
    plt.title(column + " < 100000s data")
    plt.savefig(
        "../0. Results/COSMIC_%s/" % COSMIC_num + str(getpass.getuser()) + "_" + time.strftime("%Y%m%d") + "/model/" +
        time.strftime("%Y%m%d-%H%M%S") + "_" + column + "_100000.png")

    plt.figure()  # Plot all data
    plt.hist(np.log(df[column]), bins='auto')
    plt.xlabel('Log of TimeTaken (Seconds)')
    plt.ylabel('Frequency')
    plt.title(column + " Log of all data")
    plt.savefig(
        "../0. Results/COSMIC_%s/" % COSMIC_num + str(getpass.getuser()) + "_" + time.strftime("%Y%m%d") + "/model/" +
        time.strftime("%Y%m%d-%H%M%S") + "_" + column + "_log_all.png")

    plt.figure()  # Plot times under 500,000 seconds
    plt.hist(np.log(df[df.TimeTaken < 500000][column]), bins='auto')
    plt.xlabel('Log of TimeTaken (Seconds)')
    plt.ylabel('Frequency')
    plt.title(column + " Log of < 500000s data")
    plt.savefig(
        "../0. Results/COSMIC_%s/" % COSMIC_num + str(getpass.getuser()) + "_" + time.strftime("%Y%m%d") + "/model/" +
        time.strftime("%Y%m%d-%H%M%S") + "_" + column + "_log_500000.png")

    plt.figure()  # Plot times under 100,000 seconds
    plt.hist(np.log(df[df.TimeTaken < 100000][column]), bins='auto')
    plt.xlabel('Log of TimeTaken (Seconds)')
    plt.ylabel('Frequency')
    plt.title(column + " Log of < 100000s data")
    plt.savefig(
        "../0. Results/COSMIC_%s/" % COSMIC_num + str(getpass.getuser()) + "_" + time.strftime("%Y%m%d") + "/model/" +
        time.strftime("%Y%m%d-%H%M%S") + "_" + column + "_log_100000.png")


def split_data(df, COSMIC_num):  # Split data into training and test data x, y.
    out_file_name = "../0. Results/COSMIC_%s/" % COSMIC_num + str(getpass.getuser()) + "_" + time.strftime(
        "%Y%m%d") + "/model/" + \
                    time.strftime("%Y%m%d-%H%M%S") + "_split_data.txt"  # Log file name
    out_file = open(out_file_name, "w")  # Open log file
    out_file.write("Date and time: " + time.strftime("%Y%m%d-%H%M%S") + "\n")

    distribution = 0
    i = 0
    while distribution == 0:  # Loop until the data is split well
        trainData, testData = train_test_split(df, test_size=0.1)  # Split data 80:20 randomly
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


def linear_regression(trainData_x, trainData_y, testData_x, testData_y, COSMIC_num):
    classifier = LinearRegression()
    classifier = classifier.fit(trainData_x, trainData_y)
    y_pred = classifier.predict(testData_x)
    y_train_pred = classifier.predict(trainData_x)
    results(testData_y, y_pred, trainData_y, y_train_pred, "LinearRegression", COSMIC_num)


def elastic_net(trainData_x, trainData_y, testData_x, testData_y, COSMIC_num):  # Elastic Net
    classifier = ElasticNet(alpha=0.01, l1_ratio=0.9, max_iter=100000)
    classifier = classifier.fit(trainData_x, trainData_y)
    # print(classifier.coef_)
    y_pred = classifier.predict(testData_x)
    y_train_pred = classifier.predict(trainData_x)
    results(testData_y, y_pred, trainData_y, y_train_pred, "ElasticNet", COSMIC_num)


def kernel_ridge(trainData_x, trainData_y, testData_x, testData_y, COSMIC_num):  # Kernel ridge regression
    classifier = KernelRidge(alpha=0.1)
    classifier = classifier.fit(trainData_x, trainData_y)
    y_pred = classifier.predict(testData_x)
    y_train_pred = classifier.predict(trainData_x)
    results(testData_y, y_pred, trainData_y, y_train_pred, "KernelRidge", COSMIC_num)


def results(testData_y, y_pred, trainData_y, y_train_pred, alg, COSMIC_num):
    out_file_name = "../0. Results/COSMIC_%s/" % COSMIC_num + str(getpass.getuser()) + "_" + time.strftime(
        "%Y%m%d") + "/model/" + \
                    time.strftime("%Y%m%d-%H%M%S") + "_" + alg + ".txt"  # Log file name
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
    plt.savefig(
        "../0. Results/COSMIC_%s/" % COSMIC_num + str(getpass.getuser()) + "_" + time.strftime("%Y%m%d") + "/model/" +
        time.strftime("%Y%m%d-%H%M%S") + "_" + alg + "_" + "train.png")

    plt.figure()
    plt.plot(testData_y, y_pred, 'ro')
    plt.xlabel('testData_y')
    plt.ylabel('y_pred')
    plt.title(alg + " - Test Data")
    plt.axis('equal')
    plt.ylim(0, 2000000)
    plt.xlim(0, 2000000)
    plt.savefig(
        "../0. Results/COSMIC_%s/" % COSMIC_num + str(getpass.getuser()) + "_" + time.strftime("%Y%m%d") + "/model/" +
        time.strftime("%Y%m%d-%H%M%S") + "_" + alg + "_" + "test.png")

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


if __name__ == "__main__":  # Run program
    # COSMIC_num = ask_user()
    COSMIC_num = 2  # Use the Second COSMIC dataset
    newpath = r"../0. Results/COSMIC_%s/" % COSMIC_num + str(getpass.getuser()) + "_" + time.strftime(
        "%Y%m%d") + "/model/"
    if not os.path.exists(newpath):
        os.makedirs(newpath)  # Make folder for storing results if it does not exist

    np.random.seed(12345)  # Set seed

    df = pd.read_csv("../../../Data/COSMIC_%s/vw_Incident%s_cleaned.csv" % (COSMIC_num, COSMIC_num),
                     encoding='latin-1',
                     low_memory=False)

    # df = pd.read_csv("../../../Data/COSMIC_%s/vw_Incident%s_cleaned(collinearity_thresh_0.9).csv" % (COSMIC_num,
    #                                                                                                COSMIC_num),
    #                  encoding='latin-1',
    #                  low_memory=False)

    histogram(df, "TimeTaken", COSMIC_num)  # Save histogram plots of TimeTaken

    ## Take log of y values
    # print("Y has been transformed by log . . . comment out in model code if needed\n")
    # df["TimeTaken"] = df["TimeTaken"].apply(lambda x: math.log(x))

    trainData_x, testData_x, trainData_y, testData_y = split_data(df, COSMIC_num)  # Split data

    linear_regression(trainData_x, trainData_y, testData_x, testData_y, COSMIC_num)  # Linear Regression
    elastic_net(trainData_x, trainData_y, testData_x, testData_y, COSMIC_num)  # elastic net
    # kernel_ridge(trainData_x, trainData_y, testData_x, testData_y, COSMIC_num)  # Kernel ridge regression
