"""****************************************************************************
UCD MSc Business Analytics Capstone Project - Predicting Transactions Times
*******************************************************************************
Iteration 4
Data modelling program
*******************************************************************************
Eoin Carroll
Kieron Ellis
*******************************************************************************
Working on dataset 2 from Cosmic: UCD_Data_20170623_1.xlsx
****************************************************************************"""

"""****************************************************************************
Import libraries
****************************************************************************"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os  # Used to create folders
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from math import sqrt
from sklearn.ensemble import RandomForestRegressor
from shutil import copyfile  # Used to copy parameters file to directory
from sklearn.utils import resample
from select_k_importance import select_importants, trim_df, select_top_k_importants


parameters = "../../../Data/parameters.txt"  # Parameters file


def histogram(df, column, newpath):  # Create histogram of preprocessed data
    plt.figure()  # Plot all data
    plt.hist(df[column], bins='auto')
    plt.xlabel('TimeTaken (Seconds)')
    plt.ylabel('Frequency')
    plt.title(column + " all data")
    plt.savefig(newpath + column + "_all.png")
    plt.savefig(newpath + column + "_all.pdf")

    plt.figure()  # Plot times under 500,000 seconds
    plt.hist(df[df.TimeTaken < 500000][column], bins='auto')
    plt.xlabel('TimeTaken (Seconds)')
    plt.ylabel('Frequency')
    plt.title(column + " < 500000s data")
    plt.savefig(newpath + column + "_500000.png")
    plt.savefig(newpath + column + "_500000.pdf")

    plt.figure()  # Plot times under 100,000 seconds
    plt.hist(df[df.TimeTaken < 100000][column], bins='auto')
    plt.xlabel('TimeTaken (Seconds)')
    plt.ylabel('Frequency')
    plt.title(column + " < 100000s data")
    plt.savefig(newpath + column + "_100000.png")
    plt.savefig(newpath + column + "_100000.pdf")

    plt.figure()  # Plot all data
    plt.hist(np.log(df[column]), bins='auto')
    plt.xlabel('Log of TimeTaken (Seconds)')
    plt.ylabel('Frequency')
    plt.title(column + " Log of all data")
    plt.savefig(newpath + column + "_log_all.png")
    plt.savefig(newpath + column + "_log_all.pdf")

    plt.figure()  # Plot times under 500,000 seconds
    plt.hist(np.log(df[df.TimeTaken < 500000][column]), bins='auto')
    plt.xlabel('Log of TimeTaken (Seconds)')
    plt.ylabel('Frequency')
    plt.title(column + " Log of < 500000s data")
    plt.savefig(newpath + column + "_log_500000.png")
    plt.savefig(newpath + column + "_log_500000.pdf")

    plt.figure()  # Plot times under 100,000 seconds
    plt.hist(np.log(df[df.TimeTaken < 100000][column]), bins='auto')
    plt.xlabel('Log of TimeTaken (Seconds)')
    plt.ylabel('Frequency')
    plt.title(column + " Log of < 100000s data")
    plt.savefig(newpath + column + "_log_100000.png")
    plt.savefig(newpath + column + "_log_100000.pdf")


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

    out_file.close()
    return trainData_x, testData_x, trainData_y, testData_y


def plot(x, y, alg, data, newpath):
    plt.figure()
    plt.plot(x, y, 'ro', alpha=0.1, markersize=3)
    plt.xlabel(data + " Data")
    plt.ylabel(data + " Data Prediction")
    plt.title(alg + " - " + data + " Data")
    plt.axis('equal')
    plt.ylim(0, 2000000)
    plt.xlim(0, 2000000)
    plt.tight_layout()  # Force everything to fit on figure
    plt.savefig(newpath + time.strftime("%Y%m%d-%H%M%S") + "_" + alg + "_" + data + ".png")
    plt.savefig(newpath + time.strftime("%Y%m%d-%H%M%S") + "_" + alg + "_" + data + ".pdf")


def linear_regression(trainData_x, trainData_y, testData_x, testData_y, newpath, d):
    classifier = LinearRegression()
    classifier = classifier.fit(trainData_x, trainData_y)
    y_test_pred = classifier.predict(testData_x)
    y_train_pred = classifier.predict(trainData_x)
    results(testData_y, y_test_pred, trainData_y, y_train_pred, "LinearRegression", newpath, d)


def elastic_net(trainData_x, trainData_y, testData_x, testData_y, newpath, d):  # Elastic Net
    classifier = ElasticNet(alpha=0.01, l1_ratio=0.9, max_iter=100000)
    classifier = classifier.fit(trainData_x, trainData_y)
    y_test_pred = classifier.predict(testData_x)
    y_train_pred = classifier.predict(trainData_x)
    results(testData_y, y_test_pred, trainData_y, y_train_pred, "ElasticNet", newpath, d)


def kernel_ridge(trainData_x, trainData_y, testData_x, testData_y, newpath, d):  # Kernel ridge regression
    classifier = KernelRidge(alpha=0.1)
    classifier = classifier.fit(trainData_x, trainData_y)
    y_test_pred = classifier.predict(testData_x)
    y_train_pred = classifier.predict(trainData_x)
    results(testData_y, y_test_pred, trainData_y, y_train_pred, "KernelRidge", newpath, d)


def Random_Forest_Regressor(trainData_x, trainData_y, testData_x, testData_y, newpath, d):  # Kernel ridge regression
    classifier = RandomForestRegressor(n_estimators=int(d["n_estimators"]))
    classifier = classifier.fit(trainData_x, trainData_y.values.ravel())
    y_test_pred = classifier.predict(testData_x)
    y_train_pred = classifier.predict(trainData_x)
    importances = classifier.feature_importances_
    results(testData_y, y_test_pred, trainData_y, y_train_pred, "RandomForestRegressor", newpath, d, importances,
            trainData_x)


def results(testData_y, y_test_pred, trainData_y, y_train_pred, alg, newpath, d, importances=None, trainData_x=None):
    out_file_name = newpath + time.strftime("%Y%m%d-%H%M%S") + "_" + alg + ".txt"  # Log file name
    out_file = open(out_file_name, "w")  # Open log file
    out_file.write(alg + " " + time.strftime("%Y%m%d-%H%M%S") + "\n\n")

    number_close = 0  # Use to track number of close estimations
    for i in range(len(y_test_pred)):
        if y_test_pred[i] < 0:  # Convert all negative predictions to 0
            y_test_pred[i] = 0

        # if d["holdduration"] == "y":
        #     if abs(y_test_pred[i] - testData_y.iloc[i]) <= 3600:  # Within 1 hour
        #         number_close += 1
        # else:
        if abs(y_test_pred[i] - testData_y.iloc[i, 0]) <= 3600:  # Within 1 hour
            number_close += 1

    out_file.write(alg + " Train RMSE: " + str(sqrt(mean_squared_error(trainData_y, y_train_pred))) + "\n")
    out_file.write(alg + " Test RMSE: " + str(sqrt(mean_squared_error(testData_y, y_test_pred))) + "\n")
    out_file.write(
        alg + " number test predictions within 1 hour: " + str(number_close) + " / " + str(len(y_test_pred)) + "\n")
    out_file.write(
        alg + " % test predictions within 1 hour: " + str(round(((number_close / len(y_test_pred)) * 100), 2)) + "%\n")
    out_file.write(alg + " Train R^2 scoree: " + str(r2_score(trainData_y, y_train_pred)) + "\n")
    out_file.write(alg + " Test R^2 score: " + str(r2_score(testData_y, y_test_pred)) + "\n")
    out_file.write("\n")

    print(alg, "Train rmse:", sqrt(mean_squared_error(trainData_y, y_train_pred)))  # Print Root Mean Squared Error
    print(alg, "Test rmse:", sqrt(mean_squared_error(testData_y, y_test_pred)))  # Print Root Mean Squared Error
    print(alg + " number test predictions within 1 hour: " + str(number_close) + " / " + str(len(y_test_pred)))
    print(alg + " % test predictions within 1 hour: " + str(round(((number_close / len(y_test_pred)) * 100), 2)) + "%")
    print(alg, "Train R^2 score:", r2_score(trainData_y, y_train_pred))  # Print R Squared
    print(alg, "Test R^2 score:", r2_score(testData_y, y_test_pred), "\n")  # Print R Squared

    plot(trainData_y, y_train_pred, alg, "Train", newpath)
    plot(testData_y, y_test_pred, alg, "Test", newpath)


    if importances is not None:
        print("Feature Importances:")
        dfimportances = pd.DataFrame(data=trainData_x.columns, columns=["Columns"])
        dfimportances["importances"] = importances
        dfimportances.to_csv(newpath + "importances.csv", index=False)
        dfimportances = dfimportances.sort_values("importances", ascending=False)
        print(dfimportances[:10], "\n")

        out_file.write("\nFeature Importances:\n")
        for i, (col, importance) in enumerate(zip(dfimportances["Columns"].values.tolist(), dfimportances["importances"].values.tolist())):
            out_file.write("%d. \"%s\" (%f)\n" % (i, col, importance))
    out_file.close()


if __name__ == "__main__":  # Run program
    d = {}
    with open(parameters, "r") as f:
        for line in f:
            line = line.replace(":", "")
            (key, val) = line.split()
            d[key] = val

    if d["user"] == "Kieron":
        newpath = r"../0. Results/" + d["user"] + "/model/" + time.strftime("%Y.%m.%d/") + time.strftime(
            "%H.%M.%S/")  # Log file location
    else:
        newpath = r"../0. Results/" + d["user"] + "/model/" + time.strftime("%Y.%m.%d/")  # Log file location
    if not os.path.exists(newpath):
        os.makedirs(newpath)  # Make folder for storing results if it does not exist

    np.random.seed(int(d["seed"]))  # Set seed

    if d["user"] == "Kieron":
        df = pd.read_csv(d["file_location"] + d["file_name"] + ".csv", encoding='latin-1', low_memory=False)
        # If you insist on hardcoding something in a program, keep my name out of it
    else:
        df = pd.read_csv(d["file_location"] + "vw_Incident_cleaned" + d["file_name"] + ".csv", encoding='latin-1',
                     low_memory=False)

    if d["histogram"] == "y":
        histogram(df, "TimeTaken", newpath)  # Save histogram plots of TimeTaken

    if d["log_of_y"] == "y":  # Take log of y values
        print("Y has been transformed by log . . . change parameter file to remove this feature\n")
        df["TimeTaken"] = df["TimeTaken"].apply(lambda x: math.log(x))

    if d["resample"] == "y":
        df = resample(df, n_samples=int(d["n_samples"]), random_state=int(d["seed"]))

    trainData_x, testData_x, trainData_y, testData_y = split_data(df, newpath)  # Split data

    if d["linear_regression"] == "y":
        linear_regression(trainData_x, trainData_y, testData_x, testData_y, newpath, d)  # Linear Regression
    if d["elastic_net"] == "y":
        elastic_net(trainData_x, trainData_y, testData_x, testData_y, newpath, d)  # elastic net
    if d["kernel_ridge"] == "y":
        kernel_ridge(trainData_x, trainData_y, testData_x, testData_y, newpath, d)  # Kernel ridge regression
    if d["random_forest_regressor"] == "y":
        Random_Forest_Regressor(trainData_x, trainData_y, testData_x, testData_y, newpath, d)  # Random Forest

        # cols_to_be_deleted = select_importants(newpath + "importances.csv", thresh=0.001) # keep above threshold
        k = 30
        cols_to_be_deleted = select_top_k_importants(newpath + "importances.csv", k) # keep top k
        df2 = trim_df(df, cols_to_be_deleted)
        with open(newpath + "cols_deleted_k=%s_" % k + time.strftime("%H.%M.%S.txt"), "w") as f:
            f.write(str(cols_to_be_deleted))

        trainData_x, testData_x, trainData_y, testData_y = split_data(df2, newpath)  # Split data

        if d["linear_regression"] == "y":
            linear_regression(trainData_x, trainData_y, testData_x, testData_y, newpath, d)  # Linear Regression
        if d["elastic_net"] == "y":
            elastic_net(trainData_x, trainData_y, testData_x, testData_y, newpath, d)  # elastic net
        if d["kernel_ridge"] == "y":
            kernel_ridge(trainData_x, trainData_y, testData_x, testData_y, newpath, d)  # Kernel ridge regression
        if d["random_forest_regressor"] == "y":
            Random_Forest_Regressor(trainData_x, trainData_y, testData_x, testData_y, newpath, d)  # Random Forest

    copyfile(parameters, newpath + "/" + time.strftime("%H.%M.%S") + "_parameters.txt")  # Save parameters