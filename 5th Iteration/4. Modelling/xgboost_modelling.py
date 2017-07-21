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
from sklearn.model_selection import train_test_split, KFold, cross_val_score
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
    out_file_name = newpath + time.strftime("%H.%M") + "_split_data.txt"  # Log file name
    out_file = open(out_file_name, "a")  # Open log file
    out_file.write("Date and time: " + time.strftime("%Y%m%d-%H%M%S") + "\n")

    distribution = 0
    i = 0
    while distribution == 0:  # Loop until the data is split well
        trainData, testData = train_test_split(df, test_size=0.25)  # Split data 75:25 randomly
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
    out_file.write("Standard deviation of test Y: " + str(std_test) + "\n\n")
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
    plt.savefig(newpath + time.strftime("%H.%M.%S") + "_" + alg + "_" + data + ".png")
    plt.savefig(newpath + time.strftime("%H.%M.%S") + "_" + alg + "_" + data + ".pdf")


def results(df, alg, classifier, newpath, d, RFR=False):
    out_file_name = newpath + time.strftime("%H.%M.%S") + "_" + alg + ".txt"  # Log file name
    out_file = open(out_file_name, "w")  # Open log file
    out_file.write(alg + " " + time.strftime("%Y%m%d-%H%M%S") + "\n\n")

    train_rmse = []
    test_rmse = []
    train_r_sq = []
    test_r_sq = []
    number_test = []

    for i in range(int(d["crossvalidation"])):  # Repeat tests this number of times and save min, max and average
        trainData_x, testData_x, trainData_y, testData_y = split_data(df, newpath)  # Split data
        classified = classifier.fit(trainData_x, trainData_y.values.ravel())
        y_test_pred = classified.predict(testData_x)
        y_train_pred = classified.predict(trainData_x)

        number_close = 0  # Use to track number of close estimations
        mean_time = sum(trainData_y["TimeTaken"].tolist()) / len(trainData_y["TimeTaken"].tolist())  # Calculate mean of predictions
        std_time = np.std(trainData_y["TimeTaken"].tolist())  # Calculate standard deviation of predictions
        for i in range(len(y_train_pred)):  # Convert high or low predictions to 0 or 3 std
            if y_train_pred[i] < 0:  # Convert all negative predictions to 0
                y_train_pred[i] = 0
            if y_train_pred[i] > (mean_time + 3*std_time):  # Convert all predictions > 3 std to 3std
                y_train_pred[i] = (mean_time + 3*std_time)
            if math.isnan(y_train_pred[i]):  # If NaN set to 0
                y_train_pred[i] = 0
        for i in range(len(y_test_pred)): # Convert high or low predictions to 0 or 3 std
            if y_test_pred[i] < 0:  # Convert all negative predictions to 0
                y_test_pred[i] = 0
            if y_test_pred[i] > (mean_time + 3*std_time):  # Convert all predictions > 3 std to 3std
                y_test_pred[i] = (mean_time + 3*std_time)
            if math.isnan(y_test_pred[i]):  # If NaN set to 0
                y_test_pred[i] = 0
            if abs(y_test_pred[i] - testData_y.iloc[i, 0]) <= 3600:  # Within 1 hour
                number_close += 1
        number_test.append(number_close)

        train_rmse.append(sqrt(mean_squared_error(trainData_y, y_train_pred)))
        test_rmse.append(sqrt(mean_squared_error(testData_y, y_test_pred)))
        train_r_sq.append(r2_score(trainData_y, y_train_pred))
        test_r_sq.append(r2_score(testData_y, y_test_pred))

    out_file.write(alg + " Cross Validation: " + d["crossvalidation"] + "\n")
    out_file.write(alg + " Train RMSE -> Max: " + str(round(max(train_rmse), 2)) + ", Min: " + str(round(min(
        train_rmse), 2)) + ", Avg: " + str(round(sum(train_rmse) / len(train_rmse), 2)) + "\n")  # RMSE
    out_file.write(alg + " Test RMSE -> Max: " + str(round(max(test_rmse), 2)) + ", Min: " + str(round(min(
        test_rmse), 2)) + ", Avg: " + str(round(sum(test_rmse) / len(test_rmse), 2)) + "\n")  # Save RMS
    out_file.write(alg + " Train R^2 score -> Max: " + str(round(max(train_r_sq), 2)) + ", Min: " +
                   str(round(min(train_r_sq),2)) + ", Avg: " + str(round(sum(train_r_sq) / len(train_r_sq), 2)) + "\n")
    out_file.write(alg + " Test R^2 score -> Max: " + str(round(max(test_r_sq), 2)) + ", Min: " +
                   str(round(min(test_r_sq), 2)) + ", Avg: " + str(round(sum(test_r_sq) / len(test_r_sq), 2)) + "\n")
    out_file.write(alg + " number test predictions within 1 hour -> Max: " + str(max(number_test)) + "/" +
                   str(len(y_test_pred)) + ", Min: " + str(min(number_test)) + "/" +
                   str(len(y_test_pred)) + ", Avg: " + str(sum(number_test) / len(number_test)) + "/" +
                   str(len(y_test_pred)) + "\n")
    out_file.write(alg + " % test predictions within 1 hour: -> Max: " +
                   str(round(((max(number_test) / len(y_test_pred)) * 100), 2)) + "%, Min: " +
                   str(round(((min(number_test) / len(y_test_pred)) * 100), 2)) + "%, Avg: " +
                   str(round(((sum(number_test) / len(number_test)) / len(y_test_pred) * 100), 2)) + "%" + "\n")
    out_file.write("\n")

    print(alg + " Cross Validation: " + d["crossvalidation"])
    print(alg + " Train RMSE -> Max: " + str(round(max(train_rmse), 2)) + ", Min: " +
          str(round(min(train_rmse), 2)) + ", Avg: " + str(round(sum(train_rmse) / len(train_rmse), 2)))  # Print RMSE
    print(alg + " Test RMSE -> Max: " + str(round(max(test_rmse), 2)) + ", Min: " +
          str(round(min(test_rmse), 2)) + ", Avg: " + str(round(sum(test_rmse) / len(test_rmse), 2)))  # Print RMSE
    print(alg + " Train R^2 score -> Max: " + str(round(max(train_r_sq), 2)) + ", Min: " +
          str(round(min(train_r_sq),2)) + ", Avg: " + str(round(sum(train_r_sq) / len(train_r_sq), 2)))  # Print R Sq
    print(alg + " Test R^2 score -> Max: " + str(round(max(test_r_sq), 2)) + ", Min: " +
          str(round(min(test_r_sq), 2)) + ", Avg: " + str(round(sum(test_r_sq) / len(test_r_sq), 2)))  # Print R Squared
    print(alg + " number test predictions within 1 hour -> Max: " + str(max(number_test)) + "/" +
                   str(len(y_test_pred)) + ", Min: " + str(min(number_test)) + "/" +
                   str(len(y_test_pred)) + ", Avg: " + str(sum(number_test) / len(number_test)) + "/" +
                   str(len(y_test_pred)))
    print(alg + " % test predictions within 1 hour: -> Max: " +
                   str(round(((max(number_test) / len(y_test_pred)) * 100), 2)) + "%, Min: " +
                   str(round(((min(number_test) / len(y_test_pred)) * 100), 2)) + "%, Avg: " +
                   str(round(((sum(number_test) / len(number_test)) / len(y_test_pred) * 100), 2)) + "%")
    print("Note: only last cross validation plots saved! \n")

    plot(trainData_y, y_train_pred, alg, "Train", newpath)
    plot(testData_y, y_test_pred, alg, "Test", newpath)

    if RFR == True:
        importances = classified.feature_importances_
        print("Top 10 Feature Importances:")
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
        if d["specify_subfolder"] == "n":
            newpath = r"../0. Results/" + d["user"] + "/model/" + time.strftime("%Y.%m.%d/") + time.strftime("%H.%M.%S/")# Log file location
        else:
            newpath = r"../0. Results/" + d["user"] + "/model/" + d["specify_subfolder"] + time.strftime("/%Y.%m.%d/") + \
                      time.strftime("%H.%M.%S/")  # Log file location

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

    if d["LinearRegression"] == "y":
        classifier = LinearRegression()
        results(df, "LinearRegression", classifier, newpath, d)
    if d["ElasticNet"] == "y":
        classifier = ElasticNet(alpha=0.01, l1_ratio=0.9, max_iter=100000)
        results(df, "ElasticNet", classifier, newpath, d)
    if d["KernelRidge"] == "y":
        classifier = KernelRidge(alpha=0.1)
        results(df, "KernelRidge", classifier, newpath, d)
    if d["xgboost"] == "y":
        import xgboost as xgb
        params = {
            'max_depth': 5,
            'n_estimators': 50,
            'objective': 'reg:linear'}
        classifier = xgb.XGBRegressor(**params)
        results(df, "xgboost", classifier, newpath, d)
    if d["RandomForestRegressor"] == "y":
        classifier = RandomForestRegressor(n_estimators=int(d["n_estimators"]))
        results(df, "RandomForestRegressor", classifier, newpath, d, RFR=True)

        # cols_to_be_deleted = select_importants(newpath + "importances.csv", thresh=0.001) # keep above threshold
        k = 50
        cols_to_be_deleted = select_top_k_importants(newpath + "importances.csv", k-1) # keep top k
        df = trim_df(df, cols_to_be_deleted)
        with open(newpath + "cols_kept_and_deleted_for_k=%s_" % k + time.strftime("%H.%M.%S.txt"), "w") as f:
            f.write("cols deleted = \n")
            f.write(str(cols_to_be_deleted))
            f.write("\ncols kept = \n")
            f.write(str(df.columns.tolist()))
        print("Top", len(df.columns), "Columns = ", df.columns.tolist())
        print("Bottom", len(cols_to_be_deleted), "Columns to be deleted = ", cols_to_be_deleted, "\n")

        if d["LinearRegression"] == "y":
            classifier = LinearRegression()
            results(df, "LinearRegression", classifier, newpath, d)
        if d["ElasticNet"] == "y":
            classifier = ElasticNet(alpha=0.01, l1_ratio=0.9, max_iter=100000)
            results(df, "ElasticNet", classifier, newpath, d)
        if d["KernelRidge"] == "y":
            classifier = KernelRidge(alpha=0.1)
            results(df, "KernelRidge", classifier, newpath, d)
        if d["xgboost"] == "y":
            params = {
                'max_depth': 5,
                'n_estimators': 50,
                'objective': 'reg:linear'}
            classifier = xgb.XGBRegressor(**params)
            results(df, "xgboost", classifier, newpath, d)
        if d["RandomForestRegressor"] == "y":
            classifier = RandomForestRegressor(n_estimators=int(d["n_estimators"]))
            results(df, "RandomForestRegressor", classifier, newpath, d, RFR=True)

    copyfile(parameters, newpath + "/" + time.strftime("%H.%M.%S") + "_parameters.txt")  # Save parameters
