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
Results will be saved in Iteration > 0. Results > User > prepare_dataset > Date
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
from sklearn.model_selection import KFold, cross_val_predict #, cross_val_score, train_test_split
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


def results(df, alg, regressor, newpath, d, RFR=False):
    out_file_name = newpath + time.strftime("%H.%M.%S") + "_" + alg + ".txt"  # Log file name
    out_file = open(out_file_name, "w")  # Open log file
    out_file.write(alg + " " + time.strftime("%Y%m%d-%H%M%S") + "\n\n")

    X = df.drop("TimeTaken", axis=1)
    y = df["TimeTaken"]

    numFolds = int(d["crossvalidation"])
    kf = KFold(n_splits=numFolds, shuffle=True, random_state=int(d["seed"]))

    train_rmse = []
    test_rmse = []
    train_r_sq = []
    test_r_sq = []
    number_test_1 = []  # Tracking predictions within 1 hour
    number_test_24 = []  # Tracking predictions within 24 hours

    for train_indices, test_indices in kf.split(X, y):
        # Get the dataset; this is the way to access values in a pandas DataFrame
        trainData_x = X.iloc[train_indices]
        trainData_y = y.iloc[train_indices]
        testData_x = X.iloc[test_indices]
        testData_y = y.iloc[test_indices]

        # Train the model, and evaluate it
        regr = regressor
        regr.fit(trainData_x, trainData_y.values.ravel())

        # Get predictions
        y_train_pred = regr.predict(trainData_x)
        y_test_pred = regr.predict(testData_x)

        number_close_1 = 0  # Use to track number of close estimations within 1 hour
        number_close_24 = 0  # Use to track number of close estimations within 24 hours
        mean_time = np.mean(trainData_y)#sum(trainData_y["TimeTaken"].tolist()) / len(trainData_y["TimeTaken"].tolist())  #
        # Calculate mean of predictions
        std_time = np.std(trainData_y)  # Calculate standard deviation of predictions
        for i in range(len(y_train_pred)):  # Convert high or low predictions to 0 or 3 std
            if y_train_pred[i] < 0:  # Convert all negative predictions to 0
                y_train_pred[i] = 0
            if y_train_pred[i] > (mean_time + 4*std_time):  # Convert all predictions > 3 std to 3std
                y_train_pred[i] = (mean_time + 4*std_time)
            if math.isnan(y_train_pred[i]):  # If NaN set to 0
                y_train_pred[i] = 0
        for i in range(len(y_test_pred)): # Convert high or low predictions to 0 or 3 std
            if y_test_pred[i] < 0:  # Convert all negative predictions to 0
                y_test_pred[i] = 0
            if y_test_pred[i] > (mean_time + 4*std_time):  # Convert all predictions > 3 std to 3std
                y_test_pred[i] = (mean_time + 4*std_time)
            if math.isnan(y_test_pred[i]):  # If NaN set to 0
                y_test_pred[i] = 0
            if abs(y_test_pred[i] - testData_y.iloc[i]) <= 3600:  # Within 1 hour
                number_close_1 += 1
            if abs(y_test_pred[i] - testData_y.iloc[i]) <= 3600*24:  # Within 24 hours
                number_close_24 += 1
        number_test_1.append(number_close_1)
        number_test_24.append(number_close_24)

        train_rmse.append(sqrt(mean_squared_error(trainData_y, y_train_pred)))
        test_rmse.append(sqrt(mean_squared_error(testData_y, y_test_pred)))
        train_r_sq.append(r2_score(trainData_y, y_train_pred))
        test_r_sq.append(r2_score(testData_y, y_test_pred))

    train_rmse_ave = np.mean(train_rmse)
    test_rmse_ave = np.mean(test_rmse)
    train_r2_ave = np.mean(train_r_sq)
    test_r2_ave = np.mean(test_r_sq)

    train_rmse_std = np.std(train_rmse)
    test_rmse_std = np.std(test_rmse)
    train_r2_std = np.std(train_r_sq)
    test_r2_std = np.std(test_r_sq)

    out_file.write(alg + ": Cross Validation (" + d["crossvalidation"] + " Folds)\n")

    out_file.write("\tTrain Mean RMSE: {0:.2f} (+/-{1:.2f})\n".format(train_rmse_ave, train_rmse_std))
    out_file.write("\tTest Mean RMSE: {0:.2f} (+/-{1:.2f})\n".format(test_rmse_ave, test_rmse_std))
    out_file.write("\tTrain Mean R2: {0:.5f} (+/-{1:.5f})\n".format(train_r2_ave, train_r2_std))
    out_file.write("\tTest Mean R2: {0:.5f} (+/-{1:.5f})\n".format(test_r2_ave, test_r2_std))

    out_file.write("\n\t" + alg + " number test predictions within 1 hour -> Max: " + str(max(number_test_1)) + "/" +
                   str(len(y_test_pred)) + ", Min: " + str(min(number_test_1)) + "/" +
                   str(len(y_test_pred)) + ", Avg: " + str(sum(number_test_1) / len(number_test_1)) + "/" +
                   str(len(y_test_pred)) + "\n")
    out_file.write("\t" + alg + " % test predictions within 1 hour: -> Max: " +
                   str(round(((max(number_test_1) / len(y_test_pred)) * 100), 2)) + "%, Min: " +
                   str(round(((min(number_test_1) / len(y_test_pred)) * 100), 2)) + "%, Avg: " +
                   str(round(((sum(number_test_1) / len(number_test_1)) / len(y_test_pred) * 100), 2)) + "%" + "\n")
    out_file.write("\t" + alg + " number test predictions within 24 hours -> Max: " + str(max(number_test_24)) + "/" +
                   str(len(y_test_pred)) + ", Min: " + str(min(number_test_24)) + "/" +
                   str(len(y_test_pred)) + ", Avg: " + str(sum(number_test_24) / len(number_test_24)) + "/" +
                   str(len(y_test_pred)) + "\n")
    out_file.write("\t" + alg + " % test predictions within 24 hours -> Max: " +
                   str(round(((max(number_test_24) / len(y_test_pred)) * 100), 2)) + "%, Min: " +
                   str(round(((min(number_test_24) / len(y_test_pred)) * 100), 2)) + "%, Avg: " +
                   str(round(((sum(number_test_24) / len(number_test_24)) / len(y_test_pred) * 100), 2)) + "%" + "\n")
    out_file.write("\n")

    print(alg + ": Cross Validation (" + d["crossvalidation"] + " Folds)")

    print("\tTrain Mean RMSE: {0:.2f} (+/-{1:.2f})".format(train_rmse_ave, train_rmse_std))
    print("\tTest Mean RMSE: {0:.2f} (+/-{1:.2f})".format(test_rmse_ave, test_rmse_std))
    print("\tTrain Mean R2: {0:.5f} (+/-{1:.5f})".format(train_r2_ave, train_r2_std))
    print("\tTest Mean R2: {0:.5f} (+/-{1:.5f})".format(test_r2_ave, test_r2_std))

    print("\n\t" +alg + " number test predictions within 1 hour -> Max: " + str(max(number_test_1)) + "/" +
                   str(len(y_test_pred)) + ", Min: " + str(min(number_test_1)) + "/" +
                   str(len(y_test_pred)) + ", Avg: " + str(sum(number_test_1) / len(number_test_1)) + "/" +
                   str(len(y_test_pred)))
    print("\t" + alg + " % test predictions within 1 hour: -> Max: " +
                   str(round(((max(number_test_1) / len(y_test_pred)) * 100), 2)) + "%, Min: " +
                   str(round(((min(number_test_1) / len(y_test_pred)) * 100), 2)) + "%, Avg: " +
                   str(round(((sum(number_test_1) / len(number_test_1)) / len(y_test_pred) * 100), 2)) + "%")
    print("\t" + alg + " number test predictions within 24 hours -> Max: " + str(max(number_test_24)) + "/" +
                   str(len(y_test_pred)) + ", Min: " + str(min(number_test_24)) + "/" +
                   str(len(y_test_pred)) + ", Avg: " + str(sum(number_test_24) / len(number_test_24)) + "/" +
                   str(len(y_test_pred)))
    print("\t" + alg + " % test predictions within 24 hours -> Max: " +
                   str(round(((max(number_test_24) / len(y_test_pred)) * 100), 2)) + "%, Min: " +
                   str(round(((min(number_test_24) / len(y_test_pred)) * 100), 2)) + "%, Avg: " +
                   str(round(((sum(number_test_24) / len(number_test_24)) / len(y_test_pred) * 100), 2)) + "%\n")

    # plot the results for the whole cross validation
    y_train_pred = cross_val_predict(regr, trainData_x, trainData_y, cv=int(d["crossvalidation"]))
    y_test_pred = cross_val_predict(regr, testData_x, testData_y, cv=int(d["crossvalidation"]))
    plot(trainData_y, y_train_pred, alg, "Train", newpath)
    plot(testData_y, y_test_pred, alg, "Test", newpath)

    if RFR == True:
        importances = regr.feature_importances_
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
    if d["RandomForestRegressor"] == "y":
        classifier = RandomForestRegressor(n_estimators=int(d["n_estimators"]))
        results(df, "RandomForestRegressor", classifier, newpath, d, RFR=True)
        if d["rerun_with_top_importances"] == "y":
            # cols_to_be_deleted = select_importants(newpath + "importances.csv", thresh=0.001) # keep above threshold

            k = int(d["top_k_features"])

            cols_to_be_deleted = select_top_k_importants(newpath + "importances.csv", k) # keep top k
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
            if d["RandomForestRegressor"] == "y":
                classifier = RandomForestRegressor(n_estimators=int(d["n_estimators"]))
                results(df, "RandomForestRegressor", classifier, newpath, d, RFR=True)

    copyfile(parameters, newpath + "/" + time.strftime("%H.%M.%S") + "_parameters.txt")  # Save parameters