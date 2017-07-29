"""****************************************************************************
UCD MSc Business Analytics Capstone Project - Predicting Transactions Times
*******************************************************************************
Iteration 5
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
from sklearn.model_selection import KFold, cross_val_predict#, cross_val_score, train_test_split
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from math import sqrt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from shutil import copyfile  # Used to copy parameters file to directory
from select_k_importance import trim_df, select_top_k_importants

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


def plot(x, y, alg, data, newpath, iter_no=None):
    plt.figure()
    plt.plot(x, y, 'ro', alpha=0.1, markersize=3)
    plt.xlabel(data + " Data")
    plt.ylabel(data + " Data Prediction")
    plt.title(alg + " - " + data + " Data")
    plt.axis('equal')
    plt.ylim(0, 2000000)
    plt.xlim(0, 2000000)
    plt.tight_layout()  # Force everything to fit on figure
    if d["user"] == "Kieron":
        if not os.path.exists(newpath + "PDFs/"):
            os.makedirs(newpath + "PDFs/")  # Make folder for storing results if it does not exist
        if iter_no is None:
            plt.savefig(newpath + alg + "_" + data + ".png")
            plt.savefig(newpath + "PDFs/" + alg + "_" + data + ".pdf")
        else:
            plt.savefig(newpath + d["top_k_features"] + "_" + alg + "_" + data + ".png")
            plt.savefig(newpath + "PDFs/" + d["top_k_features"] + "_" + alg + "_" + data + ".pdf")
    else:
        plt.savefig(newpath + time.strftime("%H.%M.%S") + "_" + alg + "_" + data + ".png")
        plt.savefig(newpath + time.strftime("%H.%M.%S") + "_" + alg + "_" + data + ".pdf")


def results(df, alg, in_regressor, newpath, d, iter_no=None):
    if d["user"] == "Kieron":
        if iter_no is None:
            out_file_name = newpath + alg + ".txt"  # Log file name
        else:
            out_file_name = newpath + d["top_k_features"] + "_" + alg + ".txt"  # Log file name
    else:
        out_file_name = newpath + time.strftime("%H.%M.%S") + "_" + alg + ".txt"  # Log file name
    out_file = open(out_file_name, "w")  # Open log file
    out_file.write(alg + " " + time.strftime("%Y%m%d-%H%M%S") + "\n\n")
        
    X = df.drop("TimeTaken", axis=1)
    if "Created_On" in df.columns:
        X = X.drop("Created_On", axis=1)
    if "ResolvedDate" in df.columns:
        X = X.drop("ResolvedDate", axis=1)
    y = df["TimeTaken"]

    numFolds = int(d["crossvalidation"])
    kf = KFold(n_splits=numFolds, shuffle=True, random_state=int(d["seed"]))

    train_rmse = []
    test_rmse = []
    train_r_sq = []
    test_r_sq = []
    train_mae = []
    test_mae = []
    
    # todo
    # adjusted R2
    # F-test?
    # explained_variance_score
    # median_absolute_error
    
    number_test_1 = []  # Tracking predictions within 1 hour
    number_test_4 = []  # Tracking predictions within 4 hour
    number_test_8 = []  # Tracking predictions within 8 hour
    number_test_16 = []  # Tracking predictions within 16 hour
    number_test_24 = []  # Tracking predictions within 24 hours
    number_test_48 = []  # Tracking predictions within 24 hours
    number_test_72 = []  # Tracking predictions within 24 hours
    number_test_96 = []  # Tracking predictions within 24 hours
    
    mean_time = np.mean(y)  # Calculate mean of predictions
    std_time = np.std(y)  # Calculate standard deviation of predictions
    max_time = 2000000
    
    df["TimeTaken_%s" % alg] = -1  # assign a nonsense value
    
    # todo - plot RMSE against day, month, qtr, year

    for train_indices, test_indices in kf.split(X, y):
        # Get the dataset; this is the way to access values in a pandas DataFrame
        trainData_x = X.iloc[train_indices]
        trainData_y = y.iloc[train_indices]
        testData_x = X.iloc[test_indices]
        testData_y = y.iloc[test_indices]

        # Train the model, and evaluate it
        regr = in_regressor
        regr.fit(trainData_x, trainData_y.values.ravel())

        # Get predictions
        y_train_pred = regr.predict(trainData_x)
        y_test_pred = regr.predict(testData_x)

        number_close_1 = 0  # Use to track number of close estimations within 1 hour
        number_close_4 = 0  # Use to track number of close estimations within 4 hour
        number_close_8 = 0  # Use to track number of close estimations within 8 hour
        number_close_16 = 0  # Use to track number of close estimations within 16 hour
        number_close_24 = 0  # Use to track number of close estimations within 24 hours
        number_close_48 = 0  # Use to track number of close estimations within 24 hours
        number_close_72 = 0  # Use to track number of close estimations within 24 hours
        number_close_96 = 0  # Use to track number of close estimations within 24 hours
        
        for i in range(len(y_train_pred)):  # Convert high or low predictions to 0 or 3 std
            if y_train_pred[i] < 0:  # Convert all negative predictions to 0
                y_train_pred[i] = 0
                
            if y_train_pred[i] > max_time:  # Convert all predictions > 2M to 2M
                y_train_pred[i] = max_time  
                
            
            if math.isnan(y_train_pred[i]):  # If NaN set to 0
                y_train_pred[i] = 0
        
        for i in range(len(y_test_pred)): # Convert high or low predictions to 0 or 3 std
            if y_test_pred[i] < 0:  # Convert all negative predictions to 0
                y_test_pred[i] = 0
                # neg_sum+=1
                
            # if y_test_pred[i] > (mean_time + 3*std_time):  # Convert all predictions > 3 std to 3std
                # y_test_pred[i] = (mean_time + 3*std_time)
            if y_test_pred[i] > max_time:  # Convert all predictions > max_time to max_time
                y_test_pred[i] = max_time
                
            if math.isnan(y_test_pred[i]):  # If NaN set to 0
                y_test_pred[i] = 0
            if abs(y_test_pred[i] - testData_y.iloc[i]) <= 3600:  # Within 1 hour
                number_close_1 += 1
            if abs(y_test_pred[i] - testData_y.iloc[i]) <= 4*60*60:  # Within 4 hours
                number_close_4 += 1
            if abs(y_test_pred[i] - testData_y.iloc[i]) <= 8*60*60:  # Within 8 hours
                number_close_8 += 1
            if abs(y_test_pred[i] - testData_y.iloc[i]) <= 16*60*60:  # Within 16 hours
                number_close_16 += 1
            if abs(y_test_pred[i] - testData_y.iloc[i]) <= 3600*24:  # Within 24 hours
                number_close_24 += 1
            if abs(y_test_pred[i] - testData_y.iloc[i]) <= 3600*48:  # Within 48 hours
                number_close_48 += 1
            if abs(y_test_pred[i] - testData_y.iloc[i]) <= 3600*72:  # Within 72 hours
                number_close_72 += 1
            if abs(y_test_pred[i] - testData_y.iloc[i]) <= 3600*96:  # Within 96 hours
                number_close_96 += 1
        df.loc[test_indices, "TimeTaken_%s"%alg] = y_test_pred
        
        number_test_1.append(number_close_1)
        number_test_4.append(number_close_4)
        number_test_8.append(number_close_8)
        number_test_16.append(number_close_16)
        number_test_24.append(number_close_24)
        number_test_48.append(number_close_48)
        number_test_72.append(number_close_72)
        number_test_96.append(number_close_96)

        train_rmse.append(sqrt(mean_squared_error(trainData_y, y_train_pred)))
        test_rmse.append(sqrt(mean_squared_error(testData_y, y_test_pred)))
        train_r_sq.append(r2_score(trainData_y, y_train_pred))
        test_r_sq.append(r2_score(testData_y, y_test_pred))
        train_mae.append(mean_absolute_error(trainData_y, y_train_pred))
        test_mae.append(mean_absolute_error(testData_y, y_test_pred))

    train_rmse_ave = np.mean(train_rmse)
    test_rmse_ave = np.mean(test_rmse)
    train_r2_ave = np.mean(train_r_sq)
    test_r2_ave = np.mean(test_r_sq)
    train_mae_ave = np.mean(train_mae)
    test_mae_ave = np.mean(test_mae)

    train_rmse_std = np.std(train_rmse)
    test_rmse_std = np.std(test_rmse)
    train_r2_std = np.std(train_r_sq)
    test_r2_std = np.std(test_r_sq)
    train_mae_std = np.std(train_mae)
    test_mae_std = np.std(test_mae)
    
    ave_1hour = np.mean(number_test_1)
    std_1hour = np.std(number_test_1)
    pct_ave_1hour = ave_1hour/ len(y_test_pred) * 100
    pct_std_std_1hour = std_1hour/ len(y_test_pred) * 100

    ave_4hour = np.mean(number_test_4)
    std_4hour = np.std(number_test_4)
    pct_ave_4hour = ave_4hour/ len(y_test_pred) * 100
    pct_std_std_4hour = std_4hour/ len(y_test_pred) * 100
    
    ave_8hour = np.mean(number_test_8)
    std_8hour = np.std(number_test_8)
    pct_ave_8hour = ave_8hour/ len(y_test_pred) * 100
    pct_std_std_8hour = std_8hour/ len(y_test_pred) * 100
    
    ave_16hour = np.mean(number_test_16)
    std_16hour = np.std(number_test_16)
    pct_ave_16hour = ave_16hour/ len(y_test_pred) * 100
    pct_std_std_16hour = std_16hour/ len(y_test_pred) * 100
    
    ave_24hour = np.mean(number_test_24)
    std_24hour = np.std(number_test_24)
    pct_ave_24hour = ave_24hour/ len(y_test_pred) * 100
    pct_std_std_24hour = std_24hour/ len(y_test_pred) * 100
    
    ave_48hour = np.mean(number_test_48)
    std_48hour = np.std(number_test_48)
    pct_ave_48hour = ave_48hour/ len(y_test_pred) * 100
    pct_std_std_48hour = std_48hour/ len(y_test_pred) * 100

    ave_72hour = np.mean(number_test_72)
    std_72hour = np.std(number_test_72)
    pct_ave_72hour = ave_72hour/ len(y_test_pred) * 100
    pct_std_std_72hour = std_72hour/ len(y_test_pred) * 100
    
    ave_96hour = np.mean(number_test_96)
    std_96hour = np.std(number_test_96)
    pct_ave_96hour = ave_96hour/ len(y_test_pred) * 100
    pct_std_std_96hour = std_96hour/ len(y_test_pred) * 100    
    
    out_file.write("Input file name %s:\n" % d["input_file"])
    out_file.write(alg + ": Cross Validation (" + d["crossvalidation"] + " Folds)\n")
    out_file.write("\tTrain Mean R2: {0:.5f} (+/-{1:.5f})\n".format(train_r2_ave, train_r2_std))
    out_file.write("\tTest Mean R2: {0:.5f} (+/-{1:.5f})\n".format(test_r2_ave, test_r2_std))
    out_file.write("\tTrain Mean RMSE: {0:.2f} (+/-{1:.2f})\n".format(train_rmse_ave, train_rmse_std))
    out_file.write("\tTest Mean RMSE: {0:.2f} (+/-{1:.2f})\n".format(test_rmse_ave, test_rmse_std))
    out_file.write("\tTrain Mean MAE: {0:.2f} (+/-{1:.2f})\n".format(train_mae_ave, train_mae_std))
    out_file.write("\tTest Mean MAE: {0:.2f} (+/-{1:.2f})\n".format(test_mae_ave, test_mae_std))

    out_file.write("\n\n\t{2:s} % test predictions error within 1 hour -> Mean: {0:.2f}% (+/- {1:.2f}%) of {3:d}".format(pct_ave_1hour, pct_std_std_1hour, alg, len(y_test_pred)))
    out_file.write("\n\t{2:s} % test predictions error within 4 hours -> Mean: {0:.2f}% (+/- {1:.2f}%) of {3:d}".format(pct_ave_4hour, pct_std_std_4hour, alg, len(y_test_pred)))
    out_file.write("\n\t{2:s} % test predictions error within 8 hours -> Mean: {0:.2f}% (+/- {1:.2f}%) of {3:d}".format(pct_ave_8hour, pct_std_std_8hour, alg, len(y_test_pred)))
    out_file.write("\n\t{2:s} % test predictions error within 16 hours -> Mean: {0:.2f}% (+/- {1:.2f}%) of {3:d}".format(pct_ave_16hour, pct_std_std_16hour, alg, len(y_test_pred)))
    out_file.write("\n\t{2:s} % test predictions error within 24 hours -> Mean: {0:.2f}% (+/- {1:.2f}%) of {3:d}".format(pct_ave_24hour, pct_std_std_24hour,alg, len(y_test_pred)))
    out_file.write("\n\t{2:s} % test predictions error within 48 hours -> Mean: {0:.2f}% (+/- {1:.2f}%) of {3:d}".format(pct_ave_48hour, pct_std_std_48hour,alg, len(y_test_pred)))    
    out_file.write("\n\t{2:s} % test predictions error within 72 hours -> Mean: {0:.2f}% (+/- {1:.2f}%) of {3:d}".format(pct_ave_72hour, pct_std_std_72hour,alg, len(y_test_pred)))
    out_file.write("\n\t{2:s} % test predictions error within 96 hours -> Mean: {0:.2f}% (+/- {1:.2f}%) of {3:d}\n".format(pct_ave_96hour, pct_std_std_96hour,alg, len(y_test_pred)))    
    out_file.write("\n")
    
    print(alg + ": Cross Validation (" + d["crossvalidation"] + " Folds)")
    print("\tTrain Mean R2: {0:.5f} (+/-{1:.5f})".format(train_r2_ave, train_r2_std))
    print("\tTest Mean R2: {0:.5f} (+/-{1:.5f})".format(test_r2_ave, test_r2_std))
    print("\tTrain Mean RMSE: {0:.2f} (+/-{1:.2f})".format(train_rmse_ave, train_rmse_std))
    print("\tTest Mean RMSE: {0:.2f} (+/-{1:.2f})".format(test_rmse_ave, test_rmse_std))
    print("\tTrain Mean MAE: {0:.2f} (+/-{1:.2f})".format(train_mae_ave, train_mae_std))
    print("\tTest Mean MAE: {0:.2f} (+/-{1:.2f})".format(test_mae_ave, test_mae_std))
    
    print("\n\t{2:s} % test predictions within 1 hour -> Mean: {0:.2f}% (+/- {1:.2f}%) of {3:d}".format(pct_ave_1hour, pct_std_std_1hour, alg, len(y_test_pred)))
    print("\t{2:s} % test predictions error within 4 hours -> Mean: {0:.2f}% (+/- {1:.2f}%) of {3:d}".format(pct_ave_4hour, pct_std_std_4hour, alg, len(y_test_pred)))
    print("\t{2:s} % test predictions error within 8 hours -> Mean: {0:.2f}% (+/- {1:.2f}%) of {3:d}".format(pct_ave_8hour, pct_std_std_8hour, alg, len(y_test_pred)))
    print("\t{2:s} % test predictions error within 16 hours -> Mean: {0:.2f}% (+/- {1:.2f}%) of {3:d}".format(pct_ave_16hour, pct_std_std_16hour, alg, len(y_test_pred)))
    print("\t{2:s} % test predictions error within 24 hours -> Mean: {0:.2f}% (+/- {1:.2f}%) of {3:d}".format(pct_ave_24hour, pct_std_std_24hour, alg, len(y_test_pred)))
    print("\t{2:s} % test predictions error within 48 hours -> Mean: {0:.2f}% (+/- {1:.2f}%) of {3:d}".format(pct_ave_48hour, pct_std_std_48hour, alg, len(y_test_pred)))
    print("\t{2:s} % test predictions error within 72 hours -> Mean: {0:.2f}% (+/- {1:.2f}%) of {3:d}".format(pct_ave_72hour, pct_std_std_72hour, alg, len(y_test_pred)))
    print("\t{2:s} % test predictions error within 96 hours -> Mean: {0:.2f}% (+/- {1:.2f}%) of {3:d}\n".format(pct_ave_96hour, pct_std_std_96hour, alg, len(y_test_pred)))
   
   # todo write message
    print("TimeTaken stats")
    print(mean_time)
    print(std_time)
    print(mean_time + 3*std_time)
   
   
    print("..creating concatonated cross validated predictions for plotting..")
    # create concatonated results for the whole cross validation
    y_pred = cross_val_predict(regr, X, y, cv=int(d["crossvalidation"]))
    # Final Concatonated Scores
    print("\tFinal Concatonated Test Scores:")
    print("\t\tPlotting Test RMSE %.1f" % sqrt(mean_squared_error(y, y_pred)))
    print("\t\tPlotting Test R2 %.4f" % r2_score(y, y_pred))
    print("\t\tPlotting Test MAE %.1f" % mean_absolute_error(y, y_pred))
    
    print("..plotting..")
    plot(y, y_pred, alg, "Test", newpath, iter_no)

    if alg == "RandomForestRegressor":
        print("..Calculating importances..\n")
        importances = regr.feature_importances_
        print("Most Important Features:")
        dfimportances = pd.DataFrame(data=X.columns, columns=["Columns"])
        dfimportances["importances"] = importances

        dfimportances = dfimportances.sort_values("importances", ascending=False)
        if d["export_importances_csv"] == "y":
            if iter_no == "second":
                if d["user"] == "Kieron":
                    dfimportances.to_csv(newpath + d["top_k_features"] + "_importances.csv", index=False)
                else:
                    dfimportances.to_csv(newpath + "secondround_importances.csv", index=False)
            else:
                dfimportances.to_csv(newpath + "importances.csv", index=False)

        print(dfimportances[:10], "\n")
        out_file.write("\nFeature Importances:\n")
        for i, (col, importance) in enumerate(zip(dfimportances["Columns"].values.tolist(), dfimportances["importances"].values.tolist())):
            out_file.write("%d. \"%s\" (%f)\n" % (i, col, importance))

        if d["rerun_with_top_importances"] == "y":
            out_file.close()
            return dfimportances
    print("..finished with alg: %s..\n" % alg)
    out_file.close()


if __name__ == "__main__":  # Run program
    d = {}
    with open(parameters, "r") as f:
        for line in f:
            line = line.replace(":", "")
            (key, val) = line.split()
            d[key] = val

    if d["user"] == "Kieron":
        if d["resample"] == "y":
            newpath = r"../0. Results/" + d["user"] + "/model/" + d["input_file"] + "/resample/" 
        elif d["specify_subfolder"] == "n":
            newpath = r"../0. Results/" + d["user"] + "/model/" + d["input_file"] +"/"
        else:
            newpath = r"../0. Results/" + d["user"] + "/model/" + d["input_file"]+ "/" + d["specify_subfolder"]+"/"
    else:
        newpath = r"../0. Results/" + d["user"] + "/model/" + time.strftime("%Y.%m.%d/")  # Log file location
    if not os.path.exists(newpath):
        os.makedirs(newpath)  # Make folder for storing results if it does not exist

    np.random.seed(int(d["seed"]))  # Set seed

    if d["user"] == "Kieron":
        df = pd.read_csv(d["file_location"] + d["input_file"] + ".csv", encoding='latin-1', low_memory=False)
    else:
        df = pd.read_csv(d["file_location"] + "vw_Incident_cleaned" + d["input_file"] + ".csv", encoding='latin-1',
                     low_memory=False)

    print("Input file name: %s\n" % d["input_file"])
    
    if d["histogram"] == "y":
        histogram(df, "TimeTaken", newpath)  # Save histogram plots of TimeTaken

    if d["log_of_y"] == "y":  # Take log of y values
        print("Y has been transformed by log . . . change parameter file to remove this feature\n")
        df["TimeTaken"] = df["TimeTaken"].apply(lambda x: math.log(x))
    
    if d["resample"] == "y":
        from sklearn.utils import resample
        print("..resampling")
        df = resample(df, n_samples=int(d["n_samples"]), random_state=int(d["seed"]))
    
    ####################################################################################################################
    # Modelling
    ####################################################################################################################
    if d["LinearRegression"] == "y":
        regressor = LinearRegression()
        results(df, "LinearRegression", regressor, newpath, d)

    if d["ElasticNet"] == "y":
        regressor = ElasticNet(alpha=100, l1_ratio=1, max_iter=100000)
        results(df, "ElasticNet", regressor, newpath, d)

    if d["KernelRidge"] == "y":
        regressor = KernelRidge(alpha=0.1)
        results(df, "KernelRidge", regressor, newpath, d)

    if d["MLPRegressor"] == "y":
        regressor = MLPRegressor(hidden_layer_sizes=(50,25,10,5,3), random_state=int(d["seed"]),
                                 max_iter=2000)#,
        # early_stopping=True)
        results(df, "MLPRegressor", regressor, newpath, d)

    if d["GradientBoostingRegressor"] == "y":
        regressor = GradientBoostingRegressor(random_state=int(d["seed"]))
        results(df, "GradientBoostingRegressor", regressor, newpath, d)

    if d["xgboost"] == "y":
        import xgboost as xgb
        params = {
            'max_depth': 5,
            'n_estimators': 50,
            'objective': 'reg:linear'}
        regressor = xgb.XGBRegressor(**params)
        results(df, "xgboost", regressor, newpath, d)

    if d["RandomForestRegressor"] == "y":
        regressor = RandomForestRegressor(n_estimators=int(d["n_estimators"]), random_state=int(d["seed"]), max_depth=25, n_jobs=-1)
        if d["rerun_with_top_importances"] == "n":
            results(df, "RandomForestRegressor", regressor, newpath, d)
        else:
            dfimportances = results(df, "RandomForestRegressor", regressor, newpath, d)

            if d["top_k_features"] == "half":
                k = round(len(dfimportances["Columns"])/2)+1
            else:
                k = int(d["top_k_features"])

            cols_to_be_deleted = select_top_k_importants(dfimportances, k) # keep top k important features
            df = trim_df(df, cols_to_be_deleted)

            with open(newpath + "cols_kept_and_deleted_for_k=%s.txt" % d["top_k_features"], "w") as f:
                f.write("cols deleted = \n")
                f.write(str(cols_to_be_deleted))
                f.write("\ncols kept = \n")
                f.write(str(df.columns.tolist()))
            print("Top", len(df.columns), "Columns = ", df.columns.tolist())
            print("Bottom", len(cols_to_be_deleted), "Columns to be deleted = ", cols_to_be_deleted, "\n")
            
            ############################################################################################################
            # Rerunning Model with top k features
            ############################################################################################################
            if d["LinearRegression"] == "y":
                regressor = LinearRegression()
                results(df, "LinearRegression", regressor, newpath, d, "second")

            if d["ElasticNet"] == "y":
                regressor = ElasticNet(alpha=100, l1_ratio=1, max_iter=100000)
                results(df, "ElasticNet", regressor, newpath, d, "second")

            if d["KernelRidge"] == "y":
                regressor = KernelRidge(alpha=0.1)
                results(df, "KernelRidge", regressor, newpath, d, "second")

            if d["xgboost"] == "y":
                params = {
                    'max_depth': 5,
                    'n_estimators': 50,
                    'objective': 'reg:linear'}
                regressor = xgb.XGBRegressor(**params)
                results(df, "xgboost", regressor, newpath, d, "second")

            regressor = RandomForestRegressor(n_estimators=int(d["n_estimators"]), random_state=int(d["seed"]), max_depth=25, n_jobs=-1)
            results(df, "RandomForestRegressor", regressor, newpath, d, "second")

    
    copyfile(parameters, newpath + "parameters.txt")  # Save parameters
    
    df.to_csv(d["file_location"] + d["input_file"] + "_predictions.csv", index=False)  # export file
    
    if d["beep"] == "y":
        import winsound
        Freq = 400 # Set Frequency To 2500 Hertz
        Dur = 1000 # Set Duration To 1000 ms == 1 second
        winsound.Beep(Freq,Dur)
        Freq = 300 # Set Frequency To 2500 Hertz
        winsound.Beep(Freq,Dur)