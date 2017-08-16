"""****************************************************************************
UCD MSc Business Analytics Capstone Project - Predicting Transactions Times
*******************************************************************************
Version 01
Data modelling program
*******************************************************************************
Eoin Carroll
Kieron Ellis
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
from sklearn.model_selection import KFold, train_test_split # cross_val_predict#, cross_val_score,
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from shutil import copyfile  # Used to copy parameters file to directory
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score, \
    median_absolute_error
import seaborn as sns
import datetime
from sklearn.preprocessing import StandardScaler


def plot_percent_correct(y_vals, newpath, alg_initials, input_file, std=None):
    y_vals = [0] + y_vals
    x = range(len(y_vals))
    if std is None:
        plt.plot(x, y_vals, "r")
    else:
        std = [0] + std
        plt.errorbar(x, y_vals, yerr=std, fmt="r", ecolor="b", elinewidth=0.5, capsize=1.5, errorevery=2)

    plt.title(alg_initials + " - Predictions Within Hours")

    plt.xlim(0, 96)
    plt.ylim(0, 100)

    xticks = [(x + 1) * 8 for x in range(12)]
    plt.xticks(xticks, xticks)

    yticks = [i * 10 for i in range(11)]
    plt.yticks(yticks, yticks)

    plt.grid()
    plt.xlabel("Time (hours)")
    plt.ylabel("Percentage of Correct Predictions")

    if not os.path.exists(newpath + "PDFs/"):
        os.makedirs(newpath + "PDFs/")  # Make folder for storing results if it does not exist
    plt.savefig(newpath + alg_initials + "_" + input_file + "_pct_correct.png")
    plt.savefig(newpath + "PDFs/" + alg_initials + "_" + input_file + "_pct_correct.pdf")

    plt.close()


def tree_importances(regr, X, algpath, d, out_file, alg_initials):
    importances = regr.feature_importances_

    dfimportances = pd.DataFrame(data=X.columns, columns=["Columns"])
    dfimportances["Importances"] = importances

    dfimportances = dfimportances.sort_values("Importances", ascending=False)
    if d["export_importances_csv"] == "y":
        dfimportances.to_csv(algpath + "importances.csv", index=False)

    print("Feature Importances:")
    out_file.write("\n\nFeature Importances:\n")
    out_file.write("\nThe importances for each variable used by Random Forest Regression were as follows:\n")
    for i, (col, importance) in enumerate(zip(dfimportances["Columns"].values.tolist(), dfimportances[
        "Importances"].values.tolist())):
        out_file.write("\t%d. \"%s\" (%f)\n" % (i + 1, col, importance))
        print("\t%d. \"%s\" (%f)" % (i + 1, col, importance))

    if d["export_top_k_csv"] == "y":
        df_top = return_new_top_k_df(df, dfimportances, int(d["top_k_features"]))
        if d["specify_subfolder"] == "n":
            df_top.to_csv("%s%s_%s_top_%s.csv" % (d["file_location"], d["input_file"], alg_initials,
                                                              d["top_k_features"]), index=False)
        else:
            df_top.to_csv("%s%s_%s_%s_top_%s.csv" % (d["file_location"], d["input_file"], d["specify_subfolder"],
                                                  alg_initials,d["top_k_features"]),index=False)


def regression_coef_importances(regr, X, algpath, d, out_file, alg_initials):
    scalerX = StandardScaler().fit(X)
    normed_coefs = scalerX.transform(regr.coef_.reshape(1, -1))
    normed_coefs_list = normed_coefs.tolist()[0]
    dfimportances = pd.DataFrame()
    dfimportances["Columns"] = X.columns.tolist()
    dfimportances["Importances"] = normed_coefs_list
    dfimportances["Absolute_Importances"] = abs(dfimportances["Importances"])
    dfimportances = dfimportances.sort_values("Absolute_Importances", ascending=False)

    total_importances = dfimportances["Absolute_Importances"].sum()
    dfimportances["Percentage_Importance"] = dfimportances["Absolute_Importances"] / total_importances

    if d["export_importances_csv"] == "y":
        dfimportances.to_csv(algpath + "importances.csv", index=False)
    # print(dfimportances)

    print("Feature Importances: \"column\" (magnitude of importance) [percentage of importance]")
    out_file.write("\n\nFeature Importances: \"column\" (magnitude of importance) [percentage of importance]\n")
    
    out_file.write("\nThe importances for each variable used by Linear Regression were as follows:")
    out_file.write("\n\"Variable Name\" (Standardised Regression Coefficient) [Percentage of Importance]\n")
    
    for i, (col, importance, pct) in enumerate(zip(dfimportances["Columns"].values.tolist(), dfimportances[
        "Importances"].values.tolist(), dfimportances["Percentage_Importance"].values.tolist())):
        out_file.write("\t%d. \"%s\" (%f) [%f]\n" % (i + 1, col, importance, pct))
        print("\t%d. \"%s\" (%f) [%f]" % (i + 1, col, importance, pct))

    if d["export_top_k_csv"] == "y":
        df_top = return_new_top_k_df(df, dfimportances, int(d["top_k_features"]))
        if d["specify_subfolder"] == "n":
            df_top.to_csv("%s%s_%s_top_%s.csv" % (d["file_location"], d["input_file"], alg_initials,
                                                              d["top_k_features"]), index=False)
        else:
            df_top.to_csv("%s%s_%s_%s_top_%s.csv" % (d["file_location"], d["input_file"], d["specify_subfolder"],
                                                  alg_initials, d["top_k_features"]),index=False)


def return_new_top_k_df(in_df, dfimportances, k):
    df = in_df.copy()
    top = dfimportances["Columns"].iloc[:k].values.tolist()
    top.append("TimeTaken")
    if "Created_On" in df.columns:
        top.append("Created_On")
        top.append("ResolvedDate")
    return df[top]


def histogram(df, column, newpath):  # Create histogram of preprocessed data
    plt.figure()  # Plot all data
    plt.hist(df[column], bins='auto')
    plt.xlabel('TimeTaken (Seconds)')
    plt.ylabel('Frequency')
    plt.title(column + " all data")
    if not os.path.exists(newpath + "PDFs/"):
        os.makedirs(newpath + "PDFs/")  # Make folder for storing results if it does not exist
    plt.savefig(newpath + column + "_all.png")
    plt.savefig(newpath + "PDFs/" + column + "_all.pdf")
    plt.close()

    plt.figure()  # Plot times under 500,000 seconds
    plt.hist(df[df.TimeTaken < 500000][column], bins='auto')
    plt.xlabel('TimeTaken (Seconds)')
    plt.ylabel('Frequency')
    plt.title(column + " < 500000s data")
    plt.savefig(newpath + column + "_500000.png")
    plt.savefig(newpath + "PDFs/" + column + "_500000.pdf")
    plt.close()

    plt.figure()  # Plot times under 100,000 seconds
    plt.hist(df[df.TimeTaken < 100000][column], bins='auto')
    plt.xlabel('TimeTaken (Seconds)')
    plt.ylabel('Frequency')
    plt.title(column + " < 100000s data")
    plt.savefig(newpath + column + "_100000.png")
    plt.savefig(newpath + "PDFs/" + column + "_100000.pdf")
    plt.close()

    plt.figure()  # Plot all data
    plt.hist(np.log(df[column]), bins='auto')
    plt.xlabel('Log of TimeTaken (Seconds)')
    plt.ylabel('Frequency')
    plt.title(column + " Log of all data")
    plt.savefig(newpath + column + "_log_all.png")
    plt.savefig(newpath + "PDFs/" + column + "_log_all.pdf")
    plt.close()

    plt.figure()  # Plot times under 500,000 seconds
    plt.hist(np.log(df[df.TimeTaken < 500000][column]), bins='auto')
    plt.xlabel('Log of TimeTaken (Seconds)')
    plt.ylabel('Frequency')
    plt.title(column + " Log of < 500000s data")
    plt.savefig(newpath + column + "_log_500000.png")
    plt.savefig(newpath + "PDFs/" + column + "_log_500000.pdf")
    plt.close()

    plt.figure()  # Plot times under 100,000 seconds
    plt.hist(np.log(df[df.TimeTaken < 100000][column]), bins='auto')
    plt.xlabel('Log of TimeTaken (Seconds)')
    plt.ylabel('Frequency')
    plt.title(column + " Log of < 100000s data")
    plt.savefig(newpath + column + "_log_100000.png")
    plt.savefig(newpath + "PDFs/" + column + "_log_100000.pdf")
    plt.close()

    
def plot(x, y, alg, data, newpath, alg_initials,  input_file):
    sns.reset_orig() # plt.rcParams.update(plt.rcParamsDefault)

    # Plot all
    plt.figure()
    plt.plot(x, y, 'ro', alpha=0.1, markersize=4)
    
    def f(x, a, b):
        return a*x + b
        
    coeffs = np.polyfit(x,y,1)
    x = np.array([0, 700])
    plt.plot(x, f(x, *coeffs), "b", linewidth=0.5)
    
    plt.xlabel("Actual - Time Taken (Hours)")
    plt.ylabel("Prediction - Time Taken (Hours)")
    if alg == "Simple":
        plt.title(alg_initials)

    else:
        plt.title(alg + data)
    plt.ylim([0, 700])
    plt.xlim([0, 700])
    plt.grid()
    plt.gca().set_aspect('equal', adjustable='box')
    ticks = [0, 100, 200, 300, 400, 500, 600, 700]
    tick_names = [0, 100, 200, 300, 400, 500, 600, 700]
    plt.xticks(ticks, tick_names)
    plt.yticks(ticks, tick_names)
    plt.tight_layout()  # Force everything to fit on figure
    if not os.path.exists(newpath + "PDFs/"):
        os.makedirs(newpath + "PDFs/")  # Make folder for storing results if it does not exist
    plt.savefig(newpath + alg_initials + "_" + input_file + data + "_allhrs.png")
    plt.savefig(newpath + "PDFs/" + alg_initials + "_" + input_file + data + "_allhrs.pdf")
    plt.close()
    
    #
    # # Plot up to 2500000s only (600hrs)
    # plt.figure()
    # plt.plot(x, y, 'ro', alpha=0.1, markersize=4)
    # # sns.plot(x, y, 'ro', alpha=0.1, plot_kws={"s": 3}) #scatter_kws={"s": 100}
    # # sns.lmplot(x, y, data = in_data, scatter_kws={"s": 4, 'alpha':0.3, 'color': 'red'}, line_kws={"linewidth": 1,'color': 'blue'}, fit_reg=False)
    # plt.xlabel("Actual - Time Taken (Hours)")
    # plt.ylabel("Prediction - Time Taken (Hours)")
    # if alg == "Simple":
    #     plt.title(alg_initials)
    # # elif alg == "Statsmodels_OLS":
    # #     plt.title(alg + data + " < 600hrs")
    # else:
    #     plt.title(alg + data + " < 600hrs")
    #     # plt.title(alg + " < 600hrs")
    # # plt.axis('equal')
    # plt.ylim([0, 600])
    # plt.xlim([0, 600])
    # plt.gca().set_aspect('equal', adjustable='box')
    # ticks = [0, 100, 200, 300, 400, 500, 600]
    # tick_names = [0, 100, 200, 300, 400, 500, 600]
    # plt.xticks(ticks, tick_names)
    # plt.yticks(ticks, tick_names)
    # plt.tight_layout()  # Force everything to fit on figure
    # if not os.path.exists(newpath + "PDFs/"):
    #     os.makedirs(newpath + "PDFs/")  # Make folder for storing results if it does not exist
    # # if alg == "Statsmodels_OLS":
    # plt.savefig(newpath + alg_initials + "_" + input_file + data + "_600hrs.png")
    # plt.savefig(newpath + "PDFs/" + alg_initials + "_" + input_file + data  + "_600hrs.pdf")
    # # else:
    # #     plt.savefig(newpath + alg_initials + "_" + input_file + "_600hrs.png")
    # #     plt.savefig(newpath + "PDFs/" + alg_initials + "_" + input_file + "_600hrs.pdf")
    # plt.close()
    #
    # # Plot up to 800000s only (200hrs)
    # plt.figure()
    # plt.plot(x, y, 'ro', alpha=0.1, markersize=4)
    # # sns.plot(x, y, 'ro', alpha=0.1, plot_kws={"s": 3}) #scatter_kws={"s": 100}
    # # sns.lmplot(x, y, data = in_data, scatter_kws={"s": 4, 'alpha':0.3, 'color': 'red'}, line_kws={"linewidth": 1,'color': 'blue'}, fit_reg=False)
    # plt.xlabel("Actual - Time Taken (Hours)")
    # plt.ylabel("Prediction - Time Taken (Hours)")
    # if alg == "Simple":
    #     plt.title(alg_initials)
    # # elif alg == "Statsmodels_OLS":
    # #     plt.title(alg + data + " < 200hrs")
    # else:
    #     plt.title(alg + data + " < 200hrs")
    #     # plt.title(alg + " < 200hrs")
    # # plt.axis('equal')
    # plt.ylim([0, 200])
    # plt.xlim([0, 200])
    # plt.gca().set_aspect('equal', adjustable='box')
    # ticks = [0, 50, 100, 150, 200]
    # tick_names = [0, 50, 100, 150, 200]
    # plt.xticks(ticks, tick_names)
    # plt.yticks(ticks, tick_names)
    # plt.tight_layout()  # Force everything to fit on figure
    # if not os.path.exists(newpath + "PDFs/"):
    #     os.makedirs(newpath + "PDFs/")  # Make folder for storing results if it does not exist
    # # if alg == "Statsmodels_OLS":
    # plt.savefig(newpath + alg_initials + "_" + input_file + data + "_200hrs.png")
    # plt.savefig(newpath + "PDFs/" + alg_initials + "_" + input_file + data + "_200hrs.pdf")
    # # else:
    # #     plt.savefig(newpath + alg_initials + "_" + input_file + "_200hrs.png")
    # #     plt.savefig(newpath + "PDFs/" + alg_initials + "_" + input_file + "_200hrs.pdf")
    # plt.close()
    #
    # # Plot up to 100000s only (24hrs)
    # plt.figure()
    # plt.plot(x, y, 'ro', alpha=0.1, markersize=4)
    # # sns.plot(x, y, 'ro', alpha=0.1, plot_kws={"s": 3}) #scatter_kws={"s": 100}
    # # sns.lmplot(x, y, data = in_data, scatter_kws={"s": 4, 'alpha':0.3, 'color': 'red'}, line_kws={"linewidth": 1,'color': 'blue'}, fit_reg=False)
    # plt.xlabel("Actual - Time Taken (Hours)")
    # plt.ylabel("Prediction - Time Taken (Hours)")
    # if alg == "Simple":
    #     plt.title(alg_initials)
    # # elif alg == "Statsmodels_OLS":
    # #     plt.title(alg + data + " < 24hrs")
    # else:
    #     plt.title(alg + data + " < 24hrs")
    #     # plt.title(alg + " < 24hrs")
    # # plt.axis('equal')
    # plt.ylim([0, 24])
    # plt.xlim([0, 24])
    # plt.gca().set_aspect('equal', adjustable='box')
    # ticks = [0, 4, 8, 12, 16, 20, 24]
    # tick_names = [0, 4, 8, 12, 16, 20, 24]
    # plt.xticks(ticks, tick_names)
    # plt.yticks(ticks, tick_names)
    # plt.tight_layout()  # Force everything to fit on figure
    # if not os.path.exists(newpath + "PDFs/"):
    #     os.makedirs(newpath + "PDFs/")  # Make folder for storing results if it does not exist
    # # if alg == "Statsmodels_OLS":
    # plt.savefig(newpath + alg_initials + "_" + input_file + data + "_24hrs.png")
    # plt.savefig(newpath + "PDFs/" + alg_initials + "_" + input_file + data + "_24hrs.pdf")
    # # else:
    # #     plt.savefig(newpath + alg_initials + "_" + input_file + "_24hrs.png")
    # #     plt.savefig(newpath + "PDFs/" + alg_initials + "_" + input_file + "_24hrs.pdf")
    # plt.close()


def day_in_quarter(date):
    # Function found on stack overflow
    # https://stackoverflow.com/questions/37471704/how-do-i-get-the-correspondent-day-of-the-quarter-from-a-date-field
    q2 = (datetime.datetime.strptime("4/1/{0:4d}".format(date.year), "%m/%d/%Y")).timetuple().tm_yday
    q3 = (datetime.datetime.strptime("7/1/{0:4d}".format(date.year), "%m/%d/%Y")).timetuple().tm_yday
    q4 = (datetime.datetime.strptime("10/1/{0:4d}".format(date.year), "%m/%d/%Y")).timetuple().tm_yday

    cur_day =  date.timetuple().tm_yday
    if (date.month < 4):
        return cur_day
    elif (date.month < 7):
        return cur_day - q2 + 1
    elif (date.month < 10):
        return cur_day - q3 + 1
    else:
        return cur_day - q4 + 1


def get_extra_cols(df, alg, d):
    df["Created_On"] = pd.to_datetime(df["Created_On"])
    df["ResolvedDate"] = pd.to_datetime(df["ResolvedDate"])
    df["TimeTaken_hours"] = df["TimeTaken"]/60/60


    df["TimeTaken_hours_%s"%alg] = df["TimeTaken_%s"%alg]/60/60

    df["Created_On_Day"] = df["Created_On"].apply(lambda x: int(x.strftime("%H")))  # Time of the day
    df["ResolvedDate_Day"] = df["ResolvedDate"].apply(lambda x: int(x.strftime("%H")))  # Time of the day
    df["Created_On_Week"] = df["Created_On"].apply(lambda x: int(x.strftime("%w")))  # Day of the week
    df["ResolvedDate_Week"] = df["ResolvedDate"].apply(lambda x: int(x.strftime("%w")))  # Day of the week
    df["Created_On_Month"] = df["Created_On"].apply(lambda x: int(x.strftime("%d")))  # Day of the month
    df["Created_On_Month"]-=1
    df["ResolvedDate_Month"] = df["ResolvedDate"].apply(lambda x: int(x.strftime("%d")))  # Day of the month
    df["ResolvedDate_Month"]-=1
    df["Created_On_Qtr"] = df["Created_On"].apply(lambda x: int(day_in_quarter(x)))  # Day of the Qtr
    df["Created_On_Qtr"]-=1
    df["ResolvedDate_Qtr"] = df["ResolvedDate"].apply(lambda x: int(day_in_quarter(x)))  # Day of the Qtr
    df["ResolvedDate_Qtr"]-=1

    df["Created_On_MonthOfYear"] = df["Created_On"].apply(lambda x: int(x.strftime("%m")))
    df["Created_On_MonthOfYear"] -= 1
    df["ResolvedDate_MonthOfYear"] = df["ResolvedDate"].apply(lambda x: int(x.strftime("%m")))
    df["ResolvedDate_MonthOfYear"] -= 1

    df["Created_On_Year"] = df["Created_On"].apply(lambda x: int(x.strftime("%j")))
    df["Created_On_Year"] -= 1
    df["ResolvedDate_Year"] = df["ResolvedDate"].apply(lambda x: int(x.strftime("%j")))
    df["ResolvedDate_Year"] -= 1

    return df


def get_errors(df, alg, time_range, col):
    r2_scores = []
    rmse_scores = []
    mae_scores = []
    for i in range(time_range):
        actual = df.loc[df[col] == i, "TimeTaken_hours"]
        predicted = df.loc[df[col] == i, "TimeTaken_hours_%s"%alg]

        if len(df[col][df[col] == i]) == 0:
            r2_scores.append(0)
            rmse_scores.append(0)
            mae_scores.append(0)
        else:
            # if r2_score(actual, predicted) < 0:
            #     r2_scores.append(-1)
            # else:
            #     r2_scores.append(r2_score(actual, predicted))
            r2_scores.append(r2_score(actual, predicted))
            rmse_scores.append(np.sqrt(mean_squared_error(actual, predicted)))
            mae_scores.append(mean_absolute_error(actual, predicted))
    return [r2_scores, rmse_scores, mae_scores]


def plot_errors(x_ticks, y, error_name, alg, y_label, x_label, data, alg_initials, newpath):
    if x_label == "Day of Qtr Created" or x_label == "Day of Qtr Resolved" or x_label == "Day of Year Resolved" or \
                 x_label == "Day of Year Created" or x_label == "Month Of Year Created" or x_label == "Month Of Year " \
                                                                                                      "Resolved":
        y = np.array(y)
        z = np.where(np.array(y) >= 0)
        z = z[0]
        y_z = y[z]

        x_num = [i for i in range(len(y))]

        y_np = np.array(y)

        reverse = False
        if error_name == "R2":
            reverse = True
        # pal = sns.cubehelix_palette(len(y), start=start, rot=rot,dark=.1, light=.9, reverse=reverse)
        pal = sns.cubehelix_palette(len(y), start=1, rot=0,hue=1.5, gamma=1,dark=.3, light=0.9, reverse=reverse)

        rank = y_np.argsort().argsort()
        sns.barplot(x_num, y, palette=np.array(pal[::-1])[rank])

        plt.title("%s - %s to %s" % (alg, error_name, x_label))
        plt.ylabel(y_label)
        plt.xlabel(x_label)

        if x_label == "Day of Year Resolved" or x_label == "Day of Year Created":
            plt.xticks(x_ticks, x_ticks, rotation = "vertical")
            plt.xlim(56, 210)
            # todo - change the xticks with the actual end of data date instead of guessing at 240
        elif x_label == "Month Of Year Created" or x_label == "Month Of Year Resolved":
            plt.xticks([_+1 for _ in range(6)], x_ticks)
            plt.xlim(0, 7)
        else:
            plt.xticks(x_ticks, x_ticks)

        min_ylim = min(y_z)-np.std(y_z)/3
        if min_ylim < 0:
            min_ylim = 0
        plt.ylim(min_ylim, max(y_z)+np.std(y_z)/3)

        if not os.path.exists(newpath + "PDFs/"):
            os.makedirs(newpath + "PDFs/")  # Make folder for storing results if it does not exist
        if not os.path.exists(newpath + "PDFs/" + error_name + "/"):
            os.makedirs(newpath + "PDFs/" + error_name + "/")  # Make folder for storing results if it does not exist
        if not os.path.exists(newpath + error_name + "/"):
            os.makedirs(newpath + error_name + "/")  # Make folder for storing results if it does not exist

        plt.savefig(newpath + error_name +"/"+ x_label  + ".png")
        plt.savefig(newpath + "PDFs/" + error_name +"/"+ x_label  + ".pdf")

    else:
        plt.figure()

        x_num = [i for i in range(len(x_ticks))]

        y_np = np.array(y)
        reverse = False
        if error_name == "R2":
            reverse = True

        pal = sns.cubehelix_palette(len(y), start=1, rot=0,hue=1.5, gamma=1,dark=.3, light=0.9, reverse=reverse)
        rank = y_np.argsort().argsort()
        sns.barplot(x_num, y, palette=np.array(pal[::-1])[rank])
        plt.xticks(x_num, x_ticks, rotation="vertical")
        plt.title("%s - %s to %s" % (alg, error_name, x_label))
        plt.ylabel(y_label)
        plt.xlabel(x_label)

        plt.ylim(min(y)-np.std(y)/3, max(y)+np.std(y)/3)

        if not os.path.exists(newpath + "PDFs/"):
            os.makedirs(newpath + "PDFs/")  # Make folder for storing results if it does not exist
        if not os.path.exists(newpath + "PDFs/" + error_name + "/"):
            os.makedirs(newpath + "PDFs/" + error_name + "/")  # Make folder for storing results if it does not exist
        if not os.path.exists(newpath + error_name + "/"):
            os.makedirs(newpath + error_name + "/")  # Make folder for storing results if it does not exist

        plt.savefig(newpath + error_name +"/"+ x_label  + ".png")
        plt.savefig(newpath + "PDFs/" + error_name +"/"+ x_label  + ".pdf")

    plt.close()


def plot_errors_main(df, alg, data, newpath, alg_initials):
    df = get_extra_cols(df, alg, d)
    error_names = ["R2", "RMSE", "MAE"]
    y_labels = ["R2 score", "RMSE (Hours)", "MAE (Hours)"]

    scores = get_errors(df, alg, 24, "Created_On_Day")
    x_vals = [x for x in range(24)]
    x_label = "Time of Day Created"
    for score, error_name, y_label in zip(scores, error_names, y_labels):
        plot_errors(x_vals, score, error_name, alg, y_label, x_label, data, alg_initials, newpath)

    scores = get_errors(df, alg, 24, "ResolvedDate_Day")
    x_vals = [x for x in range(24)]
    x_label = "Time of Day Resolved"
    for score, error_name, y_label in zip(scores, error_names, y_labels):
        plot_errors(x_vals, score, error_name, alg, y_label, x_label, data, alg_initials, newpath)

    scores = get_errors(df, alg, 7, "Created_On_Week")
    x_vals = ["Su", "Mo", "Tu", "We", "Th", "Fr", "Sa"]
    x_label = "Day of Week Created"
    for score, error_name, y_label in zip(scores, error_names, y_labels):
        plot_errors(x_vals, score, error_name, alg, y_label, x_label, data, alg_initials, newpath)

    scores = get_errors(df, alg, 7, "ResolvedDate_Week")
    x_vals = ["Su", "Mo", "Tu", "We", "Th", "Fr", "Sa"]
    x_label = "Day of Week Resolved"
    for score, error_name, y_label in zip(scores, error_names, y_labels):
        plot_errors(x_vals, score, error_name, alg, y_label, x_label, data, alg_initials, newpath)

    scores = get_errors(df, alg, 31, "Created_On_Month")
    x_vals = [x+1 for x in range(31)]
    x_label = "Day of Month Created"
    for score, error_name, y_label in zip(scores, error_names, y_labels):
        plot_errors(x_vals, score, error_name, alg, y_label, x_label, data, alg_initials, newpath)

    scores = get_errors(df, alg, 31, "ResolvedDate_Month")
    x_vals = [x+1 for x in range(31)]
    x_label = "Day of Month Resolved"
    for score, error_name, y_label in zip(scores, error_names, y_labels):
        plot_errors(x_vals, score, error_name, alg, y_label, x_label, data, alg_initials, newpath)

    scores = get_errors(df, alg, 89, "Created_On_Qtr")
    x_vals = [(x+1)*5 for x in range(18)]
    x_label = "Day of Qtr Created"
    for score, error_name, y_label in zip(scores, error_names, y_labels):
        plot_errors(x_vals, score, error_name, alg, y_label, x_label, data, alg_initials, newpath)

    scores = get_errors(df, alg, 89, "ResolvedDate_Qtr")
    x_vals = [(x+1)*5 for x in range(18)]
    x_label = "Day of Qtr Resolved"
    for score, error_name, y_label in zip(scores, error_names, y_labels):
        plot_errors(x_vals, score, error_name, alg, y_label, x_label, data, alg_initials, newpath)

    scores = get_errors(df, alg, 9, "Created_On_MonthOfYear")
    x_vals = [2,3,4,5,6,7]
    x_label = "Month Of Year Created"
    for score, error_name, y_label in zip(scores, error_names, y_labels):
        plot_errors(x_vals, score, error_name, alg, y_label, x_label, data, alg_initials, newpath)

    scores = get_errors(df, alg, 9, "ResolvedDate_MonthOfYear")
    x_vals = [2,3,4,5,6,7]
    x_label = "Month Of Year Resolved"
    for score, error_name, y_label in zip(scores, error_names, y_labels):
        plot_errors(x_vals, score, error_name, alg, y_label, x_label, data, alg_initials, newpath)

    scores = get_errors(df, alg, 240, "Created_On_Year")
    # todo - change the xticks with the actual end of data date instead of guessing at 240
    x_vals = [(x+1)*7 for x in range(34)]
    x_label = "Day of Year Created"
    for score, error_name, y_label in zip(scores, error_names, y_labels):
        plot_errors(x_vals, score, error_name, alg, y_label, x_label, data, alg_initials, newpath)

    scores = get_errors(df, alg, 240, "ResolvedDate_Year")
    # todo - change the xticks with the actual end of data date instead of guessing at 240
    x_vals = [(x+1)*7 for x in range(34)]
    x_label = "Day of Year Resolved"
    for score, error_name, y_label in zip(scores, error_names, y_labels):
        plot_errors(x_vals, score, error_name, alg, y_label, x_label, data, alg_initials, newpath)


def get_keepers():
    keepers = ["AmountinUSD",
               "AuditDuration",
               "Cases_created_within_past_8_hours",
               "Cases_resolved_within_past_8_hours",
               "Complexity",
               "Concurrent_open_cases",
               "CountryProcessed_africa",
               "CountryProcessed_asia",
               "CountryProcessed_australia",
               "CountryProcessed_europe",
               "CountryProcessed_northamerica",
               "CountryProcessed_other",
               "CountryProcessed_southamerica",
               "CountrySource_africa",
               "CountrySource_asia",
               "CountrySource_australia",
               "CountrySource_europe",
               "CountrySource_northamerica",
               "CountrySource_other",
               "CountrySource_southamerica",
               "Created_on_Weekend",
               "IsSOXCase",
               "Numberofreactivations",
               "Priority",
               "Queue_AOC",
               "Queue_APOC",
               "Queue_Broken",
               "Queue_E&E",
               "Queue_EOC",
               "Queue_LOC",
               "Queue_NAOC",
               "ROCName_AOC",
               "ROCName_APOC",
               "ROCName_EOC",
               "Revenutype_Advanced Billing",
               "Revenutype_Credit / Rebill",
               "Revenutype_Current Revenue",
               "Revenutype_Disputed Revenue",
               "Revenutype_Future Billing",
               "Revenutype_Future OTRRR with OLS",
               "Revenutype_Future OTRRR without OLS",
               "Revenutype_New Work Sold",
               "Revenutype_Non-revenue",
               "Revenutype_Revenue Impacting Case / Pending Revenue",
               "Revenutype_Revenue Unknown",
               "Rolling_Mean",
               "Rolling_Median",
               "Rolling_Std",
               "SalesLocation_africa",
               "SalesLocation_asia",
               "SalesLocation_australia",
               "SalesLocation_europe",
               "SalesLocation_northamerica",
               "SalesLocation_other",
               "SalesLocation_southamerica",
               "Seconds_left_Day",
               "Seconds_left_Month",
               "Seconds_left_Qtr",
               "Seconds_left_Year",
               "Source_E-mail",
               "Source_Fax",
               "Source_Hard Copy",
               "Source_Manual",
               "Source_Soft Copy",
               "Source_Web",
               "Source_eAgreement (Ele)",
               "Source_eAgreement (Phy)",
               "StageName",
               "StatusReason_3rd Party Hold",
               "StatusReason_Completed",
               "StatusReason_Customer Hold",
               "StatusReason_Final Routing",
               "StatusReason_Information Provided",
               "StatusReason_New",
               "StatusReason_New Mail",
               "StatusReason_Problem Solved",
               "StatusReason_Reactivated",
               "StatusReason_Ready for Archiving",
               "StatusReason_Ready for Audit",
               "SubReason_Additional Product Order",
               "SubReason_Basic Enterprise Commitment",
               "SubReason_Electronic Order Pend / Reject",
               "SubReason_Future Pricing Only CPS",
               "SubReason_Manual Order Entry",
               "SubReason_Meteaop",
               "SubReason_P&H Electronic Order",
               "SubReason_Tax Exemption Order",
               "SubReason_True Up",
               "SubReason_Zero Usage Order",
               "sourcesystem_AplQuest",
               "sourcesystem_Aplquest",
               "sourcesystem_CLT",
               "sourcesystem_Current Revenue",
               "sourcesystem_Moritz Jürgensen",
               "sourcesystem_NEMEC",
               "sourcesystem_NMEC",
               "sourcesystem_Web",
               "sourcesystem_`",
               "sourcesystem_clt",
               "sourcesystem_web",
               "HoldDuration", "HoldTypeName_3rd Party", "HoldTypeName_Customer", "HoldTypeName_Internal",
               "AssignedToGroup_BPO", "AssignedToGroup_CRMT",
               "IsGovernment",
                "AmountinUSD",
                "IsMagnumCase",
                "IsSignature",
               ]
    return keepers


def results(df, alg, in_regressor, newpath, d, alg_counter, alg_initials, df_test=None):
    algpath = newpath + alg_initials + "/"
    if not os.path.exists(algpath):
        os.makedirs(algpath)  # Make folder for storing results if it does not exist

    out_file_name = algpath + alg + ".txt"  # Log file name

    out_file = open(out_file_name, "w")  # Open log file
    out_file.write(alg + " " + time.strftime("%Y%m%d-%H%M%S") + "\n")
    out_file.write("\nInput file name: %s \n" % d["input_file"])

    print("DF Shape:", df.shape, "\n")

    X_train = df.drop("TimeTaken", axis=1)
    keepers = get_keepers()
    for col in X_train.columns:
        if col not in keepers:
            del X_train[col]

    # output the features being used
    if alg_counter == 1:
        print("Length of df = %s" % len(df.columns)) # todo add these 3 print messages to outfile or delete them
        print("Length of keepers = %s" % len(keepers))
        print("Length of features used = %s\n" % len(X_train.columns))
        print("Features used:")

    for i, col in enumerate(X_train.columns):
        if alg_counter == 1:
            print("\t%s - %s" % (i+1, col))

    y_train = df["TimeTaken"]

    ####################################################################################################################
    # Simple Statistics
    ####################################################################################################################
    if alg_counter == 1:
        simplepath = newpath + "Simple_Stat_Plots/"
        if not os.path.exists(simplepath):
            os.makedirs(simplepath)  # Make folder for storing results if it does not exist

        simple_out_file_name = simplepath + "simple_stats.txt"  # Log file name

        simple_out_file = open(simple_out_file_name, "w")  # Open log file
        simple_out_file.write("Simple Stats - " + time.strftime("%Y%m%d-%H%M%S") + "\n")
        simple_out_file.write("\nInput file name: %s\n" % d["input_file"])

        print("\nDF Shape:", df.shape)
        simple_out_file.write("\nDF Shape: " + str(df.shape) + "\n")

        mean_time = np.mean(y_train)  # Calculate mean of predictions
        std_time = np.std(y_train)  # Calculate standard deviation of predictions
        median_time = np.median(y_train)  # Calculate standard deviation of predictions

        simple_out_file.write("\nSimple TimeTaken stats")
        simple_out_file.write("\n\tmean_time = %s" % mean_time)
        simple_out_file.write("\n\tstd_time = %s" % std_time)
        simple_out_file.write("\n\tmedian_time = %s\n" % median_time)
        
        simple_out_file.write("\n\n\tThe mean time taken to resolve cases was %.2f hours, median was %.2f hours and the standard deviation was %.2f hours." % (mean_time, median_time, std_time))
        
        
        print("\nSimple TimeTaken stats")
        print("\tmean_time = %s" % mean_time)
        print("\tstd_time = %s" % std_time)
        print("\tmedian_time = %s\n" % median_time)

        df["Mean_TimeTaken"] = mean_time
        df["Median_TimeTaken"] = median_time

        simple_percent_close = []
        simple_number_close = [0 for _ in range(96)]

        print("\n..calculating correct predictions within hours..\n")

        for i in range(len(df["Mean_TimeTaken"])):
            for j in range(len(simple_number_close)):
                if abs(df["Mean_TimeTaken"].iloc[i] - y_train.iloc[i]) <= j + 1:  # Within 1 hour
                    simple_number_close[j] += 1

        for j in simple_number_close:
            simple_percent_close.append(j / len(y_train) * 100)

        interesting_hours = [1, 4, 8, 16, 24, 48, 72, 96]
        for hour in interesting_hours:
            hour -= 1
            print("\t{1:s} % test predictions error within {3:d} hour(s) -> Mean: {0:.2f}% of {2:d}/10".format(
                simple_percent_close[hour], "Mean", len(y_train), hour+1))
            simple_out_file.write("\n\tPredictions correct within %s hour(s): %.2f%%" % (hour+1,
                                                                                          simple_percent_close[hour]))

        mean_time_test_r2 = r2_score(y_train, df["Mean_TimeTaken"])
        mean_time_test_rmse = np.sqrt(mean_squared_error(y_train, df["Mean_TimeTaken"]))
        mean_time_test_meanae = mean_absolute_error(y_train, df["Mean_TimeTaken"])
        mean_time_test_evs = explained_variance_score(y_train, df["Mean_TimeTaken"])
        mean_time_test_medianae = median_absolute_error(y_train, df["Mean_TimeTaken"])

        median_time_test_r2 = r2_score(y_train, df["Median_TimeTaken"])
        median_time_test_rmse = np.sqrt(mean_squared_error(y_train, df["Median_TimeTaken"]))
        median_time_test_meanae = mean_absolute_error(y_train, df["Median_TimeTaken"])
        median_time_test_evs = explained_variance_score(y_train, df["Median_TimeTaken"])
        median_time_test_medianae = median_absolute_error(y_train, df["Median_TimeTaken"])

        simple_out_file.write("\n\nMean:")
        simple_out_file.write("\n\tTest R2 = %s" % mean_time_test_r2)
        simple_out_file.write("\n\tTest RMSE = %s" % mean_time_test_rmse)
        simple_out_file.write("\n\tTest MeanAE = %s" % mean_time_test_meanae)
        simple_out_file.write("\n\tTest MedianAE = %s\n" % mean_time_test_medianae)
        simple_out_file.write("\n\tTest EVS = %s" % mean_time_test_evs)

        simple_out_file.write("\n\tmedian_time_test_r2 = %s" % median_time_test_r2)
        simple_out_file.write("\n\tmedian_time_test_rmse = %s" % median_time_test_rmse)
        simple_out_file.write("\n\tmedian_time_test_meanae = %s" % median_time_test_meanae)
        simple_out_file.write("\n\tmedian_time_test_evs = %s" % median_time_test_evs)
        simple_out_file.write("\n\tmedian_time_test_medianae = %s\n" % median_time_test_medianae)

        print("\n\tmean_time_test_r2 = %s" % mean_time_test_r2)
        print("\tmean_time_test_rmse = %s" % mean_time_test_rmse)
        print("\tmean_time_test_meanae = %s " % mean_time_test_meanae)
        print("\tmean_time_test_evs = %s" % mean_time_test_evs)
        print("\tmean_time_test_medianae = %s\n " % mean_time_test_medianae)

        print("\tmedian_time_test_r2 = %s" % median_time_test_r2)
        print("\tmedian_time_test_rmse = %s" % median_time_test_rmse)
        print("\tmedian_time_test_meanae = %s" % median_time_test_meanae)
        print("\tmedian_time_test_evs = %s" % median_time_test_evs)
        print("\tmedian_time_test_medianae = %s" % median_time_test_medianae)


        if d["plotting"] == "y":
            print("\n..plotting..")

            plot(df["TimeTaken"],df["Mean_TimeTaken"], "Simple", "", simplepath, "Mean", d["input_file"])
            plot(df["TimeTaken"],df["Median_TimeTaken"], "Simple", "", simplepath, "Median",  d["input_file"])
            plot_percent_correct(simple_percent_close, simplepath, "Mean", d["input_file"])

        simple_out_file.close()
    ####################################################################################################################
    # Machine Learning
    ####################################################################################################################
    print("\n..%s.." % alg)

    # train_rmse = []
    # test_rmse = []
    # train_r_sq = []
    # test_r_sq = []
    # train_mae = []
    # test_mae = [] # Mean Absolute Error
    # train_evs = [] # Explained variance regression score function
    # test_evs = []
    # train_median_ae = [] # Median absolute error regression loss
    # test_median_ae = []

    # percent_close = [[] for _ in range(96)]
    percent_close = []

    # max_time = 2000000
    df["TimeTaken_%s" % alg] = -1000000  # assign a random value

    # for train_indices, test_indices in kf.split(X, y):
    #     # Get the dataset; this is the way to access values in a pandas DataFrame
    #     trainData_x = X.iloc[train_indices]
    #     trainData_y = y.iloc[train_indices]
    #     testData_x = X.iloc[test_indices]
    #     testData_y = y.iloc[test_indices]

    # Train the model, and evaluate it
    regr = in_regressor
    regr.fit(X_train, y_train.values.ravel())

    # Get predictions
    y_train_pred = regr.predict(X_train)

    number_close = [0 for _ in range(96)]

    for i in range(len(y_train_pred)):  # Convert high or low predictions to 0 or 3 std
        if y_train_pred[i] < 0:  # Convert all negative predictions to 0
            y_train_pred[i] = 0
        if y_train_pred[i] >= 700:
            y_train_pred[i] = 700
        if math.isnan(y_train_pred[i]):  # If NaN set to 0
            y_train_pred[i] = 0

        for j in range(len(number_close)):
            if abs(y_train_pred[i] - y_train.iloc[i]) <= j + 1:  # Within 1 hour
                number_close[j] += 1

    # store train predictions
    df["TimeTaken_%s" % alg] = y_train_pred

    train_rmse = (np.sqrt(mean_squared_error(y_train, y_train_pred)))
    train_r_sq = (r2_score(y_train, y_train_pred))
    train_mae = (mean_absolute_error(y_train, y_train_pred))
    train_evs = (explained_variance_score(y_train, y_train_pred))
    train_median_ae = (median_absolute_error(y_train, y_train_pred))

    for j in number_close:
        percent_close.append(j / len(y_train) * 100)

    ########################################################################################################################
    # Output results
    ########################################################################################################################
    out_file.write("\n%s - df Train Results\n" % alg)
    out_file.write("\tTrain R2: {0:.4f}\n".format(train_r_sq))
    out_file.write("\tTrain RMSE: {0:.2f}\n".format(train_rmse))
    out_file.write("\tTrain MeanAE: {0:.2f}\n".format(train_mae))
    out_file.write("\tTrain MedianAE: {0:.2f} \n".format(train_median_ae))
    out_file.write("\tTrain EVS: {0:.2f} \n".format(train_evs))

    print("\n%s - df Train Results" % alg)
    print("\tTrain R2: {0:.5f} ".format(train_r_sq))
    print("\tTrain RMSE: {0:.2f}".format(train_rmse))
    print("\tTrain MeanAE: {0:.2f} ".format(train_mae))
    print("\tTrain EVS: {0:.2f}".format(train_evs))
    print("\tTrain MedianAE: {0:.2f} \n".format(train_median_ae))

    interesting_hours = [1, 4, 8, 16, 24, 48, 72, 96]
    for hour in interesting_hours:
        hour -= 1
        # out_file.write(
        # "\n\t{1:s} % train predictions error within {3:d} hour(s) -> Mean: {0:.2f}% of {2:d}/10".format(
        # percent_close[hour], alg, len(y), hour + 1))
        out_file.write("\n\tPredictions correct within %s hour(s): %.2f%%" % (hour + 1, percent_close[hour]))

        print("\t{1:s} % train predictions error within {3:d} hour(s) -> Mean: {0:.2f}% of {2:d}/10".format(
            percent_close[hour], alg, len(y_train), hour + 1))

    ####################################################################################################################
    # Plotting
    ####################################################################################################################
    if d["plotting"] == "y":
        print("\n..plotting..\n")
        plot_errors_main(df, alg, "Train", algpath, alg_initials)
        plot(df["TimeTaken"], df["TimeTaken_%s" % alg], alg, "_Train", algpath, alg_initials,
             d["input_file"])
        plot_percent_correct(percent_close, algpath, alg_initials, d["input_file"])

    ####################################################################################################################
    # Importances
    ####################################################################################################################
    print("\n..Calculating importances..\n")
    if alg == "RandomForestRegressor" or alg == "GradientBoostingRegressor" or alg == "xgboost":
        tree_importances(regr, X_train, algpath, d, out_file, alg_initials)

    elif alg == "LinearRegression" or alg == "ElasticNet":
        regression_coef_importances(regr, X_train, algpath, d, out_file, alg_initials)

############################################################################################################
    ############################################################################################################
        ############################################################################################################

    # train_rmse = (np.sqrt(mean_squared_error(y_train, y_train_pred)))
    # train_r_sq = (r2_score(y_train, y_train_pred))
    # train_mae = (mean_absolute_error(y_train, y_train_pred))
    # train_evs = (explained_variance_score(y_train, y_train_pred))
    # train_median_ae = (median_absolute_error(y_train, y_train_pred))
    #
    # ########################################################################################################################
    # # output results
    # ########################################################################################################################
    # ####################################################################################################################
    # # Plotting
    # ####################################################################################################################
    # if d["plotting"] == "y":
    #     print("\n..plotting..")
    #
    #     # print("..plotting errors against time..") # plot errors against time
    #     # plot_errors_main(df, alg, "Test", algpath, alg_initials)
    #     # todo - need to first store the predicted values
    #
    #     print("..plotting train actual versus predicted..")# plot the training error for the last fold ONLY
    #     plot(y_train, y_train_pred, alg, "_Train", algpath, alg_initials, d["input_file"])
    #
    #     # print("..plotting % correct against time..")
    #     # # plot % correct against time
    #     # plot_percent_correct(average_close, algpath, alg_initials, d["input_file"], std_close)
    # ####################################################################################################################
    # # Importances
    # ####################################################################################################################
    # print("\n..Calculating importances..\n")
    # if alg == "RandomForestRegressor" or alg == "GradientBoostingRegressor" or alg == "xgboost":
    #     tree_importances(regr, X_train, algpath, d, out_file, alg_initials)
    #
    # elif alg == "LinearRegression" or alg == "ElasticNet":
    #     regression_coef_importances(regr, X_train, algpath, d, out_file, alg_initials)

    out_file.close()

    ############################################################################################################
    ############################################################################################################
    ############################################################################################################

    ####################################################################################################################
    # Extra Testing
    ####################################################################################################################
    if d["extra_testing"] == "y":
        print("\n..extra testing..\n")
        if d["prejuly_july"] == "y":
            extra_testing_path = "/July/"
            algpath = newpath + alg_initials + extra_testing_path
        elif d["prejune_june"] == "y":
            extra_testing_path = "/June/"
            algpath = newpath + alg_initials + extra_testing_path
        elif d["prejune_junejuly"] == "y":
            extra_testing_path = "/JuneJuly/"
            algpath = newpath + alg_initials + extra_testing_path
        else:
            print("..haven't specified which case study to use..")
            print("..closing program..")
            exit()
        if not os.path.exists(algpath):
            os.makedirs(algpath)  # Make folder for storing results if it does not exist

        out_file_name = algpath + alg + ".txt"  # Log file name

        out_file = open(out_file_name, "w")  # Open log file
        out_file.write(alg + " " + time.strftime("%Y%m%d-%H%M%S") + "\n")
        out_file.write("\nInput file name: %s\n" % d["input_file"])

        print("df_test Shape:", df_test.shape, "\n")
        y = df_test["TimeTaken"]
        X = df_test.drop("TimeTaken", axis=1)
        keepers = get_keepers()
        for col in X.columns:
            if col not in keepers:
                del X[col]

        y_pred = regr.predict(X)

        ####################################################################################################################
        # Extra Testing Simple Statistics
        ####################################################################################################################
        if alg_counter == 1:
            simplepath = newpath + extra_testing_path + "Simple_Stat_Plots/"
            if not os.path.exists(simplepath):
                os.makedirs(simplepath)  # Make folder for storing results if it does not exist

            simple_out_file_name = simplepath + "extra_testing_simple_stats.txt"  # Log file name

            simple_out_file = open(simple_out_file_name, "w")  # Open log file
            simple_out_file.write("Simple Stats - " + time.strftime("%Y%m%d-%H%M%S") + "\n")
            simple_out_file.write("\nInput file name: %s\n" % d["input_file"])

            print("df_test Shape:", df_test.shape)
            simple_out_file.write("\ndf_test Shape: " + str(df_test.shape) + "\n")

            mean_time = np.mean(y)  # Calculate mean of predictions
            std_time = np.std(y)  # Calculate standard deviation of predictions
            median_time = np.median(y)  # Calculate standard deviation of predictions

            simple_out_file.write("\n\nSimple df_test TimeTaken stats")
            simple_out_file.write("\n\tmean_time = %s" % mean_time)
            simple_out_file.write("\n\tstd_time = %s" % std_time)
            simple_out_file.write("\n\tmedian_time = %s\n" % median_time)


            print("\nSimple df_test TimeTaken stats")
            print("\tmean_time = %s" % mean_time)
            print("\tstd_time = %s" % std_time)
            print("\tmedian_time = %s\n" % median_time)

            
            simple_out_file.write("\n\n\tThe mean time taken to resolve cases was %.2f hours, median was %.2f hours "
                                  "and the standard deviation was %.2f hours." % (mean_time, median_time, std_time))
            
            df_test["Mean_TimeTaken"] = mean_time
            df_test["Median_TimeTaken"] = median_time


            simple_percent_close = []
            simple_number_close = [0 for _ in range(96)]

            for i in range(len(df_test["Mean_TimeTaken"])):
                for j in range(len(simple_number_close)):
                    if abs(df_test["Mean_TimeTaken"].iloc[i] - y.iloc[i]) <= j + 1:  # Within 1 hour
                        simple_number_close[j] += 1

            for j in simple_number_close:
                simple_percent_close.append(j / len(y) * 100)

            interesting_hours = [1, 4, 8, 16, 24, 48, 72, 96]
            for hour in interesting_hours:
                hour -= 1
                print("\t{1:s} % test predictions error within {3:d} hour(s) -> Mean: {0:.2f}% of {2:d}/10".format(
                    simple_percent_close[hour], "Mean", len(y), hour + 1))
                # simple_out_file.write("\n\t{1:s} % test predictions error within {3:d} hour(s) -> Mean: {0:.2f}% of {"
                                      # "2:d}/10".format(simple_percent_close[hour], "Mean", len(y), hour + 1))
                simple_out_file.write("\n\tPredictions correct within %s hour(s): %.2f%%" % (hour+1,
                                                                                              simple_percent_close[hour]))


            mean_time_test_r2 = r2_score(y, df_test["Mean_TimeTaken"])
            mean_time_test_rmse = np.sqrt(mean_squared_error(y, df_test["Mean_TimeTaken"]))
            mean_time_test_meanae = mean_absolute_error(y, df_test["Mean_TimeTaken"])
            mean_time_test_evs = explained_variance_score(y, df_test["Mean_TimeTaken"])
            mean_time_test_medianae = median_absolute_error(y, df_test["Mean_TimeTaken"])

            median_time_test_r2 = r2_score(y, df_test["Median_TimeTaken"])
            median_time_test_rmse = np.sqrt(mean_squared_error(y, df_test["Median_TimeTaken"]))
            median_time_test_meanae = mean_absolute_error(y, df_test["Median_TimeTaken"])
            median_time_test_evs = explained_variance_score(y, df_test["Median_TimeTaken"])
            median_time_test_medianae = median_absolute_error(y, df_test["Median_TimeTaken"])

            simple_out_file.write("\n\nMean:")
            simple_out_file.write("\n\tTest R2 = %s" % mean_time_test_r2)
            simple_out_file.write("\n\tTest RMSE = %s" % mean_time_test_rmse)
            simple_out_file.write("\n\tTest MeanAE = %s" % mean_time_test_meanae)
            simple_out_file.write("\n\tTest MedianAE = %s\n" % mean_time_test_medianae)
            simple_out_file.write("\n\tTest EVS = %s" % mean_time_test_evs)

            simple_out_file.write("\n\tmedian_time_test_r2 = %s" % median_time_test_r2)
            simple_out_file.write("\n\tmedian_time_test_rmse = %s" % median_time_test_rmse)
            simple_out_file.write("\n\tmedian_time_test_meanae = %s" % median_time_test_meanae)
            simple_out_file.write("\n\tmedian_time_test_evs = %s" % median_time_test_evs)
            simple_out_file.write("\n\tmedian_time_test_medianae = %s\n" % median_time_test_medianae)

            print("\n\tmean_time_test_r2 = %s" % mean_time_test_r2)
            print("\tmean_time_test_rmse = %s" % mean_time_test_rmse)
            print("\tmean_time_test_meanae = %s " % mean_time_test_meanae)
            print("\tmean_time_test_evs = %s" % mean_time_test_evs)
            print("\tmean_time_test_medianae = %s\n " % mean_time_test_medianae)

            print("\tmedian_time_test_r2 = %s" % median_time_test_r2)
            print("\tmedian_time_test_rmse = %s" % median_time_test_rmse)
            print("\tmedian_time_test_meanae = %s" % median_time_test_meanae)
            print("\tmedian_time_test_evs = %s" % median_time_test_evs)
            print("\tmedian_time_test_medianae = %s" % median_time_test_medianae)


            plot(df_test["TimeTaken"], df_test["Mean_TimeTaken"], "Simple", "All", simplepath, "Mean", d["input_file"])
            plot(df_test["TimeTaken"], df_test["Median_TimeTaken"], "Simple", "All", simplepath, "Median",
                     d["input_file"])

        ################################################################################################################
        # Extra Testing Machine Learning
        ################################################################################################################
        percent_close = []
        number_close = [0 for _ in range(96)]

        for i in range(len(y_pred)): # Convert high or low predictions to 0 or 3 std
            if y_pred[i] < 0:  # Convert all negative predictions to 0
                y_pred[i] = 0
            if math.isnan(y_pred[i]):  # If NaN set to 0
                y_pred[i] = 0

            for j in range(len(number_close)):
                if abs(y_pred[i] - y.iloc[i]) <= j+1:  # Within 1 hour
                    number_close[j] += 1

        #  append the predictions for this fold to df_test
        df_test["TimeTaken_%s" % alg] = y_pred

        test_rmse = (np.sqrt(mean_squared_error(y, y_pred)))
        test_r_sq = (r2_score(y, y_pred))
        test_mae = (mean_absolute_error(y, y_pred))
        test_evs = (explained_variance_score(y, y_pred))
        test_median_ae = (median_absolute_error(y, y_pred))

        for j in number_close:
            percent_close.append(j / len(y) * 100)


        ########################################################################################################################
        # Extra Testing  output results
        ########################################################################################################################
        out_file.write("\n%s - df Test Results\n" % alg)
        out_file.write("\tTest R2: {0:.4f}\n".format(test_r_sq))
        out_file.write("\tTest RMSE: {0:.2f}\n".format(test_rmse))
        out_file.write("\tTest MeanAE: {0:.2f}\n".format(test_mae))
        out_file.write("\tTest MedianAE: {0:.2f} \n".format(test_median_ae))
        out_file.write("\tTest EVS: {0:.2f} \n".format(test_evs))

        print("\n%s - df Test Results" % alg)
        print("\tTest R2: {0:.5f} ".format(test_r_sq))
        print("\tTest RMSE: {0:.2f}".format(test_rmse))
        print("\tTest MeanAE: {0:.2f} ".format(test_mae))
        print("\tTest EVS: {0:.2f}".format(test_evs))
        print("\tTest MedianAE: {0:.2f} \n".format(test_median_ae))

        interesting_hours = [1, 4, 8, 16, 24, 48, 72, 96]
        for hour in interesting_hours:
            hour -= 1
            # out_file.write(
                # "\n\t{1:s} % test predictions error within {3:d} hour(s) -> Mean: {0:.2f}% of {2:d}/10".format(
                    # percent_close[hour], alg, len(y), hour + 1))
            out_file.write("\n\tPredictions correct within %s hour(s): %.2f%%" % (hour+1, percent_close[hour]))
            
            print("\t{1:s} % test predictions error within {3:d} hour(s) -> Mean: {0:.2f}% of {2:d}/10".format(
                percent_close[hour], alg, len(y), hour + 1))
                
             

        ####################################################################################################################
        # Extra Testing Plotting
        ####################################################################################################################
        if d["plotting"] == "y":
            print("\n..plotting..\n")
            plot_errors_main(df_test, alg, "Test", algpath, alg_initials)
            plot(df_test["TimeTaken"], df_test["TimeTaken_%s" % alg], alg, "_Test", algpath, alg_initials,
                 d["input_file"])
            plot_percent_correct(percent_close, algpath, alg_initials,d["input_file"])

        ####################################################################################################################
        # Extra Testing Importances
        ####################################################################################################################
        print("\n..Calculating importances..\n")
        if alg == "RandomForestRegressor" or alg == "GradientBoostingRegressor" or alg == "xgboost":
            tree_importances(regr, X, algpath, d, out_file, alg_initials)

        elif alg == "LinearRegression" or alg == "ElasticNet":
            regression_coef_importances(regr, X, algpath, d, out_file, alg_initials)

        # Extra Testing Export Results

        out_file.close()
        if d["output_predictions_csv"] == "y":
            if d["specify_subfolder"] == "n":
                if d["prejuly_july"] == "y":
                    df_test.to_csv(d["file_location"] + d["input_file"] + "_%s_July_predictions.csv" % alg_initials,
                                   index=False)
                elif d["prejune_june"] == "y":
                    df_test.to_csv(d["file_location"] + d["input_file"] + "_%s_June_predictions.csv" % alg_initials,
                                   index=False)
                elif d["prejune_junejuly"] == "y":
                    df_test.to_csv(d["file_location"] + d["input_file"] + "_%s_JuneJuly_predictions.csv" % alg_initials,
                                   index=False)
                elif d["train_test_split"] == "y":
                    df_test.to_csv(d["file_location"] + d["input_file"] + "_%s_TrainTestSplit_predictions.csv" % alg_initials,
                                   index=False)
            else:
                df_test.to_csv(d["file_location"] + d["input_file"] + "_%s_%s_predictions.csv" % (d["specify_subfolder"],
                                                                                                alg_initials), index=False)


    print("\n..finished with alg: %s..\n" % alg)


if __name__ == "__main__":  # Run program
    parameters = "../../../Data/parameters.txt"  # Parameters file
    sample_parameters = "../Sample Parameter File/parameters.txt"

    print("Modeling dataset", time.strftime("%Y.%m.%d"), time.strftime("%H.%M.%S"))
    d = {}
    with open(parameters, "r") as f:
        for line in f:
            line = line.replace(":", "")
            (key, val) = line.split()
            d[key] = val

    sample_d = {}
    with open(sample_parameters, "r") as f:
        for line in f:
            line = line.replace(":", "")
            (key, val) = line.split()
            sample_d[key] = val

    for key in sample_d.keys():
        if key not in d.keys():
            print("\"%s\" not in parameters.txt - please update parameters file with this key" % key)
            print("Default key and value => \"%s: %s\"" % (key, sample_d[key]))
            exit()
    for key in d.keys():
        if key not in sample_d.keys():
            print("\"%s\" not in sample parameters" % key)

    # if resample is selected then all results are put in a resample folder
    if d["resample"] == "y":
        newpath = r"../0. Results/" + d["user"] + "/model/" + d["input_file"] + "/resample/"
    # if no subfolder is specified then the reults are put into a folder called after the input file
    elif d["specify_subfolder"] == "n":
        newpath = r"../0. Results/" + d["user"] + "/model/" + d["input_file"] +"/"
    # if a subfolder is specified then all results are put there
    else:
        newpath = r"../0. Results/" + d["user"] + "/model/" + d["input_file"]+ "/" + d["specify_subfolder"]+"/"
    if not os.path.exists(newpath):
        os.makedirs(newpath)  # Make folder for storing results if it does not exist

    copyfile(parameters, newpath + "parameters.txt")  # Save parameters

    np.random.seed(int(d["seed"]))  # Set seed

    df = pd.read_csv(d["file_location"] + d["input_file"] + ".csv", encoding='latin-1', low_memory=False)
    df["TimeTaken"] = df["TimeTaken"].apply(lambda x: x/3600)

    print("\nDF Shape:", df.shape, "\n")

    if d["extra_testing"] == "y":
        if d["delete_option_cols"] == "y":
            option_cols = ["Concurrent_open_cases", "Cases_created_within_past_8_hours",
                           "Cases_resolved_within_past_8_hours", "Seconds_left_Day", "Seconds_left_Month",
                           "Seconds_left_Qtr", "Seconds_left_Year", "Created_on_Weekend", "Rolling_Mean", "Rolling_Median",
                           "Rolling_Std"]
            for col in option_cols:
                if d[col] == "n" and col in df.columns:
                    print("deleting col %s" % col)
                    del df[col]
        print("DF Shape:", df.shape, "\n")

    if d["resample"] == "y":
        from sklearn.utils import resample
        print("..resampling\n")
        df = resample(df, n_samples=int(d["n_samples"]), random_state=int(d["seed"]))
        df = df.reset_index(drop=True)
        print("DF Shape:", df.shape, "\n")

    if d["extra_testing"] == "y":
        df["Created_On"] = pd.to_datetime(df["Created_On"])
        if d["prejuly_july"] == "y":
            print("prejuly_july")
            df_train = df[df["Created_On"] < pd.datetime(2017, 7, 1, 8)].copy()
            df_test = df[(df["Created_On"] >= pd.datetime(2017, 7, 1, 8)) & (df["Created_On"] < pd.datetime(2017, 8, 1, 8))].copy()
        elif d["prejune_june"] == "y":
            print("prejune_june")
            df_train = df[df["Created_On"] < pd.datetime(2017, 6, 1, 8)].copy()
            df_test = df[(df["Created_On"] >= pd.datetime(2017, 6, 1, 8)) &
                        (df["Created_On"] < pd.datetime(2017,7, 1,8))].copy()
        elif d["prejune_junejuly"] == "y":
            print("prejune_junejuly")
            df_train = df[df["Created_On"] < pd.datetime(2017, 6, 1, 8)].copy()
            df_test = df[(df["Created_On"] >= pd.datetime(2017, 6, 1, 8))].copy()

        if d["prejuly_july"] == "y" or d["prejune_june"] == "y" or d["prejune_junejuly"] == "y":
            print("DF Train Shape:", df_train.shape)
            print("DF Test Shape:", df_test.shape, "\n")
        if d["resample"] == "y":
            df_train = df_train.reset_index(drop=True)
            df_test = df_test.reset_index(drop=True)
    else:
        print("DF Shape:", df.shape, "\n")

    print("Input file name: %s" % d["input_file"])

    if d["histogram"] == "y":
        histogram(df, "TimeTaken", newpath)  # Save histogram plots of TimeTaken

    if d["log_of_y"] == "y":  # Take log of y values
        print("Y has been transformed by log . . . change parameter file to remove this feature\n")
        df["TimeTaken"] = df["TimeTaken"].apply(lambda x: math.log(x))
    # todo - transform predicted values back into seconds


    ####################################################################################################################
    # Modelling
    ###################################################################################################################
    alg_counter = 0  # used so the simple stats, etc. aren't printed for each algorithm used

    if d["LinearRegression"] == "y":
        alg_counter+=1
        regressor = LinearRegression()
        if d["prejuly_july"] == "y" or d["prejune_june"] == "y" or d["prejune_junejuly"] == "y":
            results(df_train, "LinearRegression", regressor, newpath, d, alg_counter, "LR", df_test)
        else:
            results(df, "LinearRegression", regressor, newpath, d, alg_counter, "LR")

    if d["ElasticNet"] == "y":
        alg_counter+=1
        regressor = ElasticNet(alpha=100, l1_ratio=1, max_iter=100000)
        if d["prejuly_july"] == "y" or d["prejune_june"] == "y" or d["prejune_junejuly"] == "y":
            results(df_train, "ElasticNet", regressor, newpath, d, alg_counter, "EN", df_test)
        else:
            results(df, "ElasticNet", regressor, newpath, d, alg_counter, "EN")

    if d["KernelRidge"] == "y":
        alg_counter+=1
        regressor = KernelRidge(alpha=0.1)
        results(df, "KernelRidge", regressor, newpath, d, alg_counter, "KR")

    if d["MLPRegressor"] == "y":
        alg_counter+=1
        regressor = MLPRegressor(hidden_layer_sizes=(50,25,10,5,3), random_state=int(d["seed"]),
                                 max_iter=2000) # early_stopping=True)
        results(df, "MLPRegressor", regressor, newpath, d, alg_counter, "MLP")

    if d["GradientBoostingRegressor"] == "y":
        alg_counter+=1
        regressor = GradientBoostingRegressor(random_state=int(d["seed"]))
        if d["prejuly_july"] == "y" or d["prejune_june"] == "y" or d["prejune_junejuly"] == "y":
            results(df_train, "GradientBoostingRegressor", regressor, newpath, d, alg_counter, "GBR", df_test)
        else:
            results(df, "GradientBoostingRegressor", regressor, newpath, d, alg_counter, "GBR")

    if d["xgboost"] == "y":
        alg_counter+=1
        import xgboost as xgb
        params = {
            'max_depth': 5,
            'n_estimators': 50,
            'objective': 'reg:linear'}
        regressor = xgb.XGBRegressor(**params)
        results(df, "xgboost", regressor, newpath, d, alg_counter, "XGB")

    if d["RandomForestRegressor"] == "y":
        alg_counter+=1
        regressor = RandomForestRegressor(n_estimators=int(d["n_estimators"]), random_state=int(d["seed"]),
                                          max_depth=25, n_jobs=-1)
        if d["prejuly_july"] == "y" or d["prejune_june"] == "y" or d["prejune_junejuly"] == "y":
            results(df_train, "RandomForestRegressor", regressor, newpath, d, alg_counter, "RFR", df_test)
        else:
            results(df, "RandomForestRegressor", regressor, newpath, d, alg_counter, "RFR")

    if d["output_predictions_csv"] == "y":
        if d["extra_testing"] == "y":
            if d["prejuly_july"] == "y":
                df.to_csv(d["file_location"] + d["input_file"] + "_prejuly_predictions.csv", index=False)  # export file
            elif d["prejune_june"] == "y":
                df.to_csv(d["file_location"] + d["input_file"] + "_prejune_predictions.csv", index=False)  # export file
            elif d["prejune_junejuly"] == "y":
                df.to_csv(d["file_location"] + d["input_file"] + "_prejune_predictions.csv", index=False)  # export file
        elif d["specify_subfolder"] != "n":
            df.to_csv(d["file_location"] + d["input_file"] + "_" + d["specify_subfolder"] + "_predictions.csv", \
                                                           index=False)  # export file
        else:
            df.to_csv(d["file_location"] + d["input_file"] + "_predictions.csv", index=False)  # export file

    if d["statsmodels_OLS"] == "y":
        import statsmodels.api as sm
        np.set_printoptions(threshold=np.inf)

        alg_initials = "OLS"
        alg = "Statsmodels_OLS"
        algpath = newpath + alg_initials + "/"
        if not os.path.exists(algpath):
            os.makedirs(algpath)  # Make folder for storing results if it does not exist

        out_file_name = algpath + alg + ".txt"  # Log file name

        out_file = open(out_file_name, "w")  # Open log file
        out_file.write(alg + " " + time.strftime("%Y%m%d-%H%M%S") + "\n")
        out_file.write("\nInput file name: %s\n" % d["input_file"])

        if d["prejuly_july"] == "y" or d["prejune_june"] == "y" or d["prejune_junejuly"] == "y":
            X = df_train.drop("TimeTaken", axis=1)
            X_test = df_test.drop("TimeTaken", axis=1)
        else:
            X = df.drop("TimeTaken", axis=1)

        keepers = get_keepers()
        for col in X.columns:
            if col not in keepers:
                del X[col]

                if d["prejuly_july"] == "y" or d["prejune_june"] == "y" or d["prejune_junejuly"] == "y":
                    del X_test[col]

        # output the features being used
        print("\nLength of df = %s" % len(df.columns)) # todo add these 3 print messages to outfile or delete them
        print("Length of keepers = %s" % len(keepers))
        print("Length of features used = %s\n" % len(X.columns))

        out_file.write("\nFeatures used:")
        for i, col in enumerate(X.columns):
            out_file.write("\n\t%s - %s" % (i+1, col))

        if d["prejuly_july"] == "y" or d["prejune_june"] == "y" or d["prejune_junejuly"] == "y":
            y_test = df_test["TimeTaken"]
            y = df_train["TimeTaken"]
        else:
            y = df["TimeTaken"]

        if d["prejuly_july"] == "y" or d["prejune_june"] == "y" or d["prejune_junejuly"] == "y":
            X_train, X_test, y_train, y_test = X, X_test, y, y_test
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=int(d["seed"]))

        model = sm.OLS(y_train, X_train)
        olsregr = model.fit()

        print("olsregr.summary():\n", olsregr.summary())
        out_file.write("\n\n olsregr.summary():\n"+ str(olsregr.summary()) + "\n")


        y_train_pred = olsregr.predict(X_train)
        y_test_pred = olsregr.predict(X_test)

        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

        print("\ntrain R2: %s" % train_r2)
        print("test R2: %s" % test_r2)
        print("train RMSE: %s" % train_rmse)
        print("test RMSE: %s" % test_rmse)

        out_file.write("\n\ntrain R2: %s" % train_r2)
        out_file.write("\ntest R2: %s" % test_r2)
        out_file.write("\ntrain RMSE: %s" % train_rmse)
        out_file.write("\ntest RMSE: %s" % test_rmse)

        out_file.write('\ny_train Predicted values: ' + str(y_train_pred))
        out_file.write('\n\ny_test Predicted values: ' + str(y_test_pred))

        plot(y_train, y_train_pred, alg, "Train Data", algpath, alg_initials, d["input_file"])
        plot(y_test, y_test_pred, alg, "Test Data", algpath, alg_initials, d["input_file"])
        out_file.close()

    if d["beep"] == "y":
        import winsound
        Freq = 400 # Set Frequency To 2500 Hertz
        Dur = 1000 # Set Duration To 1000 ms == 1 second
        winsound.Beep(Freq,Dur)
        Freq = 300 # Set Frequency To 2500 Hertz
        winsound.Beep(Freq,Dur)