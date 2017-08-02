"""****************************************************************************
UCD MSc Business Analytics Capstone Project - Predicting Transactions Times
*******************************************************************************
Version 00
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
from sklearn.model_selection import KFold, cross_val_predict#, cross_val_score, train_test_split
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.kernel_ridge import KernelRidge
from math import sqrt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from shutil import copyfile  # Used to copy parameters file to directory
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score, \
    median_absolute_error
import seaborn as sns
from calendar import monthrange
import datetime
from sklearn.preprocessing import StandardScaler

parameters = "../../../Data/parameters.txt"  # Parameters file


def tree_importances(regr, X, algpath, d, out_file, alg_initials):
    importances = regr.feature_importances_

    dfimportances = pd.DataFrame(data=X.columns, columns=["Columns"])
    dfimportances["Importances"] = importances

    dfimportances = dfimportances.sort_values("Importances", ascending=False)
    if d["export_importances_csv"] == "y":
        dfimportances.to_csv(algpath + "importances.csv", index=False)

    print("Feature Importances:")
    out_file.write("\nFeature Importances:\n")
    for i, (col, importance) in enumerate(zip(dfimportances["Columns"].values.tolist(), dfimportances[
        "Importances"].values.tolist())):
        out_file.write("\t%d. \"%s\" (%f)\n" % (i + 1, col, importance))
        print("\t%d. \"%s\" (%f)" % (i + 1, col, importance))

    if d["export_top_k_csv"] == "y":
        df_top = return_new_top_k_df(df, dfimportances, int(d["top_k_features"]))
        df_top.to_csv("../../../Data/%s_%s_top_%s.csv" % (d["input_file"], alg_initials, d["top_k_features"]),
                      index=False)


def regression_coef_importances(regr, X, algpath, d, out_file, alg_initials):
    scalerX = StandardScaler().fit(X)
    # scalery = StandardScaler().fit(y.values.reshape(-1,1)) # Have to reshape to avoid warnings
    normed_coefs = scalerX.transform(regr.coef_.reshape(1, -1))
    normed_coefs_list = normed_coefs.tolist()[0]
    dfimportances = pd.DataFrame()
    dfimportances["Columns"] = X.columns.tolist()
    dfimportances["Importances"] = normed_coefs_list
    dfimportances["Absolute_Importances"] = abs(dfimportances["Importances"])
    dfimportances = dfimportances.sort_values("Absolute_Importances", ascending=False)
    if d["export_importances_csv"] == "y":
        dfimportances.to_csv(algpath + "importances.csv", index=False)
    # print(dfimportances)

    print("Feature Importances:")
    out_file.write("\nFeature Importances:\n")
    for i, (col, importance) in enumerate(zip(dfimportances["Columns"].values.tolist(), dfimportances[
        "Importances"].values.tolist())):
        out_file.write("\t%d. \"%s\" (%f)\n" % (i + 1, col, importance))
        print("\t%d. \"%s\" (%f)" % (i + 1, col, importance))

    if d["export_top_k_csv"] == "y":
        df_top = return_new_top_k_df(df, dfimportances, int(d["top_k_features"]))
        df_top.to_csv("%s%s_%s_top_%s.csv" % (d["file_location"], d["input_file"], alg_initials, d["top_k_features"]),
                      index=False)


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


def plot(x, y, alg, data, newpath, alg_initials):
    sns.reset_orig() # plt.rcParams.update(plt.rcParamsDefault)
    plt.figure()
    plt.plot(x, y, 'ro', alpha=0.1, markersize=4)
    # sns.plot(x, y, 'ro', alpha=0.1, plot_kws={"s": 3}) #scatter_kws={"s": 100}
    # sns.lmplot(x, y, data = in_data, scatter_kws={"s": 4, 'alpha':0.3, 'color': 'red'}, line_kws={"linewidth": 1,'color': 'blue'}, fit_reg=False)
    plt.xlabel(data + " Data Actual")
    plt.ylabel(data + " Data Prediction")
    if alg == "Simple":
        plt.title(alg_initials + " - " + data + " Data")
    else:
        plt.title(alg + " - " + data + " Data")
    plt.axis('equal')
    plt.ylim(0, 2500000)
    plt.xlim(0, 2500000)
    plt.tight_layout()  # Force everything to fit on figure
    if not os.path.exists(newpath + "PDFs/"):
        os.makedirs(newpath + "PDFs/")  # Make folder for storing results if it does not exist
    plt.savefig(newpath + alg_initials + "_" + data + ".png")
    plt.savefig(newpath + "PDFs/" + alg_initials + "_" + data + ".pdf")
    plt.close()


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
    return df


def get_errors(df, alg, time_range, col):
    r2_scores = []
    rmse_scores = []
    mae_scores = []
    for i in range(time_range):
        actual = df.loc[df[col] == i, "TimeTaken_hours"]
        predicted = df.loc[df[col] == i, "TimeTaken_hours_%s"%alg]

        if len(df[col][df[col] == i]) == 0: 
            r2_scores.append(-1)
            rmse_scores.append(-1)
            mae_scores.append(-1)
        else:
            if r2_score(actual, predicted) < 0:
                r2_scores.append(0)
            else:    
                r2_scores.append(r2_score(actual, predicted))
            rmse_scores.append(np.sqrt(mean_squared_error(actual, predicted)))
            mae_scores.append(mean_absolute_error(actual, predicted))
    return [r2_scores, rmse_scores, mae_scores]


def plot_errors(x_ticks, y, error_name, alg, y_label, x_label, data, alg_initials, newpath):
    if x_label == "Day of Qtr Created" or x_label == "Day of Qtr Resolved":
        y = np.array(y)
        z = np.where(np.array(y)>=0)
        z = z[0]
        y_z = y[z]

        x_num = [i for i in range(len(y))]
        x_num = np.array(x_num)
        # x_num_z = x_num[z]

        y_np = np.array(y)
        # rot = .1
        # start = -2.5  # purple
        

        reverse = False
        if error_name == "R2":
            reverse = True
        # pal = sns.cubehelix_palette(len(y), start=start, rot=rot,dark=.1, light=.9, reverse=reverse)
        pal = sns.cubehelix_palette(len(y), start=1, rot=0,hue=1.5, gamma=1,dark=.3, light=0.9, reverse=reverse)
        
        rank = y_np.argsort().argsort() 
        sns.barplot(x_num, y, palette=np.array(pal[::-1])[rank])
        plt.xticks(x_ticks, x_ticks)
        plt.title("%s - %s to %s"% (alg, error_name, x_label))
        plt.ylabel(y_label)
        plt.xlabel(x_label)

        min_ylim = min(y_z)-np.std(y_z)/3
        if min_ylim < 0:
            min_ylim = 0
        plt.ylim(min_ylim, max(y_z)+np.std(y_z)/3)       
        
        if not os.path.exists(newpath + "PDFs/"):
            os.makedirs(newpath + "PDFs/")  # Make folder for storing results if it does not exist

        plt.savefig(newpath + error_name +"_"+ x_label  + ".png")
        plt.savefig(newpath + "PDFs/" + error_name +"_"+ x_label  + ".pdf")

    else:
        plt.figure()
        x_num = [i for i in range(len(x_ticks))]
        
        y_np = np.array(y)
            #     pal = sns.color_palette("BuGn", len(y)) # OrRd_r, GnBu_d, husl, Spectral, cubehelix, RdYlGn_r, BuGn, RdYlBu_r ;
            # pal = sns.cubehelix_palette(len(y)); pal = sns.color_palette(palette="Reds", n_colors=len(y), desat=.9)
            #     rot = .3, start = -1  # green blue ; rot = .3, start = 1.5  # green ; rot = .3, start = 2  # blue green
            #     rot = .3, start = -2.5  # red
        # rot = .1
        # start = -2.5  # purple
        reverse = False
        if error_name == "R2":
            reverse = True
        # pal = sns.cubehelix_palette(len(y), start=start, rot=rot,dark=.4, light=.7, reverse=reverse)
        pal = sns.cubehelix_palette(len(y), start=1, rot=0,hue=1.5, gamma=1,dark=.3, light=0.9, reverse=reverse)
        rank = y_np.argsort().argsort()
        sns.barplot(x_num, y, palette=np.array(pal[::-1])[rank])
        plt.xticks(x_num, x_ticks, rotation="vertical")
        plt.title("%s - %s to %s"% (alg, error_name, x_label))
        plt.ylabel(y_label)
        plt.xlabel(x_label)

        # min_ylim = min(y_z)-np.std(y_z)/3
        # if min_ylim < 0:
        #     min_ylim = 0
        # plt.ylim(min_ylim, max(y_z)+np.std(y_z)/3)

        plt.ylim(min(y)-np.std(y)/3, max(y)+np.std(y)/3)    
        if not os.path.exists(newpath + "PDFs/"):
            os.makedirs(newpath + "PDFs/")  # Make folder for storing results if it does not exist
        plt.savefig(newpath + error_name +"_"+ x_label  + ".png")
        plt.savefig(newpath + "PDFs/" + error_name +"_"+ x_label  + ".pdf")

    plt.close()


def plot_errors_main(df, alg, data, newpath, alg_initials):
    # import seaborn as sns
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
    x_vals = [x for x in range(31)]
    x_label = "Day of Month Created"
    for score, error_name, y_label in zip(scores, error_names, y_labels):
        plot_errors(x_vals, score, error_name, alg, y_label, x_label, data, alg_initials, newpath)

    scores = get_errors(df, alg, 31, "ResolvedDate_Month")
    x_vals = [x for x in range(31)]
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


def results(df, alg, in_regressor, newpath, d, alg_counter, alg_initials):
    algpath = newpath + alg_initials + "/"
    if not os.path.exists(algpath):
        os.makedirs(algpath)  # Make folder for storing results if it does not exist

    out_file_name = algpath + alg + ".txt"  # Log file name

    out_file = open(out_file_name, "w")  # Open log file
    out_file.write(alg + " " + time.strftime("%Y%m%d-%H%M%S") + "\n")
    out_file.write("\nInput file name %s:\n" % d["input_file"])

    X = df.drop("TimeTaken", axis=1)

    keepers = get_keepers()
    for col in X.columns:
        if col not in keepers:
            del X[col]

    # output the features being used
    if alg_counter == 1:
        print("Length of df = %s" % len(df.columns)) # todo add these 3 print messages to outfile or delete them
        print("Length of keepers = %s" % len(keepers))
        print("Length of features used = %s\n" % len(X.columns))
        print("Features used:")
    # out_file.write("\nFeatures used:")
    for i, col in enumerate(X.columns):
        if alg_counter == 1:
            print("\t%s - %s" % (i+1, col))
        # out_file.write("\n\t%s - %s" % (i+1, col))
    
    y = df["TimeTaken"]

    ####################################################################################################################
    # Simple Statistics
    ####################################################################################################################
    mean_time = np.mean(y)  # Calculate mean of predictions
    std_time = np.std(y)  # Calculate standard deviation of predictions
    median_time = np.median(y)  # Calculate standard deviation of predictions
    
    out_file.write("\n\nSimple TimeTaken stats")
    out_file.write("\n\tmean_time = %s" % mean_time)
    out_file.write("\n\tstd_time = %s" % std_time)
    out_file.write("\n\tmedian_time = %s\n" % median_time)
    if alg_counter == 1:
        print("\nSimple TimeTaken stats")
        print("\tmean_time = %s" % mean_time)
        print("\tstd_time = %s" % std_time)
        print("\tmedian_time = %s\n" % median_time)
    df["Mean_TimeTaken"] = mean_time
    df["Median_TimeTaken"] = median_time
    
    mean_time_test_r2 = r2_score(y, df["Mean_TimeTaken"])
    mean_time_test_rmse = sqrt(mean_squared_error(y, df["Mean_TimeTaken"]))
    mean_time_test_meanae = mean_absolute_error(y, df["Mean_TimeTaken"])
    mean_time_test_evs = explained_variance_score(y, df["Mean_TimeTaken"])
    mean_time_test_medianae = median_absolute_error(y, df["Mean_TimeTaken"])
    
    median_time_test_r2 = r2_score(y, df["Median_TimeTaken"])
    median_time_test_rmse = sqrt(mean_squared_error(y, df["Median_TimeTaken"]))
    median_time_test_meanae = mean_absolute_error(y, df["Median_TimeTaken"])
    median_time_test_evs = explained_variance_score(y, df["Median_TimeTaken"])
    median_time_test_medianae = median_absolute_error(y, df["Median_TimeTaken"])
    
    out_file.write("\n\tmean_time_test_r2 = %s" % mean_time_test_r2)
    out_file.write("\n\tmean_time_test_rmse = %s" % mean_time_test_rmse)
    out_file.write("\n\tmean_time_test_meanae = %s" % mean_time_test_meanae)
    out_file.write("\n\tmean_time_test_evs = %s" % mean_time_test_evs)
    out_file.write("\n\tmean_time_test_medianae = %s\n" % mean_time_test_medianae)
    
    out_file.write("\n\tmedian_time_test_mae = %s" % median_time_test_r2)
    out_file.write("\n\tmedian_time_test_mae = %s" % median_time_test_rmse)
    out_file.write("\n\tmedian_time_test_mae = %s" % median_time_test_meanae)
    out_file.write("\n\tmedian_time_test_mae = %s" % median_time_test_evs)
    out_file.write("\n\tmedian_time_test_mae = %s\n" % median_time_test_medianae)
    
    if alg_counter == 1:
        print("\tmean_time_test_r2 = %s" % mean_time_test_r2)
        print("\tmean_time_test_rmse = %s" % mean_time_test_rmse)
        print("\tmean_time_test_meanae = %s " % mean_time_test_meanae)
        print("\tmean_time_test_evs = %s" % mean_time_test_evs)
        print("\tmean_time_test_medianae = %s\n " % mean_time_test_medianae)

        print("\tmedian_time_test_r2 = %s" % median_time_test_r2)
        print("\tmedian_time_test_rmse = %s" % median_time_test_rmse)
        print("\tmedian_time_test_meanae = %s" % median_time_test_meanae)
        print("\tmedian_time_test_evs = %s" % median_time_test_evs)
        print("\tmedian_time_test_medianae = %s" % median_time_test_medianae)
    
    simplepath = newpath + "Simple_Stat_Plots/"
    if not os.path.exists(simplepath):
        os.makedirs(simplepath)  # Make folder for storing results if it does not exist
    
    if alg_counter == 1:
        plot(df["TimeTaken"],df["Mean_TimeTaken"], "Simple", "All", simplepath, "Mean")
        plot(df["TimeTaken"],df["Median_TimeTaken"], "Simple", "All", simplepath, "Median")
    
    ####################################################################################################################
    # Machine Learning
    ####################################################################################################################
    numFolds = int(d["crossvalidation"])
    kf = KFold(n_splits=numFolds, shuffle=True, random_state=int(d["seed"]))

    train_rmse = []
    test_rmse = []
    train_r_sq = []
    test_r_sq = []
    train_mae = []
    test_mae = [] # Mean Absolute Error
    train_evs = [] # Explained variance regression score function
    test_evs = []
    train_median_ae = [] # Median absolute error regression loss
    test_median_ae = []

    percent_within_1 = []  # Tracking predictions within 1 hour
    percent_within_4 = []  # Tracking predictions within 4 hour
    percent_within_8 = []  # Tracking predictions within 8 hour
    percent_within_16 = []  # Tracking predictions within 16 hour
    percent_within_24 = []  # Tracking predictions within 24 hours
    percent_within_48 = []  # Tracking predictions within 48 hours
    percent_within_72 = []  # Tracking predictions within 72 hours
    percent_within_96 = []  # Tracking predictions within 96 hours

    # max_time = 2000000
    df["TimeTaken_%s" % alg] = -1000000  # assign a random value

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
            # if y_train_pred[i] > max_time:  # Convert all predictions > 2M to 2M
                # y_train_pred[i] = max_time  
            if math.isnan(y_train_pred[i]):  # If NaN set to 0
                y_train_pred[i] = 0
        for i in range(len(y_test_pred)): # Convert high or low predictions to 0 or 3 std
            if y_test_pred[i] < 0:  # Convert all negative predictions to 0
                y_test_pred[i] = 0
            # if y_test_pred[i] > max_time:  # Convert all predictions > max_time to max_time
                # y_test_pred[i] = max_time
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

        #  append the predictions for this fold to df
        df.loc[test_indices, "TimeTaken_%s"%alg] = y_test_pred
        
        percent_within_1.append(number_close_1/len(y_test_pred))
        percent_within_4.append(number_close_4/len(y_test_pred))
        percent_within_8.append(number_close_8/len(y_test_pred))
        percent_within_16.append(number_close_16/len(y_test_pred))
        percent_within_24.append(number_close_24/len(y_test_pred))
        percent_within_48.append(number_close_48/len(y_test_pred))
        percent_within_72.append(number_close_72/len(y_test_pred))
        percent_within_96.append(number_close_96/len(y_test_pred))

        train_rmse.append(sqrt(mean_squared_error(trainData_y, y_train_pred)))
        test_rmse.append(sqrt(mean_squared_error(testData_y, y_test_pred)))
        train_r_sq.append(r2_score(trainData_y, y_train_pred))
        test_r_sq.append(r2_score(testData_y, y_test_pred))
        train_mae.append(mean_absolute_error(trainData_y, y_train_pred))
        test_mae.append(mean_absolute_error(testData_y, y_test_pred))
        train_evs.append(explained_variance_score(trainData_y, y_train_pred))
        test_evs.append(explained_variance_score(testData_y, y_test_pred))
        train_median_ae.append(median_absolute_error(trainData_y, y_train_pred))
        test_median_ae.append(median_absolute_error(testData_y, y_test_pred))


    ########################################################################################################################
    # Get averages and standard deviations of results
    ########################################################################################################################
    train_rmse_ave = np.mean(train_rmse)
    test_rmse_ave = np.mean(test_rmse)
    train_r2_ave = np.mean(train_r_sq)
    test_r2_ave = np.mean(test_r_sq)
    train_mae_ave = np.mean(train_mae)
    test_mae_ave = np.mean(test_mae)
    train_evs_ave = np.mean(train_evs)
    test_evs_ave = np.mean(test_evs)
    train_median_ae_ave = np.mean(train_median_ae)
    test_median_ae_ave = np.mean(test_median_ae)

    train_rmse_std = np.std(train_rmse)
    test_rmse_std = np.std(test_rmse)
    train_r2_std = np.std(train_r_sq)
    test_r2_std = np.std(test_r_sq)
    train_mae_std = np.std(train_mae)
    test_mae_std = np.std(test_mae)
    train_evs_std = np.mean(train_evs)
    test_evs_std = np.mean(test_evs)
    train_median_ae_std = np.mean(train_median_ae)
    test_median_ae_std = np.mean(test_median_ae)

    ave_1hour = np.mean(percent_within_1)
    std_1hour = np.std(percent_within_1)
    pct_ave_1hour = ave_1hour * 100
    pct_std_std_1hour = std_1hour * 100

    ave_4hour = np.mean(percent_within_4)
    std_4hour = np.std(percent_within_4)
    pct_ave_4hour = ave_4hour * 100
    pct_std_std_4hour = std_4hour * 100
    
    ave_8hour = np.mean(percent_within_8)
    std_8hour = np.std(percent_within_8)
    pct_ave_8hour = ave_8hour * 100
    pct_std_std_8hour = std_8hour * 100
    
    ave_16hour = np.mean(percent_within_16)
    std_16hour = np.std(percent_within_16)
    pct_ave_16hour = ave_16hour * 100
    pct_std_std_16hour = std_16hour * 100
    
    ave_24hour = np.mean(percent_within_24)
    std_24hour = np.std(percent_within_24)
    pct_ave_24hour = ave_24hour * 100
    pct_std_std_24hour = std_24hour * 100
    
    ave_48hour = np.mean(percent_within_48)
    std_48hour = np.std(percent_within_48)
    pct_ave_48hour = ave_48hour * 100
    pct_std_std_48hour = std_48hour * 100

    ave_72hour = np.mean(percent_within_72)
    std_72hour = np.std(percent_within_72)
    pct_ave_72hour = ave_72hour * 100
    pct_std_std_72hour = std_72hour * 100
    
    ave_96hour = np.mean(percent_within_96)
    std_96hour = np.std(percent_within_96)
    pct_ave_96hour = ave_96hour * 100
    pct_std_std_96hour = std_96hour * 100

    ########################################################################################################################
    # output results
    ########################################################################################################################
    out_file.write("\n" + alg + ": Cross Validation (" + d["crossvalidation"] + " Folds)\n")
    out_file.write("\tTrain Mean R2: {0:.5f} (+/-{1:.5f})\n".format(train_r2_ave, train_r2_std))
    out_file.write("\tTest Mean R2: {0:.5f} (+/-{1:.5f})\n".format(test_r2_ave, test_r2_std))
    out_file.write("\tTrain Mean RMSE: {0:.2f} (+/-{1:.2f})\n".format(train_rmse_ave, train_rmse_std))
    out_file.write("\tTest Mean RMSE: {0:.2f} (+/-{1:.2f})\n".format(test_rmse_ave, test_rmse_std))
    out_file.write("\tTrain Mean MeanAE: {0:.2f} (+/-{1:.2f})\n".format(train_mae_ave, train_mae_std))
    out_file.write("\tTest Mean MeanAE: {0:.2f} (+/-{1:.2f})\n".format(test_mae_ave, test_mae_std))
    out_file.write("\tTrain Mean EVS: {0:.2f} (+/-{1:.2f})\n".format(train_evs_ave, train_evs_std))
    out_file.write("\tTest Mean EVS: {0:.2f} (+/-{1:.2f})\n".format(test_evs_ave, test_evs_std))
    out_file.write("\tTrain Mean MedianAE: {0:.2f} (+/-{1:.2f})\n".format(train_median_ae_ave, train_median_ae_std))
    out_file.write("\tTest Mean MedianAE: {0:.2f} (+/-{1:.2f})\n".format(test_median_ae_ave, test_median_ae_std))


    out_file.write("\n\t{2:s} % test predictions error within 1 hour -> Mean: {0:.2f}% (+/- {1:.2f}%) of {3:d}/10".format(
        pct_ave_1hour, pct_std_std_1hour, alg, len(y)))
    out_file.write("\n\t{2:s} % test predictions error within 4 hours -> Mean: {0:.2f}% (+/- {1:.2f}%) of {3:d}/10".format(
        pct_ave_4hour, pct_std_std_4hour, alg, len(y)))
    out_file.write("\n\t{2:s} % test predictions error within 8 hours -> Mean: {0:.2f}% (+/- {1:.2f}%) of {3:d}/10".format(
        pct_ave_8hour, pct_std_std_8hour, alg, len(y)))
    out_file.write("\n\t{2:s} % test predictions error within 16 hours -> Mean: {0:.2f}% (+/- {1:.2f}%) of {3:d}/10".format(
        pct_ave_16hour, pct_std_std_16hour, alg, len(y)))
    out_file.write("\n\t{2:s} % test predictions error within 24 hours -> Mean: {0:.2f}% (+/- {1:.2f}%) of {3:d}/10".format(
        pct_ave_24hour, pct_std_std_24hour,alg, len(y)))
    out_file.write("\n\t{2:s} % test predictions error within 48 hours -> Mean: {0:.2f}% (+/- {1:.2f}%) of {3:d}/10".format(
        pct_ave_48hour, pct_std_std_48hour,alg, len(y)))
    out_file.write("\n\t{2:s} % test predictions error within 72 hours -> Mean: {0:.2f}% (+/- {1:.2f}%) of {3:d}/10".format(
        pct_ave_72hour, pct_std_std_72hour,alg, len(y)))
    out_file.write("\n\t{2:s} % test predictions error within 96 hours -> Mean: {0:.2f}% (+/- {1:.2f}%) of {3:d}/10\n".format(
        pct_ave_96hour, pct_std_std_96hour,alg, len(y)))
    out_file.write("\n")
    
    print("\n" + alg + ": Cross Validation (" + d["crossvalidation"] + " Folds)")
    print("\tTrain Mean R2: {0:.5f} (+/-{1:.5f})".format(train_r2_ave, train_r2_std))
    print("\tTest Mean R2: {0:.5f} (+/-{1:.5f})".format(test_r2_ave, test_r2_std))
    print("\tTrain Mean RMSE: {0:.2f} (+/-{1:.2f})".format(train_rmse_ave, train_rmse_std))
    print("\tTest Mean RMSE: {0:.2f} (+/-{1:.2f})".format(test_rmse_ave, test_rmse_std))
    print("\tTrain Mean MeanAE: {0:.2f} (+/-{1:.2f})".format(train_mae_ave, train_mae_std))
    print("\tTest Mean MeanAE: {0:.2f} (+/-{1:.2f})".format(test_mae_ave, test_mae_std))
    print("\tTrain Mean EVS: {0:.2f} (+/-{1:.2f})".format(train_evs_ave, train_evs_std))
    print("\tTest Mean EVS: {0:.2f} (+/-{1:.2f})".format(test_evs_ave, test_evs_std))
    print("\tTrain Mean MedianAE: {0:.2f} (+/-{1:.2f})".format(train_median_ae_ave, train_median_ae_std))
    print("\tTest Mean MedianAE: {0:.2f} (+/-{1:.2f})".format(test_median_ae_ave, test_median_ae_std))

    print("\n\t{2:s} % test predictions within 1 hour -> Mean: {0:.2f}% (+/- {1:.2f}%) of {3:d}/10".format(
        pct_ave_1hour, pct_std_std_1hour, alg, len(y)))
    print("\t{2:s} % test predictions error within 4 hours -> Mean: {0:.2f}% (+/- {1:.2f}%) of {3:d}/10".format(
        pct_ave_4hour, pct_std_std_4hour, alg, len(y)))
    print("\t{2:s} % test predictions error within 8 hours -> Mean: {0:.2f}% (+/- {1:.2f}%) of {3:d}/10".format(
        pct_ave_8hour, pct_std_std_8hour, alg, len(y)))
    print("\t{2:s} % test predictions error within 16 hours -> Mean: {0:.2f}% (+/- {1:.2f}%) of {3:d}/10".format(
        pct_ave_16hour, pct_std_std_16hour, alg, len(y)))
    print("\t{2:s} % test predictions error within 24 hours -> Mean: {0:.2f}% (+/- {1:.2f}%) of {3:d}/10".format(
        pct_ave_24hour, pct_std_std_24hour, alg, len(y)))
    print("\t{2:s} % test predictions error within 48 hours -> Mean: {0:.2f}% (+/- {1:.2f}%) of {3:d}/10".format(
        pct_ave_48hour, pct_std_std_48hour, alg, len(y)))
    print("\t{2:s} % test predictions error within 72 hours -> Mean: {0:.2f}% (+/- {1:.2f}%) of {3:d}/10".format(
        pct_ave_72hour, pct_std_std_72hour, alg, len(y)))
    print("\t{2:s} % test predictions error within 96 hours -> Mean: {0:.2f}% (+/- {1:.2f}%) of {3:d}/10\n".format(
        pct_ave_96hour, pct_std_std_96hour, alg, len(y)))

    ####################################################################################################################
    # Plotting
    ####################################################################################################################
    if d["plotting"] == "y":
        print("..plotting..\n")

        plot_errors_main(df, alg, "Test", algpath, alg_initials)
        
        # x = "TimeTaken"
        # y = "TimeTaken_%s"%alg
        # in_data = pd.DataFrame([trainData_y, y_train_pred], x, y)
        # plot(x, y, in_data, alg, "Train", algpath)
        # plot(trainData_x,y_train_pred, alg, "Train", algpath)
        # todo - Is there a way to plot the train predictions with cross validation..? Maybe just for the last fold?
        
        # x = "TimeTaken"
        # y = "TimeTaken_%s"%alg
        # in_data = pd.DataFrame(df[[x,y]], columns=[x, y])
        # plot(x,y, in_data, alg, "Test", algpath)
        plot(df["TimeTaken"],df["TimeTaken_%s"%alg], alg, "Test", algpath, alg_initials)

    ####################################################################################################################
    # Importances
    ####################################################################################################################
    print("..Calculating importances..\n")
    if alg == "RandomForestRegressor" or alg == "GradientBoostingRegressor" or alg == "xgboost":
        tree_importances(regr, X, algpath, d, out_file, alg_initials)

    elif alg == "LinearRegression" or alg == "ElasticNet":
        regression_coef_importances(regr, X, algpath, d, out_file, alg_initials)

    print("\n..finished with alg: %s.." % alg)
    out_file.close()


if __name__ == "__main__":  # Run program
    print("Modeling dataset", time.strftime("%Y.%m.%d"), time.strftime("%H.%M.%S"))
    d = {}
    with open(parameters, "r") as f:
        for line in f:
            line = line.replace(":", "")
            (key, val) = line.split()
            d[key] = val

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

    np.random.seed(int(d["seed"]))  # Set seed

    df = pd.read_csv(d["file_location"] + d["input_file"] + ".csv", encoding='latin-1', low_memory=False)

    print("Input file name: %s" % d["input_file"])
    print("DF Shape:", df.shape, "\n")
    
    if d["histogram"] == "y":
        histogram(df, "TimeTaken", newpath)  # Save histogram plots of TimeTaken

    if d["log_of_y"] == "y":  # Take log of y values
        print("Y has been transformed by log . . . change parameter file to remove this feature\n")
        df["TimeTaken"] = df["TimeTaken"].apply(lambda x: math.log(x))
    # todo - transform predicted values back into seconds

    if d["resample"] == "y":
        from sklearn.utils import resample
        print("..resampling\n")
        df = resample(df, n_samples=int(d["n_samples"]), random_state=int(d["seed"]))
        df = df.reset_index(drop=True)

    ####################################################################################################################
    # Modelling
    ###################################################################################################################
    alg_counter = 0  # used so the simple stats, etc. aren't printed for each algorithm used

    if d["LinearRegression"] == "y":
        alg_counter+=1
        regressor = LinearRegression()
        results(df, "LinearRegression", regressor, newpath, d, alg_counter, "LR")

    if d["ElasticNet"] == "y":
        alg_counter+=1
        regressor = ElasticNet(alpha=100, l1_ratio=1, max_iter=100000)
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
        results(df, "RandomForestRegressor", regressor, newpath, d, alg_counter, "RFR")
    
    copyfile(parameters, newpath + "parameters.txt")  # Save parameters

    if d["output_predictions_csv"] == "y":
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
        out_file.write("\nInput file name %s:\n" % d["input_file"])

        X = df.drop("TimeTaken", axis=1)

        keepers = get_keepers()
        for col in X.columns:
            if col not in keepers:
                del X[col]

        # output the features being used
        print("\nLength of df = %s" % len(df.columns)) # todo add these 3 print messages to outfile or delete them
        print("Length of keepers = %s" % len(keepers))
        print("Length of features used = %s\n" % len(X.columns))

        out_file.write("\nFeatures used:")
        for i, col in enumerate(X.columns):
            out_file.write("\n\t%s - %s" % (i+1, col))
        
        y = df["TimeTaken"]
        
        model = sm.OLS(y, X)
        results = model.fit()
        
        print(results.summary())
        out_file.write("\n\n" + str(results.summary()) + "\n")

        out_file.write('\nPredicted values: '+str(results.predict()))
        
        y_pred = results.predict()
        
        plot(y, y_pred, alg, "All Data", algpath, alg_initials)
        out_file.close()
    
    if d["beep"] == "y":
        import winsound
        Freq = 400 # Set Frequency To 2500 Hertz
        Dur = 1000 # Set Duration To 1000 ms == 1 second
        winsound.Beep(Freq,Dur)
        Freq = 300 # Set Frequency To 2500 Hertz
        winsound.Beep(Freq,Dur)

