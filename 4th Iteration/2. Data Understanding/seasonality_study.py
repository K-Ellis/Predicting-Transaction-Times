"""*********************************************************************************************************************
UCD MSc Business Analytics Capstone Project - Predicting Transactions Times
************************************************************************************************************************
Iteration 4
Seasonality study
************************************************************************************************************************
Eoin Carroll
Kieron Ellis
************************************************************************************************************************
Working on dataset 2 from Cosmic: UCD_Data_20170623_1.xlsx
Results will be saved in Iteration > 0. Results > User > prepare_dataset > Date
*********************************************************************************************************************"""


# Import libraries
import pandas as pd
import numpy as np
import time
import datetime
import matplotlib.pyplot as plt
import os  # Used to create folders
from sklearn import preprocessing
from shutil import copyfile  # Used to copy parameters file to directory

parameters = "../../../Data/parameters.txt"  # Parameters file


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


if __name__ == "__main__":  # Run program
    d = {}
    with open(parameters, "r") as f:
        for line in f:
            line = line.replace(":", "")
            (key, val) = line.split()
            d[key] = val

    newpath = r"../0. Results/" + d["user"] + "/prepare_dataset/" + time.strftime("%Y.%m.%d/")  # Log file location
    if not os.path.exists(newpath):
        os.makedirs(newpath)  # Make folder for storing results if it does not exist

    print("clean_Incident started")

    df = pd.read_csv(d["file_location"] + "vw_Incident" + d["file_name"] + ".csv", encoding='latin-1',
                     low_memory=False)

    df = df[["Created_On", "ResolvedDate"]]  # Delete all columns apart from . .
    df["Created_On"] = pd.to_datetime(df["Created_On"])
    df["ResolvedDate"] = pd.to_datetime(df["ResolvedDate"])
    df["TimeTaken"] = (df["ResolvedDate"] - df["Created_On"]).astype('timedelta64[s]')

    df["Created_On_Week"] = df["Created_On"].apply(lambda x: x.strftime("%w"))  # Day of the week
    df["ResolvedDate_Week"] = df["ResolvedDate"].apply(lambda x: x.strftime("%w"))  # Day of the week
    df["Created_On_Month"] = df["Created_On"].apply(lambda x: x.strftime("%d"))  # Day of the month
    df["ResolvedDate_Month"] = df["ResolvedDate"].apply(lambda x: x.strftime("%d"))  # Day of the month
    df["Created_On_Qtr"] = df["Created_On"].apply(lambda x: day_in_quarter(x))  # Day of the Qtr
    df["ResolvedDate_Qtr"] = df["ResolvedDate"].apply(lambda x: day_in_quarter(x))  # Day of the Qtr
    df["Created_On_Year"] = df["Created_On"].apply(lambda x: x.strftime("%j"))  # Day of the year
    df["ResolvedDate_Year"] = df["ResolvedDate"].apply(lambda x: x.strftime("%j"))  # Day of the year
    # Useful for finding days http://www.pythonforbeginners.com/basics/python-datetime-time-examples

    list = ["Created_On_Week", "ResolvedDate_Week", "Created_On_Month", "ResolvedDate_Month", "Created_On_Qtr",
            "ResolvedDate_Qtr", "Created_On_Year", "ResolvedDate_Year"]

    for column in list:
        plt.figure()
        plt.plot(df[column], df["TimeTaken"], 'ro', alpha=0.1, markersize=3)
        plt.xlabel(column + " - Day Number")
        plt.ylabel("TimeTaken")
        plt.title(column)
        plt.tight_layout()  # Force everything to fit on figure
        plt.savefig(newpath + time.strftime("%H.%M.%S") + "_" + column + ".png")
        plt.savefig(newpath + time.strftime("%H.%M.%S") + "_" + column + ".pdf")

    df.to_csv(d["file_location"] + "vw_Incident_cleaned" + d["file_name"] + ".csv", index=False)  # export file
    print("clean_Incident complete")
    copyfile(parameters, newpath + "/" + time.strftime("%H.%M.%S") + "_parameters.txt")  # Save params
