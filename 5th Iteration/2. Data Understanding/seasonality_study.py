"""*********************************************************************************************************************
UCD MSc Business Analytics Capstone Project - Predicting Transactions Times
************************************************************************************************************************
Iteration 5
Seasonality study
************************************************************************************************************************
Eoin Carroll
Kieron Ellis
************************************************************************************************************************
Working on dataset 2 from Cosmic: UCD_Data_20170623_1.xlsx
Results will be saved in Iteration > 0. Results > User > prepare_dataset > Date
If user==Kieron, results will be saved in Iteration > 0. Results > User > data_understanding > seasonality_study > Date
*********************************************************************************************************************"""


# Import libraries
import pandas as pd
import time
import datetime
import matplotlib.pyplot as plt
import os  # Used to create folders
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

    if d["user"] == "Kieron":
            newpath = r"../0. Results/" + d["user"] + "/data_understanding/seasonality_study/" + time.strftime("%Y.%m.%d/")
    else:
        newpath = r"../0. Results/" + d["user"] + "/prepare_dataset/" + time.strftime("%Y.%m.%d/")  # Log file location
    if not os.path.exists(newpath):
        os.makedirs(newpath)  # Make folder for storing results if it does not exist

    if d["user"] == "Kieron":
        df = pd.read_csv(d["file_location"] + d["input_file"] + ".csv", encoding='latin-1', low_memory=False)
    else:
        df = pd.read_csv(d["file_location"] + "vw_Incident" + d["input_file"] + ".csv", encoding='latin-1',
                     low_memory=False)

    col = "Program"
    if col in df.columns:
        df = df[df["Program"] == "Enterprise"]  # Program column: only interested in Enterprise

    col = "LanguageName"
    if col in df.columns:
        df = df[df["LanguageName"] == "English"]  # Only keep the rows which are English

    col = "StatusReason"
    if col in df.columns:
        df = df[df["StatusReason"] != "Rejected"]  # Remove StatusReason = rejected

    col = "ValidCase"
    if col in df.columns:
        df = df[df["ValidCase"] == 1]  # Remove ValidCase = 0

    df = df[["Created_On", "ResolvedDate"]]  # Delete all columns apart from . .
    df["Created_On"] = pd.to_datetime(df["Created_On"])
    df["ResolvedDate"] = pd.to_datetime(df["ResolvedDate"])
    df["TimeTaken"] = (df["ResolvedDate"] - df["Created_On"]).astype('timedelta64[s]')

    df["Created_On_Day"] = df["Created_On"].apply(lambda x: x.strftime("%H"))  # Time of the day
    df["ResolvedDate_Day"] = df["ResolvedDate"].apply(lambda x: x.strftime("%H"))  # Time of the day
    df["Created_On_Week"] = df["Created_On"].apply(lambda x: x.strftime("%w"))  # Day of the week
    df["ResolvedDate_Week"] = df["ResolvedDate"].apply(lambda x: x.strftime("%w"))  # Day of the week
    df["Created_On_Month"] = df["Created_On"].apply(lambda x: x.strftime("%d"))  # Day of the month
    df["ResolvedDate_Month"] = df["ResolvedDate"].apply(lambda x: x.strftime("%d"))  # Day of the month
    df["Created_On_Qtr"] = df["Created_On"].apply(lambda x: day_in_quarter(x))  # Day of the Qtr
    df["ResolvedDate_Qtr"] = df["ResolvedDate"].apply(lambda x: day_in_quarter(x))  # Day of the Qtr
    df["Created_On_Year"] = df["Created_On"].apply(lambda x: x.strftime("%j"))  # Day of the year
    df["ResolvedDate_Year"] = df["ResolvedDate"].apply(lambda x: x.strftime("%j"))  # Day of the year
    # Useful for finding days http://www.pythonforbeginners.com/basics/python-datetime-time-examples

    list = ["Created_On_Day", "ResolvedDate_Day", "Created_On_Week", "ResolvedDate_Week", "Created_On_Month",
            "ResolvedDate_Month", "Created_On_Qtr", "ResolvedDate_Qtr", "Created_On_Year", "ResolvedDate_Year"]

    for column in list:
        plt.figure()
        data_to_plot = []
        for i in df[column].unique():
            df_temp = df[df[column] == i]
            data_to_plot.append(df_temp["TimeTaken"])
        plt.boxplot(data_to_plot, showfliers=False)
        if column == "Created_On_Day" or column == "ResolvedDate_Day":
            plt.xticks([x+1 for x in range(24)])
            plt.xlabel(column + " - Time of Day (24 hours)")
        elif column == "Created_On_Week" or column == "ResolvedDate_Week":
            days_of_the_week = ["Su", "Mo", "Tu", "We", "Th", "Fr", "Sa"]
            plt.xticks([x+1 for x in range(len(days_of_the_week))], days_of_the_week)
            plt.xlabel(column + " - Day of the week")
        elif column == "Created_On_Month" or column == "ResolvedDate_Month":
            plt.xticks(rotation="vertical")
            plt.xlabel(column + " - Day Number")
        elif column == "Created_On_Qtr" or column == "ResolvedDate_Qtr":
            every_x_days = [(x+1)*5 for x in range(18)]
            plt.xticks(every_x_days, every_x_days, rotation="vertical")
            plt.xlabel(column + " - Day Number")
        else:
            every_x_days = [(x + 1) * 30 for x in range(4)]
            every_x_days.append(140)
            plt.xticks(every_x_days, every_x_days, rotation="vertical")
            plt.ylim(0, 2500000)
            plt.xlabel(column + " - Day Number")
        plt.ylabel("TimeTaken")
        plt.title(column)
        plt.tight_layout()  # Force everything to fit on figure

        if d["user"] == "Kieron":
            plt.savefig(newpath + column + ".png")
            if not os.path.exists(newpath + "PDFs/"):
                os.makedirs(newpath + "PDFs/")  # Make folder for storing results if it does not exist
            plt.savefig(newpath + "PDFs/" + column + ".pdf")
        else:
            plt.savefig(newpath + time.strftime("%H.%M.%S") + "_" + column + ".png")
            plt.savefig(newpath + time.strftime("%H.%M.%S") + "_" + column + ".pdf")
        print(column, "plot created")

    if d["user"] == "Kieron":
        df.to_csv(d["file_location"] + "seasonality_study.csv", index=False)  # export file
        copyfile(parameters, newpath + "parameters.txt")  # Save params
    else:
        df.to_csv(d["file_location"] + "vw_Incident_cleaned" + d["output_file"] + ".csv", index=False)  # export
        # file
        copyfile(parameters, newpath + "/" + time.strftime("%H.%M.%S") + "_parameters.txt")  # Save params
