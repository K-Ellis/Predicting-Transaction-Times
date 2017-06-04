"""
************************************************************************************************************************
UCD MSc Business Analytics Capstone Project - Predicting Transactions Times
************************************************************************************************************************
Iteration 2
Data pre-processing program
************************************************************************************************************************
Eoin Carroll
Kieron Ellis
************************************************************************************************************************
"""


# Import libraries
import pandas as pd
import time


# Multipurpose Pre-Processing Functions
def time_taken(df, out_file, start, finish): # replace Created_On, Receiveddate and ResolvedDate with one new column, "TimeTaken"
    df[start] = pd.to_datetime(df[start])
    df[finish] = pd.to_datetime(df[finish])
    df2 = pd.DataFrame()  # create new dataframe, df2, to store answer
    df2["TimeTaken"] = (df[finish] - df[start]).astype('timedelta64[m]')
    del df[start]
    del df[finish]
    df = pd.concat([df2, df], axis=1)
    out_file.write("Time Taken column calculated" + "\n\n")
    return df


def min_entries(df, out_file, min=3):  # Delete columns that have less than min entries regardless of number rows
    out_file.write("min_entries function - min: " + str(min) + "\n")
    for column in df:
        if df[column].count() < min:
            # print("Column deletion:", column, df[column].count())
            out_file.write("Column deletion: " + str(column) + " -> Entry Count: " + str(df[column].count()) + "\n")
            del df[column]  # delete column
    out_file.write("\n")
    return df


def min_variable_types(df, out_file, min=2):  # Delete columns with less than min variable types in that column
    out_file.write("min_variable_types function - min: " + str(min) + "\n")
    for column in df:
        if len(df[column].value_counts().index.tolist()) < min:
            # print("Column deletion: ", column, len(df[column].value_counts().index.tolist()))
            out_file.write("Column deletion: " + str(column) + " -> Variable Type Count: " +
                           str(len(df[column].value_counts().index.tolist())) + "\n")
            del df[column]  # delete column
    out_file.write("\n")
    return df


def drop_NULL(df, out_file, x=0.30):  # Remove columns where there is a proportion of NULL,NaN,blank values > tol
    # todo - this is very similar to min entries, might be able to combine the two
    # df.dropna(axis=1, thresh=int(len(df) * x)) .... an alternative method I could not get to work
    out_file.write("drop_NULL - max ratio: " + str(x) + "\n")
    for column in df:
        if (len(df) - df[column].count()) / len(df) >= x:
            # print("Drop Null:", column, df[column].count())
            out_file.write("Column deletion: " + str(column) + " -> Ratio: " +
                           str((len(df) - df[column].count()) / len(df)) + "\n")
            del df[column]  # delete column
    out_file.write("\n")
    return df


def drop_zeros(df, out_file, x=0.90):  # Remove columns where there is a proportion of 0 values greater than tol
    out_file.write("drop_zeros - max ratio: " + str(x) + "\n")
    for column in df:
        if 0 in df[column].value_counts():  # Make sure 0 is in column
            if df[column].value_counts()[0] / len(df) >= x:  # If there are more 0s than out limit
                # print("Drop Zeros:", column, df[column].value_counts()[0] / len(df))
                out_file.write("Column deletion: " + str(column) + " -> Ratio: " +
                               str(df[column].value_counts()[0] / len(df)) + "\n")
                del df[column]  # delete column
    out_file.write("\n")
    return df


def drop_ones(df, out_file, x=0.90):  # Remove columns where there is a proportion of 1 values greater than tol
    out_file.write("drop_ones - max ratio: " + str(x) + "\n")
    for column in df:
        if 1 in df[column].value_counts():  # Make sure 0 is in column
            if df[column].value_counts()[1] / len(df) >= x:  # If there are more 1s than out limit
                # print("Drop Ones:", column, df[column].value_counts()[0] / len(df))
                out_file.write("Column deletion: " + str(column) + " -> Ratio: " +
                               str(df[column].value_counts()[1] / len(df)) + "\n")
                del df[column]  # delete column
    out_file.write("\n")
    return df


# Excel Sheet Specific functions
def clean_Incident():

    print("clean_Incident started")
    out_file_name = "../../../Logs/" + time.strftime("%Y%m%d-%H%M%S") + "_clean_Incident" + ".txt"  # Log file name
    out_file = open(out_file_name, "w")  # Open log file
    out_file.write("Date and time: " + time.strftime("%Y%m%d-%H%M%S") + "\n")
    out_file.write("clean_Incident started" + "\n\n")

    df = pd.read_csv("../../../Data/vw_Incident.csv", encoding='latin-1', low_memory=False)
    # todo - Was getting error:
    # "sys:1: DtypeWarning: Columns (16,65) have mixed types. Specify dtype option on import or set low_memory=False."
    # Solution - Added low_memory=False to read in function. Not sure what this does . . . need to check

    # Create Time Variable
    df = time_taken(df, out_file, "Created_On", "ResolvedDate")

    # Domain knowledge processing
    # Program column: only interested in Enterprise
    df = df[df.Program == "Enterprise"]
    # Only keep the rows which are English
    df = df[df.LanguageName == "English"]
    # Remove StatusReason = rejected
    df = df[df.StatusReason != "Rejected"]
    # Remove ValidCase = 0
    df = df[df.ValidCase == 1]
    # Duplicate column - we will keep IsoCurrencyCode
    del df["CurrencyName"]
    # don't understand what it does
    del df["caseOriginCode"]
    del df["WorkbenchGroup"]
    # Not using received
    del df["Receiveddate"]
    # IsSOXCase contains lots of NULLS - converting to 0 with the assumption that NULL and 0 means not SOX
    df["IsSOXCase"].fillna(0, inplace=True)
    # change the priorities to nominal variables
    df["Priority"] = df["Priority"].map({"Low":0, "Normal":1, "High":2, "Immediate":3})
    # change the Complexities to nominal variables
    df["Complexity"] = df["Complexity"].map({"Low":0, "Medium":1, "High":2})

    # Data mining processing - where there is not enough meaningful information
    df = min_entries(df, out_file)  # Delete columns that have less than x=3 entries
    df = min_variable_types(df, out_file)  # Delete columns with less than x=2 variable types in that column
    df = drop_NULL(df, out_file)  # Remove columns where there is a proportion of NULL,NaN,blank values > tol
    df = drop_zeros(df, out_file)  # Remove columns where there is a proportion of 0 values greater than tol
    # todo - all drop zero columns had a ratio of 0.014 . . . . need to look at further
    df = drop_ones(df, out_file)  # Remove columns where there is a proportion of 1 values greater than tol

    # todo combine the transactions into their respective cases?
    # delete for now, not sure what to do with it..
    # del df["ParentCase"]

    # todo combine variables with less than 100 entries into one variable, call it
    # "other" or something
    # CountrySource
    # CountryProcessed
    # SalesLocation
    # CurrencyName
    # sourcesystem
    # Workbench
    # Revenutype . . . . note, drop_NULL removes this (Rev NULL % is 31)

    # all unique entries - these are needed to map across sheets
    # We need these to sum predictions etc
    # del df["TicketNumber"]
    # del df["IncidentId"]

    # replace the Null values with the mean for the column
    # todo, had to comment out this one  df["CaseRevenue"] = df["CaseRevenue"].fillna(df["CaseRevenue"].mean())

    # export file
    df.to_csv("../../../Data/vw_Incident_cleaned.csv", index = False)

    out_file.write("clean_Incident complete")
    out_file.close()
    print("clean_Incident complete")


def clean_AuditHistory():

    print("clean_AuditHistory started")
    out_file_name = "../../../Logs/" + time.strftime("%Y%m%d-%H%M%S") + "_clean_AuditHistory" + ".txt"  # Log file name
    out_file = open(out_file_name, "w")  # Open log file
    out_file.write("Date and time: " + time.strftime("%Y%m%d-%H%M%S") + "\n")
    out_file.write("clean_AuditHistory started" + "\n\n")

    df = pd.read_csv("../../../Data/vw_AuditHistory.csv", encoding='latin-1', low_memory=False)

    # Create Time Variable
    # df = time_taken(df, out_file, "Created_On", "Modified_On")
    # todo - to_datetime not working for audit history

    # Domain knowledge processing
    # Not using TimeStamp
    del df["TimeStamp"]

    # Data mining processing - where there is not enough meaningful information
    df = min_entries(df, out_file)  # Delete columns that have less than x=3 entries
    df = min_variable_types(df, out_file)  # Delete columns with less than x=2 variable types in that column
    df = drop_NULL(df, out_file)  # Remove columns where there is a proportion of NULL,NaN,blank values > tol
    df = drop_zeros(df, out_file)  # Remove columns where there is a proportion of 0 values greater than tol
    df = drop_ones(df, out_file)  # Remove columns where there is a proportion of 1 values greater than tol

    # todo combine the transactions into their respective cases?
    # delete for now, not sure what to do with it..
    # del df["ParentCase"]

    # export file
    df.to_csv("../../../Data/vw_AuditHistory_cleaned.csv", index = False)

    out_file.write("clean_AuditHistory complete")
    out_file.close()
    print("clean_AuditHistory complete")


def clean_HoldActivity():

    print("clean_HoldActivity started")
    out_file_name = "../../../Logs/" + time.strftime("%Y%m%d-%H%M%S") + "_clean_HoldActivity" + ".txt"  # Log file name
    out_file = open(out_file_name, "w")  # Open log file
    out_file.write("Date and time: " + time.strftime("%Y%m%d-%H%M%S") + "\n")
    out_file.write("clean_HoldActivity started" + "\n\n")

    df = pd.read_csv("../../../Data/vw_HoldActivity.csv", encoding='latin-1', low_memory=False)

    # Domain knowledge processing
    # Use hold duration as time
    del df["StartTime"]
    del df["EndTime"]
    del df["Modified_On"]
    # HoldTypeName is only 3rd party and customer (not Internal . . . assumption that there are no other types)
    df = df[df.HoldTypeName != "Internal"]
    # Duplicate columns, keep Statuscode
    del df["Statecode"]
    df = df[df.Statuscode == "Completed"]

    # Data mining processing - where there is not enough meaningful information
    df = min_entries(df, out_file)  # Delete columns that have less than x=3 entries
    df = min_variable_types(df, out_file)  # Delete columns with less than x=2 variable types in that column
    df = drop_NULL(df, out_file)  # Remove columns where there is a proportion of NULL,NaN,blank values > tol
    df = drop_zeros(df, out_file)  # Remove columns where there is a proportion of 0 values greater than tol
    df = drop_ones(df, out_file)  # Remove columns where there is a proportion of 1 values greater than tol

    # todo combine the transactions into their respective cases?
    # delete for now, not sure what to do with it..
    # del df["ParentCase"]

    # export file
    df.to_csv("../../../Data/vw_HoldActivity_cleaned.csv", index = False)

    out_file.write("clean_AuditHistory complete")
    out_file.close()
    print("clean_HoldActivity complete")


def clean_PackageTriageEntry():

    print("clean_PackageTriageEntry started")
    out_file_name = "../../../Logs/" + time.strftime("%Y%m%d-%H%M%S") + "_clean_PackageTriageEntry" + ".txt"  # Log file name
    out_file = open(out_file_name, "w")  # Open log file
    out_file.write("Date and time: " + time.strftime("%Y%m%d-%H%M%S") + "\n")
    out_file.write("clean_PackageTriageEntry started" + "\n\n")

    df = pd.read_csv("../../../Data/vw_PackageTriageEntry.csv", encoding='latin-1', low_memory=False)

    # Create Time Variable
    # df = time_taken(df, out_file, "Created_On", "Modified_On")
    # todo - to_datetime not working

    # Domain knowledge processing
    # Not using TimeStamp
    # del df["TimeStamp"]

    # Data mining processing - where there is not enough meaningful information
    df = min_entries(df, out_file)  # Delete columns that have less than x=3 entries
    df = min_variable_types(df, out_file)  # Delete columns with less than x=2 variable types in that column
    df = drop_NULL(df, out_file)  # Remove columns where there is a proportion of NULL,NaN,blank values > tol
    df = drop_zeros(df, out_file)  # Remove columns where there is a proportion of 0 values greater than tol
    df = drop_ones(df, out_file)  # Remove columns where there is a proportion of 1 values greater than tol

    # todo combine the transactions into their respective cases?
    # delete for now, not sure what to do with it..
    # del df["ParentCase"]

    # export file
    df.to_csv("../../../Data/vw_PackageTriageEntry_cleaned.csv", index = False)

    out_file.write("clean_PackageTriageEntry complete")
    out_file.close()
    print("clean_PackageTriageEntry complete")


# Run program
if __name__ == "__main__":
    clean_Incident()
    clean_AuditHistory()
    clean_HoldActivity()
    clean_PackageTriageEntry()