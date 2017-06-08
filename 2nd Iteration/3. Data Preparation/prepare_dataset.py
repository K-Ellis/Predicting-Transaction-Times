"""****************************************************************************
UCD MSc Business Analytics Capstone Project - Predicting Transactions Times
*******************************************************************************
Iteration 2
Data pre-processing program
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
import time
from sklearn import preprocessing


"""****************************************************************************
Multipurpose Pre-Processing Functions
****************************************************************************"""


def fill_nulls(df, column, out_file):  # Fill in NULL values for all columns
    df[column].fillna(0, inplace=True)
    out_file.write("NULL Values replaced with 0's in " + column + "\n\n")
    return df


def fill_nulls_dfc(dfc, fill_value, out_file):  # Fill in NULL values for one column
    dfc.fillna(fill_value, inplace=True)
    out_file.write("All NULL Values for \"%s\" replaced with most frequent value, %s"%(dfc.name, fill_value) +
                   "\n\n")


def fill_nulls_dfcs(df, dfcs, out_file):
    for dfc in dfcs:
        fill_nulls_dfc(df[dfc], df[dfc].mode()[0], out_file)


def find_dfcs_with_nulls_in_threshold(df, min_thres, max_thres):
    dfcs = []
    if min_thres == None and max_thres == None:
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                dfcs.append(col)
    else:
        for col in df.columns:
            if df[col].isnull().sum() > min_thres and df[col].isnull().sum() < max_thres:
                dfcs.append(col)
    return dfcs


def time_taken(df, out_file, start, finish): # replace Created_On, Receiveddate and ResolvedDate with one new column, "TimeTaken"
    df[start] = pd.to_datetime(df[start])
    df[finish] = pd.to_datetime(df[finish])
    df2 = pd.DataFrame()  # create new dataframe, df2, to store answer
    df2["TimeTaken"] = (df[finish] - df[start]).astype('timedelta64[s]')
    del df[start]
    del df[finish]
    df = pd.concat([df2, df], axis=1)
    out_file.write("Time Taken column calculated" + "\n")
    mean_time = sum(df["TimeTaken"].tolist()) / len(df)  # Calculate mean of time taken
    std_time = np.std(df["TimeTaken"].tolist())  # Calculate standard deviation of time taken
    df = df[df.TimeTaken < (mean_time + 3*std_time)]  # Remove outliers that are > 3 std from mean
    out_file.write("Outliers removed > 3 sd from mean of TimeTaken" + "\n\n")
    return df


def map_variables(dfc, out_file, column="Column"):  # Map categorical variables to numeric
    var_map = dict(enumerate(dfc.value_counts().index.tolist()))
    new_var_map = {}
    new_var_map[0] = 0
    for entry in var_map:  # Shift all so we can include a 0 entry for NULL values
        new_var_map[var_map[entry]] = entry + 1
    dfc = dfc.map(new_var_map)
    out_file.write("Variable map for " + column + ": " + str(new_var_map) + "\n\n")
    # todo - bug if one of the categories is 0 . . . it overwrites the NULL 0 default, need to workaround
    return dfc


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


def drop_null(df, out_file, x=0.95):  # Remove columns where there is a proportion of NULL,NaN,blank values > tol
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


def drop_zeros(df, out_file, x=0.95):  # Remove columns where there is a proportion of 0 values greater than tol
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


def drop_ones(df, out_file, x=0.95):  # Remove columns where there is a proportion of 1 values greater than tol
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


def one_hot_encoding(df, column, out_file):  # One hot encoding
    df = pd.concat([df, pd.get_dummies(df[column], prefix=column, drop_first=True)], axis=1)
    del df[column]
    out_file.write("One hot encoding completed for " + str(column) + "\n\n")
    return df


def transform_country(dfc, out_file, column="Column"):  # Convert country into continents
    # See excel sheet called countrylist in iteration 2 - preprocessing folder for decision process
    africa = ["Algeria","Angola","Botswana","Burundi","Cameroon","Congo (DRC)","Côte d’Ivoire","Egypt","Gabon","Ghana",
              "Ivory Coast","Kenya","Macau SAR","Mauritius","Mozambique","Namibia","Nigeria",
              "Rest of East & Southern Africa","Rwanda","Senegal","Sierra Leone","South Africa","Swaziland","Tanzania",
              "Togo","Uganda","West and Central Africa","Zambia","Zimbabwe"]
    asia = ["Afghanistan","Azerbaijan","Bahrain","Bangladesh","Brunei","China","Hong Kong","Hong Kong SAR","India",
            "Indian Ocean Islands","Indonesia","Iraq","Israel","Japan","Jordan","Kazakhstan","Korea","Kuwait",
            "Kyrgyzstan","Lebanon","Levant","Libya","Malaysia","MEA HQ","Morocco","Myanmar","Nepal","North Gulf",
            "Oman","Pakistan","Palestinian Authority","Philippines","Qatar","Russia","Saudi Arabia","Serbia",
            "Singapore","South Gulf","Sri Lanka","Taiwan","Thailand","Tunisia","United Arab Emirates","Uzbekistan",
            "Vietnam"]
    australia = ["Australia","Cook Islands","New Zealand","Norfolk Island","Samoa"]
    europe = ["Austria","Belarus","Belgium","Bosnia and Herzegovina","Bulgaria","Croatia","Cyprus","Czech Republic",
              "Denmark","Estonia","Finland","France","Georgia","Germany","Greece","Hungary","Iceland","Ireland",
              "Ireland Sales Mktg","Italy","Latvia","Lithuania","Luxembourg","Macedonia, Former Yugoslav Rep",
              "Macedonia, FYR","Malta","Moldova","Montenegro","NEPA Indirect Markets","Netherlands","Norway",
              "Poland","Portugal","Romania","Slovakia","Slovenia","Spain","Sweden","Switzerland","Turkey",
              "Ukraine","United Kingdom"]
    northamerica = ["Canada","United States"]
    southamerica = ["Argentina","Bahamas, The","Barbados","Bermuda","Bolivia","Brazil","caribbean","Central America",
                    "Chile","Colombia","Costa Rica","Dominican Rep.","Ecuador","El Salvador","French Guiana",
                    "French Polynesia","Guadeloupe","Guatemala","Honduras","Jamaica","Jamaica & BCBB","Mexico",
                    "Nicaragua","Panama","Paraguay","Peru","Puerto Rico","St. Lucia","Trinidad and Tobago","Uruguay",
                    "Venezuela"]
    pd.options.mode.chained_assignment = None  # default='warn' . . . . this disables an overwriting warning
    for i in range(len(dfc)):
        if dfc.iloc[i] in africa:
            dfc.iloc[i] = "africa"
        elif dfc.iloc[i] in asia:
            dfc.iloc[i] = "asia"
        elif dfc.iloc[i] in australia:
            dfc.iloc[i] = "australia"
        elif dfc.iloc[i] in europe:
            dfc.iloc[i] = "europe"
        elif dfc.iloc[i] in northamerica:
            dfc.iloc[i] = "northamerica"
        elif dfc.iloc[i] in southamerica:
            dfc.iloc[i] = "southamerica"
        else:
            dfc.iloc[i] = "other"
    out_file.write("Continents assigned for " + column + "\n\n")
    return dfc


def scale_quant_cols(df, quant_cols, out_file):
    # Scale quantitative variables
    df_num = df[quant_cols]
    for col in quant_cols:
        del df[col]

    min_max_scaler = preprocessing.MinMaxScaler()
    # df_num = min_max_scaler.fit_transform(df_num)
    x_scaled = min_max_scaler.fit_transform(df_num)
    df_x_scaled = pd.DataFrame(x_scaled)
    df_x_scaled.columns = df_num.keys().tolist()

    df = pd.concat([df_x_scaled, df], axis=1)
    out_file.write("columns scaled = " + str(df_num.keys().tolist()) + "\n\n")
    return df


"""****************************************************************************
Excel Sheet functions
****************************************************************************"""


def clean_Incident():

    print("clean_Incident started")
    out_file_name = "../../../Logs/" + time.strftime("%Y%m%d-%H%M%S") + "_clean_Incident" + ".txt"  # Log file name
    out_file = open(out_file_name, "w")  # Open log file
    out_file.write("Date and time: " + time.strftime("%Y%m%d-%H%M%S") + "\n")
    out_file.write("clean_Incident started" + "\n\n")

    df = pd.read_csv("../../../Data/vw_Incident.csv", encoding='latin-1', low_memory=False)

    df = time_taken(df, out_file, "Created_On", "ResolvedDate")  # Create Time Variable

    ############################################
    # Queue: One hot encoding in buckets
    ############################################
    substr_list = ["NAOC", "EOC", "AOC", "APOC", "LOC", "E&E", "Xbox", "OpsPM"]
    # Create a list of 8 unique substrings located in the categorical
    # variables. These will become the new one-hot encoded column names.
    val_list = df.Queue.value_counts().index.tolist()# List the categorical
    #  vales in Queue
    cat_list = [[] for item in substr_list]# Create a list of 8 lists (the
    # same size as substr_list)
    for i, substr in enumerate(substr_list):
        for j, val in enumerate(val_list):
            if substr in val: # If one of the 8 substrings is located in a
                # categorical variable, overwrite the variable with a
                # nonsense value and append the variable name to cat_list
                val_list[j] = "n"
                cat_list[i].append(val)
    for i in range(len(substr_list)):
        df.Queue = df.Queue.replace(cat_list[i], substr_list[i]) # Replace
        # the categorical variables in Queue with the substrings

    # df = one_hot_encoding(df, "Queue", out_file)
    ############################################
    ############################################

    # Filtering for the data we want
    df = df[df.Program == "Enterprise"]  # Program column: only interested in Enterprise
    df = df[df.LanguageName == "English"]  # Only keep the rows which are English
    df = df[df.StatusReason != "Rejected"]  # Remove StatusReason = rejected
    df = df[df.ValidCase == 1]  # Remove ValidCase = 0

    # Data mining processing - where there is not enough meaningful information
    df = min_entries(df, out_file)  # Delete columns that have less than x=3 entries
    df = min_variable_types(df, out_file)  # Delete columns with less than x=2 variable types in that column
    df = drop_null(df, out_file)  # Remove columns where there is a proportion of NULL,NaN,blank values > tol
    df = drop_zeros(df, out_file)  # Remove columns where there is a proportion of 0 values greater than tol
    # todo - all drop zero columns had a ratio of 0.014 . . . . need to look at further
    df = drop_ones(df, out_file)  # Remove columns where there is a proportion of 1 values greater than tol

    # df = fill_nulls(df, "CurrencyName", out_file)  # Fill in NULL values with 0s

    # fill nulls for columns with 50>null entries>10000 with most frequent
    # value
    # dfcs = find_dfcs_with_nulls_in_threshold(df, 50, len(df)-50)

    dfcs = find_dfcs_with_nulls_in_threshold(df, None, None)
    fill_nulls_dfcs(df, dfcs, out_file)

    df = time_taken(df, out_file, "Created_On", "ResolvedDate")  # Create Time Variable


    # Domain knowledge processing
    del df["CurrencyName"]  # Duplicate column - we will keep IsoCurrencyCode
    del df["caseOriginCode"]  # Don't understand what it does
    del df["WorkbenchGroup"]  # Don't understand what it does
    del df["Receiveddate"]  # Not using received
    del df["IsoCurrencyCode"]  # Not using IsoCurrencyCode - we have all values in USD
    del df["RelatedCases"] # useless until we link cases together

    # TODO - convert TotalIdleTime and TotalWaitTime to seconds

    # Note pd.get_dummies(df) may be useful for hot encoding
    # Map to nominal variables - need to decide which ones we want

    ############################################
    # Priority and Complexity - nominal variables
    ############################################
    # df.Priority = df.Priority.fillna("Normal") # Fill Priority's nulls with
    # the most frequent value in the Series, Normal
    # ..already done by fill_nulls_dfc() above
    df["Priority"] = df["Priority"].map({"Low": 0, "Normal": 1, "High": 2,
                                         "Immediate": 3})
    out_file.write("map Priority column to nominal variables: Low: 0, "
                   "Normal: 1, High: 2, Immediate: 3 \n\n")

    df["Complexity"] = df["Complexity"].map({"Low": 0, "Medium": 1, "High": 2})
    out_file.write("map Complexity column to nominal variables: Low: 0, "
                   "Normal: 1, High: 2 \n\n")
    ############################################
    ############################################

    # Transform countries into continents and then one hot encode
    df["CountrySource"] = transform_country(df["CountrySource"], out_file, column="CountrySource")
    df = one_hot_encoding(df, "CountrySource", out_file)
    df["CountryProcessed"] = transform_country(df["CountryProcessed"], out_file, column="CountryProcessed")
    df = one_hot_encoding(df, "CountryProcessed", out_file)
    df["SalesLocation"] = transform_country(df["SalesLocation"], out_file, column="SalesLocation")
    df = one_hot_encoding(df, "SalesLocation", out_file)

    # TODO - have a closer look at SubReason, sourcesystem, Source,
    # Workbench, Revenutype
    # can we reduce the number of one-hot columns?
    # Group levels together?
    # Combine infrequent levels as "Other"?

    cat_vars_to_one_hot = ["SubReason", "ROCName", "sourcesystem", "Source",
                           "Workbench", "StageName", "Revenutype",
                           "StatusReason"]
    # for var in cat_vars_to_one_hot:
    #    df = one_hot_encoding(df, var, out_file)

    # todo combine variables with less than 100 entries into one variable, call it
    # "other" or something
    # CountrySource
    # CountryProcessed
    # SalesLocation
    # CurrencyName
    # sourcesystem
    # Workbench
    # Revenutype . . . . note, drop_NULL removes this (Rev NULL % is 31)

    # All unique entries, needed to map across sheets - remember to include later on when we are combining sheets
    del df["TicketNumber"]
    del df["IncidentId"]

    # replace the Null values with the mean for the column
    df["CaseRevenue"] = df["CaseRevenue"].fillna(df["CaseRevenue"].mean())
    out_file.write("fill nulls with CaseRevenue mean \n\n")

    num_cols = ["CaseRevenue", "AmountinUSD"]
    df = scale_quant_cols(df, num_cols, out_file)

    df.dropna(inplace = True)

    """
    # Used for testing model program - can delete whenever
    one_hot_encoding(df, "CountrySource", out_file)
    one_hot_encoding(df, "CountryProcessed", out_file)
    one_hot_encoding(df, "SalesLocation", out_file)
    del df["CountrySource"]
    del df["CountryProcessed"]
    del df["SalesLocation"]
    """


    df.to_csv("../../../Data/vw_Incident_cleaned.csv", index = False)   # export file

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
    del df["TimeStamp"]  # Not using TimeStamp

    # Data mining processing - where there is not enough meaningful information
    df = min_entries(df, out_file)  # Delete columns that have less than x=3 entries
    df = min_variable_types(df, out_file)  # Delete columns with less than x=2 variable types in that column
    df = drop_null(df, out_file)  # Remove columns where there is a proportion of NULL,NaN,blank values > tol
    df = drop_zeros(df, out_file)  # Remove columns where there is a proportion of 0 values greater than tol
    df = drop_ones(df, out_file)  # Remove columns where there is a proportion of 1 values greater than tol

    # Note pd.get_dummies(df) may be useful for hot encoding
    # Map to nominal variables - need to decide which ones we want
    df["Action"] = map_variables(df["Action"], out_file, "Action")

    # df = fill_nulls(df, "Action", out_file)  # Fill in NULL values with 0s

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
    df = df[df.HoldTypeName != "Internal"]  # HoldTypeName is only 3rd party and customer
    del df["Statecode"]  # Duplicate columns, keep Statuscode
    df = df[df.Statuscode == "Completed"]  # Only interested in completed

    # Data mining processing - where there is not enough meaningful information
    df = min_entries(df, out_file)  # Delete columns that have less than x=3 entries
    df = min_variable_types(df, out_file)  # Delete columns with less than x=2 variable types in that column
    df = drop_null(df, out_file)  # Remove columns where there is a proportion of NULL,NaN,blank values > tol
    df = drop_zeros(df, out_file)  # Remove columns where there is a proportion of 0 values greater than tol
    df = drop_ones(df, out_file)  # Remove columns where there is a proportion of 1 values greater than tol

    # Note pd.get_dummies(df) may be useful for hot encoding
    # Map to nominal variables - need to decide which ones we want
    df["HoldTypeName"] = map_variables(df["HoldTypeName"], out_file, "HoldTypeName")
    df["Reason"] = map_variables(df["Reason"], out_file, "Reason")
    df["AssignedToGroup"] = map_variables(df["AssignedToGroup"], out_file, "AssignedToGroup")

    # df = fill_nulls(df, "AssignedToGroup", out_file)  # Fill in NULL values with 0s

    # todo combine the transactions into their respective cases?
    # delete for now, not sure what to do with it..
    # del df["ParentCase"]

    df.to_csv("../../../Data/vw_HoldActivity_cleaned.csv", index = False)  # export file

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
    df = drop_null(df, out_file)  # Remove columns where there is a proportion of NULL,NaN,blank values > tol
    df = drop_zeros(df, out_file)  # Remove columns where there is a proportion of 0 values greater than tol
    df = drop_ones(df, out_file)  # Remove columns where there is a proportion of 1 values greater than tol

    # Note pd.get_dummies(df) may be useful for hot encoding
    # Map to nominal variables - need to decide which ones we want
    df["EntryType"] = map_variables(df["EntryType"], out_file, "EntryType")
    df["EntryLevel"] = map_variables(df["EntryLevel"], out_file, "EntryLevel")
    df["EntryProcess"] = map_variables(df["EntryProcess"], out_file, "EntryProcess")

    # df = fill_nulls(df, "EntryProcess", out_file)  # Fill in NULL values with 0s

    df.to_csv("../../../Data/vw_PackageTriageEntry_cleaned.csv", index = False)  # export file

    out_file.write("clean_PackageTriageEntry complete")
    out_file.close()
    print("clean_PackageTriageEntry complete")


"""****************************************************************************
Run All Code
****************************************************************************"""


if __name__ == "__main__":  # Run program
    clean_Incident()
    # clean_AuditHistory()
    # clean_HoldActivity()
    # clean_PackageTriageEntry()
