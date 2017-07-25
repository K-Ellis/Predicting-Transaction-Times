"""*********************************************************************************************************************
UCD MSc Business Analytics Capstone Project - Predicting Transactions Times
************************************************************************************************************************
Iteration 5
Data pre-processing program
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
from calendar import monthrange
import os  # Used to create folders
from sklearn import preprocessing
from shutil import copyfile  # Used to copy parameters file to directory

parameters = "../../../Data/parameters.txt"  # Parameters file


"""*********************************************************************************************************************
Reusable Pre-Processing Functions
*********************************************************************************************************************"""


def fill_nulls(df, column, out_file):  # Fill in NULL values for that column
    df[column].fillna(0, inplace=True)
    out_file.write("NULL Values replaced with 0's in " + column + "\n\n")
    return df


def fill_nulls_dfc(dfc, fill_value, out_file):  # Fill in NULL values for one column
    dfc.fillna(fill_value, inplace=True)
    out_file.write("All NULL Values for \"%s\" replaced with most frequent value, %s" % (dfc.name, fill_value) + "\n")


def fill_nulls_dfcs(df, dfcs, fill_value, out_file): # Fill in Nulls given a set of dataframe columns
    for dfc in dfcs:
        if fill_value == "mode":
            fill_nulls_dfc(df[dfc], df[dfc].mode()[0], out_file)
        if fill_value == "mean":
            fill_nulls_dfc(df[dfc], df[dfc].mean(), out_file)


def find_dfcs_with_nulls_in_threshold(df, min_thres, max_thres, exclude):
    dfcs = []  # DataFrameColumns
    if min_thres is None and max_thres is None:  # if there is no min and max threshold
        for col in df.columns:
            if col not in exclude:
                if df[col].isnull().sum() > 0:  # if the column has Null entries append it to dfcs
                    dfcs.append(col)
    else:  # if the col has nulls within a given range, append it to dfcs
        for col in df.columns:
            if col not in exclude:
                if df[col].isnull().sum() > min_thres and df[col].isnull().sum() < max_thres:
                    dfcs.append(col)
    return dfcs


# custom month end
def custom_month_end(date):
    wkday, days_in_month = monthrange(date.year, date.month)
    lastBDay = days_in_month - max(((wkday + days_in_month - 1) % 7) - 4, 0)
    m = 0
    d = 31
    increment_day = False
    if date.day == days_in_month and date.day == lastBDay:
        m = 1
        d = 1
        increment_day = False
    elif date.day > lastBDay + 1:
        m = 1
        d = 31
        increment_day = True
    elif date.day > lastBDay and date.hour >= 8:
        m = 1
        d = 31
        increment_day = True
    elif date.day == 1 and date.hour < 8:
        d = 1
        increment_day = False
    elif lastBDay == days_in_month:
        increment_day = False
        m = 1
        d = 1

    date += pd.DateOffset(months=m, day=d)

    if date.weekday() > 4:
        date -= pd.offsets.BDay()
        date += pd.offsets.Day()
    elif increment_day == True:
        date += pd.offsets.Day()
    return date.normalize() + pd.DateOffset(hours=8)

def seconds_left_in_month(date):
    cutoff = custom_month_end(date)
    seconds_left = (cutoff-date).seconds
    days_left = (cutoff-date).days
    total_seconds_left = seconds_left + days_left*24*60*60
    return total_seconds_left


def time_taken(df, out_file, start, finish, d):  # replace start & finish with one new column, "TimeTaken"
    df[start] = pd.to_datetime(df[start])
    df[finish] = pd.to_datetime(df[finish])
    df2 = pd.DataFrame()  # create new dataframe, df2, to store answer  # todo - no need for df2
    df2["TimeTaken"] = (df[finish] - df[start]).astype('timedelta64[s]')
    # del df[start]  # Removed so we can include time to month and qtr end
    if d["delete_created_resolved"] == "y":
        del df[finish]
    df = pd.concat([df2, df], axis=1)
    out_file.write("\nTime Taken column calculated" + "\n")
    mean_time = sum(df["TimeTaken"].tolist()) / len(df["TimeTaken"])  # Calculate mean of time taken
    std_time = np.std(df["TimeTaken"].tolist())  # Calculate standard deviation of time taken
    df = df[df["TimeTaken"] < (mean_time + 3*std_time)]  # Remove outliers that are > 3 std from mean
    # df = df[df["TimeTaken"] < 2000000]  # Remove outliers that are > 2000000
    out_file.write("Outliers removed > 3 sd from mean of TimeTaken" + "\n")

    # df["Days_left_Month"] = df["Created_On"].apply(lambda x: int(days_left_in_month(x)))  # Day of the month
    # out_file.write("Day of month calculated" + "\n")
    df["Seconds_left_month"] = df["Created_On"].apply(lambda x: int(seconds_left_in_month(x)))  # seconds left to
    # month end
    out_file.write("seconds to month end calculated" + "\n")

    df["Days_left_QTR"] = df["Created_On"].apply(lambda x: int(days_left_in_quarter(x)))  # Day of the Qtr
    if d["delete_created_resolved"] == "y":
        del df["Created_On"]
    out_file.write("Day of quarter calculated" + "\n\n")
    return df


def min_entries(df, out_file, min_entries=3):  # Delete columns that have less than min entries regardless of number
    # rows
    out_file.write("min_entries function - min: " + str(min_entries) + "\n")
    for column in df:
        if df[column].count() < min_entries:
            out_file.write("Column deletion: " + str(column) + " -> Entry Count: " + str(df[column].count()) + "\n")
            del df[column]  # delete column
    out_file.write("\n")
    return df


def min_variable_types(df, out_file, min_var=2):  # Delete columns with less than min variable types in that column
    out_file.write("min_variable_types function - min: " + str(min_var) + "\n")
    for column in df:
        if len(df[column].value_counts().index.tolist()) < min_var:
            out_file.write("Column deletion: " + str(column) + " -> Variable Type Count: " +
                           str(len(df[column].value_counts().index.tolist())) + "\n")
            del df[column]  # delete column
    out_file.write("\n")
    return df


def drop_null(df, out_file, x=0.99):  # Remove columns where there is a proportion of NULL,NaN,blank values > tol
    # todo - this is very similar to min entries, might be able to combine the two
    # df.dropna(axis=1, thresh=int(len(df) * x)) .... an alternative method I could not get to work
    out_file.write("drop_NULL - max ratio: " + str(x) + "\n")
    for column in df:
        if (len(df) - df[column].count()) / len(df) >= x:
            out_file.write("Column deletion: " + str(column) + " -> Ratio: " +
                           str((len(df) - df[column].count()) / len(df)) + "\n")
            del df[column]  # delete column
    out_file.write("\n")
    return df


def drop_zeros(df, out_file, x=0.99):  # Remove columns where there is a proportion of 0 values greater than tol
    out_file.write("drop_zeros - max ratio: " + str(x) + "\n")
    for column in df:
        if 0 in df[column].value_counts():  # Make sure 0 is in column
            if df[column].value_counts()[0] / len(df) >= x:  # If there are more 0s than out limit
                out_file.write("Column deletion: " + str(column) + " -> Ratio: " +
                               str(df[column].value_counts()[0] / len(df)) + "\n")
                del df[column]  # delete column
    out_file.write("\n")
    return df


def drop_ones(df, out_file, x=0.99):  # Remove columns where there is a proportion of 1 values greater than tol
    out_file.write("drop_ones - max ratio: " + str(x) + "\n")
    for column in df:
        if 1 in df[column].value_counts():  # Make sure 0 is in column
            if df[column].value_counts()[1] / len(df) >= x:  # If there are more 1s than out limit
                out_file.write("Column deletion: " + str(column) + " -> Ratio: " +
                               str(df[column].value_counts()[1] / len(df)) + "\n")
                del df[column]  # delete column
    out_file.write("\n")
    return df


def one_hot_encoding(df, column, out_file):  # One hot encoding
    df = pd.concat([df, pd.get_dummies(df[column], prefix=column, drop_first=True)], axis=1)
    del df[column]
    out_file.write("One hot encoding completed for " + str(column) + "\n")
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
    out_file.write("Continents assigned for " + column + "\n")
    return dfc


def scale_quant_cols(df, quant_cols, out_file):  # Scale quantitative variables
    df_num = df[quant_cols]
    for col in quant_cols:
        del df[col]
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(df_num)
    df_x_scaled = pd.DataFrame(x_scaled)
    df_x_scaled.columns = df_num.keys().tolist()

    df.reset_index(drop=True, inplace=True)

    df = pd.concat([df_x_scaled, df], axis=1)
    out_file.write("columns scaled = " + str(df_num.keys().tolist()) + "\n\n")
    return df


def days_left_in_quarter(date):
    # Function found on stack overflow
    # https://stackoverflow.com/questions/37471704/how-do-i-get-the-correspondent-day-of-the-quarter-from-a-date-field
    q2 = (datetime.datetime.strptime("4/1/{0:4d}".format(date.year), "%m/%d/%Y")).timetuple().tm_yday
    q3 = (datetime.datetime.strptime("7/1/{0:4d}".format(date.year), "%m/%d/%Y")).timetuple().tm_yday
    q4 = (datetime.datetime.strptime("10/1/{0:4d}".format(date.year), "%m/%d/%Y")).timetuple().tm_yday
    q5 = (datetime.datetime.strptime("12/31/{0:4d}".format(date.year), "%m/%d/%Y")).timetuple().tm_yday + 1 # 1st Jan

    cur_day =  date.timetuple().tm_yday
    if (date.month < 4):
        return q2 - (cur_day)
    elif (date.month < 7):
        return q3 - (cur_day - q2 + 1)
    elif (date.month < 10):
        return q4 - (cur_day - q3 + 1)
    else:
        return q5 - (cur_day - q4 + 1)


def days_left_in_month(date):
    return int(monthrange(date.year, date.month)[1]) - int(date.strftime("%d"))


"""*********************************************************************************************************************
Excel Sheet functions
*********************************************************************************************************************"""


def clean_Incident(d, newpath):

    print("clean_Incident started")

    out_file_name = newpath + time.strftime("%H.%M.%S") + "_clean_Incident.txt"  # Log file name
    out_file = open(out_file_name, "w")  # Open log file
    out_file.write("Date and time: " + time.strftime("%Y%m%d-%H%M%S") + "\n")
    out_file.write("clean_Incident started" + "\n")

    if d["user"] == "Kieron":
        df = pd.read_csv(d["file_location"] + d["input_file"] + ".csv", encoding='latin-1', low_memory=False)
    else:
        df = pd.read_csv(d["file_location"] + "vw_Incident" + d["input_file"] + ".csv", encoding='latin-1', low_memory=False)

    if d["resample"] == "y":
        from sklearn.utils import resample
        df = resample(df, n_samples=int(d["n_samples"]), random_state=int(d["seed"]))

    ####################################################################################################################
    # Use only cases created in the last 4 business days of the month
    ####################################################################################################################
    if d["last_4_BDays"] == "y":
        from pandas.tseries.offsets import BDay
        from pandas.tseries.offsets import MonthEnd
        import datetime
        df["Created_On_DateTime"] = pd.to_datetime(df["Created_On"])

        start = pd.datetime(2017, 2, 1)  # todo some day in the future - avoid hardcoding dates
        end = pd.datetime(2017, 7, 1)
        end_of_months = pd.date_range(start, end, freq='BM')

        last_bdayslist = []
        for end_of_month in end_of_months:
            last_bdayslist += pd.bdate_range(end_of_month- BDay(3),end_of_month- BDay(0))

        dflast_bdays = pd.DataFrame(last_bdayslist, columns=["Last_4_BDays"])

        # dfincident_trimmed = df.copy()

        dflast_bdays["Last_4_BDays_String"] = dflast_bdays["Last_4_BDays"].map(lambda x: x.strftime('%Y-%m-%d'))
        df["Created_On_String"] = df["Created_On_DateTime"].map(lambda x: x.strftime('%Y-%m-%d'))

        for date in df["Created_On_String"]:
            if date not in dflast_bdays["Last_4_BDays_String"].values:
                df = df[df["Created_On_String"] != date]

        del df["Created_On_String"]
        del df["Created_On_DateTime"]

    ####################################################################################################################
    # Generate workload variables
    ####################################################################################################################
    if d["workload"] == "y":
        df["Concurrent_open_cases"] = 0  # Add number of cases that were open at the same time
        for i in range(len(df)):
            # todo current date in real program
            df.loc[i, "Concurrent_open_cases"] = len(df[(df.Created_On < df.iloc[i]["Created_On"]) & (
                df.ResolvedDate > df.iloc[i]["ResolvedDate"])])

    ####################################################################################################################
    # Date and time - calculate time taken and time remaining before month and Qtr end
    ####################################################################################################################
    df = time_taken(df, out_file, "Created_On", "ResolvedDate", d)  # Create Time Variable and filter outliers

    ####################################################################################################################
    # Queue: One hot encoding in buckets
    ####################################################################################################################
    if d["minimum_data"] != "y":
        substr_list = ["NAOC", "EOC", "AOC", "APOC", "LOC", "Broken", "E&E"]
        check_substr_exists = [False for _ in substr_list]
        # Create a list of 8 unique substrings located in the categorical variables. These will become the new one-hot
        # encoded column names.
        val_list = df.Queue.value_counts().index.tolist()  # List the categorical values in Queue
        cat_list = [[] for _ in substr_list]  # Create a list of 8 lists (the same size as substr_list)
        for i, substr in enumerate(substr_list):
            for j, val in enumerate(val_list):
                if substr in val:  # If one of the 8 substrings is located in a categorical variable, overwrite the
                    # variable with a nonsense value and append the variable name to cat_list
                    val_list[j] = "n"
                    cat_list[i].append(val)
                    check_substr_exists[i] = True
        for i in range(len(substr_list)):
            if check_substr_exists[i] == True:
                df.Queue = df.Queue.replace(cat_list[i], substr_list[i])  # Replace the categorical variables in Queue with
            # the substrings
        extra_queues = ["<WWCS - EMEA Admin>", "<3P Xbox AR Operations>", "<VL OpsPM program support>", "<CLT Duplicates>"]
        # combine uncommon queue types
        for extra in extra_queues:
            if extra in df["Queue"].values:
                df["Queue"] = df["Queue"].replace(extra, "Other")
        df = one_hot_encoding(df, "Queue", out_file)
        out_file.write("\n")

    ####################################################################################################################
    # Filtering for the data MS want
    ####################################################################################################################
    df = df[df["Program"] == "Enterprise"]  # Program column: only interested in Enterprise
    df = df[df["LanguageName"] == "English"]  # Only keep the rows which are English
    df = df[df["StatusReason"] != "Rejected"]  # Remove StatusReason = rejected
    df = df[df["ValidCase"] == 1]  # Remove ValidCase = 0

    ####################################################################################################################
    # Combine based on ticket numbers
    ####################################################################################################################
    if d["minimum_data"] != "y":

        if d["append_HoldDuration"] == "y" or d["append_AuditDuration"] == "y":
            df["TicketNumber"] = [x.lstrip('5-') for x in df["TicketNumber"]]
            df["TicketNumber"] = df["TicketNumber"].astype(int)

        if d["append_HoldDuration"] == "y":

            if d["user"] == "Kieron":
                dfholdactivity = pd.read_csv("../../../Data/vw_HoldActivity.csv", encoding='latin-1',low_memory=False)
            else:
                dfholdactivity = pd.read_csv(d["file_location"] + "vw_HoldActivity" + d["input_file"] + ".csv",
                                             encoding='latin-1', low_memory=False)

            dfholdactivity["TicketNumber"] = [x.lstrip('5-') for x in dfholdactivity["TicketNumber"]]
            dfholdactivity["TicketNumber"] = dfholdactivity["TicketNumber"].astype(int)

            columns = ["TicketNumber", "HoldDuration", "HoldTypeName", "AssignedToGroup"]
            for col in dfholdactivity:
                if col not in columns:
                    del dfholdactivity[col]

            dfdummies = pd.get_dummies(data=dfholdactivity, columns=["HoldTypeName", "AssignedToGroup"])
            dfdummies["HoldCount"] = 1

            unique_tickets = pd.DataFrame(dfholdactivity["TicketNumber"].unique().tolist(), columns=["TicketNumber"])

            columns_to_count = ["HoldDuration", "HoldTypeName_3rd Party", "HoldTypeName_Customer", "HoldTypeName_Internal",
                                "AssignedToGroup_BPO", "AssignedToGroup_CRMT", "AssignedToGroup_Internal",
                                "AssignedToGroup_Microsoft IT", "AssignedToGroup_Ops. Program Manager",
                                "AssignedToGroup_Submitter (Contact)", "HoldCount"]
            for col in columns_to_count:
                unique_tickets[col] = 0

            for i, row in unique_tickets.iterrows():
                ticket = row["TicketNumber"]
                summed = dfdummies[dfdummies["TicketNumber"] == ticket].sum()

                summed = summed.to_frame().T

                for j in range(len(summed.columns) - 1):
                    j += 1
                    unique_tickets.iloc[i, j] += summed.iloc[0, j]
            # merge new dfduration df with dfincident based on ticket number
            df = df.merge(right=unique_tickets, how="left", on="TicketNumber")

            # fill the NANs with 0's
            for col in columns_to_count:
                df[col].fillna(0, inplace=True)

        if d["append_AuditDuration"] == "y":
            from datetime import timedelta

            if d["user"] == "Kieron":
                dfaudithistory = pd.read_csv("../../../Data/vw_AuditHistory.csv", encoding='latin-1',low_memory=False)
            else:
                dfaudithistory = pd.read_csv(d["file_location"] + "vw_AuditHistory" + d["input_file"] + ".csv",
                                             encoding='latin-1', low_memory=False)

            dfaudithistory["TicketNumber"] = [x.lstrip('5-') for x in dfaudithistory["TicketNumber"]]
            dfaudithistory["TicketNumber"] = dfaudithistory["TicketNumber"].astype(int)

            dfaudithistory["Created_On"] = pd.to_datetime(dfaudithistory["Created_On"])

            dfaudithistory_uniqueticketsonly = pd.DataFrame(dfaudithistory["TicketNumber"].unique(), columns=["TicketNumber"])
            dfaudithistory_uniqueticketsonly["AuditDuration"] = None

            for ticket in dfaudithistory_uniqueticketsonly["TicketNumber"].tolist():
                dfaudithistory_uniqueticketsonly.loc[dfaudithistory_uniqueticketsonly["TicketNumber"] == ticket, "AuditDuration"] = \
                timedelta.total_seconds(dfaudithistory.loc[dfaudithistory["TicketNumber"] == ticket, "Created_On"].max() - \
                    dfaudithistory.loc[dfaudithistory["TicketNumber"] == ticket, "Created_On"].min())

            # merge new dfduration df with dfincident based on ticket number
            df = df.merge(dfaudithistory_uniqueticketsonly, how='left', left_on='TicketNumber', right_on='TicketNumber')

            # fill the NANs with 0's
            df["AuditDuration"].fillna(0, inplace=True)

    ####################################################################################################################
    # Domain knowledge processing
    ####################################################################################################################
    if d["delete_ticketNumber"] == "y":
        del df["TicketNumber"]  # Delete for first iteration
    del df["IncidentId"]  # Delete for first iteration
    del df["Receiveddate"]  # Not using received
    del df["CaseRevenue"] # In local currency. - we have all values in USD
    del df["CurrencyName"]  # Not using column - we have all values in USD
    del df["IsoCurrencyCode"]  # Not using IsoCurrencyCode - we have all values in USD
    del df["caseOriginCode"]  # Don't understand what it does
    del df["pendingemails"]  # Don't understand what it does
    del df["WorkbenchGroup"]  # Don't understand what it does
    del df["Workbench"]  # Don't understand what it does
    del df["RelatedCases"]  # useless until we link cases together
    del df["TotalIdleTime"]  # can be used for real world predictions?
    del df["TotalWaitTime"]  # can be used for real world predictions?

    ####################################################################################################################
    # Not enough unique entries
    ####################################################################################################################
    del df["RevenueImpactAmount"]  # Mostly NULL values
    del df["Auditresult"]  # Mostly NULL values
    del df["PendingRevenue"]
    del df["Requestspercase"]
    del df["Totalbillabletime"]
    del df["Totaltime"]
    del df["CreditAmount"]
    del df["DebitAmount"]
    del df["OrderAmount"]
    del df["InvoiceAmount"]
    del df["Deleted"]
    del df["RejectionReason"]
    del df["RejectionSubReason"]
    del df["PackageNumber"]
    del df["RequiredThreshold"]
    del df["Slipped"]
    del df["DefectiveCase"]
    del df["ProcessName"]
    del df["NumberofChildIncidents"]
    del df["ParentCase"]
    del df["Referencesystem"]
    del df["StateCode"]
    del df["Isrevenueimpacting"]

    del df["ValidCase"]
    del df["BusinessFunction"]
    del df["LineOfBusiness"]
    del df["Program"]
    del df["CaseType"]
    del df["CaseSubTypes"]
    del df["Reason"]
    del df["Language"]
    del df["LanguageName"]
    del df["IsAudited"]
    del df["SubSubReason"]

    ####################################################################################################################
    # Data mining processing - where there is not enough meaningful information.
    ####################################################################################################################
    # df = min_entries(df, out_file)  # Delete columns that have less than x=3 entries
    # df = min_variable_types(df, out_file)  # Delete columns with less than x=2 variable types in that column
    # df = drop_null(df, out_file)  # Remove columns where there is a proportion of NULL,NaN,blank values > tol
    # df = drop_zeros(df, out_file)  # Remove columns where there is a proportion of 0 values greater than tol
    # df = drop_ones(df, out_file)  # Remove columns where there is a proportion of 1 values greater than tol

    # Important variable with ~6500 Null values in COSMIC 2, therefore can't assume that Null entry's values?

    # TODO - For Null entries, find the corrosponding ticketnumber in the audithistory sheet. If the ticketnumber exists
    # there and has gone through each of the stages, then we can assume the isSOXcase Null value should be a 1.
    # Deleting 6500 rows from incident is quite a lot.
    if d["minimum_data"] != "y":
        df["IsSOXCase"].fillna(2, inplace=True)
        df = df[df["IsSOXCase"] != 2]

    ####################################################################################################################
    # Priority, Complexity, StageName - ordinal variable mapping
    ####################################################################################################################
    if d["minimum_data"] != "y":
        df["Priority"] = df["Priority"].map({"Low": 0, "Normal": 1, "High": 2, "Immediate": 3})
        out_file.write("Map Priority column to ordinal variables: Low: 0, Normal: 1, High: 2, Immediate: 3 \n")
        df["Complexity"] = df["Complexity"].map({"Low": 0, "Medium": 1, "High": 2})
        out_file.write("Map Complexity column to ordinal variables: Low: 0, Normal: 1, High: 2 \n")
        df["StageName"] = df["StageName"].map({"Ops In": 0, "Triage And Validation": 1, "Data Entry": 2, "Submission": 3,
                                               "Ops Out": 4})
        out_file.write("Map StageName column to ordinal variables: Ops In: 0, Triage And Validation: 1, Data Entry: 2, "
                       "Submission: 3, Ops Out: 4 \n\n")

    ####################################################################################################################
    # Fill Categorical and numerical nulls. And Scale numerical variables.
    ####################################################################################################################
    if d["minimum_data"] != "y":
        quant_cols = ["AmountinUSD", "Priority", "Complexity", "StageName"]
        if d["append_HoldDuration"] == "y":
            quant_cols.append("HoldDuration")
        if d["append_AuditDuration"] == "y":
            quant_cols.append("AuditDuration")
        if d["workload"] == "y":
            quant_cols.append("Concurrent_open_cases")

        exclude_from_mode_fill = quant_cols
        dfcs = find_dfcs_with_nulls_in_threshold(df, None, None, exclude_from_mode_fill)
        fill_nulls_dfcs(df, dfcs, "mode", out_file)
        fill_nulls_dfcs(df, ["AmountinUSD", "Priority", "Complexity", "StageName"], "mean", out_file)
        out_file.write("\n")
        df = scale_quant_cols(df, quant_cols, out_file)

        df.IsSOXCase = df.IsSOXCase.astype(int)
        df.Numberofreactivations = df.Numberofreactivations.astype(int)

    ####################################################################################################################
    # Transform countries into continents and then one hot encode
    ####################################################################################################################
    if d["minimum_data"] != "y":
        df["CountrySource"] = transform_country(df["CountrySource"], out_file, column="CountrySource")
        df["CountryProcessed"] = transform_country(df["CountryProcessed"], out_file, column="CountryProcessed")
        df["SalesLocation"] = transform_country(df["SalesLocation"], out_file, column="SalesLocation")
        out_file.write("\n")

    ####################################################################################################################
    # One-hot encode categorical variables
    ####################################################################################################################
    if d["minimum_data"] != "y":
        cat_vars_to_one_hot = ["CountrySource", "CountryProcessed", "SalesLocation", "StatusReason", "SubReason",
                               "ROCName", "sourcesystem", "Source", "Revenutype"]
        for var in cat_vars_to_one_hot:
            df = one_hot_encoding(df, var, out_file)
        out_file.write("\n")

    # TODO - have a closer look at SubReason, sourcesystem, Source, Workbench, Revenutype
        #  can we reduce the number of one-hot columns?
        #  Group levels together?
        #  Combine infrequent levels as "Other"?

    # ####################################################################################################################
    # # Delete Collinear Variables above a given threshold
    # ####################################################################################################################
    # df, cols_deleted = find_and_delete_corr(df, correlation_threshold)

    # out_file.write("\n Delete one of the columns when correlation between a pair is above %s:" %
    # correlation_threshold)
    # for col in cols_deleted:
    #    out_file.write("%s" % col)
    # out_file.write("\n\n")

    ####################################################################################################################
    # Export final df
    ####################################################################################################################
    minimum = ["TicketNumber", "TimeTaken", "Concurrent_open_cases", "Days_left_Month", "Days_left_QTR", "Seconds_left_month"]
    for col in df.columns:
        if col not in minimum:
            del df[col]
    # df.dropna(inplace=True)

    # Sort columns alphabetically and put TimeTaken first
    df = df.reindex_axis(sorted(df.columns), axis=1)
    y = df.pop("TimeTaken")
    df = pd.concat([y, df], axis=1)

    if d["user"] == "Kieron":
        df.to_csv(d["file_location"] + d["output_file"] + ".csv", index=False)  # export file
        out_file.write("file saved as " + d["file_location"] + d["output_file"] + ".csv" + "\n")
    else:
        df.to_csv(d["file_location"] + "vw_Incident_cleaned" + d["output_file"] + ".csv", index=False)  # export file
        out_file.write("file saved as " + d["file_location"] + "vw_Incident_cleaned" + d["output_file"] + ".csv" + "\n")

    out_file.write("clean_Incident complete")
    out_file.close()
    print("clean_Incident complete")


def clean_AuditHistory(d, newpath):

    print("clean_AuditHistory started")
    out_file_name = newpath + time.strftime("%H.%M.%S") + "_clean_AuditHistory.txt"  # Log file name
    out_file = open(out_file_name, "w")  # Open log file
    out_file.write("Date and time: " + time.strftime("%Y%m%d-%H%M%S") + "\n")
    out_file.write("clean_AuditHistory started" + "\n\n")
    if d["user"] == "Kieron":
        df = pd.read_csv(d["file_location"] + d["input_file"] + ".csv", encoding='latin-1', low_memory=False)
    else:
        df = pd.read_csv(d["file_location"] + "vw_AuditHistory" + d["input_file"] + ".csv", encoding='latin-1',
                         low_memory=False)

    # Create Time Variable
    # df = time_taken(df, out_file, "Created_On", "Modified_On")
    # todo - to_datetime not working for audit history

    # Domain knowledge processing
    del df["TimeStamp"]  # Not using TimeStamp
    df = one_hot_encoding(df, "Action", out_file)

    # Data mining processing - where there is not enough meaningful information
    df = min_entries(df, out_file)  # Delete columns that have less than x=3 entries
    df = min_variable_types(df, out_file)  # Delete columns with less than x=2 variable types in that column
    df = drop_null(df, out_file)  # Remove columns where there is a proportion of NULL,NaN,blank values > tol
    df = drop_zeros(df, out_file)  # Remove columns where there is a proportion of 0 values greater than tol
    df = drop_ones(df, out_file)  # Remove columns where there is a proportion of 1 values greater than tol

    dfstageid = pd.read_csv(d["file_location"] + "vw_StageTable" + d["input_file"] + ".csv", encoding='latin-1', low_memory=False)

    # replace NewValue and OldValue with their respective StageNames from the StageID table
    dfstageid["StageId"] = dfstageid["StageId"].str.lower()  # upper case only in stage ID table
    # Give new names so they can be deleted after the merge
    dfstageidnew = dfstageid.rename(columns={'StageId': 'NewStageId', 'StageName': 'NewValueName'})
    dfstageidold = dfstageid.rename(columns={'StageId': 'OldStageId', 'StageName': 'OldValueName'})
    df = df.merge(dfstageidnew, how="left", left_on='NewValue', right_on='NewStageId')
    df = df.merge(dfstageidold, how="left", left_on='OldValue', right_on='OldStageId')
    to_be_deleted = ["NewValue", "OldValue", "NewStageId", "OldStageId"]
    for item in to_be_deleted:
        del df[item]

    new_cols = ["NewValueName", "OldValueName"]
    fill_nulls_dfcs(df, new_cols, "mode", out_file)
    for col in new_cols:
        df = one_hot_encoding(df, col, out_file)

    if d["user"] == "Kieron":
        df.to_csv(d["file_location"] + d["output_file"] + ".csv", index=False)  # export file
        out_file.write("file saved as " + d["file_location"] + d["output_file"] + ".csv" + "\n")
    else:
        df.to_csv(d["file_location"] + "vw_AuditHistory_cleaned" + d["output_file"] + ".csv", index=False)  # export file
        out_file.write("file saved as " + d["file_location"] + "vw_AuditHistory_cleaned" + d["output_file"] + ".csv" + "\n")

    out_file.write("clean_AuditHistory complete")
    out_file.close()
    print("clean_AuditHistory complete")


def clean_HoldActivity(d, newpath):

    print("clean_HoldActivity started")

    out_file_name = newpath + time.strftime("%H.%M.%S") + "_clean_HoldActivity.txt"  # Log file name
    out_file = open(out_file_name, "w")  # Open log file
    out_file.write("Date and time: " + time.strftime("%Y%m%d-%H%M%S") + "\n")
    out_file.write("clean_HoldActivity started" + "\n\n")

    if d["user"] == "Kieron":
        df = pd.read_csv(d["file_location"] + d["input_file"] + ".csv", encoding='latin-1',
                         low_memory=False)
    else:
        df = pd.read_csv(d["file_location"] + "vw_HoldActivity" + d["input_file"] + ".csv", encoding='latin-1',
                         low_memory=False)

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
    # Map to ordinal variables - need to decide which ones we want
    df = one_hot_encoding(df, "HoldTypeName", out_file)
    df = one_hot_encoding(df, "Reason", out_file)
    df = one_hot_encoding(df, "AssignedToGroup", out_file)

    # todo - fill TimeZoneRuleVersionNumber nulls
    fill_nulls_dfcs(df, ["TimeZoneRuleVersionNumber"], "mode", out_file)
    # fill_nulls_dfcs(df, dfcs, fill_value, out_file)

    # todo combine the transactions into their respective cases?
    # delete for now, not sure what to do with it..
    # del df["ParentCase"]

    if d["user"] == "Kieron":
        df.to_csv(d["file_location"] + d["output_file"] + ".csv", index=False)  # export file
        out_file.write("file saved as " + d["file_location"] + d["output_file"] + ".csv" + "\n")
    else:
        df.to_csv(d["file_location"] + "vw_HoldActivity_cleaned" + d["output_file"] + ".csv", index=False)  # export file
        out_file.write("file saved as " + d["file_location"] + "vw_HoldActivity_cleaned" + d["output_file"] + ".csv" + "\n")

    out_file.write("clean_HoldActivity complete")
    out_file.close()
    print("clean_HoldActivity complete")


def clean_PackageTriageEntry(d, newpath):

    print("clean_PackageTriageEntry started")

    out_file_name = newpath + time.strftime("%H.%M.%S") + "_clean_PackageTriageEntry.txt"  # Log file name
    out_file = open(out_file_name, "w")  # Open log file
    out_file.write("Date and time: " + time.strftime("%Y%m%d-%H%M%S") + "\n")
    out_file.write("clean_PackageTriageEntry started" + "\n\n")

    if d["user"] == "Kieron":
        df = pd.read_csv(d["file_location"] + d["input_file"] + ".csv", encoding='latin-1',
                         low_memory=False)
    else:
        df = pd.read_csv(d["file_location"] + "vw_PackageTriageEntry" + d["input_file"] + ".csv", encoding='latin-1',
                         low_memory=False)

    # Create Time Variable
    # df = time_taken(df, out_file, "Created_On", "Modified_On")
    # todo - to_datetime not working

    # Domain knowledge processing
    # Not using TimeStamp
    del df["PCNStatus"]
    del df["SAPStatus"]
    del df["StateCode"]
    del df["StatusCode"]

    # Data mining processing - where there is not enough meaningful information
    df = min_entries(df, out_file)  # Delete columns that have less than x=3 entries
    df = min_variable_types(df, out_file)  # Delete columns with less than x=2 variable types in that column
    df = drop_null(df, out_file)  # Remove columns where there is a proportion of NULL,NaN,blank values > tol
    df = drop_zeros(df, out_file)  # Remove columns where there is a proportion of 0 values greater than tol
    df = drop_ones(df, out_file)  # Remove columns where there is a proportion of 1 values greater than tol

    df = one_hot_encoding(df, "EntryType", out_file)
    df = one_hot_encoding(df, "EntryLevel", out_file)
    df = one_hot_encoding(df, "EntryProcess", out_file)

    # df = fill_nulls(df, "EntryProcess", out_file)  # Fill in NULL values with 0s

    if d["user"] == "Kieron":
        df.to_csv(d["file_location"] + d["output_file"] + ".csv", index=False)  # export file
        out_file.write("file saved as " + d["file_location"] + d["output_file"] + ".csv" + "\n")
    else:
        df.to_csv(d["file_location"] + "vw_PackageTriageEntry_cleaned" + d["output_file"] + ".csv", index=False)  # export file
        out_file.write(
            "file saved as " + d["file_location"] + "vw_PackageTriageEntry_cleaned" + d["output_file"] + ".csv" + "\n")

    out_file.write("clean_PackageTriageEntry complete")
    out_file.close()
    print("clean_PackageTriageEntry complete")


"""****************************************************************************
Run All Code
****************************************************************************"""


if __name__ == "__main__":  # Run program
    d = {}
    with open(parameters, "r") as f:
        for line in f:
            line = line.replace(":", "")
            (key, val) = line.split()
            d[key] = val
            # print(key)

    newpath = r"../0. Results/" + d["user"] + "/prepare_dataset/" + time.strftime("%Y.%m.%d/")  # Log file location
    if not os.path.exists(newpath):
        os.makedirs(newpath)  # Make folder for storing results if it does not exist

    if d["clean_Incident"] == "y":
        clean_Incident(d, newpath)
    if d["clean_AuditHistory"] == "y":
        clean_AuditHistory(d, newpath)
    if d["clean_HoldActivity"] == "y":
        clean_HoldActivity(d, newpath)
    if d["clean_PackageTriageEntry"] == "y":
        clean_PackageTriageEntry(d, newpath)

    copyfile(parameters, newpath + "/" + time.strftime("%H.%M.%S") + "_parameters.txt")  # Save params
