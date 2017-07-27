"""*********************************************************************************************************************
UCD MSc Business Analytics Capstone Project - Predicting Transactions Times
************************************************************************************************************************
Data pre-processing program
Version 00
************************************************************************************************************************
Eoin Carroll
Kieron Ellis
*********************************************************************************************************************"""


# Import libraries
import pandas as pd
import time
import datetime
from calendar import monthrange
from sklearn.utils import resample
from sklearn import preprocessing
from shutil import copyfile  # Used to copy parameters file to directory


def fill_nulls_dfcs(df, dfcs, fill_value): # Fill in Nulls given a set of dataframe columns
    for dfc in dfcs:
        if fill_value == "mode":
            df[dfc].fillna(df[dfc].mode()[0], inplace=True)
        if fill_value == "mean":
            df[dfc].fillna(df[dfc].mean(), inplace=True)


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


def custom_month_end(date):  # custom month end
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


def time_taken(df):  # replace start & finish with one new column, "TimeTaken"
    df["Created_On"] = pd.to_datetime(df["Created_On"])
    df["ResolvedDate"] = pd.to_datetime(df["ResolvedDate"])
    df2 = pd.DataFrame()  # create new dataframe, df2, to store answer  # todo - no need for df2
    df2["TimeTaken"] = (df["ResolvedDate"] - df["Created_On"]).astype('timedelta64[s]')
    df = pd.concat([df2, df], axis=1)
    print("TimeTaken added")
    return df


def transform_country(dfc):  # Convert country into continents
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
    return dfc


def scale_quant_cols(df, quant_cols):  # Scale quantitative variables
    df_num = df[quant_cols]
    for col in quant_cols:
        del df[col]  # todo del df[quant_cols]???
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(df_num)
    df_x_scaled = pd.DataFrame(x_scaled)
    df_x_scaled.columns = df_num.keys().tolist()
    df.reset_index(drop=True, inplace=True)
    df = pd.concat([df_x_scaled, df], axis=1)
    return df


def days_left_in_quarter(date):
    # Function found on stack overflow
    # https://stackoverflow.com/questions/37471704/how-do-i-get-the-correspondent-day-of-the-quarter-from-a-date-field
    q2 = (datetime.datetime.strptime("4/1/{0:4d}".format(date.year), "%m/%d/%Y")).timetuple().tm_yday
    q3 = (datetime.datetime.strptime("7/1/{0:4d}".format(date.year), "%m/%d/%Y")).timetuple().tm_yday
    q4 = (datetime.datetime.strptime("10/1/{0:4d}".format(date.year), "%m/%d/%Y")).timetuple().tm_yday
    q5 = (datetime.datetime.strptime("12/31/{0:4d}".format(date.year), "%m/%d/%Y")).timetuple().tm_yday + 1  # 1st Jan

    cur_day = date.timetuple().tm_yday
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


def deletions(df):
    ####################################################################################################################
    # Domain knowledge processing
    ####################################################################################################################
    del df["IncidentId"]  # Delete for first iteration
    del df["Receiveddate"]  # Not using received
    del df["CaseRevenue"]  # In local currency. - we have all values in USD
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

    del df["TicketNumber"]

    return df


def clean_data(d):
    df = pd.read_csv(d["file_location"] + d["input_file"] + ".csv", encoding='latin-1', low_memory=False)
    print("Data read in, dataframe shape:", df.shape)

    if d["resample"] == "y":  # Resample option - select a smaller sample from dataset
        df = resample(df, n_samples=int(d["n_samples"]), random_state=int(d["seed"]))
        print("Dataset resampled, dataframe shape:", df.shape)

    ####################################################################################################################
    # Filtering for the data MS want
    ####################################################################################################################
    df["ResolvedDate"].fillna(2, inplace=True)  # Remove any cases with no resolved date
    df = df[df["ResolvedDate"] != 2]
    print("Unresolved cases removed, dataframe shape:", df.shape)

    df = df[df["Program"] == "Enterprise"]  # Program column: only interested in Enterprise
    df = df[df["LanguageName"] == "English"]  # Only keep the rows which are English
    df = df[df["StatusReason"] != "Rejected"]  # Remove StatusReason = rejected
    df = df[df["ValidCase"] == 1]  # Remove ValidCase = 0
    print("Filtering Program/Language/Status/Valid, dataframe shape:", df.shape)

    ####################################################################################################################
    # Generate Concurrent_open_cases variable
    ####################################################################################################################
    if d["Concurrent_open_cases"] == "y":
        df["Concurrent_open_cases"] = 0  # Add number of cases that were open at the same time
        for i in range(len(df)):
            df.loc[i, "Concurrent_open_cases"] = len(df[(df.Created_On < df.iloc[i]["Created_On"]) & (
                df.ResolvedDate > df.iloc[i]["ResolvedDate"])])
        print("Concurrent_open_cases added")

    ####################################################################################################################
    # Date and time - calculate time taken and time remaining before month and Qtr end
    ####################################################################################################################
    df = time_taken(df)  # Create Time Variable

    if d["Days_left_Month"] == "y":
        df["Days_left_Month"] = df["Created_On"].apply(lambda x: int(days_left_in_month(x)))  # Day of the month
        print("Days_left_Month added")
    if d["Days_left_QTR"] == "y":
        df["Days_left_QTR"] = df["Created_On"].apply(lambda x: int(days_left_in_quarter(x)))  # Day of the Qtr
        print("Days_left_QTR added")
    if d["Seconds_left_month"] == "y":
        df["Seconds_left_month"] = df["Created_On"].apply(
            lambda x: int(seconds_left_in_month(x)))  # seconds to month end
        print("Seconds_left_month added")

    ####################################################################################################################
    # Combine based on ticket numbers
    ####################################################################################################################
    if d["append_HoldDuration"] == "y":
        dfholdactivity = pd.read_csv("../../../Data/vw_HoldActivity.csv", encoding='latin-1', low_memory=False)
        dfholdactivity["TicketNumber"] = [x.lstrip('5-') for x in dfholdactivity["TicketNumber"]]
        dfholdactivity["TicketNumber"] = dfholdactivity["TicketNumber"].astype(int)
        columns = ["TicketNumber", "HoldDuration", "HoldTypeName", "AssignedToGroup"]
        for col in dfholdactivity:
            if col not in columns:
                del dfholdactivity[col]

        dfdummies = pd.get_dummies(data=dfholdactivity, columns=["HoldTypeName", "AssignedToGroup"])
        dfdummies["HoldCount"] = 1

        unique_tickets = pd.DataFrame(dfholdactivity["TicketNumber"].unique().tolist(), columns=["TicketNumber"])

        columns_to_count = ["HoldDuration", "HoldTypeName_3rd Party", "HoldTypeName_Customer",
                            "HoldTypeName_Internal",
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

        for col in columns_to_count:  # fill the NANs with 0's
            df[col].fillna(0, inplace=True)

    if d["append_AuditDuration"] == "y":
        dfaudithistory = pd.read_csv("../../../Data/vw_AuditHistory.csv", encoding='latin-1', low_memory=False)
        dfaudithistory["TicketNumber"] = [x.lstrip('5-') for x in dfaudithistory["TicketNumber"]]
        dfaudithistory["TicketNumber"] = dfaudithistory["TicketNumber"].astype(int)

        dfaudithistory["Created_On"] = pd.to_datetime(dfaudithistory["Created_On"])

        dfaudithistory_uniqueticketsonly = pd.DataFrame(dfaudithistory["TicketNumber"].unique(),
                                                        columns=["TicketNumber"])
        dfaudithistory_uniqueticketsonly["AuditDuration"] = None

        for ticket in dfaudithistory_uniqueticketsonly["TicketNumber"].tolist():
            dfaudithistory_uniqueticketsonly.loc[
                dfaudithistory_uniqueticketsonly["TicketNumber"] == ticket, "AuditDuration"] = \
                datetime.timedelta.total_seconds(
                    dfaudithistory.loc[dfaudithistory["TicketNumber"] == ticket, "Created_On"].max() - \
                    dfaudithistory.loc[dfaudithistory["TicketNumber"] == ticket, "Created_On"].min())

        # merge new dfduration df with dfincident based on ticket number
        df = df.merge(dfaudithistory_uniqueticketsonly, how='left', left_on='TicketNumber', right_on='TicketNumber')
        df["AuditDuration"].fillna(0, inplace=True)  # fill the NANs with 0's

    ####################################################################################################################
    # Deletions
    ####################################################################################################################
    if d["del_Unnamed"] == "y":
        del df["Unnamed: 69"]
    if d["delete_created_resolved"] == "y":
        del df["Created_On"]
        del df["ResolvedDate"]
    df = deletions(df)  # Use deletions function to clear lots of columns  #todo only keep ones we want instead

    ####################################################################################################################
    # Filtering todo add in parameters
    ####################################################################################################################
    # mean_time = sum(df["TimeTaken"].tolist()) / len(df["TimeTaken"])  # Calculate mean of time taken
    # std_time = np.std(df["TimeTaken"].tolist())  # Calculate standard deviation of time taken
    # df = df[df["TimeTaken"] < (mean_time + 3*std_time)]  # Remove outliers that are > 3 std from mean
    # df = df[df["TimeTaken"] < 2000000]  # Remove outliers that are > 2000000

    ####################################################################################################################
    # IsSox case transformation to filter na's
    ####################################################################################################################
    df["IsSOXCase"].fillna(2, inplace=True)
    df.IsSOXCase = df.IsSOXCase.astype(int)
    df = df[df["IsSOXCase"] != 2]

    df["Numberofreactivations"].fillna(0, inplace=True)
    df.Numberofreactivations = df.Numberofreactivations.astype(int)  # Also convert to ints

    ####################################################################################################################
    # Ordinal variable mapping. Priority, Complexity, StageName
    ####################################################################################################################
    df["Priority"] = df["Priority"].map({"Low": 0, "Normal": 1, "High": 2, "Immediate": 3})
    df["Complexity"] = df["Complexity"].map({"Low": 0, "Medium": 1, "High": 2})
    df["StageName"] = df["StageName"].map({"Ops In": 0, "Triage And Validation": 1, "Data Entry": 2, "Submission": 3,
                                           "Ops Out": 4})

    ####################################################################################################################
    # Fill Categorical and numerical nulls. Scale numerical variables.
    ####################################################################################################################
    quant_cols = ["AmountinUSD", "Priority", "Complexity", "StageName"]  # todo confirm seconds
    if d["append_HoldDuration"] == "y":
        quant_cols.append("HoldDuration")
    if d["append_AuditDuration"] == "y":
        quant_cols.append("AuditDuration")
    if d["Concurrent_open_cases"] == "y":
        quant_cols.append("Concurrent_open_cases")
    if d["Days_left_Month"] == "y":
        quant_cols.append("Days_left_Month")
    if d["Days_left_QTR"] == "y":
        quant_cols.append("Days_left_QTR")
    if d["Seconds_left_month"] == "y":
        quant_cols.append("Seconds_left_month")

    exclude_from_mode_fill = quant_cols
    dfcs = find_dfcs_with_nulls_in_threshold(df, None, None, exclude_from_mode_fill)
    fill_nulls_dfcs(df, dfcs, "mode")
    fill_nulls_dfcs(df, ["AmountinUSD", "Priority", "Complexity", "StageName"], "mean")
    print("fill_nulls_dfcs done")

    df = scale_quant_cols(df, quant_cols)
    print("scale_quant_cols done")

    ####################################################################################################################
    # Transform countries into continents
    ####################################################################################################################
    transform_countrys = ["CountrySource", "CountryProcessed", "SalesLocation"]
    for column in transform_countrys:
        df[column] = transform_country(df[column])
    print("Transformed countries into continents")

    ####################################################################################################################
    # Queue: Prepare for one hot encoding in buckets
    ####################################################################################################################
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
            df.Queue = df.Queue.replace(cat_list[i],
                                        substr_list[i])  # Replace the categorical variables in Queue with
            # the substrings
    extra_queues = ["<WWCS - EMEA Admin>", "<3P Xbox AR Operations>", "<VL OpsPM program support>",
                    "<CLT Duplicates>"]
    # combine uncommon queue types
    for extra in extra_queues:
        if extra in df["Queue"].values:
            df["Queue"] = df["Queue"].replace(extra, "Other")

    ####################################################################################################################
    # One-hot encode categorical variables
    ####################################################################################################################
    cat_vars_to_one_hot = ["CountrySource", "CountryProcessed", "SalesLocation", "StatusReason", "SubReason",
                           "ROCName", "sourcesystem", "Source", "Revenutype", "Queue"]
    for var in cat_vars_to_one_hot:
        df = pd.concat([df, pd.get_dummies(df[var], prefix=var, drop_first=True)], axis=1)
        del df[var]
    print("One hot encoding complete")

    ####################################################################################################################
    # If we only want minimum data
    ####################################################################################################################
    if d["minimum_data"] == "y":
        minimum = ["TimeTaken", "Concurrent_open_cases", "Days_left_Month", "Days_left_QTR", "Seconds_left_month"]
        # todo - include seconds to qtr/month
        for col in df.columns:
            if col not in minimum:
                del df[col]
        print("Minimum data only - all other columns deleted")

    ####################################################################################################################
    # Sort columns alphabetically and put TimeTaken first, export file
    ####################################################################################################################
    df = df.reindex_axis(sorted(df.columns), axis=1)
    df = pd.concat([df.pop("TimeTaken"), df], axis=1)
    df.to_csv(d["file_location"] + d["output_file"] + ".csv", index=False)  # export file


if __name__ == "__main__":  # Run program
    print("Cleaning dataset", time.strftime("%Y.%m.%d"), time.strftime("%H.%M.%S"))
    parameters = "../../../Data/parameters.txt"  # Parameters file
    d = {}  # Read in parameters from file
    with open(parameters, "r") as f:
        for line in f:
            line = line.replace(":", "")
            (key, val) = line.split()
            d[key] = val

    clean_data(d)  # Carry out pre-processing
    copyfile(parameters, "../../../Data/" + time.strftime("%Y.%m.%d.%H.%M.%S") + "_parameters.txt")  # save params
    print("Cleaned file saved as " + d["file_location"] + d["output_file"] + ".csv")