import pandas as pd


def timetaken(df): # replace Created_On, Receiveddate and ResolvedDate with one new column, "TimeTaken"
    del df["Receiveddate"]
    df["Created_On"] = pd.to_datetime(df["Created_On"])
    df["ResolvedDate"] = pd.to_datetime(df["ResolvedDate"])
    df["TimeTaken"] = (df["ResolvedDate"] - df["Created_On"]).astype('timedelta64[m]')
    del df["Created_On"]
    del df["ResolvedDate"]
    return df


def min_entries(df, min=3):  # Delete columns that have less than min entries regardless of number rows
    for column in df:
        if df[column].count() < min:
            # print("Min entries: ", column, df[column].count())
            del df[column]  # delete column
    return df


def min_variable_types(df, min=2):  # Delete columns with less than min variable types in that column
    for column in df:
        if len(df[column].value_counts().index.tolist()) < min:
            # print("Min var type: ", column, len(df[column].value_counts().index.tolist()))
            del df[column]  # delete column
    return df


def drop_NULL(df, x=0.25):  # Remove columns where there is a proportion of NULL,NaN,blank values greater than tol
    # todo - this is very similar to min entries, might be able to combine the two
    # df.dropna(axis=1, thresh=int(len(df) * x)) .... an alternative method I could not get to work
    for column in df:
        if df[column].count() <= len(df) * x:
            # print("Drop Null:", column, df[column].count())
            del df[column]  # delete column
    return df


def drop_zeros(df, x=0.90):  # Remove columns where there is a proportion of 0 values greater than tol
    for column in df:
        if 0 in df[column].value_counts():  # Make sure 0 is in column
            if df[column].value_counts()[0] / len(df) >= x:  # If there are more 0s than out limit
                # print("Drop Zeros:", column, df[column].value_counts()[0] / len(df))
                del df[column]  # delete column
    return df


def drop_ones(df, x=0.90):  # Remove columns where there is a proportion of 1 values greater than tol
    for column in df:
        if 1 in df[column].value_counts():  # Make sure 0 is in column
            if df[column].value_counts()[1] / len(df) >= x:  # If there are more 1s than out limit
                # print("Drop Ones:", column, df[column].value_counts()[0] / len(df))
                del df[column]  # delete column
    return df


def clean_Incident():

    print("clean_Incident started")
    df = pd.read_csv("../../../Data/vw_Incident.csv", encoding='latin-1', low_memory=False)
    # todo - Was getting error:
    # "sys:1: DtypeWarning: Columns (16,65) have mixed types. Specify dtype option on import or set low_memory=False."
    # Solution - Added low_memory=False to read in function. Not sure what this does . . . need to check

    df = timetaken(df)
    df = min_entries(df)  # Delete columns that have less than 3 entries
    df = min_variable_types(df) # Delete columns with less than 2 variable types in that column
    df = drop_NULL(df)
    df = drop_zeros(df)  # todo - all drop zero columns had a ratio of 0.014 . . . . need to look at further
    df = drop_ones(df)

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
    # Revenutype

    # all unique entries
    # todo these are needed to map across sheets, keep for now
    # del df["TicketNumber"]
    # del df["IncidentId"]

    # Program column: only interested in Enterprise
    df = df[df.Program == "Enterprise"]
    # Only keep the rows which are English
    df = df[df.LanguageName == "English"]

    # Duplicate column - we will keep IsoCurrencyCode
    del df["CurrencyName"]

    # del df["LanguageName"]  todo deleting this column causes an error later when we try to filter by English

    # don't understand what it does
    del df["caseOriginCode"]
    del df["WorkbenchGroup"]

    # replace the Null values with the mean for the column
    df["CaseRevenue"] = df["CaseRevenue"].fillna(df["CaseRevenue"].mean())

    # change the priorities to nominal variables
    df["Priority"].map({"Low":0, "Normal":1, "High":2, "Immediate":3})

    # export file
    df.to_csv("../../../Data/vw_Incident_cleaned.csv", index = False)

    print("clean_Incident complete")

    # todo save log file


def clean_AuditHistory():
    print("clean_AuditHistory started")
    print("clean_AuditHistory complete")

def clean_HoldActivity():
    print("clean_HoldActivity started")
    print("clean_HoldActivity complete")


def clean_PackageTriageEntry():
    print("clean_PackageTriageEntry started")
    print("clean_PackageTriageEntry complete")


clean_Incident()
# clean_AuditHistory()
# clean_HoldActivity()
# clean_PackageTriageEntry()