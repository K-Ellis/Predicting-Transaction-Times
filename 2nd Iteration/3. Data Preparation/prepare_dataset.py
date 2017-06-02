import pandas as pd

def clean_Incident():

    print("clean_Incident started")

    df = pd.read_csv("../../../Data/vw_Incident.csv", encoding='latin-1', low_memory=False)

    # one entry
    # todo write code to delete columns where there are less than 3 entries instead of naming headings - more scalable
    del df["SubSubReason"]
    # <3 entries
    del df["Referencesystem"]

    # Program column: only interested in Enterprise
    df = df[df.Program == "Enterprise"]

    # only one variable type
    # todo write code to delete columns where there are only one variable type
    del df["BusinessFunction"]
    del df["LineOfBusiness"]
    del df["CaseType"]
    del df["CaseSubTypes"]
    del df["Reason"]
    del df["ProcessName"]
    del df["StateCode"]

    # combine the transactions into their respective cases?
    # delete for now, not sure what to do with it..
    del df["ParentCase"]

    # combine variables with less than 100 entries into one variable, call it
    # "other" or something
    # CountrySource
    # CountryProcessed
    # SalesLocation
    # CurrencyName
    # sourcesystem
    # Workbench
    # Revenutype

    # duplicate column
    del df["CurrencyName"]
    del df["LanguageName"]
    del df["Totaltime"]

    # not enough data
    # todo convert this into code - might be different for new datasets (maybe use <50 entries)
    del df["RevenueImpactAmount"] # < 50 entries
    del df["IsAudited"] # 4 true audited entries
    del df["Auditresult"] # 1 true audit result
    del df["PendingRevenue"] # 13 true results
    del df["Requestspercase"] # 5 true results
    del df["RequiredThreshold"] # 11 true results
    del df["Slipped"] # 11 true results
    del df["DefectiveCase"] # 33 true results
    del df["Isrevenueimpacting"] # 13 true results

    # only keep the rows which are English (12912 English entries)
    # todo couldn't get this line to work
    # df = df[df.LanguageName == "English"]

    # don't understand what it does
    del df["caseOriginCode"]
    del df["WorkbenchGroup"]
    del df["RejectionSubReason"]
    del df["PackageNumber"]
    del df["Numberofreactivations"]
    del df["NumberofChildIncidents"]

    # seems unnecessary
    del df["pendingemails"]

    # change to binary, either 0 or >0
    del df["Totalbillabletime"]

    # all unique entries
    del df["TicketNumber"]
    del df["IncidentId"]

    # replace Created_On, Receiveddate and ResolvedDate with one new column,
    # "TimeTaken"
    del df["Receiveddate"]
    df["Created_On"] = pd.to_datetime(df["Created_On"])
    df["ResolvedDate"] = pd.to_datetime(df["ResolvedDate"])
    df2 = pd.DataFrame()# create new dataframe, df2, to store answer
    df2["TimeTaken"] = (df["ResolvedDate"] - df["Created_On"]).astype(
        'timedelta64[m]')
    del df["Created_On"]
    del df["ResolvedDate"]
    df = pd.concat([df2, df], axis=1)

    # replace the Null values with the mean for the column
    df["CaseRevenue"] = df["CaseRevenue"].fillna(df["CaseRevenue"].mean())

    # too many nulls
    del df["RelatedCases"]
    del df["CreditAmount"]
    del df["DebitAmount"]
    del df["OrderAmount"]
    del df["InvoiceAmount"]
    del df["Deleted"]

    # change the priorities to nominal variables
    df["Priority"].map({"Low":0, "Normal":1, "High":2, "Immediate":3})



    # export file
    df.to_csv("../../../Data/vw_Incident_cleaned.csv", index = False)


    # Was getting error "sys:1: DtypeWarning: Columns (16,65) have mixed types. Specify dtype option on import or set low_memory=False."
    # added low_memory=False to read in function
    print("clean_Incident complete")


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
clean_AuditHistory()
clean_HoldActivity()
clean_PackageTriageEntry()