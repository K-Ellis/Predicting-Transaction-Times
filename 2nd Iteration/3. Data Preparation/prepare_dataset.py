import pandas as pd

df = pd.read_csv("../../../Data/vw_Incident.csv", encoding='latin-1')

# one entry
del df["SubSubReason"]
# <3 entries
del df["Referencesystem"]

# Program column: only interested in Enterprise
df = df[df.Program == "Enterprise"]

# only one variable type
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
del df["IsoCurrencyCode"]
del df["Language"]
del df["Totaltime"]

# not enough data
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
df = df[df.LanguageName == "English"]

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

# create new dataframe, df2 to store answer
df["Created_On"] = pd.to_datetime(df["Created_On"])
df["ResolvedDate"] = pd.to_datetime(df["ResolvedDate"])
df2 = pd.DataFrame()
df2["TimeTaken"] = (df["ResolvedDate"] - df["Created_On"]).astype(
    'timedelta64[m]')
del df["Created_On"]
del df["ResolvedDate"]

# # get x
# x = pd.DataFrame()
# for col in list(df):
#     dummies = pd.get_dummies(df[col]).iloc[:, 1:]
#     x = pd.concat([x, dummies], axis = 1)

df = pd.concat([df2, df], axis=1)



print(df.head())
df.to_csv("../../../Data/vw_Incident_cleaned.csv", index = False)