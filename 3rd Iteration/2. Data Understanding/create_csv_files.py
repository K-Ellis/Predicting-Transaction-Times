import pandas as pd

df = pd.read_excel("../../../Data/UCD_Data_20170420_1.xlsx", sheetname=0)
df.to_csv("../../../Data/vw_Incident.csv", index = False)

df = pd.read_excel("../../../Data/UCD_Data_20170420_1.xlsx", sheetname=1)
df.to_csv("../../../Data/vw_HoldActivity.csv", index = False)

df = pd.read_excel("../../../Data/UCD_Data_20170420_1.xlsx", sheetname=2)
df.to_csv("../../../Data/vw_AuditHistory.csv", index = False)

df = pd.read_excel("../../../Data/UCD_Data_20170420_1.xlsx", sheetname=3)
df.to_csv("../../../Data/vw_PackageTriageEntry.csv", index = False)

df = pd.read_excel("../../../Data/UCD_Data_20170420_1.xlsx", sheetname=4)
df.to_csv("../../../Data/vw_StageTable.csv", index = False)