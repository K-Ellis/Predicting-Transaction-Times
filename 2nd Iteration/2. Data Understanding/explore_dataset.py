import pandas as pd

df = pd.read_csv("../../../Data/vw_Incident.csv", encoding='latin-1')
#
# print(df.head())
#
# print(df.describe())

# print(df.iloc[4589,16])
# print(df[df["SubSubReason"] == "RR Contig"])
print(df["SubSubReason"].loc[df["SubSubReason"] == "RR Contig"])

print(df["Referencesystem"].loc[df["Referencesystem"] == "eMSL"])