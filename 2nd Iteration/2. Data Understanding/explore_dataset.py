import pandas as pd

df = pd.read_csv("../../../Data/vw_Incident.csv", encoding='latin-1')
#
# print(df.head())
#
print(df.iloc[:,0].describe())

# print(df.iloc[4589,16])
# print(df[df["SubSubReason"] == "RR Contig"])

## This only appears once in the dataset, therefore it should probably be
# removed for modelling.
print(df["SubSubReason"].loc[df["SubSubReason"] == "RR Contig"])

## This only appears twice, remove it.
print(df["Referencesystem"].loc[df["Referencesystem"] == "eMSL"])