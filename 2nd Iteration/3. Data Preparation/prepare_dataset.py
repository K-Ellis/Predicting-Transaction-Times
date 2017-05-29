import pandas as pd

df = pd.read_csv("../../../Data/vw_Incident.csv", encoding='latin-1')

del df["SubSubReason"]
del df["Referencesystem"]

df.to_csv("../../../Data/vw_Incident_cleaned.csv", index = False)