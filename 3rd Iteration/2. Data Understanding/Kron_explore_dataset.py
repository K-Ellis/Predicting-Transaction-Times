import pandas as pd

pd.set_option("display.max_columns",105)
# df = pd.read_csv("../../../Data/vw_Incident.csv", encoding='latin-1')
df = pd.read_csv("../../../Data/vw_Incident_cleaned.csv", encoding='latin-1')
print(df.head())

