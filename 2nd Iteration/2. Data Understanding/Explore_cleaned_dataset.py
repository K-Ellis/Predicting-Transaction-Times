import pandas as pd

df = pd.read_csv("../../../Data/vw_Incident_cleaned.csv", encoding='latin-1')

pd.set_option("display.max_columns",50)
print(df.head())