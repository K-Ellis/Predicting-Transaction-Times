import pandas as pd

df = pd.read_csv("../../../Data/vw_Incident_cleaned.csv", encoding='latin-1')

# print(df.CaseRevenue)
# print(df.AmountinUSD)
for col in df:
    print(df[col].isnull().sum())