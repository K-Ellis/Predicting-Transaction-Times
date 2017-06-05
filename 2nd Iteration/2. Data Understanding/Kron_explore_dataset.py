import pandas as pd

# pd.set_option("display.max_columns",66)
df = pd.read_csv("../../../Data/vw_Incident_cleaned.csv", encoding='latin-1')

substr_list = ["NAOC", "EOC", "AOC", "APOC", "LOC", "E&E", "Xbox"]
val_list = df.Queue.value_counts().index.tolist()
cat_list = [[] for item in substr_list]

for i, substr in enumerate(substr_list):
    for j, val in enumerate(val_list):
        if substr in val:
            val_list[j] = "n"
            cat_list[i].append(val)

for i, item in enumerate(cat_list):
    dfseries = df.Queue.isin(item)
    dfseries = dfseries.astype(int)
    dfseries.name = substr_list[i]
    df = pd.concat([dfseries, df], axis=1)

del df["Xbox"]
del df["Queue"]
print(df.head())