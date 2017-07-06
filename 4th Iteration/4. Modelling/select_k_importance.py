"""
Used in model.py
Add in at some point
"""

import pandas as pd

def select_importants(csvfile, thesh):
    dfimportances = pd.read_csv(csvfile, encoding='latin-1', low_memory=False)
    cols_to_be_deleted = dfimportances["Columns"][dfimportances["importances"] < thesh].values.tolist()
    return cols_to_be_deleted

def select_top_k_importants(csvfile, k):
    dfimportances = pd.read_csv(csvfile, encoding='latin-1', low_memory=False)
    dfimportances.sort_values("importances", ascending=False, inplace=True)
    top = dfimportances["Columns"].iloc[:k].values.tolist()
    bottom = []
    for col in dfimportances["Columns"].values.tolist():
        if col not in top:
            bottom.append(col)
    return bottom
    # cols_to_be_deleted = dfimportances["Columns"][dfimportances["importances"] < thesh].values.tolist()
    # return cols_to_be_deleted

def trim_df(df, cols_to_be_deleted):
    newdf = df.copy()
    for col in cols_to_be_deleted:
        del newdf[col]
    return newdf

if __name__ == "__main__":  # used for testing
    select_top_k_importants("../0. Results/Kieron/model/2017.07.05/11.47.45/importances.csv", 20)