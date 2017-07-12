"""
Used in model.py
Add in at some point
"""

import pandas as pd

# given a threshold, return the columns which are less important to be deleted
def select_importants(csvfile, thesh):
    dfimportances = pd.read_csv(csvfile, encoding='latin-1', low_memory=False)
    cols_to_be_deleted = dfimportances["Columns"][dfimportances["importances"] < thesh].values.tolist()
    return cols_to_be_deleted

# given N columns, keep the top k important columns, return the bottom N-k columns to be deleted
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

# trim the df of the cols which are least important and are to be deleted, returning a new df
def trim_df(df, cols_to_be_deleted):
    newdf = df.copy()
    for col in cols_to_be_deleted:
        del newdf[col]
    return newdf

if __name__ == "__main__":  # used for testing
    select_top_k_importants("../0. Results/Kieron/model/2017.07.05/11.47.45/importances.csv", 20)