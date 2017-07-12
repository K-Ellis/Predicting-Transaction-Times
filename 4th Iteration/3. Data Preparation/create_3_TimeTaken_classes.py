import pandas as pd

def split_df_into_3(bins, df):
    dflist = []
    prev_bin = 0
    for bin0 in bins:
        dflist.append(pd.DataFrame(df.loc[(df["TimeTaken"] < bin0) & (df["TimeTaken"] >= prev_bin)]))
        prev_bin = bin0
    dflist.append(pd.DataFrame(df.loc[df["TimeTaken"] >= bins[-1]]))
    return dflist

df = pd.read_csv("../../../Data/ABT_Incident_HoldDuration_cleaned.csv",encoding='latin-1',low_memory=False)

bins = [92000, 370000]
print(92000/60/60)
print(370000/60/60/24)
split_df_3 = split_df_into_3(bins, df)

print(len(split_df_3))
lengthsdfs = [len(split_df_3[i]) for i in range(len(split_df_3))]
print(lengthsdfs)
print(sum(lengthsdfs))
print(len(df))

split_df_3[0].to_csv("../../../Data/split_small.csv", index=False)
split_df_3[1].to_csv("../../../Data/split_medium.csv", index=False)
split_df_3[2].to_csv("../../../Data/split_large.csv", index=False)

# 1.
    # if TimeTaken is in range(x1, y1) assign the row the label "Short"
    # if TimeTaken is in range(x2, y2) assign the row the label "Medium"
    # if TimeTaken is in range(x3, y3) assign the row the label "Long"

# 2.
    # split the data into these 3 classes, creating 3 csv's

# 3.
    # delete the class labels (to avoid correlation/collinearity with TimeTaken)

# 4.
    # train 3 regression models, one for each class

# 5.
    # create model to classify unseen data into these 3 classes

