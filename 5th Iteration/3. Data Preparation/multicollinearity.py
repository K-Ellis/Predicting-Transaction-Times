"""****************************************************************************
UCD MSc Business Analytics Capstone Project - Predicting Transactions Times
*******************************************************************************
Iteration 4
Colinearity testing
*******************************************************************************
Eoin Carroll
Kieron Ellis
*******************************************************************************
Code to remove highly correlated variables
Results were the same as not removing these variables
Plan to come back and test more at the end
****************************************************************************"""

import pandas as pd
import numpy as np

def cols_with_corr_above_threshold(df, threshold):
    corr = df.corr().abs()
    indices = np.where(corr > threshold)
    col_names_indices = []
    for x, y in zip(*indices):
        if x != y and x < y:
            col_names_indices.append((df.columns[x], df.columns[y], x, y, corr[df.columns[x]][df.columns[y]]))
    col_names_indices.sort(key=lambda x: x[4], reverse=True)
    return col_names_indices


def print_corr_names_indices(col_names_indices):
    print("Correlation Score, (Column names), [Indices]:")
    for i, item in enumerate(col_names_indices):
        print("     %s\t- %s, (\"%s\", \"%s\"), [%s, %s]" % (i + 1, item[4], item[0], item[1], item[2], item[3]))


def count_corr_cols_which_appear_multiple_times(col_names_indices):
    count_dict = {}
    for name_set in col_names_indices:
        if name_set[0] not in count_dict:
            count_dict[name_set[0]] = 1
        else:
            count_dict[name_set[0]] += 1

        if name_set[1] not in count_dict:
            count_dict[name_set[1]] = 1
        else:
            count_dict[name_set[1]] += 1
    return count_dict


def print_cols_which_appear_multiple_times(count_dict):
    print("\nColumns which appear more than once above the threshold:")
    for item in count_dict:
        if count_dict[item] > 1:
            print(" -", item, "=", count_dict[item])


def find_corr_cols_to_delete(col_names_indices, count_dict):
    cols_to_delete = []
    for name_set in col_names_indices:
        if name_set[0] not in cols_to_delete and name_set[1] not in cols_to_delete:
            if count_dict[name_set[0]] > count_dict[name_set[0]]:
                cols_to_delete.append(name_set[0])
            else:
                cols_to_delete.append(name_set[1])
    return cols_to_delete


def print_cols_to_delete(cols_to_delete):
    print("\nColumns to be deleted:")
    for i, item in enumerate(cols_to_delete):
        print(i+1, "-", item)


def delete_corr_cols(df, cols_to_delete):
    df_cleaned = df.copy()
    for item in cols_to_delete:
        del df_cleaned[item]
    return df_cleaned


def find_and_delete_corr(df, threshold):
    col_names_indices = cols_with_corr_above_threshold(df, threshold)
    count_dict = count_corr_cols_which_appear_multiple_times(col_names_indices)
    cols_to_delete = find_corr_cols_to_delete(col_names_indices, count_dict)
    df_cleaned = delete_corr_cols(df, cols_to_delete)
    return df_cleaned, cols_to_delete


if __name__ == "__main__":
    COSMIC_num = 2
    df = pd.read_csv("../../../Data/COSMIC_%s/vw_Incident%s_cleaned.csv" % (COSMIC_num, COSMIC_num), encoding='latin-1',
                     low_memory=False)

    collinearity_threshold = 0.9

    col_names_indices = cols_with_corr_above_threshold(df, collinearity_threshold)
    print_corr_names_indices(col_names_indices)

    count_dict = count_corr_cols_which_appear_multiple_times(col_names_indices)
    print_cols_which_appear_multiple_times(count_dict)

    cols_to_delete = find_corr_cols_to_delete(col_names_indices, count_dict)
    print_cols_to_delete(cols_to_delete)

    df_cleaned = delete_corr_cols(df, cols_to_delete)

    df.to_csv("../../../Data/COSMIC_%s/vw_Incident%s_cleaned(collinearity_thresh_%s).csv" % (COSMIC_num, COSMIC_num,
                                                                                collinearity_threshold), index=False)  # export