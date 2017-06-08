import pandas as pd
from sklearn import preprocessing

df = pd.read_csv("../../../Data/vw_Incident.csv", encoding='latin-1')

# def find_numerical_cols(df):
#     for col in df:
#         if df[col].dtype != object:
#             print(col, df[col].dtype)

def scale_quant_cols(df, quant_cols):#, outfile):
    # Scale quantitative variables
    df_num = df[quant_cols]
    for col in quant_cols:
        del df[col]

    min_max_scaler = preprocessing.MinMaxScaler()
    # df_num = min_max_scaler.fit_transform(df_num)
    x_scaled = min_max_scaler.fit_transform(df_num)
    df_x_scaled = pd.DataFrame(x_scaled)
    df_x_scaled.columns = df_num.keys().tolist()

    df = pd.concat([df_x_scaled, df], axis=1)
    # out_file.write("columns scaled = " + str(df_num.keys().tolist()) + "\n\n")
    print("columns scaled = " + str(df_num.keys().tolist()) + "\n\n")
    return df

def fill_nulls_dfc(dfc, fill_value):#, out_file):  # Fill in NULL values for
# one column
    dfc.fillna(fill_value, inplace=True)
    # out_file.write("All NULL Values for \"%s\" replaced with most frequent value, %s"%(dfc.name, fill_value) +
    #                "\n\n")

def fill_nulls_dfcs(df, dfcs):#, out_file):
    for dfc in dfcs:
        fill_nulls_dfc(df[dfc], df[dfc].mode()[0])#, out_file)

fill_nulls_dfcs(df, num_cols)
num_cols = ["CaseRevenue", "AmountinUSD"]
df = scale_quant_cols(df, num_cols)

print(df.head())
# print(df.CaseRevenue)

