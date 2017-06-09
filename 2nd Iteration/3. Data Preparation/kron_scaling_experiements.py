import pandas as pd
from sklearn import preprocessing

df = pd.read_csv("../../../Data/vw_Incident.csv", encoding='latin-1')

def scale_quant_cols(df, quant_cols):#, outfile):
    # Scale quantitative variables
    # df_num = df[quant_cols]
    df_num = pd.DataFrame()
    for col in quant_cols:
        df_num = pd.concat([df_num, df[col]], axis=1)
        del df[col]
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(df_num)
    df_x_scaled = pd.DataFrame(x_scaled)
    df_x_scaled.columns = df_num.keys().tolist()

    # print(df_num)
    # print(df_x_scaled)
    df.reset_index(drop=True, inplace=True)

    df = pd.concat([df_x_scaled, df], axis=1)#, join="inner")
    # out_file.write("columns scaled = " + str(df_num.keys().tolist()) + "\n\n")
    # print("columns scaled = " + str(df_num.keys().tolist()) + "\n\n")
    return df

# def scale_quant_cols2(df, quant_cols):#, out_file):  # Scale quantitative variables
#     min_max_scaler = preprocessing.MinMaxScaler()
#     x_scaled = min_max_scaler.fit_transform(df[quant_cols])
#     df_x_scaled = pd.DataFrame(x_scaled)
#     df_x_scaled.columns = df[quant_cols].keys().tolist()
#     df = pd.concat([df_x_scaled, df], axis=1)
#     # out_file.write("columns scaled = " + str(df[quant_cols].keys().tolist()) + "\n\n")
#     return df

def fill_nulls_dfc(dfc, fill_value):#, out_file):
    dfc.fillna(fill_value, inplace=True)
def fill_nulls_dfcs(df, dfcs):#, out_file):
    for dfc in dfcs:
        fill_nulls_dfc(df[dfc], df[dfc].mean())#, out_file)

def find_dfcs_with_nulls_in_threshold(df, min_thres, max_thres, exclude):
    dfcs = []
    if min_thres == None and max_thres == None:
        for col in df.columns:
            if col not in exclude:
                if df[col].isnull().sum() > 0:
                    dfcs.append(col)
    else:
        for col in df.columns:
            if col not in exclude:
                if df[col].isnull().sum() > min_thres and df[col].isnull().sum() < max_thres:
                    dfcs.append(col)
    return dfcs

df = df[df["Program"] == "Enterprise"]  # Program column: only interested in Enterprise
df = df[df["LanguageName"] == "English"]  # Only keep the rows which are English
df = df[df["StatusReason"] != "Rejected"]  # Remove StatusReason = rejected
df = df[df["ValidCase"] == 1]  # Remove ValidCase = 0

quant_cols = ["CaseRevenue", "AmountinUSD"]
find_dfcs_with_nulls_in_threshold(df, None, None, quant_cols)


fill_nulls_dfcs(df, quant_cols)

df = scale_quant_cols(df, quant_cols)

print(len(df))
print(df.CaseRevenue)
# print(df)
