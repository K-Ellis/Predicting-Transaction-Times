import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os  # Used to create folders
from shutil import copyfile  # Used to copy parameters file to directory
from sklearn.metrics import mean_squared_error
import math
from matplotlib.ticker import FuncFormatter

def get_parameters(parameters):
    d = {}
    with open(parameters, "r") as f:
        for line in f:
            line = line.replace(":", "")
            line = line.replace(",", "")
            line = line.split()
            key = line.pop(0)
            if len(line) > 1:
                d[key] = line
            else:
                d[key] = line[0]
    return d
    
def get_dfs(d, i):
    if i is None:
        df1 = pd.read_csv(d["file_location"] + d["input_files"] + ".csv", encoding='latin-1', low_memory=False)
        return df1
    else:
        if d["input_files"][i] == "July":
            df1 = pd.read_csv(d["file_location"] + "man_PreJuly_July_LR_predictions.csv", encoding='latin-1', low_memory=False)
            df2 = pd.read_csv(d["file_location"] + "man_PreJuly_July_EN_predictions.csv", encoding='latin-1', low_memory=False)
            df3 = pd.read_csv(d["file_location"] + "man_PreJuly_July_GBR_predictions.csv", encoding='latin-1', low_memory=False)
            df4 = pd.read_csv(d["file_location"] + "man_PreJuly_July_RFR_predictions.csv", encoding='latin-1', low_memory=False)

            maindf = pd.DataFrame()
            maindf["TimeTaken"] = df1["TimeTaken"]
            maindf["Mean_TimeTaken"] = df1["Mean_TimeTaken"]
            maindf["TimeTaken_LinearRegression"] = df1["TimeTaken_LinearRegression"]
            maindf["TimeTaken_ElasticNet"] = df2["TimeTaken_ElasticNet"]
            maindf["TimeTaken_GradientBoostingRegressor"] = df3["TimeTaken_GradientBoostingRegressor"]
            maindf["TimeTaken_RandomForestRegressor"] = df4["TimeTaken_RandomForestRegressor"]

            return maindf
        elif d["input_files"][i] == "June":
            df1 = pd.read_csv(d["file_location"] + "man_PreJune_June_LR_predictions.csv", encoding='latin-1',
                              low_memory=False)
            df2 = pd.read_csv(d["file_location"] + "man_PreJune_June_EN_predictions.csv", encoding='latin-1',
                              low_memory=False)
            df3 = pd.read_csv(d["file_location"] + "man_PreJune_June_GBR_predictions.csv", encoding='latin-1',
                              low_memory=False)
            df4 = pd.read_csv(d["file_location"] + "man_PreJune_June_RFR_predictions.csv", encoding='latin-1',
                              low_memory=False)

            maindf = pd.DataFrame()
            maindf["TimeTaken"] = df1["TimeTaken"]
            maindf["Mean_TimeTaken"] = df1["Mean_TimeTaken"]
            maindf["TimeTaken_LinearRegression"] = df1["TimeTaken_LinearRegression"]
            maindf["TimeTaken_ElasticNet"] = df2["TimeTaken_ElasticNet"]
            maindf["TimeTaken_GradientBoostingRegressor"] = df3["TimeTaken_GradientBoostingRegressor"]
            maindf["TimeTaken_RandomForestRegressor"] = df4["TimeTaken_RandomForestRegressor"]

            return maindf

        else:
            df1 = pd.read_csv(d["file_location"] + d["input_files"][i] + ".csv", encoding='latin-1', low_memory=False)
    return df1

def get_RMSEs(df, pred_cols):
    RMSEs = []
    for col in pred_cols:
        RMSEs.append(np.sqrt(mean_squared_error(df["TimeTaken"], df[col])))
    return RMSEs

def get_stats(df):
    pred_cols = ["Mean_TimeTaken", "TimeTaken_LinearRegression", "TimeTaken_ElasticNet", "TimeTaken_GradientBoostingRegressor", "TimeTaken_RandomForestRegressor"]

    # pred_cols = ["Mean_TimeTaken", "TimeTaken_RandomForestRegressor"]

    number_close = [[0 for _ in range(24)] for _ in range(len(pred_cols))]
    percent_close = [[] for _ in range(len(pred_cols))]

    interesting_hours = [(x+1)*2 for x in range(24)]
    # todo - 96 / this = what the xticks has to be for pct correct plot

    number_close96 = [0 for _ in range(len(pred_cols))]
    percent_close96 = []

    for i in range(len(df[pred_cols[0]])):
        for k, col in enumerate(pred_cols):
            if abs(df[col].iloc[i] - df["TimeTaken"].iloc[i]) <= 48:  # Within 1 hour
                number_close96[k] += 1

        for j, hour in enumerate(interesting_hours):
            for k, col in enumerate(pred_cols):
                if abs(df[col].iloc[i] - df["TimeTaken"].iloc[i]) <= hour:  # Within 1 hour
                    number_close[k][j] += 1

    for i in range(len(number_close)):
        for j in number_close[i]:
            percent_close[i].append(j / len(df["TimeTaken"]) * 100)

    for i in number_close96:
        percent_close96.append(i / len(df["TimeTaken"]) * 100)

    RMSEs = get_RMSEs(df, pred_cols)
    return percent_close, percent_close96, RMSEs
    
def multi_plot_pct_correct(ys, newpath, d, title):
    fig, ax = plt.subplots(dpi=300) # figsize=(3.841, 7.195),
    ys_plot = ys.copy()
    for i in range(len(ys_plot)):
        ys_plot[i] = [0] + ys[i]

    x = [x*2 for x in range(len(ys_plot[0]))]
    # todo - change to reflect how many datapoints there are with np.where

    alg_initial_list = ["B", "LR", "EN", "GBR", "RFR"]
    colours = [".m--", ".r-", ".g-", ".y-", ".b-"]

    for y, alg_initial, colour in zip(ys_plot, alg_initial_list, colours):
        plt.plot(x, y, colour, label=alg_initial, alpha=0.8)

    plt.title("%% Correct Predictions +/- Given Time (%s)" % (title))

    plt.xlim(0, 48)
    plt.ylim(0, 100)
    
    # plt.legend(loc=5, bbox_to_anchor=(1.2, 0.5))
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], title='Algorithms', loc=5, bbox_to_anchor=(1.21, 0.5))

    xticks = [(x + 1) * 4 for x in range(12)]

    plt.xticks(xticks, xticks)

    yticks_i = [i * 10 for i in range(11)]
    yticks = ["0%", "10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%", "90%", "100%"]
    plt.yticks(yticks_i, yticks)

    # plt.grid()
    plt.xlabel("Time Buckets (hours)")
    plt.ylabel("% Correct Predictions")
    
    if not os.path.exists(newpath + "PDFs/"):
        os.makedirs(newpath + "PDFs/")  # Make folder for storing results if it does not exist

    plt.savefig("%s%s_pct_correct.png" % (newpath, title), bbox_inches='tight', dpi=300)
    plt.savefig("%sPDFs/%s_pct_correct.pdf" % (newpath, title), bbox_inches='tight')

    try:
        get_ipython
        plt.show()
    except:
        print("excepting")
        plt.close()

    fig, ax = plt.subplots(dpi=300)

    x = [x*2 for x in range(len(ys_plot[0]))]
    # todo - change to reflect how many datapoints there are with np.where

    alg_initial_list = ["Average Case Processing Time", "LR", "EN", "GBR", "Recommended Model"]
    colours = [".m--", ".r-", ".g-", ".y-", ".b-"]

    for y, alg_initial, colour in zip(ys_plot, alg_initial_list, colours):
        if alg_initial != "LR" and alg_initial != "EN" and alg_initial != "GBR" :
            plt.plot(x, y, colour, label=alg_initial, alpha=0.8)

    plt.title("% Correct Predictions +/- Given Time") # (%s)" % (title))

    plt.xlim(0, 48)
    plt.ylim(0, 100)
    
    # plt.legend(bbox_to_anchor=(1.2, 0.5), loc="center", fontsize=12)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc="center", fontsize=12)#, loc=5, bbox_to_anchor=(1.21, 0.5))

    xticks = [(x + 1) * 8 for x in range(6)]
    # xticks = [(x + 1) * 4 for x in range(12)]

    plt.xticks(xticks, xticks)

    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y / 100)))
    # yticks_i = [i * 10 for i in range(11)]
    # yticks = ["0%", "10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%", "90%", "100%"]
    # plt.yticks(yticks_i, yticks)

    # plt.grid()
    plt.xlabel("Time (hours)")
    plt.ylabel("% of Correct Predictions")
    
    if not os.path.exists(newpath + "PDFs/"):
        os.makedirs(newpath + "PDFs/")  # Make folder for storing results if it does not exist

    plt.savefig("%s%s_pct_correct_B_RF.png" % (newpath, title), bbox_inches='tight', dpi=300)
    plt.savefig("%sPDFs/%s_pct_correct_B_RF.pdf" % (newpath, title), bbox_inches='tight')

    try:
        get_ipython
        plt.show()
    except:
        plt.close()

def multi_plot_RMSEs_bar(ys, newpath, d, title, actual_title):
    fig, ax = plt.subplots(dpi=300)
    x = range(len(ys[0]))
    x = np.array(x)
    ys = np.array(ys)

    alg_initial_list = ["B", "LR", "EN", "GBR", "RFR"]
    colours = ["#F1948A", "#85C1E9"]
    if len(ys) == 3:
        colours += ["#F7DC6F"] #b30000 AF000E


    width = len(x)/(len(x) * (len(title)+1))
    newwidth = width
    for y, colour, exp in zip(ys, colours, title):
        plt.bar(x+newwidth, y, width, color=colour, label=exp, alpha=1, align="center")
        newwidth += width

    plt.title(actual_title)

    plt.legend(loc=5, bbox_to_anchor=(1.2, 0.5))

    plt.xticks(x+(width*(len(title)+1) /2), alg_initial_list)

    plt.xlabel("Algorithms")
    plt.ylabel("RMSE (hours)")
    if not os.path.exists(newpath + "PDFs/"):
        os.makedirs(newpath + "PDFs/")  # Make folder for storing results if it does not exist

    plt.savefig("%s%s_RMSEs_bar.png" % (newpath, title), bbox_inches='tight', dpi=300)
    plt.savefig("%sPDFs/%s_RMSEs_bar.pdf" % (newpath, title), bbox_inches='tight')

    try:
        get_ipython
        plt.show()
    except:
        plt.close()

def multi_plot_RMSEs_line(ys, newpath, d, title):
    fig, ax = plt.subplots(dpi=300)
    x = range(len(title))
    x = np.array(x)
    ys = np.array(ys)

    alg_initial_list = ["B", "LR", "EN", "GBR", "RFR"]
    colours = [".m--", ".r-", ".g-", ".y-", ".b-"]

    for y, alg_initial, colour in zip(ys, alg_initial_list, colours):
        plt.plot(x, y, colour, label=alg_initial, alpha=0.8)

    plt.title("RMSE with Decreasing Features")

    plt.legend(loc=5, bbox_to_anchor=(1.2, 0.5))

    plt.xticks(x, title)

    plt.xlabel("Decreasing Features")
    plt.ylabel("RMSE (hours)")

    if not os.path.exists(newpath + "PDFs/"):
        os.makedirs(newpath + "PDFs/")  # Make folder for storing results if it does not exist

    plt.savefig("%s%s_RMSEs_line.png" % (newpath, title), bbox_inches='tight', dpi=300)
    plt.savefig("%sPDFs/%s_RMSEs_line.pdf" % (newpath, title), bbox_inches='tight')

    try:
        get_ipython
        plt.show()
    except:
        plt.close()

def multi_plot_within96(ys, newpath, d, title):
    fig, ax = plt.subplots(dpi=300)
    x = range(len(title))

    alg_initial_list = ["B", "LR", "EN", "GBR", "RFR"]
    colours = [".m--", ".r-", ".g-", ".y-", ".b-"]

    for y, alg_initial, colour in zip(ys, alg_initial_list, colours):
        plt.plot(x, y, colour, label=alg_initial, alpha=0.8)

    plt.title("Correct Predictions Within 96 Hours with Decreasing Features")

    plt.legend(loc=5, bbox_to_anchor=(1.2, 0.5))
    plt.xticks(x, title)

    plt.xlabel("Decreasing Features")
    plt.ylabel("Correct Predictions Within 96 hours")

    if not os.path.exists(newpath + "PDFs/"):
        os.makedirs(newpath + "PDFs/")  # Make folder for storing results if it does not exist

    plt.savefig("%s%s_within96.png" % (newpath, title), bbox_inches='tight', dpi=300)
    plt.savefig("%sPDFs/%s_within96.pdf" % (newpath, title), bbox_inches='tight')

    try:
        get_ipython
        plt.show()
    except:
        plt.close()

def multi_plot_within96_bar(ys, newpath, d, title, actual_title):
    fig, ax = plt.subplots(dpi=300)
    x = range(len(ys[0]))
    x = np.array(x)
    ys = np.array(ys)

    alg_initial_list = ["B", "LR", "EN", "GBR", "RFR"]
    colours = ["#F1948A", "#85C1E9"]
    if len(ys) == 3:
        colours += ["#F7DC6F"] #b30000 AF000E

    width = len(x)/(len(x) * (len(title)+1))
    newwidth = width
    minsofar = 100
    maxsofar = 0
    for y, colour, exp in zip(ys, colours, title):
        if min(y) < minsofar:
            minsofar = min(y)
        if max(y) > maxsofar:
            maxsofar = max(y)
        plt.bar(x+newwidth, y, width, color=colour, label=exp, alpha=1, align="center")
        newwidth += width

    plt.title(actual_title)

    plt.legend(loc=5, bbox_to_anchor=(1.2, 0.5))
    plt.xticks(x+(width*(len(title)+1) /2), alg_initial_list)

    plt.xlabel("Algorithms")
    plt.ylabel("Correct Predictions Within 96 hours")

    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y/100)))
    # yticks_i = [i * 5 for i in range(21)]
    # # yticks = ["0%", "10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%", "90%", "100%"]
    # yticks = ["0%","5%", "10%", "15%", "20%", "25%", "30%", "35%", "40%", "45%", "50%", "55%",
    #           "60%", "65%", "70%", "75%", "80%", "85%", "90%", "95%", "100%"]
    # plt.yticks(yticks_i, yticks)

    # if minsofar < 10:
    #     low = 0
    # else:
    #     low = math.ceil(minsofar-5)
    #
    # if maxsofar > 90:
    #     high = 100
    # else:
    #     high = math.ceil(maxsofar+5)
    #
    # plt.ylim(low, high)

    if not os.path.exists(newpath + "PDFs/"):
        os.makedirs(newpath + "PDFs/")  # Make folder for storing results if it does not exist

    plt.savefig("%s%s_within96_bar.png" % (newpath, title), bbox_inches='tight', dpi=300)
    plt.savefig("%sPDFs/%s_within96_bar.pdf" % (newpath, title), bbox_inches='tight')

    try:
        get_ipython
        plt.show()
    except:
        plt.close()

if __name__ == "__main__":  # Run program
    parameters = "../../../Data/parameters.txt"  # Parameters file
    sample_parameters = "../Sample Parameter File/parameters.txt"

    print("Modeling dataset", time.strftime("%Y.%m.%d"), time.strftime("%H.%M.%S"))

    d = get_parameters(parameters)
    sample_d = get_parameters(parameters)

    for key in sample_d.keys():
        if key not in d.keys():
            print("\"%s\" not in parameters.txt - please update parameters file with this key" % key)
            print("Default key and value => \"%s: %s\"" % (key, sample_d[key]))
            exit()
    for key in d.keys():
        if key not in sample_d.keys():
            print("\"%s\" not in sample parameters" % key)

    if d["resample"] == "y":# if resample is selected then all results are put in a resample folder
        newpath = r"../0. Results/" + d["user"] + "/multi_plot/" + "/resample/"
    elif d["specify_subfolder"] == "n":
        print("please specify a subfolder inside multiplot")
        print("..exiting")
        exit()
    else:
        newpath = r"../0. Results/" + d["user"] + "/multi_plot/" +  d["specify_subfolder"] +"/"
    if not os.path.exists(newpath):
        os.makedirs(newpath)  # Make folder for storing results if it does not exist

    copyfile(parameters, newpath + "parameters.txt")  # Save parameters
    np.random.seed(int(d["seed"]))  # Set seed

    dfs = []
    if type(d["input_files"]) == str:
        dfs.append(get_dfs(d, None))
        print("df shape : " , dfs[0].shape)
        input_file_names = [d["input_file_names"]]
    else:
        for i in range(len(d["input_files"])):
            dfs.append(get_dfs(d, i))
            print("dfs[%s] shape : " % i, dfs[i].shape)
        input_file_names = d["input_file_names"]

    if d["resample"] == "y":
        from sklearn.utils import resample
        print("\n..resampling\n")
        for i in range(len(dfs)):
            dfs[i] = resample(dfs[i], n_samples=int(d["n_samples"]), random_state=int(d["seed"]))
            dfs[i] = dfs[i].reset_index(drop=True)
            print("DF Shape:", dfs[i].shape)

    dfs_pct_close96 = []
    df_pct_closes = []
    dfs_RMSEs = []

    print("\n..calculating stats..\n")
    for i, df in enumerate(dfs):
        percent_close, number_close96, RMSEs = get_stats(df)
        dfs_pct_close96.append(number_close96)
        dfs_RMSEs.append(RMSEs)
        df_pct_closes.append(percent_close) # for 1 df

    # if len(dfs) == 1:
        # for pct_close, title in zip(df_pct_closes, input_file_names):
            # multi_plot_pct_correct(pct_close, newpath, d, title)
    # else:
    print("\n..plotting pct correct..\n")
    for pct_close, title in zip(df_pct_closes, input_file_names):
        multi_plot_pct_correct(pct_close, newpath, d, title)

    print("\n..plotting RMSE bar..\n")

    if "July" in input_file_names or "June" in input_file_names:
        actual_title = "RMSE for Training and Testing Stages"
    else:
        actual_title = "RMSE for each Algorithm"

    multi_plot_RMSEs_bar(dfs_RMSEs, newpath, d, input_file_names, actual_title)

    print("\n..plotting RMSE line..\n")
    dfs_RMSEs_T = np.transpose(dfs_RMSEs)
    multi_plot_RMSEs_line(dfs_RMSEs_T, newpath, d, input_file_names)

    print("\n..plotting pct correct within 96 hours..\n")
    dfs_pct_close96_T = np.transpose(dfs_pct_close96)
    multi_plot_within96(dfs_pct_close96_T, newpath, d, input_file_names)

    actual_title = "Correct Predictions Within 96 Hours"
    multi_plot_within96_bar(dfs_pct_close96, newpath, d, input_file_names, actual_title=actual_title)
