import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os  # Used to create folders
from shutil import copyfile  # Used to copy parameters file to directory

def get_dfs(d):
    df1 = pd.read_csv(d["file_location"] + d["input1"] + ".csv", encoding='latin-1', low_memory=False)
    df1["TimeTaken"] = df1["TimeTaken"].apply(lambda x: x/3600)
    
    df2 = pd.read_csv(d["file_location"] + d["input2"] + ".csv", encoding='latin-1', low_memory=False)
    df2["TimeTaken"] = df2["TimeTaken"].apply(lambda x: x/3600)
    
    return df1, df2
    
def get_pct_correct(df, col):
    percent_close = []
    number_close = [0 for _ in range(96)]
    
    for i in range(len(df[col])):
        for j in range(len(number_close)):
            if abs(df[col].iloc[i]/3600 - df["TimeTaken"].iloc[i]) <= j+1:  # Within 1 hour
                    number_close[j] += 1
    for j in number_close:
        percent_close.append(j / len(df[col]) * 100)    
    return percent_close
    
def multi_plot_pct_correct(y1, y2, alg_initials, newpath, d, df_num=None):
    y1 = [0] + y1
    y2 = [0] + y2
    x = range(len(y1))
    print(x[:10])
    print(y1[:10])
    print(y2[:10])
    
    if df_num == None:
        label1 = d["df1_name"]
        label2 = d["df2_name"]
    else:
        label1 = "LR"
        label2 = "RFR"
    
    plt.plot(x, y1, "r", label=label1)
    plt.plot(x, y2, "b", label=label2)

    if df_num == None:
        plt.title("%s %s - %s Predictions Within Hours" % (d["df1_name"], d["df2_name"], alg_initials))
    elif df_num == 1:
        plt.title("%s - %s Predictions Within Hours" % (d["df1_name"], alg_initials))
    else:
        plt.title("%s - %s Predictions Within Hours" % (d["df2_name"], alg_initials))

    plt.xlim(0, 96)
    plt.ylim(0, 100)
    
    plt.legend(loc=4)

    xticks = [(x + 1) * 8 for x in range(12)]
    plt.xticks(xticks, xticks)

    yticks = [i * 10 for i in range(11)]
    plt.yticks(yticks, yticks)

    plt.grid()
    plt.xlabel("Time (hours)")
    plt.ylabel("Percentage of Correct Predictions")

    if not os.path.exists(newpath + "PDFs/"):
        os.makedirs(newpath + "PDFs/")  # Make folder for storing results if it does not exist
    
    if df_num == None:
        plt.savefig("%s%s_%s_%s_pct_correct.png" % (newpath, d["df1_name"], d["df2_name"], alg_initials))
        plt.savefig("%sPDFs/%s_%s_%s_pct_correct.pdf" % (newpath, d["df1_name"], d["df2_name"], alg_initials))
    elif df_num == 1:
        plt.savefig("%s%s_%s_pct_correct.png" % (newpath, d["df1_name"], alg_initials))
        plt.savefig("%sPDFs/%s_%s_pct_correct.pdf" % (newpath, d["df1_name"], alg_initials))
    else:
        plt.savefig("%s%s_%s_pct_correct.png" % (newpath, d["df2_name"], alg_initials))
        plt.savefig("%sPDFs/%s_%s_pct_correct.pdf" % (newpath, d["df2_name"], alg_initials))
    plt.close()
    
 
 
if __name__ == "__main__":  # Run program
    parameters = "../../../Data/parameters.txt"  # Parameters file
    sample_parameters = "../Sample Parameter File/parameters.txt"

    print("Modeling dataset", time.strftime("%Y.%m.%d"), time.strftime("%H.%M.%S"))
    d = {}
    with open(parameters, "r") as f:
        for line in f:
            line = line.replace(":", "")
            (key, val) = line.split()
            d[key] = val

    sample_d = {}
    with open(sample_parameters, "r") as f:
        for line in f:
            line = line.replace(":", "")
            (key, val) = line.split()
            sample_d[key] = val

    for key in sample_d.keys():
        if key not in d.keys():
            print("\"%s\" not in parameters.txt - please update parameters file with this key" % key)
            print("Default key and value => \"%s: %s\"" % (key, sample_d[key]))
            exit()
    for key in d.keys():
        if key not in sample_d.keys():
            print("\"%s\" not in sample parameters" % key)
            
    # if resample is selected then all results are put in a resample folder
    if d["resample"] == "y":
        newpath = r"../0. Results/" + d["user"] + "/multi_plot/" + d["df1_name"] + d["df2_name"] + "/resample/"
    # if no subfolder is specified then the reults are put into a folder called after the input file
    elif d["specify_subfolder"] == "n":
        newpath = r"../0. Results/" + d["user"] + "/multi_plot/" + d["df1_name"] + d["df2_name"] + "/"
    # if a subfolder is specified then all results are put there
    else:
        newpath = r"../0. Results/" + d["user"] + "/multi_plot/" + + d["df1_name"] + d["df2_name"] + "/" + d["specify_subfolder"] +"/"
    if not os.path.exists(newpath):
        os.makedirs(newpath)  # Make folder for storing results if it does not exist

    copyfile(parameters, newpath + "parameters.txt")  # Save parameters

    np.random.seed(int(d["seed"]))  # Set seed

    df1, df2 = get_dfs(d)

    print("\nDF1 Shape:", df1.shape)
    print("DF2 Shape:", df2.shape, "\n")

    if d["LinearRegression"] == "y":
        col = "TimeTaken_LinearRegression"
        alg_initials = "LR"
        pct_correctsLR1 = get_pct_correct(df1, col)
        pct_correctsLR2 = get_pct_correct(df2, col)        
        multi_plot_pct_correct(pct_correctsLR1, pct_correctsLR2, alg_initials, newpath, d)
     
    if d["RandomForestRegressor"] == "y":
        col = "TimeTaken_RandomForestRegressor"
        alg_initials = "RFR"
        pct_correctsRFR1 = get_pct_correct(df1, col)
        pct_correctsRFR2 = get_pct_correct(df2, col)    
        multi_plot_pct_correct(pct_correctsRFR1, pct_correctsRFR2, alg_initials, newpath, d)
        
    if d["LinearRegression"] == "y" and d["RandomForestRegressor"] == "y":
        alg_initials = "LR RFR"
        multi_plot_pct_correct(pct_correctsLR1, pct_correctsRFR1, alg_initials, newpath, d, 1)
        multi_plot_pct_correct(pct_correctsLR2, pct_correctsRFR2, alg_initials, newpath, d, 2)