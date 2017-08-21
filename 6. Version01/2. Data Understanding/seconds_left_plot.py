import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np



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

parameters = "../../../Data/parameters.txt"  # Parameters file
sample_parameters = "../Sample Parameter File/parameters.txt"

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

if d["specify_subfolder"] == "n":
    newpath = r"../0. Results/" + d["user"] + "/seconds_left_plot/"
else:
    newpath = r"../0. Results/" + d["user"] + "/seconds_left_plot/" +  d["specify_subfolder"] +"/"
if not os.path.exists(newpath):
    os.makedirs(newpath)  # Make folder for storing results if it does not exist

if not os.path.exists(newpath + "PDFs/"):
    os.makedirs(newpath + "PDFs/")  # Make folder for storing results if it does not exist

df = pd.read_csv(d["file_location"] + d["plotting_input_file"] + ".csv", encoding='latin-1', low_memory=False)

df["Seconds_left_Year"] /= 3600
df["TimeTaken"] /= 3600

df["Seconds_left_Month"] /= 3600
df["TimeTaken"] /= 3600
df["Seconds_left_Qtr"] /= 3600
df["Seconds_left_Day"] /= 3600


def plot_ave_timetaken(ave_timetaken, std_timetaken, title, xlabel, ylabel, savefig, errorevery, xticks=None,
                       xticks_loc=None):
    y = np.array(ave_timetaken)
    z = np.where(y >= 0)[0]
    y = y[z]

    stdy = np.array(std_timetaken)
    stdz = np.where(stdy >= 0)[0]
    stdy = stdy[z]

    x = [x for x in range(len(y))]

    plt.errorbar(x, y, yerr=stdy, fmt="r",
                 ecolor="b",
                 elinewidth=1, capsize=1000, errorevery=errorevery)

    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)

    if xticks is not None:
        plt.xticks(xticks_loc, xticks)

    plt.savefig("%s%s.png" % (newpath, savefig))
    plt.savefig("%s%s%s.pdf" % (newpath, "PDFs/", savefig))

    plt.close()

lastx = 0
ave_timetaken = []
std_timetaken = []
for x in range(1, int(max(df["Seconds_left_Qtr"]))):
    x *= 10
    ave_timetaken.append(df["TimeTaken"][(df["Seconds_left_Qtr"] >= lastx) & (df["Seconds_left_Qtr"] < x)].mean())
    std_timetaken.append(df["TimeTaken"][(df["Seconds_left_Qtr"] >= lastx) & (df["Seconds_left_Qtr"] < x)].std())
    lastx = x

xticks_loc = [x * 30 for x in range(8)]
xticks = [x * 30 * 10 for x in range(8)]
plot_ave_timetaken(ave_timetaken, std_timetaken, title="Average TimeTaken against Seconds_left_Qtr",
                   xlabel="Hours left until end of the Qtr", ylabel="Average TimeTaken (hours)",
                   savefig="average_SLQ_err", errorevery=10,
                   xticks=xticks, xticks_loc=xticks_loc)


lastx= 0
ave_timetaken = []
std_timetaken = []
for x in range(1, int(max(df["Seconds_left_Month"]))):
    x *= 10
    ave_timetaken.append(df["TimeTaken"][(df["Seconds_left_Month"]>=lastx) & (df["Seconds_left_Month"]<x)].mean())
    std_timetaken.append(df["TimeTaken"][(df["Seconds_left_Month"]>=lastx) & (df["Seconds_left_Month"]<x)].std())
    lastx=x

xticks_loc = [x * 10 for x in range(8)]
xticks = [x * 10 * 10 for x in range(8)]
plot_ave_timetaken(ave_timetaken, std_timetaken, title="Average TimeTaken against Seconds_left_Month",
                   xlabel="Hours left until end of the Month", ylabel="Average TimeTaken (hours)",
                   savefig="average_SLM_err", errorevery=5,
                   xticks=xticks, xticks_loc=xticks_loc)


lastx= 0
ave_timetaken = []
std_timetaken = []
for x in range(1, int(max(df["Seconds_left_Day"]))):
    x *= 1
    ave_timetaken.append(df["TimeTaken"][(df["Seconds_left_Day"]>=lastx) & (df["Seconds_left_Day"]<x)].mean())
    std_timetaken.append(df["TimeTaken"][(df["Seconds_left_Day"]>=lastx) & (df["Seconds_left_Day"]<x)].std())
    lastx=x

xticks_loc = None
xticks = None
plot_ave_timetaken(ave_timetaken, std_timetaken, title="Average TimeTaken against Seconds_left_Day",
                   xlabel="Hours left until end of the Day", ylabel="Average TimeTaken (hours)",
                   savefig="average_SLD_err", errorevery=4,
                   xticks=xticks, xticks_loc=xticks_loc)


df["Seconds_left_Year"].hist(bins=50)
plt.title("Seconds_left_Year Histogram")
plt.xlabel("Hours left until the end of the Year")
plt.ylabel("Frequency")
plt.savefig("%sSeconds_left_Year_hist_50.png" % newpath)
plt.savefig("%s%sSeconds_left_Year_hist_50.pdf" % (newpath, "PDFs/"))
plt.close()


df["Seconds_left_Year"].hist(bins=25)
plt.title("Seconds_left_Year Histogram")
plt.xlabel("Hours left until the end of the Year")
plt.ylabel("Frequency")
plt.savefig("%sSeconds_left_Year_hist_25.png" % newpath)
plt.savefig("%s%sSeconds_left_Year_hist_25.pdf" % (newpath, "PDFs/"))
plt.close()


df["Seconds_left_Qtr"].hist(bins=20)
plt.title("Seconds_left_Qtr Histogram")
plt.xlabel("Hours left until the end of the Quarter")
plt.ylabel("Frequency")
plt.savefig("%sSeconds_left_Qtr_hist.png" % newpath)
plt.savefig("%s%sSeconds_left_Qtr_hist.pdf" % (newpath, "PDFs/"))
plt.close()


df["Seconds_left_Month"].hist(bins=40)
plt.title("Seconds_left_Month Histogram")
plt.xlabel("Hours left until the end of the Month")
plt.ylabel("Frequency")
plt.savefig("%sSeconds_left_Month_hist.png" % newpath)
plt.savefig("%s%sSeconds_left_Month_hist.pdf" % (newpath, "PDFs/"))
plt.close()


df["Seconds_left_Day"].hist(bins=50)
plt.title("Seconds_left_Day Histogram")
plt.xlabel("Hours left until the end of the Day")
plt.ylabel("Frequency")
plt.savefig("%sSeconds_left_Day_hist.png" % newpath)
plt.savefig("%s%sSeconds_left_Day_hist.pdf" % (newpath, "PDFs/"))
plt.close()