import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

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

df["Seconds_left_Qtr"] /= 3600

df["Seconds_left_Qtr"].hist(bins=20)
plt.title("Seconds_left_Qtr Histogram")
plt.xlabel("Hours left until the end of the Quarter")
plt.ylabel("Frequency")
plt.savefig("%sSeconds_left_Qtr_hist.png" % newpath)
plt.savefig("%s%sSeconds_left_Qtr_hist.pdf" % (newpath, "PDFs/"))
plt.close()


df["Seconds_left_Month"] /= 3600

df["Seconds_left_Month"].hist(bins=40)
plt.title("Seconds_left_Month Histogram")
plt.xlabel("Hours left until the end of the Month")
plt.ylabel("Frequency")
plt.savefig("%sSeconds_left_Month_hist.png" % newpath)
plt.savefig("%s%sSeconds_left_Month_hist.pdf" % (newpath, "PDFs/"))
plt.close()



df["Seconds_left_Day"] /= 3600

df["Seconds_left_Day"].hist(bins=50)
plt.title("Seconds_left_Day Histogram")
plt.xlabel("Hours left until the end of the Day")
plt.ylabel("Frequency")
plt.savefig("%sSeconds_left_Day_hist.png" % newpath)
plt.savefig("%s%sSeconds_left_Day_hist.pdf" % (newpath, "PDFs/"))
plt.close()