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
    newpath = r"../0. Results/" + d["user"] + "/within_past_8_hours_plot/"
else:
    newpath = r"../0. Results/" + d["user"] + "/within_past_8_hours_plot/" +  d["specify_subfolder"] +"/"
if not os.path.exists(newpath):
    os.makedirs(newpath)  # Make folder for storing results if it does not exist

if not os.path.exists(newpath + "PDFs/"):
    os.makedirs(newpath + "PDFs/")  # Make folder for storing results if it does not exist

df = pd.read_csv(d["file_location"] + d["plotting_input_file"] + ".csv", encoding='latin-1', low_memory=False)

df["Cases_created_within_past_8_hours"].hist(bins=int(300/10))
plt.title("Cases_created_within_past_8_hours Histogram")
plt.xlabel("Cases_created_within_past_8_hours")
plt.ylabel("Frequency")
plt.savefig("%sCases_created_within_past_8_hours_hist.png" % newpath)
plt.savefig("%s%sCases_created_within_past_8_hours_hist.pdf" % (newpath, "PDFs/"))
plt.close()

df["Cases_resolved_within_past_8_hours"].hist(bins=int(300/10))
plt.title("Cases_resolved_within_past_8_hours Histogram")
plt.xlabel("Cases_resolved_within_past_8_hours")
plt.ylabel("Frequency")
plt.savefig("%sCases_resolved_within_past_8_hours_hist.png" % newpath)
plt.savefig("%s%sCases_resolved_within_past_8_hours_hist.pdf" % (newpath, "PDFs/"))
plt.close()