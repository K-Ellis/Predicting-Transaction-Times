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
    newpath = r"../0. Results/" + d["user"] + "/volume_plot/"
else:
    newpath = r"../0. Results/" + d["user"] + "/volume_plot/" +  d["specify_subfolder"] +"/"
if not os.path.exists(newpath):
    os.makedirs(newpath)  # Make folder for storing results if it does not exist

df = pd.read_csv(d["file_location"] + d["plotting_input_file"] + ".csv", encoding='latin-1', low_memory=False)

def get_day_into_year(date):
    day_into_year = date.timetuple().tm_yday
    return day_into_year
df["ResolvedDate"] = pd.to_datetime(df["ResolvedDate"])
df["ResolvedDate_Days_into_year"] = df["ResolvedDate"].apply(lambda x: int(get_day_into_year(x)))
df["Created_On"] = pd.to_datetime(df["Created_On"])
df["Created_on_Days_into_year"] = df["Created_On"].apply(lambda x: int(get_day_into_year(x)))

months = [31, 60, 91, 121, 152, 182, 213, 244]
x = 0
month_list = []
VolumePerMonth = []
ResolvedVolumePerMonth = []
for i, month in enumerate(months):
    y = month
    month_list.append(i + 1)
    VolumePerMonth.append(len(df["Created_on_Days_into_year"][
                                  (df["Created_on_Days_into_year"] > x) & (df["Created_on_Days_into_year"] <= y)]))
    ResolvedVolumePerMonth.append(len(df["ResolvedDate_Days_into_year"][
                                  (df["ResolvedDate_Days_into_year"] > x) & (df["ResolvedDate_Days_into_year"] <= y)]))

    x = month
    y = x + month

plt.bar(month_list, VolumePerMonth)

plt.xlim(1.5, 8.5)
plt.ylabel("Volume of Cases")
plt.xlabel("Month of the Year")
plt.title("Volume of Cases Created in Each Month")

if not os.path.exists(newpath + "PDFs/"):
    os.makedirs(newpath + "PDFs/")  # Make folder for storing results if it does not exist

plt.savefig("%scase_volume_created.png" % newpath)
plt.savefig("%s%scase_volume_created.pdf" % (newpath, "PDFs/"))

plt.close()


plt.bar(month_list, ResolvedVolumePerMonth)

plt.xlim(1.5, 8.5)
plt.ylabel("Volume of Cases")
plt.xlabel("Month of the Year")
plt.title("Volume of Cases Resolved in Each Month")

if not os.path.exists(newpath + "PDFs/"):
    os.makedirs(newpath + "PDFs/")  # Make folder for storing results if it does not exist

plt.savefig("%scase_volume_resolved.png" % newpath)
plt.savefig("%s%scase_volume_resolved.pdf" % (newpath, "PDFs/"))

plt.close()