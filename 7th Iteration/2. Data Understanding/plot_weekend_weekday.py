
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
    newpath = r"../0. Results/" + d["user"] + "/plot_weekend/"
else:
    newpath = r"../0. Results/" + d["user"] + "/plot_weekend/" +  d["specify_subfolder"] +"/"
if not os.path.exists(newpath):
    os.makedirs(newpath)  # Make folder for storing results if it does not exist

df = pd.read_csv(d["file_location"] + d["plotting_input_file"] + ".csv", encoding='latin-1', low_memory=False)


# plotting created on weekend versus weekday

def get_last_bdays_months_just_date():
    last_bdays = pd.date_range("2017.01.01", periods=11, freq='BM')
    last_bdays_offset = []
    for last_bday in last_bdays:
        last_bdays_offset.append((last_bday + pd.DateOffset(days=1,hours=8)).date())
    return last_bdays_offset

def created_on_weekend(date, last_bdays):
    day_of_the_week = date.weekday()
    if day_of_the_week == 0 or day_of_the_week == 6:
        # but have to check if the date is the day after last business day of the month!
        if date.date in last_bdays and date.time()<8:
            return 0
        else:
            return 1
    else:
        return 0
last_bdays = get_last_bdays_months_just_date()
df["Created_On"] = pd.to_datetime(df["Created_On"])
df["Created_on_weekend"] = df["Created_On"].apply(lambda x: int(created_on_weekend(x, last_bdays)))

df["TimeTaken"] /= 3600

meanstuff = dict(color="yellow", linewidth=2)
sns.boxplot(df["Created_on_weekend"], df["TimeTaken"], showfliers=False, showmeans=True, meanline=True, meanprops=meanstuff)
plt.title("TimeTaken for Cases Created On Weekday and Weekend")
plt.xlabel("Created On Weekday or Weekend")
plt.ylabel("TimeTaken (hours)")
xticks = ["Weekday", "Weekend"]
plt.xticks([x for x in range(len(xticks))], xticks)

if not os.path.exists(newpath + "PDFs/"):
    os.makedirs(newpath + "PDFs/")  # Make folder for storing results if it does not exist

plt.savefig("%splot_weekend.png" % newpath)
plt.savefig("%s%splot_weekend.pdf" % (newpath, "PDFs/"))

plt.close()


