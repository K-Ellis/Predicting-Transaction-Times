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
    newpath = r"../0. Results/" + d["user"] + "/concurrent_plot/"
else:
    newpath = r"../0. Results/" + d["user"] + "/concurrent_plot/" +  d["specify_subfolder"] +"/"
if not os.path.exists(newpath):
    os.makedirs(newpath)  # Make folder for storing results if it does not exist

if not os.path.exists(newpath + "PDFs/"):
    os.makedirs(newpath + "PDFs/")  # Make folder for storing results if it does not exist

df = pd.read_csv(d["file_location"] + d["plotting_input_file"] + ".csv", encoding='latin-1', low_memory=False)

df["TimeTaken"] /= 3600

lastx= 0
ave_timetaken = []
std_timetaken = []
for x in range(1,1281):
    ave_timetaken.append(df["TimeTaken"][(df["Concurrent_open_cases"]>=lastx) & (df["Concurrent_open_cases"]<x)].mean())
    std_timetaken.append(df["TimeTaken"][(df["Concurrent_open_cases"]>=lastx) & (df["Concurrent_open_cases"]<x)].std())
    lastx=x

# plt.plot([x for x in range(1280)],ave_timetaken, color="orange")
# plt.title("Average TimeTaken against Concurrent_open_cases")
# plt.ylabel("Average TimeTaken")
# plt.xlabel("Concurrently Open Cases")
# plt.savefig("%saverage_concurrent.png" % newpath)
# plt.savefig("%s%saverage_concurrent.pdf" % (newpath, "PDFs/"))
# plt.close()

plt.errorbar([x for x in range(1280)],ave_timetaken,yerr=std_timetaken, fmt="r", ecolor="b", elinewidth=0.5, capsize=1000, errorevery=10)
plt.title("Average TimeTaken against Concurrent_open_cases")
plt.ylabel("Average TimeTaken (hours)")
plt.xlabel("Concurrently Open Cases")
plt.savefig("%saverage_concurrent_err.png" % newpath)
plt.savefig("%s%saverage_concurrent_err.pdf" % (newpath, "PDFs/"))
plt.close()

#
# df["Concurrent_open_cases"].hist(bins=int(1280/10))
# plt.title("Concurrent_open_cases Histogram")
# plt.ylabel("Frequency")
# plt.xlabel("Concurrently Open Cases")
# plt.savefig("%sconcurrent_hist.png" % newpath)
# plt.savefig("%s%ssconcurrent_hist.pdf" % (newpath, "PDFs/"))
# plt.close()
#
# df["Concurrent_open_cases"].hist(bins=int(1280/10))
# plt.plot([x for x in range(1280)],ave_timetaken, alpha=0.8, color="orange")
# plt.title("Average TimeTaken against Concurrent_open_cases Histogram")
# plt.ylabel("Concurrent Frequency and Average TimeTaken")
# plt.xlabel("Concurrently Open Cases")
# plt.savefig("%saverage_concurrent_hist.png" % newpath)
# plt.savefig("%s%ssaverage_concurrent_hist.pdf" % (newpath, "PDFs/"))
# plt.close()
