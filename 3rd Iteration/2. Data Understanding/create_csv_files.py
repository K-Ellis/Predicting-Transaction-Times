import pandas as pd
import time
import os  # Used to create folders
import getpass  # Used to check PC name

def create_csv(dataset):  # Convert files to CSV
    newpath = r"../5. Results/" + str(getpass.getuser()) + "_" + time.strftime("%Y%m%d")
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    out_file_name = "../5. Results/" + str(getpass.getuser()) + "_" + time.strftime("%Y%m%d") + "/" + \
                    time.strftime("%Y%m%d-%H%M%S") + "_create_csv" + ".txt"  # Log file name
    out_file = open(out_file_name, "w")  # Open log file
    out_file.write("Date and time: " + time.strftime("%Y%m%d-%H%M%S") + "\n")

    df = pd.read_excel("../../../Data/" + dataset, sheetname=0)
    df.to_csv("../../../Data/vw_Incident.csv", index = False)
    out_file.write("vw_Incident saved" + "\n")

    df = pd.read_excel("../../../Data/" + dataset, sheetname=1)
    df.to_csv("../../../Data/vw_HoldActivity.csv", index = False)
    out_file.write("vw_HoldActivity saved" + "\n")

    df = pd.read_excel("../../../Data/" + dataset, sheetname=2)
    df.to_csv("../../../Data/vw_AuditHistory.csv", index = False)
    out_file.write("vw_AuditHistory" + "\n")

    df = pd.read_excel("../../../Data/" + dataset, sheetname=3)
    df.to_csv("../../../Data/vw_PackageTriageEntry.csv", index = False)
    out_file.write("vw_PackageTriageEntry saved" + "\n")

    df = pd.read_excel("../../../Data/" + dataset, sheetname=4)
    df.to_csv("../../../Data/vw_StageTable.csv", index = False)
    out_file.write("vw_StageTable saved" + "\n")
    out_file.close()

if __name__ == "__main__":  # Run program
    dataset = "UCD_Data_20170420_1.xlsx"
    create_csv(dataset)
    print("CSVs created for", dataset)