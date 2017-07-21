"""*********************************************************************************************************************
UCD MSc Business Analytics Capstone Project - Predicting Transactions Times
************************************************************************************************************************
Iteration 5
Create CSV file program
************************************************************************************************************************
Eoin Carroll
Kieron Ellis
************************************************************************************************************************
Working on dataset 2 from Cosmic: UCD_Data_20170623_1.xlsx
************************************************************************************************************************
Parameters file "parameters.txt" required
*********************************************************************************************************************"""

import pandas as pd
import time
import os  # Used to create folders
from shutil import copyfile  # Used to copy parameters file to directory

parameters = "../../../Data/parameters.txt"  # Parameters file

if __name__ == "__main__":  # Run program
    d = {}
    with open(parameters, "r") as f:
        for line in f:
            line = line.replace(":", "")
            (key, val) = line.split()
            d[key] = val

    # This code sets up logging
    newpath = r"../0. Results/" + d["user"] + "/create_csv_files/" + time.strftime("%Y.%m.%d/")
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    file_name = newpath + time.strftime("%H.%M.%S") + "_create_csv_log.txt"  # Log file name
    out_file = open(file_name, "w")  # Open log file
    out_file.write("Date and time: " + time.strftime("%Y%m%d-%H%M%S") + "\n\n")

    sheets = ["vw_Incident", "vw_HoldActivity", "vw_AuditHistory", "vw_PackageTriageEntry", "vw_StageTable"]
    for i in range(len(sheets)):  # This generates the csv files
        df = pd.read_excel(d["raw_data_location"] + d["raw_data"], sheetname=i)
        df.to_csv(d["file_location"] + "/" + sheets[i] + d["file_name"] + ".csv", index=False)
        out_file.write("Output:" + d["file_location"] + sheets[i] + d["file_name"] + " saved" + "\n")
        out_file.write("Date and time: " + time.strftime("%Y%m%d-%H%M%S") + "\n\n")
    out_file.close()

    copyfile(parameters, newpath + "/" + time.strftime("%H.%M.%S") + "_parameters.txt")  # Save parameters
    print("CSVs created for", d["raw_data_location"] + d["raw_data"])