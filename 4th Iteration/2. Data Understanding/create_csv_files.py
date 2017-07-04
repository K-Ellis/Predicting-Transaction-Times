"""*********************************************************************************************************************
UCD MSc Business Analytics Capstone Project - Predicting Transactions Times
************************************************************************************************************************
Iteration 4
Create CSV file program
************************************************************************************************************************
Eoin Carroll
Kieron Ellis
************************************************************************************************************************
Working on dataset 2 from Cosmic: UCD_Data_20170623_1.xlsx
************************************************************************************************************************
Parameters file "create_csv_files.txt" required
user: Yourname . . . eg. Eoin
raw_data: Dataset . . . eg. UCD_Data_20170623_1.xlsx
raw_data_location: eg. ../../../Data/ or ../../../Data/Cosmic_1
outfile_name: Appended to the sheet name. Eg. _1
outfile_location: ../../../Data/
*********************************************************************************************************************"""

import pandas as pd
import time
import os  # Used to create folders
from shutil import copyfile  # Used to copy parameters file to directory

parameters = "../../../Data/create_csv_files.txt"  # Parameters file for this program

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
    out_file_name = newpath + time.strftime("%H.%M.%S") + "_create_csv_log.txt"  # Log file name
    out_file = open(out_file_name, "w")  # Open log file
    out_file.write("Date and time: " + time.strftime("%Y%m%d-%H%M%S") + "\n\n")

    sheets = ["vw_Incident", "vw_HoldActivity", "vw_AuditHistory", "vw_PackageTriageEntry", "vw_StageTable"]
    for i in range(len(sheets)):  # This generates the csv files
        df = pd.read_excel(d["raw_data_location"] + d["raw_data"], sheetname=i)
        df.to_csv(d["outfile_location"] + "/" + sheets[i] + d["outfile_name"] + ".csv", index=False)
        out_file.write("Output:" + d["outfile_location"] + sheets[i] + d["outfile_name"] + " saved" + "\n")
        out_file.write("Date and time: " + time.strftime("%Y%m%d-%H%M%S") + "\n\n")
    out_file.close()

    copyfile(parameters, newpath + "/" + time.strftime("%H.%M.%S") + "_create_csv_parameters.txt")  # Save parameters
    print("CSVs created for", d["raw_data_location"] + d["raw_data"])