"""*********************************************************************************************************************
UCD MSc Business Analytics Capstone Project - Predicting Transactions Times
************************************************************************************************************************
Version 00
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

    sheets = ["vw_Incident", "vw_HoldActivity", "vw_AuditHistory", "vw_PackageTriageEntry", "vw_StageTable"]
    for i in range(len(sheets)):  # This generates the csv files
        df = pd.read_excel(d["raw_data_location"] + d["raw_data"], sheetname=i)
        df.to_csv(d["file_location"] + "/" + sheets[i] + d["output_file"] + ".csv", index=False)

    print("CSVs created for", d["raw_data_location"] + d["raw_data"])