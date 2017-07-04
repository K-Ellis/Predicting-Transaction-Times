import pandas as pd
import time
import os  # Used to create folders


def create_csv(user, dataset, outfile_name, outfile_location):  # Convert files to CSV
    newpath = r"../0. Results/" + user + "/create_csv_files/" + time.strftime("%Y.%m.%d/")
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    out_file_name = newpath + time.strftime("%H.%M.%S") + "_create_csv.txt"  # Log file name
    out_file = open(out_file_name, "w")  # Open log file
    out_file.write("Date and time: " + time.strftime("%Y%m%d-%H%M%S") + "\n")

    df = pd.read_excel(dataset, sheetname=0)
    df.to_csv(outfile_location + "/vw_Incident" + outfile_name + ".csv", index=False)
    out_file.write(outfile_location + "vw_Incident saved" + outfile_name + " saved" + "\n")

    df = pd.read_excel(dataset, sheetname=1)
    df.to_csv(outfile_location + "/vw_HoldActivity" + outfile_name + ".csv", index=False)
    out_file.write(outfile_location + "vw_HoldActivity saved" + outfile_name + " saved" + "\n")

    df = pd.read_excel(dataset, sheetname=2)
    df.to_csv(outfile_location + "/vw_AuditHistory" + outfile_name + ".csv", index=False)
    out_file.write(outfile_location + "vw_AuditHistory" + outfile_name + " saved" + "\n")

    df = pd.read_excel(dataset, sheetname=3)
    df.to_csv(outfile_location + "/vw_PackageTriageEntry" + outfile_name + ".csv", index=False)
    out_file.write(outfile_location + "vw_PackageTriageEntry" + outfile_name + " saved" + "\n")

    df = pd.read_excel(dataset, sheetname=4)
    df.to_csv(outfile_location + "/vw_StageTable" + outfile_name + ".csv", index=False)
    out_file.write(outfile_location + "vw_StageTable" + outfile_name + " saved" + "\n")
    out_file.write("Date and time: " + time.strftime("%Y%m%d-%H%M%S") + "\n")
    out_file.close()

if __name__ == "__main__":  # Run program

    d = {}
    with open("../../../Data/create_csv_files.txt", "r") as f:
        for line in f:
            line = line.replace(":", "")
            (key, val) = line.split()
            d[key] = val

    create_csv(d["user"], d["raw_data_location"] + d["raw_data"], d["outfile_name"], d["outfile_location"])
    print("CSVs created for", d["raw_data"])

    # Parameters file "create_csv_files.txt" required
    # user: Yourname . . . eg. Eoin
    # raw_data: Dataset . . . eg. UCD_Data_20170623_1.xlsx
    # raw_data_location: eg. ../../../Data/ or ../../../Data/Cosmic_1
    # outfile_name: Appended to the sheet name. Eg. _1
    # outfile_location: ../../../Data/
    # Remember to update this if any changes are made