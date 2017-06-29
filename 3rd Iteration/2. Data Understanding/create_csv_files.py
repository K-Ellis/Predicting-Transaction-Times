import pandas as pd
import time
import os  # Used to create folders
import getpass  # Used to check PC name


def create_csv(dataset, COSMIC_num):  # Convert files to CSV
    newpath = r"../5. Results/" + str(getpass.getuser()) + "_" + time.strftime("%Y%m%d")
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    out_file_name = "../5. Results/" + str(getpass.getuser()) + "_" + time.strftime("%Y%m%d") + "/" + \
                    time.strftime("%Y%m%d-%H%M%S") + "_create_csv" + ".txt"  # Log file name
    out_file = open(out_file_name, "w")  # Open log file
    out_file.write("Date and time: " + time.strftime("%Y%m%d-%H%M%S") + "\n")

    df = pd.read_excel("../../../Data/COSMIC_%s/"%(COSMIC_num) + dataset, sheetname=0)
    df.to_csv("../../../Data/COSMIC_%s/vw_Incident%s.csv"%(COSMIC_num,COSMIC_num), index = False)
    out_file.write("vw_Incident saved" + "\n")

    df = pd.read_excel("../../../Data/COSMIC_%s/"%(COSMIC_num) + dataset, sheetname=1)
    df.to_csv("../../../Data/COSMIC_%s/vw_HoldActivity%s.csv"%(COSMIC_num,COSMIC_num), index = False)
    out_file.write("vw_HoldActivity saved" + "\n")

    df = pd.read_excel("../../../Data/COSMIC_%s/"%(COSMIC_num) + dataset, sheetname=2)
    df.to_csv("../../../Data/COSMIC_%s/vw_AuditHistory%s.csv"%(COSMIC_num,COSMIC_num), index = False)
    out_file.write("vw_AuditHistory" + "\n")

    df = pd.read_excel("../../../Data/COSMIC_%s/"%(COSMIC_num) + dataset, sheetname=3)
    df.to_csv("../../../Data/COSMIC_%s/vw_PackageTriageEntry%s.csv"%(COSMIC_num,COSMIC_num), index = False)
    out_file.write("vw_PackageTriageEntry saved" + "\n")

    df = pd.read_excel("../../../Data/COSMIC_%s/"%(COSMIC_num) + dataset, sheetname=4)
    df.to_csv("../../../Data/COSMIC_%s/vw_StageTable%s.csv"%(COSMIC_num,COSMIC_num), index = False)
    out_file.write("vw_StageTable saved" + "\n")
    out_file.close()

if __name__ == "__main__":  # Run program
    dataset = "UCD_Data_20170420_1.xlsx"
    create_csv(dataset, 1)
    print("CSVs created for", dataset)

    dataset = "UCD_Data_20170623_1.xlsx"
    create_csv(dataset, 2)
    print("CSVs created for", dataset)