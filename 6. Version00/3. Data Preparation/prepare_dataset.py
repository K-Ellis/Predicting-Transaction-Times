"""*********************************************************************************************************************
UCD MSc Business Analytics Capstone Project - Predicting Transactions Times
************************************************************************************************************************
Data pre-processing program
Version 00 - Program for experiments after last iteration
************************************************************************************************************************
Eoin Carroll
Kieron Ellis
*********************************************************************************************************************"""


import pandas as pd
import time
import datetime
import numpy as np
from sklearn.utils import resample
from sklearn import preprocessing
from shutil import copyfile  # Used to copy parameters file to directory
from pandas.tseries.offsets import BDay


def fill_nulls_dfcs(df, dfcs, fill_value): # Fill in Nulls given a set of dataframe columns
    for dfc in dfcs:
        if fill_value == "mode":
            df[dfc].fillna(df[dfc].mode()[0], inplace=True)
        if fill_value == "mean":
            df[dfc].fillna(df[dfc].mean(), inplace=True)


def time_taken(df):  # replace start & finish with one new column, "TimeTaken"
    df2 = pd.DataFrame()  # create new dataframe, df2, to store answer  # todo - no need for df2
    df2["TimeTaken"] = (df["ResolvedDate"] - df["Created_On"]).astype('timedelta64[s]')
    df = pd.concat([df2, df], axis=1)
    print("TimeTaken added:", df.shape)
    return df


def transform_country(dfc):  # Convert country into continents
    # See excel sheet called countrylist in iteration 2 - preprocessing folder for decision process
    africa = ["Algeria","Angola","Botswana","Burundi","Cameroon","Congo (DRC)","Côte d’Ivoire","Egypt","Gabon","Ghana",
              "Ivory Coast","Kenya","Macau SAR","Mauritius","Mozambique","Namibia","Nigeria",
              "Rest of East & Southern Africa","Rwanda","Senegal","Sierra Leone","South Africa","Swaziland","Tanzania",
              "Togo","Uganda","West and Central Africa","Zambia","Zimbabwe"]
    asia = ["Afghanistan","Azerbaijan","Bahrain","Bangladesh","Brunei","China","Hong Kong","Hong Kong SAR","India",
            "Indian Ocean Islands","Indonesia","Iraq","Israel","Japan","Jordan","Kazakhstan","Korea","Kuwait",
            "Kyrgyzstan","Lebanon","Levant","Libya","Malaysia","MEA HQ","Morocco","Myanmar","Nepal","North Gulf",
            "Oman","Pakistan","Palestinian Authority","Philippines","Qatar","Russia","Saudi Arabia","Serbia",
            "Singapore","South Gulf","Sri Lanka","Taiwan","Thailand","Tunisia","United Arab Emirates","Uzbekistan",
            "Vietnam"]
    australia = ["Australia","Cook Islands","New Zealand","Norfolk Island","Samoa"]
    europe = ["Austria","Belarus","Belgium","Bosnia and Herzegovina","Bulgaria","Croatia","Cyprus","Czech Republic",
              "Denmark","Estonia","Finland","France","Georgia","Germany","Greece","Hungary","Iceland","Ireland",
              "Ireland Sales Mktg","Italy","Latvia","Lithuania","Luxembourg","Macedonia, Former Yugoslav Rep",
              "Macedonia, FYR","Malta","Moldova","Montenegro","NEPA Indirect Markets","Netherlands","Norway",
              "Poland","Portugal","Romania","Slovakia","Slovenia","Spain","Sweden","Switzerland","Turkey",
              "Ukraine","United Kingdom"]
    northamerica = ["Canada","United States"]
    southamerica = ["Argentina","Bahamas, The","Barbados","Bermuda","Bolivia","Brazil","caribbean","Central America",
                    "Chile","Colombia","Costa Rica","Dominican Rep.","Ecuador","El Salvador","French Guiana",
                    "French Polynesia","Guadeloupe","Guatemala","Honduras","Jamaica","Jamaica & BCBB","Mexico",
                    "Nicaragua","Panama","Paraguay","Peru","Puerto Rico","St. Lucia","Trinidad and Tobago","Uruguay",
                    "Venezuela"]
    pd.options.mode.chained_assignment = None  # default='warn' . . . . this disables an overwriting warning
    for i in range(len(dfc)):
        if dfc.iloc[i] in africa: dfc.iloc[i] = "africa"
        elif dfc.iloc[i] in asia: dfc.iloc[i] = "asia"
        elif dfc.iloc[i] in australia: dfc.iloc[i] = "australia"
        elif dfc.iloc[i] in europe: dfc.iloc[i] = "europe"
        elif dfc.iloc[i] in northamerica: dfc.iloc[i] = "northamerica"
        elif dfc.iloc[i] in southamerica: dfc.iloc[i] = "southamerica"
        else: dfc.iloc[i] = "other"
    return dfc


def scale_quant_cols(df, quant_cols):  # Scale quantitative variables
    df_num = df[quant_cols]
    for col in quant_cols:
        del df[col]  # todo del df[quant_cols]???
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(df_num)
    df_x_scaled = pd.DataFrame(x_scaled)
    df_x_scaled.columns = df_num.keys().tolist()
    df.reset_index(drop=True, inplace=True)
    df = pd.concat([df_x_scaled, df], axis=1)
    return df


def deletions(df, d):  # Delete all columns apart from those listed
    keepers = ["TimeTaken", "Created_On", "ResolvedDate", "IsSOXCase", "AmountinUSD", "IsGovernment", "IsMagnumCase",
               "IsSignature"]
               
    if d["Concurrent_open_cases"] == "y": keepers.append("Concurrent_open_cases")
    if d["Cases_created_within_past_8_hours"] == "y": keepers.append("Cases_created_within_past_8_hours")
    if d["Cases_resolved_within_past_8_hours"] == "y": keepers.append("Cases_resolved_within_past_8_hours")

    if d["Seconds_left_Day"] == "y": keepers.append("Seconds_left_Day")
    if d["Seconds_left_Month"] == "y": keepers.append("Seconds_left_Month")
    if d["Seconds_left_Qtr"] == "y": keepers.append("Seconds_left_Qtr")
    if d["Seconds_left_Year"] == "y": keepers.append("Seconds_left_Year")
    
    if d["Created_on_Weekend"] == "y": keepers.append("Created_on_Weekend")
    
    if d["Rolling_Mean"] == "y": keepers.append("Rolling_Mean")
    if d["Rolling_Median"] == "y": keepers.append("Rolling_Median")
    if d["Rolling_Std"] == "y": keepers.append("Rolling_Std")

    if d["ordinal_mapping"] == "y":
        cols = ["Priority", "Complexity", "StageName"]
        for col in cols:
            keepers.append(col)

    if d["onehot_encoding"] == "y":
        cols = ["CountrySource", "CountryProcessed", "SalesLocation", "StatusReason", "SubReason",
                "ROCName", "sourcesystem", "Source", "Revenutype", "Queue"]
        for col in cols:
            keepers.append(col)

    # New variables created using Created_On and ResolvedDate:
    # "Concurrent_open_cases", "Cases_created_within_past_8_hours", "Cases_resolved_within_past_8_hours", 
    # "Seconds_left_Day", "Seconds_left_Month", "Seconds_left_Qtr", "Seconds_left_Year", "Created_on_Weekend", 
    # "Rolling_Mean", "Rolling_Median", "Rolling_Std"
    
    if d["append_HoldDuration"] == "y":
        for col in ["HoldDuration", "HoldTypeName_3rd Party", "HoldTypeName_Customer", "HoldTypeName_Internal",
                    "AssignedToGroup_BPO", "AssignedToGroup_CRMT"]:
            keepers.append(col)
    if d["append_AuditDuration"] == "y": keepers.append("AuditDuration")

    print("Keepers:", keepers)
    for col in df.columns:
        if col not in keepers: del df[col]
    print("Deletions:", df.shape)

    return df


def get_last_bdays_months():
    last_bdays = pd.date_range("2017.01.01", periods=11, freq='BM')
    last_bdays_offset = []
    for last_bday in last_bdays:
        last_bdays_offset.append(last_bday + pd.DateOffset(days=1,hours=8))
    return last_bdays_offset


def get_last_bdays_qtr():
    start_date = pd.datetime(2017, 1, 1) 
    last_bdays = pd.date_range(start_date, periods=3, freq='Q')
    last_bdays_offset = []
    for last_bday in last_bdays:
        last_bdays_offset.append(last_bday + pd.DateOffset(days=1,hours=8))
    last_bdays_offset = [start_date] + last_bdays_offset
    return last_bdays_offset
    

def return_end_of_year():
    last_bdays = []
    last_bdays.append(pd.datetime(2017, 1, 1))
    last_bdays.append(pd.datetime(2017, 7, 1, 8))
    last_bdays.append(pd.datetime(2018, 6, 30, 8))
    return last_bdays

    
def get_seconds_left(date, last_bdays_offset):
    for i in range(len(last_bdays_offset)):       
        if date >= last_bdays_offset[i] and date <last_bdays_offset[i+1]:
            seconds_left = (last_bdays_offset[i+1] - date).seconds
            days_left = (last_bdays_offset[i+1] - date).days
            total_seconds_left = seconds_left + days_left*24*60*60        
            return total_seconds_left
       
       
def get_daily_cutoffs(last_bdays):
    start_of_the_year = pd.datetime(2017,1,1)  # todo hardcoded date
    last_bdays_offset = [start_of_the_year]
    for last_bday in last_bdays:
        end_fourth_last = (last_bday - BDay(1) - BDay(1) - BDay(1) - BDay(1)).normalize() +pd.DateOffset(hours=17)
        last_bdays_offset.append(end_fourth_last)
        end_third_last = (last_bday - BDay(1) - BDay(1) - BDay(1)).normalize() +pd.DateOffset(hours=23)
        last_bdays_offset.append(end_third_last)
        end_second_last = (last_bday - BDay(1) - BDay(1)).normalize() +pd.DateOffset(hours=23)
        last_bdays_offset.append(end_second_last)
        last_bdays_offset.append(last_bday)
    return last_bdays_offset

    
def get_seconds_until_end_of_day(date, cutoffs):
    for i in range(len(cutoffs)):    
        if date >= cutoffs[i] and date < cutoffs[i+1]:
            hour = cutoffs[i+1].hour
            minute = cutoffs[i+1].minute
            second = cutoffs[i+1].second
            cutoff_in_seconds = hour*60*60 + minute*60 + second

            hour = date.hour
            minute = date.minute
            second = date.second
            date_in_seconds = hour*60*60 + minute*60 + second
         
            if date.weekday() < 5:
                if cutoffs[i+1].time() > date.time():
                    return cutoff_in_seconds - date_in_seconds
                elif (date + pd.DateOffset(days=1)).weekday() <5:
                    return cutoff_in_seconds + 24*60*60 - date_in_seconds
                else:
                    return cutoff_in_seconds + (24*60*60)*3 - date_in_seconds
            else:
                if date.date() == cutoffs[i+1].date():
                    if cutoffs[i+1].hour == 23:
                        return cutoff_in_seconds - date_in_seconds
                    else:
                        return cutoff_in_seconds +24*60*60 - date_in_seconds
                else:
                    if date.weekday() == 5:
                        return (24*60*60)*2 - date_in_seconds + cutoff_in_seconds
                    else:
                        return (24*60*60) - date_in_seconds + cutoff_in_seconds


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


def get_rolling_stats(df, window):
    dfcopy = df.copy()
    dfcopy.sort_values(by="Created_On", inplace=True)

    dfcopy["Rolling_Mean"] = dfcopy["TimeTaken"].rolling(window=window).mean()
    dfcopy["Rolling_Median"] = dfcopy["TimeTaken"].rolling(window=window).median()
    dfcopy["Rolling_Std"] = dfcopy["TimeTaken"].rolling(window=window).std()

    dfcopy.dropna(subset=["Rolling_Mean"], inplace=True)
    return dfcopy
    

def get_GCO_start_cutoffs(end_cutoffs):
    start_cutoffs = []
    for cutoff in end_cutoffs:
        if cutoff.weekday()==6:
            # take away 5 bdays
            start_cutoff = cutoff - BDay(5)
            start_cutoffs.append(start_cutoff)
        else:
            # take away 4 bdays
            start_cutoff = cutoff - BDay(4)
            start_cutoffs.append(start_cutoff)
    return start_cutoffs

    
def get_GCO_df(date, start_cutoffs, end_cutoffs):
    date_in_GCO = False
    for i in range(len(start_cutoffs)):
        if date >= start_cutoffs[i] and date <= end_cutoffs[i]:
            date_in_GCO = True
    if date_in_GCO:
        return date
    else:
        return None    

    
def clean_data(d):
    df = pd.read_csv(d["file_location"] + d["prepare_input_file"] + ".csv", encoding='latin-1', low_memory=False)
    print("prepare_input_file:", d["prepare_input_file"])
    print("Data read in:", df.shape)

    ####################################################################################################################
    # Filtering for the data MS want
    ####################################################################################################################
    df["ResolvedDate"].fillna(2, inplace=True)  # Remove any cases with no resolved date
    df = df[df["ResolvedDate"] != 2]
    print("Unresolved cases removed:", df.shape)

    df = df[df["Program"] == "Enterprise"]  # Program column: only interested in Enterprise
    df = df[df["LanguageName"] == "English"]  # Only keep the rows which are English
    df = df[df["StatusReason"] != "Rejected"]  # Remove StatusReason = rejected
    df = df[df["ValidCase"] == 1]  # Remove ValidCase = 0
    print("Filtered Program/Language/Status/Valid:", df.shape)
    df.reset_index(drop=True, inplace=True)

    # Change to datetime
    df["Created_On"] = pd.to_datetime(df["Created_On"])
    df["ResolvedDate"] = pd.to_datetime(df["ResolvedDate"])

    ####################################################################################################################
    # IsSox case transformation to filter na's
    ####################################################################################################################
    df["IsSOXCase"].fillna(2, inplace=True)
    df.IsSOXCase = df.IsSOXCase.astype(int)
    df = df[df["IsSOXCase"] != 2]
    df.reset_index(drop=True, inplace=True)
    print("IsSOXCase Filtered", df.shape)
    # if d["remove_null_IsSOXCase"] == "y":
    # else:
        # del df["IsSOXCase"]

    ####################################################################################################################
    # Resample option
    ####################################################################################################################
    if d["resample"] == "y":  # Resample option - select a smaller sample from dataset
        df = resample(df, n_samples=int(d["n_samples"]), random_state=int(d["seed"]))
        df = df.reset_index(drop=True)
        print("Dataset resampled:", df.shape)

    ####################################################################################################################
    # Date and time - calculate time taken
    ####################################################################################################################
    df = time_taken(df)  # Create Time Variable

    ####################################################################################################################
    # Filtering remove_outliers
    ####################################################################################################################
    if d["remove_outliers"] == "y":
        mean_time = sum(df["TimeTaken"].tolist()) / len(df["TimeTaken"])  # Calculate mean of time taken
        std_time = np.std(df["TimeTaken"].tolist())  # Calculate standard deviation of time taken
        df = df[df["TimeTaken"] < (mean_time + 3*std_time)]  # Remove outliers that are > 3 std from mean
        df = df.reset_index(drop=True)
        print("Removed Outliers 3std:", df.shape)

    ####################################################################################################################
    # Last 4 Business Days Only
    ####################################################################################################################    
    if d["last_4_BDays"] == "y":
        end_cutoffs  = get_last_bdays_months()
        start_cutoffs = get_GCO_start_cutoffs(end_cutoffs)
        df["Created_On"] = df["Created_On"].apply(lambda x: get_GCO_df(x, start_cutoffs, end_cutoffs))
        df.dropna(subset=["Created_On"], inplace=True)
        df.reset_index(drop=True, inplace=True)
        print("last_4_BDays done:", df.shape)
        
    ####################################################################################################################
    # Generate Concurrent_open_cases variable.  
    ####################################################################################################################
    if d["Concurrent_open_cases"] == "y":
        df["Concurrent_open_cases"] = 0  # Add number of cases that were open at the same time
        for i in range(len(df)):
            if d["Concurrent_open_cases"] == "y":
                df.loc[i, "Concurrent_open_cases"] = len(df[(df.Created_On < df.iloc[i]["Created_On"]) & (
                                                                        df.ResolvedDate > df.iloc[i]["ResolvedDate"])])
        df.dropna(subset=["TimeTaken"], inplace=True)
        df.reset_index(drop=True, inplace=True)
        print("Concurrent_open_cases added:", df.shape)
        
    ####################################################################################################################
    # Generate Cases_created_within_past_8_hours and/or Cases_resolved_within_past_8_hours variables.      
    ####################################################################################################################
    if d["Cases_created_within_past_8_hours"] == "y" or d["Cases_resolved_within_past_8_hours"] == "y":
        hours_to_search = 8
        if d["Cases_created_within_past_8_hours"] == "y":
            df["Cases_created_within_past_8_hours"] = 0 
            
        if d["Cases_resolved_within_past_8_hours"] == "y":        
            df["Cases_resolved_within_past_8_hours"] = 0 
            
        for i in range(len(df)):    
            if d["Cases_created_within_past_8_hours"] == "y":
                df.loc[i, "Cases_created_within_past_8_hours"] = len(df[(df.Created_On <= df.iloc[i]["Created_On"]) & 
                                    (df.Created_On >= df.iloc[i]["Created_On"]-pd.DateOffset(hours=hours_to_search))])
            
            if d["Cases_resolved_within_past_8_hours"] == "y":    
                df.loc[i, "Cases_resolved_within_past_8_hours"] = len(df[(df.ResolvedDate <= df.iloc[i]["Created_On"]) & 
                                    (df.ResolvedDate >= df.iloc[i]["Created_On"]-pd.DateOffset(hours=hours_to_search))])
        
        if d["Cases_created_within_past_8_hours"] == "y":
            df.dropna(subset=["TimeTaken"], inplace=True)
            df.reset_index(drop=True, inplace=True)
            print("Cases_created_within_past_8_hours added:", df.shape)
        
        if d["Cases_resolved_within_past_8_hours"] == "y": 
            df.dropna(subset=["TimeTaken"], inplace=True)
            df.reset_index(drop=True, inplace=True)
            print("Cases_resolved_within_past_8_hours added:", df.shape)

    ####################################################################################################################
    # Time remaining before Day, Month, Qtr and Year end. 
    ####################################################################################################################
    if d["Seconds_left_Day"] == "y":
        last_bdays_months = get_last_bdays_months()
        daily_cutoffs = get_daily_cutoffs(last_bdays_months)
        df["Seconds_left_Day"] = df["Created_On"].apply(lambda x: int(get_seconds_until_end_of_day(x, daily_cutoffs)))  # Day of the Qtr
        print("Seconds_left_Day added:", df.shape)
        
    if d["Seconds_left_Month"] == "y":
        last_bdays_months = get_last_bdays_months()
        df["Seconds_left_Month"] = df["Created_On"].apply(lambda x: int(get_seconds_left(x, last_bdays_months)))  
        print("Seconds_left_Month added:", df.shape)
    
    if d["Seconds_left_Qtr"] == "y":
        last_bdays_qtr = get_last_bdays_qtr()
        df["Seconds_left_Qtr"] = df["Created_On"].apply(lambda x: int(get_seconds_left(x, last_bdays_qtr)))  
        print("Seconds_left_Qtr added:", df.shape)
    
    if d["Seconds_left_Year"] == "y":
        end_of_year_dates = return_end_of_year()
        df["Seconds_left_Year"] = df["Created_On"].apply(lambda x: int(get_seconds_left(x, end_of_year_dates)))
        print("Seconds_left_Year added:", df.shape)
        
    if d["Created_on_Weekend"] == "y":
        last_bdays = get_last_bdays_months_just_date()
        df["Created_on_Weekend"] = df["Created_On"].apply(lambda x: int(created_on_weekend(x, last_bdays)))
        print("Created_on_Weekend added:", df.shape)
        
    ####################################################################################################################
    # Rolling stats
    ####################################################################################################################
    if d["Rolling_Mean"] == "y" or d["Rolling_Median"] == "y" or d["Rolling_Std"] == "y":
        window = 10
        df = get_rolling_stats(df, window)  # ..Don't want to do this for each of them, because it deletes the first 9 
                                                                                                # entries each time.
        print("All Rolling Stats added:", df.shape)
        
        if d["Rolling_Mean"] != "y":
            del df["Rolling_Mean"]
            
        if d["Rolling_Median"] != "y":
            del df["Rolling_Median"]
            
        if d["Rolling_Std"] != "y":
            del df["Rolling_Std"]
        
        print("Specified Rolling Stats kept:", df.shape)
   
    ####################################################################################################################
    # Combine based on ticket numbers
    ####################################################################################################################
    if d["append_HoldDuration"] == "y":
        dfholdactivity = pd.read_csv("../../../Data/vw_HoldActivity.csv", encoding='latin-1', low_memory=False)
        dfholdactivity["TicketNumber"] = [x.lstrip('5-') for x in dfholdactivity["TicketNumber"]]
        dfholdactivity["TicketNumber"] = dfholdactivity["TicketNumber"].astype(int)
        columns = ["TicketNumber", "HoldDuration", "HoldTypeName", "AssignedToGroup"]
        for col in dfholdactivity:
            if col not in columns: del dfholdactivity[col]

        dfdummies = pd.get_dummies(data=dfholdactivity, columns=["HoldTypeName", "AssignedToGroup"])
        dfdummies["HoldCount"] = 1

        unique_tickets = pd.DataFrame(dfholdactivity["TicketNumber"].unique().tolist(), columns=["TicketNumber"])

        columns_to_count = ["HoldDuration", "HoldTypeName_3rd Party", "HoldTypeName_Customer",
                            "HoldTypeName_Internal",
                            "AssignedToGroup_BPO", "AssignedToGroup_CRMT", "AssignedToGroup_Internal",
                            "AssignedToGroup_Microsoft IT", "AssignedToGroup_Ops. Program Manager",
                            "AssignedToGroup_Submitter (Contact)", "HoldCount"]
        for col in columns_to_count:
            unique_tickets[col] = 0

        for i, row in unique_tickets.iterrows():
            ticket = row["TicketNumber"]
            summed = dfdummies[dfdummies["TicketNumber"] == ticket].sum()

            summed = summed.to_frame().T

            for j in range(len(summed.columns) - 1):
                j += 1
                unique_tickets.iloc[i, j] += summed.iloc[0, j]
        # merge new dfduration df with dfincident based on ticket number
        df = df.merge(right=unique_tickets, how="left", on="TicketNumber")

        for col in columns_to_count:  # fill the NANs with 0's
            df[col].fillna(0, inplace=True)
        print("append_HoldDuration added:", df.shape)

    if d["append_AuditDuration"] == "y":
        dfaudithistory = pd.read_csv("../../../Data/vw_AuditHistory.csv", encoding='latin-1', low_memory=False)
        dfaudithistory["TicketNumber"] = [x.lstrip('5-') for x in dfaudithistory["TicketNumber"]]
        dfaudithistory["TicketNumber"] = dfaudithistory["TicketNumber"].astype(int)
        dfaudithistory["Created_On"] = pd.to_datetime(dfaudithistory["Created_On"])
        dfaudithistory_uniqueticketsonly = pd.DataFrame(dfaudithistory["TicketNumber"].unique(),
                                                        columns=["TicketNumber"])
        dfaudithistory_uniqueticketsonly["AuditDuration"] = None

        for ticket in dfaudithistory_uniqueticketsonly["TicketNumber"].tolist():
            dfaudithistory_uniqueticketsonly.loc[
                dfaudithistory_uniqueticketsonly["TicketNumber"] == ticket, "AuditDuration"] = \
                datetime.timedelta.total_seconds(
                    dfaudithistory.loc[dfaudithistory["TicketNumber"] == ticket, "Created_On"].max() - \
                    dfaudithistory.loc[dfaudithistory["TicketNumber"] == ticket, "Created_On"].min())

        # merge new dfduration df with dfincident based on ticket number
        df = df.merge(dfaudithistory_uniqueticketsonly, how='left', left_on='TicketNumber', right_on='TicketNumber')
        df["AuditDuration"].fillna(0, inplace=True)  # fill the NANs with 0's
        print("append_AuditDuration added:", df.shape)

    ####################################################################################################################
    # Deletions
    ####################################################################################################################
    df = deletions(df, d)

    ####################################################################################################################
    # Ordinal variable mapping. Priority, Complexity, StageName
    ####################################################################################################################
    if d["ordinal_mapping"] == "y":
        df["Priority"] = df["Priority"].map({"Low": 0, "Normal": 1, "High": 2, "Immediate": 3})
        df["Complexity"] = df["Complexity"].map({"Low": 0, "Medium": 1, "High": 2})
        df["StageName"] = df["StageName"].map({"Ops In": 0, "Triage And Validation": 1, "Data Entry": 2, "Submission": 3,
                                               "Ops Out": 4})
        print("Ordinal variable mapping done:", df.shape)

    ####################################################################################################################
    # Transform countries into continents
    ####################################################################################################################
    if d["onehot_encoding"] == "y":
        transform_countrys = ["CountrySource", "CountryProcessed", "SalesLocation"]
        for column in transform_countrys:
            df[column] = transform_country(df[column])
        print("Transformed countries into continents:", df.shape)

    ####################################################################################################################
    # Queue: Prepare for one hot encoding in buckets
    ####################################################################################################################
    if d["onehot_encoding"] == "y":
        substr_list = ["NAOC", "EOC", "AOC", "APOC", "LOC", "Broken", "E&E"]
        check_substr_exists = [False for _ in substr_list]
        # Create a list of 8 unique substrings located in the categorical variables. These will become the new one-hot
        # encoded column names.
        val_list = df.Queue.value_counts().index.tolist()  # List the categorical values in Queue
        cat_list = [[] for _ in substr_list]  # Create a list of 8 lists (the same size as substr_list)
        for i, substr in enumerate(substr_list):
            for j, val in enumerate(val_list):
                if substr in val:  # If one of the 8 substrings is located in a categorical variable, overwrite the
                    # variable with a nonsense value and append the variable name to cat_list
                    val_list[j] = "n"
                    cat_list[i].append(val)
                    check_substr_exists[i] = True
        for i in range(len(substr_list)):
            if check_substr_exists[i] == True:
                df.Queue = df.Queue.replace(cat_list[i], substr_list[i])  # Rplace categorical vars in Queue with substrings
        extra_queues = ["<WWCS - EMEA Admin>", "<3P Xbox AR Operations>", "<VL OpsPM program support>",
                        "<CLT Duplicates>"]
        # combine uncommon queue types
        for extra in extra_queues:
            if extra in df["Queue"].values:
                df["Queue"] = df["Queue"].replace(extra, "Other")
        print("Queue transformation:", df.shape)

    ####################################################################################################################
    # Fill Categorical and numerical nulls. Scale numerical variables.
    ####################################################################################################################
    quant_cols = ["AmountinUSD"]#, "Priority", "Complexity", "StageName"]
    if d["append_HoldDuration"] == "y": quant_cols.append("HoldDuration")
    if d["append_AuditDuration"] == "y":  quant_cols.append("AuditDuration")
    
    if d["Concurrent_open_cases"] == "y": quant_cols.append("Concurrent_open_cases")
    if d["Cases_created_within_past_8_hours"] == "y": quant_cols.append("Cases_created_within_past_8_hours")  
    if d["Cases_resolved_within_past_8_hours"] == "y": quant_cols.append("Cases_resolved_within_past_8_hours")
        
    if d["Seconds_left_Day"] == "y": quant_cols.append("Seconds_left_Day")  
    if d["Seconds_left_Month"] == "y": quant_cols.append("Seconds_left_Month")  
    if d["Seconds_left_Qtr"] == "y": quant_cols.append("Seconds_left_Qtr")  
    if d["Seconds_left_Year"] == "y": quant_cols.append("Seconds_left_Year")  
    
    if d["Rolling_Mean"] == "y": quant_cols.append("Rolling_Mean")  
    if d["Rolling_Median"] == "y": quant_cols.append("Rolling_Median")  
    if d["Rolling_Std"] == "y": quant_cols.append("Rolling_Std")

    do_not_fill = ["TimeTaken", "Created_on_Weekend", "Created_On", "ResolvedDate"]
    categorical_cols = []
    for col in list(df):
        if col not in quant_cols and col not in do_not_fill:
            categorical_cols.append(col)
    fill_nulls_dfcs(df, categorical_cols, "mode")
    print("fill categoricals done:", df.shape)

    # print(len(df["AmountinUSD"]))
    df["AmountinUSD"] = df["AmountinUSD"].apply(lambda x: float(x)) # was getting an error when filling amountinusd
    # here but this fixed it
    # print(len(df["AmountinUSD"]))

    fill_nulls_dfcs(df, quant_cols, "mean")
    # fill_nulls_dfcs(df, ["AmountinUSD", "Priority", "Complexity", "StageName"], "mean")
    print("fill quants done:", df.shape)

    # df = scale_quant_cols(df, quant_cols)
    # print("scale_quant_cols done:", df.shape)
        
    ####################################################################################################################
    # If we only want mandatory data
    ####################################################################################################################
    if d["mandatory_data"] == "y":
        minimum = ["TimeTaken", "Created_On", "ResolvedDate"]
        mandatory = ["Queue", "ROCName", "CountrySource", "CountryProcessed", "SalesLocation"]        
        minimum+=mandatory
        
        if d["append_HoldDuration"] == "y": minimum.append("HoldDuration")
        if d["append_AuditDuration"] == "y": minimum.append("AuditDuration")
            
        if d["Concurrent_open_cases"] == "y": minimum.append("Concurrent_open_cases")
        if d["Cases_created_within_past_8_hours"] == "y": minimum.append("Cases_created_within_past_8_hours")
        if d["Cases_resolved_within_past_8_hours"] == "y": minimum.append("Cases_resolved_within_past_8_hours")
        
        if d["Seconds_left_Day"] == "y": minimum.append("Seconds_left_Day")
        if d["Seconds_left_Month"] == "y": minimum.append("Seconds_left_Month")
        if d["Seconds_left_Qtr"] == "y": minimum.append("Seconds_left_Qtr")
        if d["Seconds_left_Year"] == "y": minimum.append("Seconds_left_Year")
                    
        if d["Created_on_Weekend"] == "y": minimum.append("Created_on_Weekend")                
                   
        if d["Rolling_Mean"] == "y": minimum.append("Rolling_Mean")
        if d["Rolling_Median"] == "y": minimum.append("Rolling_Median")
        if d["Rolling_Std"] == "y": minimum.append("Rolling_Std")
    
        if d["keep_created_resolved"] == "y":
            minimum.append("Created_On")
            minimum.append("ResolvedDate")
    
        for col in df.columns:
            if col not in minimum: del df[col]
        df.reset_index(drop=True, inplace=True)
        print("mandatory_data data only - all other columns deleted", df.shape)

    ####################################################################################################################
    # If we only want minimum data
    ####################################################################################################################
    if d["minimum_data"] == "y":
        minimum = ["TimeTaken", "Created_On", "ResolvedDate"]
        if d["append_HoldDuration"] == "y": minimum.append("HoldDuration")
        if d["append_AuditDuration"] == "y": minimum.append("AuditDuration")
            
        if d["Concurrent_open_cases"] == "y": minimum.append("Concurrent_open_cases")
        if d["Cases_created_within_past_8_hours"] == "y": minimum.append("Cases_created_within_past_8_hours")
        if d["Cases_resolved_within_past_8_hours"] == "y": minimum.append("Cases_resolved_within_past_8_hours")
        
        if d["Seconds_left_Day"] == "y": minimum.append("Seconds_left_Day")
        if d["Seconds_left_Month"] == "y": minimum.append("Seconds_left_Month")
        if d["Seconds_left_Qtr"] == "y": minimum.append("Seconds_left_Qtr")
        if d["Seconds_left_Year"] == "y": minimum.append("Seconds_left_Year")
                    
        if d["Created_on_Weekend"] == "y": minimum.append("Created_on_Weekend")                
                   
        if d["Rolling_Mean"] == "y": minimum.append("Rolling_Mean")
        if d["Rolling_Median"] == "y": minimum.append("Rolling_Median")
        if d["Rolling_Std"] == "y": minimum.append("Rolling_Std")
    
        if d["keep_created_resolved"] == "y":
            minimum.append("Created_On")
            minimum.append("ResolvedDate")
    
        for col in df.columns:
            if col not in minimum: del df[col]
        df.reset_index(drop=True, inplace=True)
        print("Minimum data only - all other columns deleted", df.shape)

    ####################################################################################################################
    # One-hot encode categorical variables
    ####################################################################################################################
    if d["onehot_encoding"] == "y":
        cat_vars_to_one_hot = ["CountrySource", "CountryProcessed", "SalesLocation", "StatusReason", "SubReason",
                               "ROCName", "sourcesystem", "Source", "Revenutype", "Queue"]
        for var in cat_vars_to_one_hot:
            if var in df.columns:
                df = pd.concat([df, pd.get_dummies(df[var], prefix=var, drop_first=True)], axis=1)
                del df[var]
        print("One hot encoding complete:", df.shape)

    ####################################################################################################################
    # If we want to keep created and resolved
    ####################################################################################################################
    if d["keep_created_resolved"] == "n":
        del df["Created_On"]
        del df["ResolvedDate"]
    print("delete_created_resolved:", df.shape)

    ####################################################################################################################
    # Sort columns alphabetically and put TimeTaken first, export file
    ####################################################################################################################
    df = df.reindex_axis(sorted(df.columns), axis=1)
    df = pd.concat([df.pop("TimeTaken"), df], axis=1)
    df.reset_index(drop=True, inplace=True)
    df.to_csv(d["file_location"] + d["prepare_output_file"] + ".csv", index=False)  # export file
    
    # if d["extra_testing"] == "y":
        # dfprejuly = df[df["Created_On"]<pd.datetime(2017,7,1)]
        # dfjuly = df[df["Created_On"]>=pd.datetime(2017,7,1)]
        # dfprejuly.to_csv(d["file_location"] + d["prepare_output_file"] + "_preJuly.csv", index=False)  # export file
        # dfjuly.to_csv(d["file_location"] + d["prepare_output_file"] + "_July.csv", index=False)  # export file
        


if __name__ == "__main__":  # Run program
    print("Cleaning dataset", time.strftime("%Y.%m.%d"), time.strftime("%H.%M.%S"))
    parameters = "../../../Data/parameters.txt"  # Parameters file
    d = {}  # Read in parameters from file
    with open(parameters, "r") as f:
        for line in f:
            line = line.replace(":", "")
            (key, val) = line.split()
            d[key] = val
    clean_data(d)  # Carry out pre-processing
    if d["save_parameters"] == "y":
        copyfile(parameters, "../../../Data/Parameters/" + time.strftime("%Y.%m.%d.%H.%M.%S") + "_parameters.txt")
    print("Cleaned file saved as " + d["file_location"] + d["prepare_output_file"] + ".csv")
    
    if d["beep"] == "y":
        import winsound
        Freq = 400 # Set Frequency To 2500 Hertz
        Dur = 1000 # Set Duration To 1000 ms == 1 second
        winsound.Beep(Freq,Dur)
        Freq = 300 # Set Frequency To 2500 Hertz
        winsound.Beep(Freq,Dur)