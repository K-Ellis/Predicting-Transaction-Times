import pandas as pd

# Convert transactions with multiple ticketnumbers to binaries and merge these into the incident worksheet.
# Need the incident worksheet to contain TicketNumbers, so specify
        # Delete_TicketNumber: y
# in the parameters file for prepare_dataset.py

# given a df with transactions, columns from it to convert
def show_existance_of_variable(dftransactions, cols):
    # create a df made of just the unique ticket numbers
    unique_tickets = dftransactions["TicketNumber"].unique()
    # Create Analytical Base Table
    ABT = pd.DataFrame(unique_tickets, columns=["TicketNumber"])
    # for each column to be examined
    for col in cols:
        # create a zero entry in ABT
        ABT[col] = 0
        # For each ticket number
        for ticket in unique_tickets:
            # find the rows in the df where that ticket number exists and sum the columns values
            exists = dftransactions.loc[dftransactions["TicketNumber"] == ticket, col].sum()
            # if the column has a one in it, it will be > 0
            if exists > 0:
                # therefore in ABT, create a new column with the value one where the row is the same as the ticket
                # number
                ABT.loc[ABT["TicketNumber"] == ticket, col] = 1
            else:
                # otherwise set it to zero
                ABT.loc[ABT["TicketNumber"] == ticket, col] = 0
    # merge the Incident sheet with into the ABT
    # ABT[col].fillna(0, inplace=True)
    # ABT.to_csv("../../../Data/ABT.csv", index=False)
    return ABT

if __name__ == "__main__":

    dfincident = pd.read_csv("../../../Data/vw_Incident_cleaned_with_TicketNumber.csv",
                          encoding='latin-1',
                         low_memory=False)
    dfholdactivity = pd.read_csv("../../../Data/vw_HoldActivity_cleaned.csv",
                          encoding='latin-1',
                         low_memory=False)
    dfaudithistory = pd.read_csv("../../../Data/vw_AuditHistory_cleaned.csv",
                          encoding='latin-1',
                         low_memory=False)
    dfpackagetiageetry = pd.read_csv("../../../Data/vw_PackageTriageEntry_cleaned.csv",
                          encoding='latin-1',
                         low_memory=False)

    # list the columns
    holdactivitycolumns = dfholdactivity.columns.tolist()
    # not included in this are the non-categorical columns
    not_included = ["TicketNumber","ActivityId", "RegardingObjectId", "HoldDuration", "TimeZoneRuleVersionNumber", "HoldTypeName_Customer"]
    for item in not_included:
        holdactivitycolumns.remove(item)
    ABT1 = show_existance_of_variable(dfholdactivity,holdactivitycolumns)
    ABT1.to_csv("../../../Data/ABT1.csv", index=False)

    dfaudithistorycolumns = dfaudithistory.columns.tolist()
    not_included = ["TicketNumber", "AuditHistoryId", "AuditId", "EntityId", "CaseId", "UTCConversionTimeZoneCode",
                    "Created_On", "Modified_On",
                    "Action_Update"]
    for item in not_included:
        dfaudithistorycolumns.remove(item)
    ABT2 = show_existance_of_variable(dfaudithistory, dfaudithistorycolumns)
    ABT2.to_csv("../../../Data/ABT2.csv", index=False)

    packagetiageetrycolumns = dfpackagetiageetry.columns.tolist()
    not_included = ["TicketNumber", "PackageTriageEntryId", "PackageTriageFormId", "Created_On", "Modified_On"]
    for item in not_included:
        packagetiageetrycolumns.remove(item)
    ABT3 = show_existance_of_variable(dfpackagetiageetry, packagetiageetrycolumns)
    ABT3.to_csv("../../../Data/ABT.csv", index=False)

    # merge the dfs together
	finalABT = ABT1.merge(dfincident, how='right', left_on='TicketNumber', right_on='TicketNumber')
	finalABT = ABT2.merge(finalABT, how='right', left_on='TicketNumber', right_on='TicketNumber')
	finalABT = ABT3.merge(finalABT, how='right', left_on='TicketNumber', right_on='TicketNumber')

	del finalABT["TicketNumber"]
	
	finalABT_filled = finalABT.fillna(0)
	finalABT_filled.to_csv("../../../Data/ABT_filled.csv", index=False)
	
	finalABT_dropped = finalABT.dropna()
	finalABT_dropped.to_csv("../../../Data/ABT_dropped.csv", index=False)

	# hold activity info merged with incident
	ABT_HOLD = ABT1.merge(dfincident, how='right', left_on='TicketNumber', right_on='TicketNumber')
	del ABT_HOLD["TicketNumber"]
	ABT_HOLD_dropped = ABT_HOLD.dropna()
	ABT_HOLD_dropped.to_csv("../../../Data/ABT_HOLD_dropped.csv", index=False)
	ABT_HOLD_filled = ABT_HOLD.fillna(0)
	ABT_HOLD_filled.to_csv("../../../Data/ABT_HOLD_filled.csv", index=False)
	
	# audit history info merged with incident
	ABT_AUD = ABT2.merge(dfincident, how='right', left_on='TicketNumber', right_on='TicketNumber')
	del ABT_AUD["TicketNumber"]
	ABT_AUD_dropped = ABT_AUD.dropna()
	ABT_AUD_dropped.to_csv("../../../Data/ABT_AUD_dropped.csv", index=False)
	ABT_AUD_filled = ABT_AUD.fillna(0)
	ABT_AUD_filled.to_csv("../../../Data/ABT_AUD_filled.csv", index=False)
	
	
	# package triage info merged with incident
	ABT_PAC = ABT3.merge(dfincident, how='right', left_on='TicketNumber', right_on='TicketNumber')
	del ABT_PAC["TicketNumber"]
	ABT_PAC_dropped = ABT_PAC.dropna()
	ABT_PAC_dropped.to_csv("../../../Data/ABT_PAC_dropped.csv", index=False)
	ABT_PAC_filled = ABT_PAC.fillna(0)
	ABT_PAC_filled.to_csv("../../../Data/ABT_PAC_filled.csv", index=False)
	
		
	# audit history and hold activity info merged with incident
	ABT_HOLD_AUD = ABT2.merge(dfincident, how='right', left_on='TicketNumber', right_on='TicketNumber')
	ABT_HOLD_AUD = ABT1.merge(ABT_HOLD_AUD, how='right', left_on='TicketNumber', right_on='TicketNumber')
	del ABT_HOLD_AUD["TicketNumber"]
	ABT_HOLD_AUD_dropped = ABT_HOLD_AUD.dropna()
	ABT_HOLD_AUD_dropped.to_csv("../../../Data/ABT_HOLD_AUD_dropped.csv", index=False)
	ABT_HOLD_AUD_filled = ABT_HOLD_AUD.fillna(0)
	ABT_HOLD_AUD_filled.to_csv("../../../Data/ABT_HOLD_AUD_filled.csv", index=False)