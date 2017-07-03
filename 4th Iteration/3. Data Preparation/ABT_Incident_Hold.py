import pandas as pd
COSMIC_num = 2
dfincident = pd.read_csv("../../../Data/COSMIC_%s/vw_Incident%s.csv" % (COSMIC_num, COSMIC_num),
                      encoding='latin-1',
                     low_memory=False)
dfholdactivity = pd.read_csv("../../../Data/COSMIC_%s/vw_HoldActivity%s.csv" % (COSMIC_num, COSMIC_num),
                      encoding='latin-1',
                     low_memory=False)

# Create a new df with the unique Ticket numbers and 0 for hold durations
dfduration = pd.DataFrame(dfholdactivity["TicketNumber"].unique(), columns=["TicketNumber"])
dfduration["HoldDuration"] = 0

uniques = dfholdactivity["TicketNumber"].unique()
# For each of the unique ticket numbers in holdactivity: sum the hold durations and store them next to the equivalent
#  ticket in the new dfduration df
for ticket in uniques:
    duration = dfholdactivity.loc[dfholdactivity["TicketNumber"] == ticket, 'HoldDuration'].sum()
    dfduration.loc[dfduration["TicketNumber"] == ticket, "HoldDuration"] = duration

# merge new dfduration df with dfincident based on ticket number
dfincident = dfincident.merge(dfduration,how='left', left_on='TicketNumber', right_on='TicketNumber')

# fill the NANs with 0's
dfincident["HoldDuration"].fillna(0, inplace=True)

# save the new Analytical Base Table
dfincident.to_csv("../../../Data/COSMIC_%s/ABT_Incident_HoldDuration.csv" % (COSMIC_num), index=False)